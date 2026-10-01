"""T017: opt-in DragFlow execution patches. Off by default; on means recorded as a modified path."""

import copy
import inspect
import json
import os
import sys
import types
from pathlib import Path

import pytest

from advv.backends import dragflow_speedups as sp
from advv.config import validate_config
from advv.errors import ConfigError
from advv.reporting import report_run, speedup_line
from advv.storage import read_json
from conftest import PROJECT, FakeBackend


def all_on(**overrides):
    return {**{key: True for key in sp.SPEEDUP_KEYS}, **overrides}


def only(*names):
    return sp.parse_speedups({name: True for name in names})


class FakeTorch:
    def __init__(self, devices=2):
        self.set_devices = []
        self.grad = True
        self.cuda = types.SimpleNamespace(
            is_available=lambda: True, device_count=lambda: devices, set_device=self.set_devices.append
        )
        self.backends = types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
            cudnn=types.SimpleNamespace(allow_tf32=True),
        )

    def is_grad_enabled(self):
        return self.grad


class Blocks(list):
    pass


class Processor:
    """Same defaulted device parameters as upstream FluxIPAttnProcessor (attn_processor.py:24-37, 137-144)."""

    def __init__(self):
        self.device = "cuda:1"

    def to(self, device):
        self.device = device
        return self

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        emb_dict={},
        subject_emb_dict={},
        device_0="cuda:0",
        device_1="cuda:1",
        *args,
        **kwargs,
    ):
        return device_0, device_1, self._get_ip_hidden_states(attn, hidden_states, None)

    def _get_ip_hidden_states(self, attn, img_query, ip_hidden_states, device_0="cuda:0", device_1="cuda:1"):
        return device_0, device_1


class Transformer:
    def __init__(self, processors=2):
        self.single_transformer_blocks = Blocks(["s0", "s1", "s2"])
        self.gradient_checkpointing = True
        self.attn_processors = {f"block{i}.attn.processor": Processor() for i in range(processors)}

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True


FEATURES = ["DOUBLE-17-0", "DOUBLE-17-2", "DOUBLE-18-0", "DOUBLE-18-2"]


class Dragger:
    device_0, device_1 = "cuda:0", "cuda:1"

    def __init__(self):
        self.conf = {"use_optimizer": False, "use_adapter": True, "target_block_feature_ids_flux": list(FEATURES)}
        self.pipeline = types.SimpleNamespace(transformer=Transformer())
        self.calls = []

    def _extract_latent_features(self, timestep, z, target_inputs):
        self.calls.append(len(self.pipeline.transformer.single_transformer_blocks))
        return "noise_prediction", "features"


def upstream(reclaim=None):
    def official_reclaim():
        return "official"

    modules = {
        name: types.SimpleNamespace(reclaim_memory=reclaim or official_reclaim) for name in sp.RECLAIM_MODULES
    }
    modules["overrider_DiT"].config = {"target_block_feature_ids_flux": list(FEATURES)}
    modules["attn_processor"] = types.SimpleNamespace(FluxIPAttnProcessor=Processor)
    return modules


def latent(requires_grad):
    return types.SimpleNamespace(requires_grad=requires_grad)


def test_parse_defaults_to_official_path_and_rejects_unknown_or_non_bool():
    assert sp.parse_speedups(None) == {key: False for key in sp.SPEEDUP_KEYS}
    assert sp.parse_speedups({"tf32": True})["tf32"] is True
    assert sum(sp.parse_speedups({"tf32": True}).values()) == 1
    for bad in ({"fast": True}, {"tf32": 1}, {"tf32": "yes"}, ["tf32"]):
        with pytest.raises(ConfigError):
            sp.parse_speedups(bad)


def test_example_config_has_every_flag_off_and_old_configs_still_validate(cfg):
    assert cfg["generator"]["speedups"] == {key: False for key in sp.SPEEDUP_KEYS}
    old = copy.deepcopy(cfg)
    del old["generator"]["speedups"]  # runs created before T017 have no speedups key
    validate_config(old)
    bad = copy.deepcopy(cfg)
    bad["generator"]["speedups"]["bf16"] = True
    with pytest.raises(ConfigError, match="Unknown generator.speedups"):
        validate_config(bad)


def test_record_marks_patched_and_precision_variant_paths():
    official = sp.speedup_record(sp.parse_speedups(None))
    assert official == {
        "official_execution_path": True,
        "enabled": [],
        "numerically_equivalent_patches": [],
        "precision_variants": [],
    }
    record = sp.speedup_record(only("skip_unused_single_blocks", "tf32"), FakeTorch())
    assert record["official_execution_path"] is False
    assert record["numerically_equivalent_patches"] == ["skip_unused_single_blocks"]
    assert record["precision_variants"] == ["tf32"]
    assert record["tf32_allowed"] == {"cuda_matmul": False, "cudnn": True}


def test_all_off_installs_nothing():
    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"official path touched dragger.{name}")

    installed = sp.install_speedups(sp.parse_speedups(None), torch=FakeTorch(), dragger=Untouchable(), modules={})
    assert installed.applied == []


def test_single_blocks_are_skipped_only_for_the_grad_forward_and_restored():
    torch, dragger = FakeTorch(), Dragger()
    blocks = dragger.pipeline.transformer.single_transformer_blocks
    installed = sp.install_speedups(only("skip_unused_single_blocks"), torch=torch, dragger=dragger, modules=upstream())
    # F_drag: grad enabled and a latent that requires grad -> no single blocks, no noise prediction.
    assert dragger._extract_latent_features(timestep=0.5, z=latent(True), target_inputs={}) == (None, "features")
    # F_orig runs under no_grad and its noise prediction is used (dragger.py:320-326).
    torch.grad = False
    assert dragger._extract_latent_features(timestep=0.5, z=latent(False), target_inputs={}) == (
        "noise_prediction",
        "features",
    )
    torch.grad = True
    dragger._extract_latent_features(timestep=0.5, z=latent(False), target_inputs={})
    assert dragger.calls == [0, 3, 3]
    assert dragger.pipeline.transformer.single_transformer_blocks is blocks
    installed.undo()
    assert "_extract_latent_features" not in vars(dragger)


def test_single_blocks_restored_when_forward_fails():
    dragger = Dragger()
    blocks = dragger.pipeline.transformer.single_transformer_blocks

    def boom(timestep, z, target_inputs):
        raise RuntimeError("oom")

    extract = sp.skip_single_blocks(boom, dragger, lambda: True)
    with pytest.raises(RuntimeError):
        extract(timestep=0.1, z=latent(True), target_inputs={})
    assert dragger.pipeline.transformer.single_transformer_blocks is blocks


@pytest.mark.parametrize(
    "conf_change, overrider_ids",
    [
        ({"use_optimizer": True}, FEATURES),
        ({"target_block_feature_ids_flux": FEATURES + ["SINGLE-3-0"]}, FEATURES),
        ({}, ["DOUBLE-18-0", "FINAL"]),
    ],
)
def test_single_block_skip_refuses_when_loss_could_use_later_blocks(conf_change, overrider_ids):
    dragger, modules = Dragger(), upstream()
    dragger.conf.update(conf_change)
    modules["overrider_DiT"].config = {"target_block_feature_ids_flux": overrider_ids}
    with pytest.raises(ConfigError, match="skip_unused_single_blocks"):
        sp.install_speedups(only("skip_unused_single_blocks"), torch=FakeTorch(), dragger=dragger, modules=modules)
    assert "_extract_latent_features" not in vars(dragger)


def test_gradient_checkpointing_toggle_and_precondition():
    dragger = Dragger()
    transformer = dragger.pipeline.transformer
    installed = sp.install_speedups(
        only("disable_gradient_checkpointing"), torch=FakeTorch(), dragger=dragger, modules=upstream()
    )
    assert transformer.gradient_checkpointing is False
    installed.undo()
    assert transformer.gradient_checkpointing is True
    transformer.gradient_checkpointing = False
    with pytest.raises(ConfigError):
        sp.install_speedups(only("disable_gradient_checkpointing"), torch=FakeTorch(), dragger=dragger, modules={})


def test_ip_processors_move_to_transformer_device_with_same_signature():
    dragger = Dragger()
    processors = list(dragger.pipeline.transformer.attn_processors.values())
    before = inspect.signature(processors[0].__call__)
    installed = sp.install_speedups(
        only("ip_adapter_on_transformer_device"), torch=FakeTorch(), dragger=dragger, modules=upstream()
    )
    for processor in processors:
        assert processor.device == "cuda:0" and isinstance(processor, Processor)
        assert processor("attn", "q") == ("cuda:0", "cuda:0", ("cuda:0", "cuda:0"))
        # diffusers filters processor kwargs by these parameter names (attention_processor.py:577-586).
        assert list(inspect.signature(processor.__call__).parameters) == list(before.parameters)
        assert processor("attn", "q", device_1="cuda:1")[1] == "cuda:1"
    assert Processor()("attn", "q") == ("cuda:0", "cuda:1", ("cuda:0", "cuda:1"))  # class itself untouched
    installed.undo()
    assert all(type(p) is Processor and p.device == "cuda:1" for p in processors)


def test_ip_processor_patch_refuses_unknown_processors_and_rolls_back_earlier_patches():
    dragger = Dragger()
    dragger.pipeline.transformer.attn_processors["x"] = object()
    flags = only("skip_unused_single_blocks", "disable_gradient_checkpointing", "ip_adapter_on_transformer_device")
    with pytest.raises(ConfigError, match="FluxIPAttnProcessor"):
        sp.install_speedups(flags, torch=FakeTorch(), dragger=dragger, modules=upstream())
    assert "_extract_latent_features" not in vars(dragger)
    assert dragger.pipeline.transformer.gradient_checkpointing is True
    dragger.conf["use_adapter"] = False
    with pytest.raises(ConfigError, match="use_adapter"):
        sp.install_speedups(only("ip_adapter_on_transformer_device"), torch=FakeTorch(), dragger=dragger, modules={})


def test_with_defaults_only_changes_named_defaults():
    def f(a, b=1, *args, c=2, d=3, **kwargs):
        return a, b, c, d

    g = sp.with_defaults(f, b=10)
    assert g(0) == (0, 10, 2, 3) and f(0) == (0, 1, 2, 3)
    assert inspect.signature(g).parameters.keys() == inspect.signature(f).parameters.keys()
    with pytest.raises(ConfigError):
        sp.with_defaults(f, c=5)  # keyword-only defaults are not the upstream pattern


def test_light_reclaim_memory_keeps_only_the_current_device_side_effect():
    torch, modules = FakeTorch(devices=2), upstream()
    installed = sp.install_speedups(only("light_reclaim_memory"), torch=torch, dragger=Dragger(), modules=modules)
    replacements = {modules[name].reclaim_memory for name in sp.RECLAIM_MODULES}
    assert len(replacements) == 1
    assert modules["dragger"].reclaim_memory() is None and torch.set_devices == [1]
    installed.undo()
    assert all(modules[name].reclaim_memory() == "official" for name in sp.RECLAIM_MODULES)
    del modules["pipeline_flux"].reclaim_memory
    with pytest.raises(ConfigError):
        sp.install_speedups(only("light_reclaim_memory"), torch=torch, dragger=Dragger(), modules=modules)
    assert modules["dragger"].reclaim_memory() == "official"


def test_tf32_sets_and_restores_both_flags():
    torch = FakeTorch()
    installed = sp.install_speedups(only("tf32"), torch=torch, dragger=Dragger(), modules=upstream())
    assert torch.backends.cuda.matmul.allow_tf32 is True and torch.backends.cudnn.allow_tf32 is True
    assert sp.speedup_record(only("tf32"), torch)["tf32_allowed"]["cuda_matmul"] is True
    installed.undo()
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is True


def test_all_flags_install_and_undo_in_reverse():
    torch, dragger, modules = FakeTorch(), Dragger(), upstream()
    installed = sp.install_speedups(all_on(), torch=torch, dragger=dragger, modules=modules)
    assert installed.applied == list(sp.SPEEDUP_KEYS)
    installed.undo()
    assert installed.applied == [] and "_extract_latent_features" not in vars(dragger)
    assert dragger.pipeline.transformer.gradient_checkpointing is True
    assert torch.backends.cuda.matmul.allow_tf32 is False


def fake_upstream(monkeypatch, tmp_path):
    """Stand-ins for torch and the pinned upstream modules so DragFlow.__init__ runs on CPU."""
    torch = FakeTorch()
    torch.float32 = "float32"
    loaded = []

    class FakeDragger(Dragger):
        def __init__(self, conf, dtype):
            super().__init__()
            self.conf.update(conf)

        def load_pipeline(self):
            loaded.append(self)

    dashboard = types.SimpleNamespace(HFEmbedder=object, hf_hub_download=None, load_data=None, reclaim_memory=None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "dashboard_utils", dashboard)
    monkeypatch.setitem(sys.modules, "dragger", types.SimpleNamespace(Dragger=FakeDragger, reclaim_memory=None))
    monkeypatch.setitem(sys.modules, "pipeline_flux", types.SimpleNamespace(reclaim_memory=None))
    monkeypatch.setitem(
        sys.modules,
        "overrider_DiT",
        types.SimpleNamespace(reclaim_memory=None, config={"target_block_feature_ids_flux": FEATURES}),
    )
    monkeypatch.setitem(sys.modules, "adapter.attn_processor", types.SimpleNamespace(FluxIPAttnProcessor=Processor))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    lock = tmp_path / "weights.json"
    repos = [
        "black-forest-labs/FLUX.1-dev",
        "google/siglip-so400m-patch14-384",
        "facebook/dinov2-giant",
        "Tencent/InstantCharacter",
    ]
    lock.write_text(json.dumps({"models": {r: {"path": str(tmp_path / r)} for r in repos}}))
    return torch, loaded, lock


def wrapper_cfg(lock, tmp_path, speedups):
    generator = {"repo_path": str(tmp_path), "weights_lock": str(lock), "parameters": {}}
    if speedups is not None:
        generator["speedups"] = speedups
    return {"generator": generator, "_upstream_config": {"lr": 1000.0, "use_optimizer": False}}


def test_wrapper_leaves_official_path_untouched_by_default(monkeypatch, tmp_path):
    from advv.backends.dragflow import DragFlow

    _, loaded, lock = fake_upstream(monkeypatch, tmp_path)
    for speedups in (None, {}, {"tf32": False}):
        flow = DragFlow(wrapper_cfg(lock, tmp_path, speedups), tmp_path)
        assert flow.installed.applied == [] and "_extract_latent_features" not in vars(flow.dragger)
    assert len(loaded) == 3


def test_wrapper_installs_enabled_patches_after_loading(monkeypatch, tmp_path):
    from advv.backends.dragflow import DragFlow

    torch, loaded, lock = fake_upstream(monkeypatch, tmp_path)
    flow = DragFlow(wrapper_cfg(lock, tmp_path, {"skip_unused_single_blocks": True, "tf32": True}), tmp_path)
    assert loaded == [flow.dragger]
    assert flow.installed.applied == ["skip_unused_single_blocks", "tf32"]
    assert "_extract_latent_features" in vars(flow.dragger)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    with pytest.raises(ConfigError):
        DragFlow(wrapper_cfg(lock, tmp_path, {"fast": True}), tmp_path)


def test_report_states_whether_dragflow_ran_the_official_path(cfg):
    from advv.pipeline import Pipeline, create_run

    cfg["generator"]["speedups"]["skip_unused_single_blocks"] = True
    cfg["generator"]["speedups"]["tf32"] = True
    root = create_run(cfg, "speedups", provenance="fake")
    assert read_json(root / "config.json")["generator"]["speedups"]["tf32"] is True
    Pipeline(root, FakeBackend(["YES", "YES"], [(30, 40, 50)])).run()
    report = read_json(report_run(root).with_suffix(".json"))
    assert report["generator_speedups"]["enabled"] == ["skip_unused_single_blocks", "tf32"]
    assert report["generator_speedups"]["precision_variants"] == ["tf32"]
    text = (root / "report.md").read_text()
    assert "**modified**" in text and "Precision variant" in text
    assert speedup_line(sp.speedup_record(sp.parse_speedups(None))).endswith("official path (no speedup patches).")


def test_default_report_says_official_path(cfg):
    from advv.pipeline import Pipeline, create_run

    root = create_run(cfg, "official", provenance="fake")
    Pipeline(root, FakeBackend(["YES", "YES"], [(30, 40, 50)])).run()
    assert read_json(root / "report.json")["generator_speedups"]["official_execution_path"] is True
    assert "official path" in (root / "report.md").read_text()


def test_export_rows_carry_the_execution_path(cfg):
    """T017 QA L5: images.jsonl tells a tf32 or patched candidate apart from the official path."""
    from advv.pipeline import Pipeline, create_run
    from advv.reporting import export_run, records_for
    from advv.storage import atomic_json
    from test_human_review import as_production

    cfg["generator"]["speedups"]["tf32"] = True
    cfg["run"]["target_count"] = 2
    root = create_run(cfg, "export_speedups", provenance="fake")
    Pipeline(root, FakeBackend(["YES"] * 4, [(30, 40, 50), (31, 40, 50)])).run()
    as_production(root)
    first = records_for(root)[0]
    # A record's own generation_info (written by the real backend) wins over the run config.
    first["generation_info"] = {**(first.get("generation_info") or {}), "speedups": sp.speedup_record(only())}
    atomic_json(root / f"records/{first['candidate_id']}.json", first)
    rows = {r["candidate_id"]: r for r in map(json.loads, export_run(root).read_text().splitlines())}
    assert rows.pop(first["candidate_id"])["generator_speedups"] == {
        "official_execution_path": True,
        "enabled": [],
        "tf32": False,
    }
    [(_, row)] = rows.items()
    assert row["generator_speedups"] == {"official_execution_path": False, "enabled": ["tf32"], "tf32": True}


@pytest.mark.integration
def test_real_ip_processor_keeps_signature_and_moves_defaults():
    """Needs the DragFlow venv (torch, diffusers, einops); no GPU or weights."""
    framework = PROJECT / "third_party/DragFlow/framework"
    sys.path.insert(0, str(framework))
    try:
        from adapter.attn_processor import FluxIPAttnProcessor
    finally:
        sys.path.remove(str(framework))
    same = sp.same_device_processor_class(FluxIPAttnProcessor, "cuda:0")
    for name in ("__call__", "_get_ip_hidden_states"):
        original, patched = inspect.signature(getattr(FluxIPAttnProcessor, name)), inspect.signature(getattr(same, name))
        assert list(original.parameters) == list(patched.parameters)
        assert patched.parameters["device_1"].default == "cuda:0"
        assert original.parameters["device_1"].default == "cuda:1"
    assert Path(inspect.getsourcefile(same.__call__)).resolve() == (framework / "adapter/attn_processor.py").resolve()


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    import dragflow_harness

    return dragflow_harness


def test_harness_gpu_and_output_guards(harness, monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    for key in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"):
        monkeypatch.delenv(key, raising=False)
    for bad in ("8", "8,8", "8,9,10", "a,b"):
        with pytest.raises(SystemExit):
            harness.expose_gpus(bad)
    # T017 QA L4: the script never picks GPUs itself; the command line must show them.
    with pytest.raises(SystemExit, match="Set CUDA_VISIBLE_DEVICES=8,9"):
        harness.expose_gpus("8,9")
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(SystemExit, match="disagrees"):
        harness.expose_gpus("8,9")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "8, 9")
    assert harness.expose_gpus("8,9") == ["8", "9"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "8, 9" and os.environ["HF_HUB_OFFLINE"] == "1"


def test_harness_output_must_be_under_workspace_or_explicitly_external(harness, tmp_path):
    """T017 QA L3/L2: allow-list instead of a deny-list; nothing is created for a refused path."""
    probe = "_t017_output_guard_probe"
    for inside in ("src", "tests", "docs", ".git", ".venv-dragflow", "runs", "third_party", "configs", ""):
        target = PROJECT / inside / probe
        with pytest.raises(SystemExit, match="under _workspace/"):
            harness.new_output_dir(target)
        assert not target.exists()
    with pytest.raises(SystemExit, match="under _workspace/"):
        harness.new_output_dir(harness.WORKSPACE)
    with pytest.raises(SystemExit, match="allow-external-out"):
        harness.new_output_dir(tmp_path / "out")
    assert not (tmp_path / "out").exists()
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit, match="must not be inside"):
        harness.new_output_dir(run_dir / "out", allow_external=True, forbid=(run_dir,))
    assert harness.new_output_dir(tmp_path / "out", allow_external=True).is_dir()
    with pytest.raises(SystemExit, match="already exists"):
        harness.new_output_dir(tmp_path / "out", allow_external=True)


def test_harness_output_under_workspace(harness, monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "PROJECT", tmp_path)
    monkeypatch.setattr(harness, "WORKSPACE", tmp_path / "_workspace")
    assert harness.new_output_dir(tmp_path / "_workspace/T018/a").is_dir()
    with pytest.raises(SystemExit, match="under _workspace/"):
        harness.new_output_dir(tmp_path / "src/a")


def test_harness_rounds_stay_in_transport(harness):
    """T017 QA L1: a harness round is one K-loop operation; INTENSIFY (lr reset) starts at max_dragging_num."""
    cfg = {"_upstream_config": {"max_dragging_num": 50, "use_grad_mask": True}, "generator": {"parameters": {}}}
    assert harness.drag_round_limit(cfg) == 50
    harness.check_rounds(cfg, 50, 2)
    for rounds, minimum in ((51, 2), (1, 2), (0, 1)):
        with pytest.raises(SystemExit, match="--rounds"):
            harness.check_rounds(cfg, rounds, minimum)
    cfg["generator"]["parameters"]["max_dragging_num"] = 10
    with pytest.raises(SystemExit, match=r"\[2, 10\]"):
        harness.check_rounds(cfg, 11, 2)
    cfg["_upstream_config"]["use_grad_mask"] = False
    with pytest.raises(SystemExit, match="use_grad_mask"):
        harness.check_rounds(cfg, 5, 2)


def test_harness_arm_flags(harness):
    assert not any(harness.parse_flags("none").values())
    exact = harness.parse_flags("exact_all")
    assert [k for k, v in exact.items() if v] == list(sp.EXACT_SPEEDUPS)
    assert harness.parse_flags("exact_all,tf32") == all_on()
    with pytest.raises(ConfigError):
        harness.parse_flags("bf16")
