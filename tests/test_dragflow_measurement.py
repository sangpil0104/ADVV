"""T017-fix: measurement scripts keep arm states apart and judge exact arms against the GPU noise floor.

CPU only: a fake Dragger with the upstream call order stands in for DragFlow, and numpy arrays for latents.
"""

import types

import numpy as np
import pytest

from advv.backends import dragflow_speedups as sp
from conftest import PROJECT


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    import dragflow_harness

    return dragflow_harness


@pytest.fixture
def eq(harness):
    import check_dragflow_equivalence

    return check_dragflow_equivalence


class OOM(RuntimeError):
    pass


class Latent:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def __float__(self):
        return float(self.value)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self.value


class Torch:
    float32 = "float32"

    def __init__(self):
        self.cuda = types.SimpleNamespace(
            device_count=lambda: 2,
            synchronize=lambda device: None,
            reset_peak_memory_stats=lambda device: None,
            max_memory_allocated=lambda device: 1,
            max_memory_reserved=lambda device: 2,
            manual_seed_all=lambda seed: None,
            empty_cache=lambda: None,
            OutOfMemoryError=OOM,
        )
        self.backends = types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
            cudnn=types.SimpleNamespace(allow_tf32=True),
        )

    def manual_seed(self, seed):
        pass

    def is_grad_enabled(self):
        return True


class Functional:
    def l1_loss(self, a, b):
        return Latent(abs(a - b))


class Dragger:
    """Upstream call order of Dragger.__call__ up to the K-loop (dragger.py:297-482), with a hook to fail."""

    def __init__(self, conf, fail=None, operations=70):
        self.conf, self.fail, self.operations = conf, fail, operations
        self.pipeline = types.SimpleNamespace(transformer=types.SimpleNamespace(gradient_checkpointing=True))

    def process_inversion(self, source_inputs):
        return Latent([1.0, 2.0])

    def dragger_step(self):
        for index in range(self.operations):
            if index >= self.conf["max_dragging_num"]:
                self.conf["lr"] = 1000.0
            if self.fail == index:
                raise self.error
            self._extract_latent_features(timestep=0, z=types.SimpleNamespace(requires_grad=True), target_inputs=None)
            self.modules["dragger"].F.l1_loss(3.0, 1.0 + index)
            self._combine_latents(z=index)

    def _extract_latent_features(self, timestep, z, target_inputs):
        return None, None

    def _combine_latents(self, z):
        return Latent([float(z), 0.5])

    def __call__(self, raw, instruction, image_name):
        self.process_inversion(source_inputs=raw)
        self.conf["lr"] = 7.0  # any in-run change must not leak into the next arm
        self.dragger_step()


def flow_for(dragger):
    return types.SimpleNamespace(
        conf=dragger.conf, dragger=dragger, load_data=lambda folder, out, device, dtype: ("raw", "instruction")
    )


def setup(tmp_path, fail=None, error=None, operations=70):
    modules = {"dragger": types.SimpleNamespace(F=Functional())}
    dragger = Dragger({"use_grad_mask": True, "max_dragging_num": 50, "lr": 1000.0, "seed": 0}, fail, operations)
    dragger.modules, dragger.error = modules, error
    candidate = {"seed": 3, "input_folder": tmp_path, "candidate": "c"}
    return flow_for(dragger), modules, candidate


def test_run_arm_completes_and_restores_conf_and_patches(harness, tmp_path):
    flow, modules, candidate = setup(tmp_path)
    torch = Torch()
    result = harness.run_arm(torch, flow, modules, candidate, tmp_path, 3, sp.parse_speedups({"tf32": True}))
    assert result["status"] == "completed" and result["applied"] == ["tf32"]
    assert result["losses"] == [2.0, 1.0, 0.0] and len(result["_latents"]) == 3
    assert flow.conf == {"use_grad_mask": True, "max_dragging_num": 50, "lr": 1000.0, "seed": 0}
    assert torch.backends.cuda.matmul.allow_tf32 is False and isinstance(modules["dragger"].F, Functional)
    assert "_combine_latents" not in vars(flow.dragger)


@pytest.mark.parametrize(
    "error, status, message",
    [(OOM("CUDA out of memory"), "oom", "out of memory"), (ValueError("bad shape"), "error", "ValueError: bad shape")],
)
def test_run_arm_records_oom_and_other_errors_without_raising(harness, tmp_path, error, status, message):
    flow, modules, candidate = setup(tmp_path, fail=1, error=error)
    torch = Torch()
    result = harness.run_arm(torch, flow, modules, candidate, tmp_path, 3, sp.parse_speedups({"tf32": True}))
    assert result["status"] == status and message in result["error"]
    assert len(result["_latents"]) == 1 and flow.conf["lr"] == 1000.0
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_run_arm_reports_failed_install_and_early_finish_as_errors(harness, tmp_path):
    flow, modules, candidate = setup(tmp_path)
    flow.dragger.pipeline.transformer.gradient_checkpointing = False
    flags = sp.parse_speedups({"disable_gradient_checkpointing": True})
    result = harness.run_arm(Torch(), flow, modules, candidate, tmp_path, 3, flags)
    assert result["status"] == "error" and "ConfigError" in result["error"] and result["applied"] == []
    flow, modules, candidate = setup(tmp_path, operations=2)
    result = harness.run_arm(Torch(), flow, modules, candidate, tmp_path, 3, sp.parse_speedups(None))
    assert result["status"] == "error" and "finished after 2 of 3" in result["error"]


SETTINGS = {"rtol": 1e-5, "noise_multiplier": 10.0, "tf32_loss_rtol": 1e-2, "tf32_latent_rtol": 5e-2}


def arm(harness, status="completed", scale=0.0, flags="none"):
    latents = [np.array([1.0 + scale, 2.0]), np.array([3.0, 4.0 - scale])]
    return {
        "status": status,
        "speedups": sp.speedup_record(harness.parse_flags(flags)),
        "losses": [2.0 * (1 + scale), 1.0],
        "_latents": latents if status == "completed" else latents[:1],
        "_z_inverted": np.array([10.0, 20.0]) if status == "completed" else None,
        "round_seconds": [5.0, 2.0, 2.0],
        **({} if status == "completed" else {"error": "CUDA out of memory"}),
    }


def judge(eq, results, planned=None):
    return eq.build_report(results, planned or list(results), SETTINGS)


def test_noise_floor_sets_exact_tolerance_and_control_is_not_judged(eq, harness):
    results = {
        "baseline": arm(harness),
        "control_baseline": arm(harness, scale=1e-5),
        "skip_unused_single_blocks": arm(harness, scale=5e-5, flags="skip_unused_single_blocks"),
    }
    report = judge(eq, results)
    floor = report["noise_floor"]
    assert floor["source"] == "control_baseline" and floor["loss_relative"] == pytest.approx(1e-5)
    # max(rtol, 10 x floor): the loss tolerance follows the floor, the latent tolerance stays at rtol or above.
    assert report["exact_tolerance"]["loss_relative"] == pytest.approx(1e-4)
    assert report["exact_tolerance"]["latent_relative"] >= 1e-5
    assert report["arms"]["control_baseline"]["judged_as"] == "noise_floor"
    assert "control_baseline" not in report["exact_arms_outside_tolerance"] + report["exact_arms_within_tolerance"]
    assert report["exact_arms_within_tolerance"] == ["skip_unused_single_blocks"]
    assert report["exit_code"] == 0 and report["finished"] is True


def test_without_control_exact_arms_use_rtol_alone(eq, harness):
    results = {"baseline": arm(harness), "exact_all": arm(harness, scale=5e-5, flags="exact_all")}
    report = judge(eq, results)
    assert report["noise_floor"] is None and "--no-control" in report["noise_floor_note"]
    assert report["exact_arms_outside_tolerance"] == ["exact_all"] and report["exit_code"] == 2


def test_oom_arm_is_incomplete_not_a_numerical_difference(eq, harness):
    """T017 QA M1: an OOM exact arm used to land in exact_arms_outside_tolerance with exit 2."""
    results = {
        "baseline": arm(harness),
        "control_baseline": arm(harness),
        "exact_all": arm(harness, status="oom", flags="exact_all"),
    }
    report = judge(eq, results)
    assert report["exact_arms_outside_tolerance"] == []
    assert report["incomplete_arms"] == {"exact_all": "oom"}
    assert "comparison" not in report["arms"]["exact_all"] and report["arms"]["exact_all"]["error"]
    assert report["exit_code"] == 3


def test_numerical_difference_wins_over_incomplete_and_tf32_never_decides(eq, harness):
    results = {
        "baseline": arm(harness),
        "control_baseline": arm(harness),
        "exact_all": arm(harness, scale=1e-3, flags="exact_all"),
        "light_reclaim_memory": arm(harness, status="error", flags="light_reclaim_memory"),
        "exact_all,tf32": arm(harness, scale=0.5, flags="exact_all,tf32"),
    }
    report = judge(eq, results)
    assert report["exact_arms_outside_tolerance"] == ["exact_all"] and report["exit_code"] == 2
    assert report["precision_variant_arms_outside_diagnostic_tolerance"] == ["exact_all,tf32"]
    del results["exact_all"]
    assert judge(eq, results)["exit_code"] == 3


def test_failed_baseline_still_reports_other_arms(eq, harness):
    results = {"baseline": arm(harness, status="oom"), "exact_all": arm(harness, flags="exact_all")}
    report = judge(eq, results)
    assert report["exit_code"] == 4 and report["incomplete_arms"] == {"baseline": "oom"}
    assert report["arms"]["exact_all"]["speed"]["round_seconds"] == [5.0, 2.0, 2.0]
    assert "comparison" not in report["arms"]["exact_all"]


def test_partial_report_while_arms_are_pending(eq, harness):
    report = judge(eq, {"baseline": arm(harness)}, ["baseline", "control_baseline", "exact_all"])
    assert report["finished"] is False and report["exit_code"] is None
    assert report["pending_arms"] == ["control_baseline", "exact_all"]


def test_usage_errors_do_not_use_exit_code_2(eq):
    with pytest.raises(SystemExit) as raised:
        eq.build_parser().parse_args(["--gpus", "8,9"])
    assert raised.value.code not in (0, 2)
