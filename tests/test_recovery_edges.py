import copy
from dataclasses import replace

import pytest
from PIL import Image

from advv.config import load_config
from advv.errors import DataError
from advv.ingest import scan_sources
from advv.pipeline import Pipeline, create_run
from advv.sampler import sample_edit, validate_plan
from advv.storage import atomic_json, read_json, run_lock
from conftest import FakeBackend, PROJECT


def test_normal_no_sequence_has_no_attempt_cap(cfg):
    cfg["visualization"]["enabled"] = False
    backend = FakeBackend(["NO"] * 15, [(i, 2, 3) for i in range(15)] + [KeyboardInterrupt()])
    root = create_run(cfg, "many_no", provenance="fake")
    state = Pipeline(root, backend).run()
    assert state["status"] == "interrupted" and state["attempts"] == 16
    assert state["accepted_count"] == 0


def test_consecutive_technical_errors_stop_but_do_not_count_as_no(cfg):
    cfg["visualization"]["enabled"] = False
    backend = FakeBackend(["invalid"] * 6, [(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    root = create_run(cfg, "errors", provenance="fake")
    state = Pipeline(root, backend).run()
    assert state["status"] == "failed" and state["attempts"] == 3
    assert "Consecutive technical" in state["last_error"]["message"]


def test_changed_cfg_blocks_resume(cfg):
    root = create_run(cfg, "changed", provenance="fake")
    changed = copy.deepcopy(cfg)
    changed["run"]["seed"] += 1
    atomic_json(root / "config.json", changed)
    with pytest.raises(DataError, match="configuration changed"):
        Pipeline(root, FakeBackend())


def test_two_writers_blocked(cfg):
    root = create_run(cfg, "locking", provenance="fake")
    with run_lock(root):
        with pytest.raises(DataError, match="Another writer"):
            with run_lock(root):
                pytest.fail("Second lock must not be acquired")


def test_exif_normalization_before_coordinates(cfg, tmp_path):
    image = Image.new("RGB", (80, 120), (20, 80, 130))
    exif = Image.Exif()
    exif[274] = 6
    image.save(tmp_path / "input/rotated.jpg", exif=exif)
    sources = scan_sources(cfg)[0]
    source = [s for s in sources if s.original_path.endswith("rotated.jpg")][0]
    assert (source.width, source.height) == (120, 80)
    plan = sample_edit(source, {"summary": "subject"}, cfg, 0, tmp_path / "plans")
    validate_plan(plan, source, tmp_path / "plans")


def test_feature_grid_rejects_rounded_outside(cfg, tmp_path):
    source = scan_sources(cfg)[0][0]
    plan = sample_edit(source, {"summary": "subject"}, cfg, 0, tmp_path / "plans")
    plan = replace(plan, target_point=[source.width - 0.01, source.height - 0.01])
    with pytest.raises(DataError, match="feature grid"):
        validate_plan(plan, source, tmp_path / "plans")


def test_python_venv_symlink_is_not_resolved(cfg, tmp_path):
    import yaml

    folder = tmp_path / "env/bin"
    folder.mkdir(parents=True)
    (folder / "python").symlink_to("/usr/bin/python3")
    data = yaml.safe_load((PROJECT / "configs/advv.example.yaml").read_text())
    data["execution"]["generator_python"] = str(folder / "python")
    # Keep referenced prompts relative to the real config directory.
    config_path = PROJECT / "configs/advv.example.yaml"
    value = load_config(config_path, input_dir=tmp_path / "input", target_count=1)
    from advv.config import PATH_FIELDS

    assert "generator_python" in PATH_FIELDS["execution"]
    data = value
    data.pop("_assets")
    data["execution"]["generator_python"] = str(folder / "python")
    custom = tmp_path / "config.yaml"
    custom.write_text(yaml.safe_dump(data))
    loaded = load_config(custom)
    assert loaded["execution"]["generator_python"] == str(folder / "python")


def test_effective_geometry_coordinate_inverse():
    from advv.backends.dragflow import to_original, upstream_instruction

    assert to_original([8, 4], [16, 8], [128, 64]) == [64, 32]
    value = upstream_instruction(
        {
            "operation": "relocation",
            "source_point": [10, 12],
            "target_point": [20, 25],
            "anchor_point": None,
            "source_prompt": "source",
            "target_prompt": "edit",
        }
    )
    assert value["region_operations"]["0"]["task"] == "transformation"


def test_uncertain_skips_semantic_and_retains_quarantine(cfg):
    backend = FakeBackend(["UNCERTAIN", "YES", "YES"], [(1, 2, 3), (4, 5, 6)])
    root = create_run(cfg, "uncertain", provenance="fake")
    state = Pipeline(root, backend).run()
    assert state["status"] == "completed"
    from advv.reporting import records_for

    record = records_for(root)[0]
    assert record["checks"]["semantic"]["status"] == "not_run"
    assert record["image_path"].startswith("quarantine/")
    assert (root / record["image_path"]).is_file()
    assert read_json(root / "report.json")["status_counts"]["uncertain"] == 1


def test_interrupted_accepted_copy_recovers_without_reinference(cfg, monkeypatch):
    import advv.pipeline as module

    real_write = module.atomic_json

    def interrupt_before_commit(path, value):
        if path.parent.name == "records" and value.get("finalized") and value["status"] == "accepted":
            raise KeyboardInterrupt()
        real_write(path, value)

    root = create_run(cfg, "accepted_crash", provenance="fake")
    monkeypatch.setattr(module, "atomic_json", interrupt_before_commit)
    assert Pipeline(root, FakeBackend()).run()["status"] == "interrupted"
    assert len(list((root / "accepted").glob("*.png"))) == 1
    monkeypatch.setattr(module, "atomic_json", real_write)
    backend = FakeBackend([], [])
    state = Pipeline(root, backend).run()
    assert state["status"] == "completed" and state["accepted_count"] == 1
    assert backend.generations == [] and backend.calls == []


def test_owned_atomic_image_temporary_cleanup(tmp_path):
    from advv.storage import delete_generated, generated_temporary_paths

    folder = tmp_path / "candidates/candidate_1"
    folder.mkdir(parents=True)
    transient = folder / ".generated.png.123abcd"
    transient.write_bytes(b"uncommitted generated image")
    paths = generated_temporary_paths(tmp_path, "candidate_1")
    assert len(paths) == 1
    delete_generated(tmp_path, "candidate_1", paths[0])
    assert not transient.exists()
