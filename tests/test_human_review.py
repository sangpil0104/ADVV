"""Final human pass/fail review after both Qwen checks; FAILs are discarded without regeneration."""

import io
import json
import os

import pytest
import yaml

from advv import cli
from advv.config import validate_config
from advv.errors import ConfigError, DataError
from advv.human_review import FAIL, PASS, QUIT, read_key, review_run
from advv.pipeline import Pipeline, create_run
from advv.reporting import export_run, records_for, report_run
from advv.storage import atomic_json, delete_generated, read_json
from conftest import FakeBackend


def accepted_run(cfg, count=2, run_id="review", review=True):
    cfg["run"]["target_count"] = count
    cfg["human_review"] = {"enabled": review}
    root = create_run(cfg, run_id, provenance="fake")
    colors = [(30, 40 + i, 50) for i in range(count)]
    state = Pipeline(root, FakeBackend(["YES"] * (2 * count), colors)).run()
    assert state["status"] == "completed"
    return root


def as_production(root):
    """Test-only: relabel fixture provenance so the production exporter can be exercised."""
    manifest = read_json(root / "manifest.json")
    manifest["backend"] = "dragflow+qwen_local"
    atomic_json(root / "manifest.json", manifest)
    for record in records_for(root):
        record["backend"] = manifest["backend"]
        atomic_json(root / f"records/{record['candidate_id']}.json", record)


def test_pass_keeps_fail_discards_and_export_has_passed_only(cfg):
    root = accepted_run(cfg)
    kept, dropped = records_for(root)
    result = review_run(root, decisions=[PASS, FAIL], reviewer="tester", out=io.StringIO())
    assert result == {"pass": 1, "fail": 1, "pending": 0}
    kept, dropped = records_for(root)
    assert kept["human_review"]["decision"] == PASS and (root / kept["image_path"]).is_file()
    cid = dropped["candidate_id"]
    assert dropped["export_status"] == "human_rejected" and dropped["image_path"] is None
    assert not dropped["cleanup_pending"]
    assert not (root / f"accepted/{cid}.png").exists()
    assert not (root / f"visualizations/{cid}/comparison.png").exists()
    assert (root / f"visualizations/{cid}/drag_plan.png").is_file()
    assert (root / "review/current.png").is_file()
    # Qwen's YES answers stay on record next to the human decision.
    assert dropped["checks"]["semantic"]["response"]["answer"] == "YES"
    as_production(root)
    rows = [json.loads(line) for line in export_run(root).read_text().splitlines()]
    assert [r["candidate_id"] for r in rows] == [kept["candidate_id"]]
    assert rows[0]["human_review"]["decision"] == PASS
    report = read_json(report_run(root).with_suffix(".json"))
    assert report["human_review"] == {"enabled": True, "pass": 1, "fail": 1, "pending": 0}


def test_unreviewed_images_are_not_exported(cfg):
    root = accepted_run(cfg, count=1)
    as_production(root)
    assert export_run(root).read_text() == ""


def test_quit_pauses_and_next_session_continues(cfg):
    root = accepted_run(cfg)
    assert review_run(root, decisions=[PASS, QUIT], out=io.StringIO())["pending"] == 1
    assert review_run(root, decisions=[], out=io.StringIO())["pending"] == 1  # Exhausted input quits.
    assert review_run(root, decisions=[FAIL], out=io.StringIO()) == {"pass": 1, "fail": 1, "pending": 0}


def test_review_blocks_regeneration_by_resume(cfg):
    root = accepted_run(cfg, count=1)
    review_run(root, decisions=[FAIL], out=io.StringIO())
    backend = FakeBackend()
    with pytest.raises(DataError, match="human review"):
        Pipeline(root, backend).run()
    assert backend.generations == []
    assert read_json(root / "state.json")["status"] == "completed"


def test_review_requires_completed_qwen_stage(cfg):
    cfg["human_review"] = {"enabled": True}
    root = create_run(cfg, "unfinished", provenance="fake")
    with pytest.raises(DataError, match="completes"):
        review_run(root, decisions=[], out=io.StringIO())


def test_changed_accepted_image_is_not_shown(cfg):
    root = accepted_run(cfg, count=1)
    (root / records_for(root)[0]["image_path"]).write_bytes(b"tampered")
    with pytest.raises(DataError, match="changed"):
        review_run(root, decisions=[PASS], out=io.StringIO())
    assert "human_review" not in records_for(root)[0]


def test_failed_deletion_is_retried_on_next_review(cfg):
    root = accepted_run(cfg, count=1)
    path = root / records_for(root)[0]["image_path"]
    link = root / "extra_link.png"
    os.link(path, link)  # Hard links are refused, so deletion must wait.
    review_run(root, decisions=[FAIL], out=io.StringIO())
    record = records_for(root)[0]
    assert record["cleanup_pending"] and "cleanup_error" in record and path.exists()
    link.unlink()
    review_run(root, decisions=[], out=io.StringIO())
    record = records_for(root)[0]
    assert not record["cleanup_pending"] and not path.exists()


def test_accepted_copy_needs_human_fail_to_delete(tmp_path):
    (tmp_path / "accepted").mkdir()
    (tmp_path / "accepted/a.png").write_bytes(b"x")
    with pytest.raises(DataError):
        delete_generated(tmp_path, "a", "accepted/a.png")
    with pytest.raises(DataError):
        delete_generated(tmp_path, "a", "accepted/b.png", human_rejected=True)
    delete_generated(tmp_path, "a", "accepted/a.png", human_rejected=True)
    assert not (tmp_path / "accepted/a.png").exists()


@pytest.mark.parametrize(
    "data, key",
    [(b"\x1b[D", PASS), (b"\x1bOD", PASS), (b"\x1b[C", FAIL), (b"\x1bOC", FAIL), (b"q", QUIT), (b"x", None)],
)
def test_arrow_keys(data, key):
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, data)
        assert read_key(read_fd) == key
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_disabled_keeps_qwen_only_pipeline(cfg):
    assert cfg["human_review"] == {"enabled": False}  # Example config default.
    root = accepted_run(cfg, review=False)
    with pytest.raises(DataError, match="disabled"):
        review_run(root, decisions=[FAIL], out=io.StringIO())
    as_production(root)
    rows = [json.loads(line) for line in export_run(root).read_text().splitlines()]
    assert len(rows) == 2 and all(r["human_review"] is None for r in rows)
    assert read_json(report_run(root).with_suffix(".json"))["human_review"] == {"enabled": False}


def test_runs_without_the_setting_default_to_disabled(cfg):
    del cfg["human_review"]
    root = accepted_run(cfg, count=1, review=False)
    config = read_json(root / "config.json")
    del config["human_review"]
    atomic_json(root / "config.json", config)
    as_production(root)
    assert len(export_run(root).read_text().splitlines()) == 1


@pytest.mark.parametrize("value", ["yes", None, 1])
def test_invalid_setting_rejected(cfg, value):
    cfg["human_review"] = {"enabled": value}
    with pytest.raises(ConfigError, match="human_review"):
        validate_config(cfg)


@pytest.mark.parametrize("flag, expected", [(["--human-review"], True), (["--no-human-review"], False), ([], False)])
def test_cli_flag_fixes_setting_per_run(cfg, tmp_path, monkeypatch, flag, expected):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    class Backend(FakeBackend):
        def __init__(self, resolved, root):
            super().__init__()

    monkeypatch.setattr(cli, "LocalBackend", Backend)
    monkeypatch.setattr(cli, "preflight", lambda *args, **kwargs: {"status": "ready"})
    argv = ["run", "1", "--config", str(config_path), "--gpus", "0,1", "--run-id", "flag", *flag]
    assert cli.main(argv) == 0
    root = tmp_path / "runs/flag"
    assert read_json(root / "config.json")["human_review"] == {"enabled": expected}
    with pytest.raises(SystemExit):
        cli.main(["run", "--run-dir", str(root), "--resume", "--gpus", "0,1", "--human-review"])
