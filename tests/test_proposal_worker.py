"""Region proposal worker: process contract, receipts and error paths with a fake SAM 3 (CPU only)."""

import copy
import os
import sys

import numpy as np
import pytest

import advv.backends.process as process
from advv.backends.process import LocalBackend
from advv.contracts import Source
from advv.errors import BackendError, ConfigError, FatalBackendError
from advv.pipeline import Pipeline, create_run
from advv.storage import read_json
from conftest import PROJECT, FakeBackend

FAKE_SAM3 = PROJECT / "tests/fixtures/fake_sam3"
SUBJECTS = ["kitten", "box"]
# Only the kitten has named parts: the box is never asked for a tail or head (v4 per-subject parts).
PARTS = [{"subject": "kitten", "parts": ["tail", "head"]}]


def rect(x0, y0, x1, y1):
    mask = np.zeros((80, 120), bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


@pytest.fixture
def worker_cfg(v2cfg, tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam3_weights"
    checkpoint.mkdir()
    (checkpoint / "sam3.pt").write_bytes(b"fixture checkpoint")
    cfg = copy.deepcopy(v2cfg)
    cfg["region_proposal"].update(model_path=str(checkpoint), model_revision="a" * 40, code_revision="b" * 40)
    cfg["execution"].update(proposal_python=sys.executable, worker_timeout_seconds=60)
    # The worker subprocess imports the fake torch/sam3 instead of real ones.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(FAKE_SAM3), os.environ.get("PYTHONPATH")])))
    monkeypatch.delenv("ADVV_FAKE_SAM3", raising=False)
    return cfg


def new_run(cfg, run_id):
    root = create_run(cfg, run_id, provenance="fake")
    return root, Source(**read_json(root / "sources.json")[0])


class WorkerBackend(FakeBackend):
    """Fake DragFlow/Qwen, real proposal worker process (with the fake SAM 3 package)."""

    def __init__(self, cfg, root, **kwargs):
        super().__init__(subjects=SUBJECTS, parts=PARTS, **kwargs)
        self.local = LocalBackend(cfg, root)
        self.events = []

    def propose(self, source, subjects, parts, run_dir):
        self.events.append("propose")
        return self.local.propose(source, subjects, parts, run_dir)

    def generate(self, source, plan, run_dir):
        self.events.append("generate")
        return super().generate(source, plan, run_dir)

    def close(self):
        self.events.append("close")
        self.local.close()
        super().close()


def test_worker_receipt_feeds_frozen_proposals(worker_cfg):
    root, source = new_run(worker_cfg, "worker")
    backend = WorkerBackend(worker_cfg, root)
    state = Pipeline(root, backend).run()
    assert state["status"] == "completed" and state["accepted_count"] == 1
    sid = source.source_id
    folder = root / f"proposals/{sid}"
    data = read_json(folder / "proposals.json")
    # The fake package marks itself, so its masks can never be taken for real SAM 3 output.
    assert data["backend"] == "fixture" and data["subjects"] == SUBJECTS and data["parts"] == PARTS
    assert data["models"] == {"facebook/sam3": "a" * 40, "facebookresearch/sam3": "b" * 40}
    assert [e["phrase"] for e in data["entities"]] == ["kitten", "box"]
    assert data["entities"][0]["parts"] and data["rejected"]["part_score"] > 0
    assert len(data["entities"]) == 2  # The 0.2 box never leaves SAM (confidence_threshold).
    # The named tail comes first, with its phrase; "tail" alone finds the same mask (duplicate) and a tail
    # outside every entity (containment); "kitten head" passes SAM 3's lowered threshold but not ADVV's.
    kitten_parts = data["entities"][0]["parts"]
    assert kitten_parts[0]["source"] == "text" and kitten_parts[0]["phrase"] == "kitten tail"
    assert kitten_parts[0]["bbox"] == [20, 40, 35, 59]
    # The kitten keeps a named part, so its point parts are suppressed; the box has none and keeps points.
    assert len(kitten_parts) == 1
    assert data["entities"][1]["parts"] and all(
        p["source"] == "point" and "point" in p for p in data["entities"][1]["parts"]
    )
    rejected = data["rejected"]
    assert rejected["text_part_duplicate"] == 1 and rejected["text_part_score"] == 1
    assert rejected["text_part_containment"] == 1  # the tail outside the kitten; the box asks for no part
    raw = read_json(folder / "receipt/raw.json")
    assert raw["backend"] == "fixture" and raw["source_pixel_sha256"] == source.pixel_sha256
    assert raw["parts"] == PARTS and raw["info"]["confidence_threshold"] == pytest.approx(0.49)
    parts = [p for p in raw["entities"][0]["parts"] if p["source"] == "point"]
    assert len(parts) == 3 * len({tuple(p["point"]) for p in parts})
    assert all((p["mask"] is None) == (p["score"] < 0.8) for p in parts)
    assert rejected["point_suppressed_by_text"] == len(parts)  # the receipt still holds every point prompt
    text = [p for p in raw["entities"][0]["parts"] if p["source"] == "text"]
    assert [(p["phrase"], p["instance_index"]) for p in text] == [
        ("kitten tail", 0),
        ("tail", 0),
        ("tail", 1),
        ("kitten head", 0),
    ]
    assert [p["mask"] is None for p in text] == [False, False, True, True]
    assert [p["containment"] for p in text] == [1.0, 1.0, 0.0, 1.0]
    # The kitten's part names are never asked for the box (v2_pilot_001 asked "tree front bumper").
    assert [e["phrase"] for e in raw["entities"]] == ["kitten", "box"]
    assert not [p for p in raw["entities"][1]["parts"] if p["source"] == "text"]
    response = read_json(folder / "receipt/response.json")
    assert response["ok"] and response["result"]["raw_sha256"]
    assert state["region_proposal_generation"][sid][-1]["status"] == "completed"
    assert state["segments"][-1]["proposal_gpu_ids"] == ["0"]
    assert (root / "logs/proposal.log").exists()
    # One proposal pass per source, and the proposal worker is gone before DragFlow starts.
    assert backend.events.count("propose") == 1
    assert "close" in backend.events[backend.events.index("propose") : backend.events.index("generate")]
    assert backend.local.worker is None


def test_finished_receipt_is_reused_without_a_worker(worker_cfg, monkeypatch):
    root, source = new_run(worker_cfg, "reuse")
    first = LocalBackend(worker_cfg, root)
    raw = first.propose(source, SUBJECTS, PARTS, root)
    first.close()

    def no_worker(*args, **kwargs):
        raise AssertionError("receipt should be reused")

    monkeypatch.setattr(process, "Worker", no_worker)
    again = LocalBackend(worker_cfg, root).propose(source, SUBJECTS, PARTS, root)
    assert [e["score"] for e in again["entities"]] == [e["score"] for e in raw["entities"]]
    assert all(np.array_equal(a["mask"], b["mask"]) for a, b in zip(again["entities"], raw["entities"]))
    with pytest.raises(AssertionError, match="reused"):
        LocalBackend(worker_cfg, root).propose(source, ["kitten"], PARTS, root)  # other phrases: new request
    with pytest.raises(AssertionError, match="reused"):
        LocalBackend(worker_cfg, root).propose(source, SUBJECTS, [{"subject": "kitten", "parts": ["tail"]}], root)  # other parts: new request


@pytest.mark.parametrize(
    "mode, error, message",
    [
        ("missing_keys", FatalBackendError, "did not load cleanly"),
        ("no_cuda", FatalBackendError, "CUDA"),
        ("text_error", FatalBackendError, "RuntimeError: fake SAM 3 text failure"),
    ],
)
def test_worker_failures_are_fatal_and_receipted(worker_cfg, monkeypatch, mode, error, message):
    monkeypatch.setenv("ADVV_FAKE_SAM3", mode)
    root, source = new_run(worker_cfg, f"fail_{mode}")
    backend = LocalBackend(worker_cfg, root)
    with pytest.raises(error, match=message):
        backend.propose(source, SUBJECTS, PARTS, root)
    backend.close()
    response = read_json(root / f"proposals/{source.source_id}/receipt/response.json")
    assert not response["ok"] and response["error"]["fatal"]
    assert not (root / f"proposals/{source.source_id}/receipt/raw.json").exists()


def test_worker_timeout_is_a_retryable_error(worker_cfg, monkeypatch):
    monkeypatch.setenv("ADVV_FAKE_SAM3", "hang")
    worker_cfg["execution"]["worker_timeout_seconds"] = 2
    root, source = new_run(worker_cfg, "hang")
    backend = LocalBackend(worker_cfg, root)
    with pytest.raises(BackendError, match="timeout") as caught:
        backend.propose(source, SUBJECTS, PARTS, root)
    assert not isinstance(caught.value, FatalBackendError) and backend.worker is None


def test_comparison_backend_is_not_implemented(worker_cfg):
    worker_cfg["region_proposal"]["backend"] = "grounding_dino_sam2.1"
    root, source = new_run(worker_cfg, "gdino")
    backend = LocalBackend(worker_cfg, root)
    with pytest.raises(FatalBackendError, match="grounding_dino_sam2.1 .*not implemented"):
        backend.propose(source, SUBJECTS, PARTS, root)
    backend.close()


def test_missing_proposal_python_is_a_config_error(worker_cfg):
    worker_cfg["execution"]["proposal_python"] = None
    root, source = new_run(worker_cfg, "nopython")
    with pytest.raises(ConfigError, match="proposal_python"):
        LocalBackend(worker_cfg, root).propose(source, SUBJECTS, PARTS, root)


RAW = {
    "phrases": ["kitten"],
    "entities": [
        {"phrase": "kitten", "score": 0.9, "mask": rect(20, 20, 59, 59), "parts": []},
    ],
}


def test_failed_generation_is_missing_not_no_region_and_resumes(v2cfg):
    root = create_run(v2cfg, "gen_fail", provenance="fake")
    sid = read_json(root / "sources.json")[0]["source_id"]
    backend = FakeBackend(subjects=["kitten"], proposals=BackendError("worker hiccup"))
    state = Pipeline(root, backend).run()
    assert state["status"] == "failed" and "region_proposals_missing" in state["last_error"]["message"]
    assert "worker hiccup" in state["last_error"]["message"]
    attempts = state["region_proposal_generation"][sid]
    assert [a["status"] for a in attempts] == ["error", "error"]  # 1 + max_technical_retries
    assert "region_proposals" not in state or sid not in state["region_proposals"]
    assert backend.generations == [] and not (root / f"proposals/{sid}/proposals.json").exists()
    backend = FakeBackend(subjects=["kitten"], proposals=RAW)
    state = Pipeline(root, backend).run()
    assert state["status"] == "completed" and len(backend.proposal_calls) == 1
    assert state["region_proposals"][sid]["status"] == "ready"
    backend = FakeBackend(subjects=["kitten"], proposals=BackendError("must not be called"))
    state = Pipeline(root, backend).run()  # Frozen proposals are loaded, never regenerated.
    assert state["status"] == "completed" and backend.proposal_calls == []


def test_fatal_generation_error_stops_at_once(v2cfg):
    root = create_run(v2cfg, "gen_fatal", provenance="fake")
    backend = FakeBackend(subjects=["kitten"], proposals=FatalBackendError("CUDA OOM"))
    state = Pipeline(root, backend).run()
    assert state["status"] == "failed" and state["last_error"]["type"] == "FatalBackendError"
    assert len(backend.proposal_calls) == 1


def test_generated_empty_proposals_hold_the_source(v2cfg):
    raw = {"phrases": ["kitten"], "entities": [{"phrase": "kitten", "score": 0.1, "mask": rect(0, 0, 9, 9)}]}
    root = create_run(v2cfg, "gen_empty", provenance="fake")
    sid = read_json(root / "sources.json")[0]["source_id"]
    backend = FakeBackend(subjects=["kitten"], proposals=raw)
    state = Pipeline(root, backend).run()
    assert state["status"] == "failed" and "no_eligible_sources" in state["last_error"]["message"]
    assert state["region_proposals"][sid]["status"] == "source_no_region"
    assert read_json(root / f"proposals/{sid}/proposals.json")["status"] == "no_region"
    assert backend.generations == []
