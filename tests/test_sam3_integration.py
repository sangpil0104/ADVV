"""GPU integration check of the real SAM 3 proposal worker. Excluded from the default pytest run.

Run only on one GPU whose owner you checked (advv-gpu-safety), e.g.
  ADVV_SAM3_GPU=7 ADVV_SAM3_SUBJECTS=cat ADVV_SAM3_PARTS="head,front leg,tail" \
    python -m pytest -m integration tests/test_sam3_integration.py -s
ADVV_SAM3_SUBJECTS / ADVV_SAM3_PARTS are comma-separated and stand in for a frozen profile's subjects / parts
(an empty ADVV_SAM3_PARTS means no named parts: point prompts only).
Needs .venv-sam3 with `pip install --no-deps -e .`, third_party/sam3 and the pinned checkpoint.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from advv.backends.process import LocalBackend
from advv.config import load_config
from advv.contracts import Source
from advv.pipeline import create_run
from advv.proposals import build_proposals, load_proposals, write_proposals
from advv.storage import read_json
from conftest import PROJECT

pytestmark = pytest.mark.integration


def test_sam3_worker_on_selected_gpu(tmp_path):
    gpu = os.environ.get("ADVV_SAM3_GPU")
    if not gpu:
        pytest.skip("Set ADVV_SAM3_GPU to one GPU you checked is free")
    lock = read_json(PROJECT / "configs/upstream.lock.json")["region_proposal"]["sam3"]
    image = Path(os.environ.get("ADVV_SAM3_IMAGE", PROJECT / "assets/cat_stretched.jpg"))
    subjects = os.environ.get("ADVV_SAM3_SUBJECTS", "cat").split(",")
    parts = [p for p in os.environ.get("ADVV_SAM3_PARTS", "head,front leg,tail").split(",") if p]
    inputs = tmp_path / "input"
    inputs.mkdir()
    shutil.copy(image, inputs / image.name)
    cfg = load_config(PROJECT / "configs/advv.example.yaml", input_dir=inputs, target_count=1, gpus=gpu)
    cfg["run"]["output_root"] = str(tmp_path / "runs")
    cfg["execution"]["proposal_python"] = str(PROJECT / ".venv-sam3/bin/python")
    cfg["region_proposal"].update(
        model_path=str(PROJECT / "models/facebook__sam3" / lock["revision"]),
        model_revision=lock["revision"],
        code_revision=lock["code_commit"],
    )
    root = create_run(cfg, "sam3_integration", provenance="fake")  # proposals only; nothing is exported
    source = Source(**read_json(root / "sources.json")[0])
    backend = LocalBackend(cfg, root)
    try:
        raw = backend.propose(source, subjects, parts, root)
    finally:
        backend.close()
    assert raw["backend"] == "sam3"
    assert raw["models"] == {"facebook/sam3": lock["revision"], "facebookresearch/sam3": lock["code_commit"]}
    assert raw["info"]["load_stdout"] == "" and raw["info"]["peak_vram_bytes"] > 0
    assert raw["entities"], f"SAM 3 found no {subjects} in {image}"
    settings = cfg["region_proposal"]
    built = build_proposals((source.width, source.height), raw, settings)
    profile = {"response": {"subjects": subjects, "parts": parts}, "frozen_sha256": "integration-check"}
    write_proposals(
        root, source, built, backend=raw["backend"], models=raw["models"], settings=settings, profile=profile
    )
    loaded = load_proposals(root, source, settings, profile=profile, allow_fixture=False)
    summary = {
        "image": str(image),
        "run_dir": str(root),
        "raw_entities": [(e["phrase"], round(e["score"], 4), len(e["parts"])) for e in raw["entities"]],
        "raw_text_parts": [
            (i, p["phrase"], round(p["score"], 4), round(p["containment"], 3))
            for i, e in enumerate(raw["entities"])
            for p in e["parts"]
            if p["source"] == "text"
        ],
        "entities": [(e["proposal_id"], e["phrase"], len(e["parts"])) for e in loaded.data["entities"]],
        "parts": [
            (p["proposal_id"], p["source"], p.get("phrase"), round(p["score"], 4))
            for e in loaded.data["entities"]
            for p in e["parts"]
        ],
        "rejected": loaded.data["rejected"],
        "info": {k: v for k, v in raw["info"].items() if k != "load_stdout"},
    }
    print(json.dumps(summary, indent=2))
    assert loaded.data["status"] in ("ready", "no_region")
