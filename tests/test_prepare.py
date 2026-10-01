"""advv prepare --only sam3 and the preflight weight check, without downloads."""

import copy
import hashlib
import json
import sys
import types

import pytest

from advv.errors import ConfigError
from advv.prepare import prepare, verify_sam3
from advv.storage import read_json
from conftest import PROJECT

REVISION = "3" * 40
FILES = {"sam3.pt": b"weights", "config.json": b"{}"}


def lock_entry():
    return {
        "code_repo": "https://github.com/facebookresearch/sam3.git",
        "code_commit": "2" * 40,
        "model_id": "facebook/sam3",
        "revision": REVISION,
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in FILES.items()
        },
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    lock = json.loads((PROJECT / "configs/upstream.lock.json").read_text())
    assert lock["region_proposal"]["sam3"]["revision"] == "3c879f39826c281e95690f02c7821c4de09afae7"
    lock["region_proposal"]["sam3"] = lock_entry()
    (tmp_path / "configs/upstream.lock.json").write_text(json.dumps(lock))
    downloads = []

    def hf_hub_download(repo_id, filename, revision, local_dir):
        downloads.append((repo_id, filename, revision))
        path = tmp_path / local_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FILES[filename])
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download))
    return tmp_path, downloads


def test_prepare_sam3_downloads_missing_and_verifies_existing(project):
    root, downloads = project
    folder = root / f"models/facebook__sam3/{REVISION}"
    folder.mkdir(parents=True)
    (folder / "sam3.pt").write_bytes(FILES["sam3.pt"])  # already present: hash check only
    saved = prepare(root, only="sam3")
    assert downloads == [("facebook/sam3", "config.json", REVISION)]
    entry = read_json(root / "models/weights.lock.json")["models"]["facebook/sam3"]
    assert entry == saved["models"]["facebook/sam3"] and entry["path"] == str(folder)
    assert entry["revision"] == REVISION and set(entry["files"]) == set(FILES)
    assert not (root / "configs/advv.local.yaml").exists()  # DragFlow/Qwen are not prepared here


def test_prepare_sam3_refuses_a_mismatching_file(project):
    root, downloads = project
    folder = root / f"models/facebook__sam3/{REVISION}"
    folder.mkdir(parents=True)
    (folder / "sam3.pt").write_bytes(b"other bytes")
    with pytest.raises(ConfigError, match="does not match the pinned SHA256"):
        prepare(root, only="sam3")
    assert (folder / "sam3.pt").read_bytes() == b"other bytes"  # never deleted
    assert not (root / "models/weights.lock.json").exists()


def test_verify_sam3_matches_config_and_lock(project, v2cfg):
    root, _ = project
    prepare(root, only="sam3")
    entry = read_json(root / "models/weights.lock.json")["models"]["facebook/sam3"]
    cfg = copy.deepcopy(v2cfg)
    cfg["generator"]["weights_lock"] = str(root / "models/weights.lock.json")
    cfg["region_proposal"].update(model_path=entry["path"], model_revision=REVISION)
    assert verify_sam3(cfg)["revision"] == REVISION
    for key, value in (("model_revision", "4" * 40), ("model_path", str(root)), ("model_path", None)):
        bad = copy.deepcopy(cfg)
        bad["region_proposal"][key] = value
        with pytest.raises(ConfigError, match="disagree"):
            verify_sam3(bad)
    (root / f"models/facebook__sam3/{REVISION}/sam3.pt").write_bytes(b"trunc")
    with pytest.raises(ConfigError, match="incomplete SAM 3 file"):
        verify_sam3(cfg)
