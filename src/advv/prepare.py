"""Explicit model preparation; inference itself never downloads weights."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from .errors import ConfigError
from .storage import atomic_bytes, atomic_json, file_hash, read_json

QWEN = "Qwen/Qwen3.5-4B"
ONLY = ("all", "qwen", "dragflow", "support", "sam3")
FLUX = "black-forest-labs/FLUX.1-dev"
PATTERNS = {
    QWEN: ["*.json", "*.safetensors", "*.jinja", "*.txt", "vocab.*", "merges.*"],
    FLUX: [
        "*.json",
        "ae.safetensors",
        "transformer/*.safetensors",
        "text_encoder/*.safetensors",
        "text_encoder_2/*.safetensors",
        "vae/*.safetensors",
        "tokenizer/*",
        "tokenizer_2/*",
    ],
    "Tencent/InstantCharacter": ["instantcharacter_ip-adapter.bin"],
    "google/siglip-so400m-patch14-384": ["*.json", "*.safetensors", "*.model", "*.txt"],
    "facebook/dinov2-giant": ["*.json", "*.safetensors"],
    "openai/clip-vit-large-patch14": ["*.json", "*.safetensors", "*.txt"],
}


def prepare(project: Path, *, only: str = "all") -> dict:
    """Download pinned weights. "all" covers DragFlow and Qwen; gated SAM 3 needs --only sam3."""
    project = project.resolve()
    lock = read_json(project / "configs/upstream.lock.json")
    destination = project / "models/weights.lock.json"
    saved = read_json(destination) if destination.exists() else {"schema_version": "1.1", "models": {}}
    if only == "sam3":
        return prepare_sam3(project, lock, saved, destination)
    from huggingface_hub import snapshot_download

    for repo, revision in lock["models"].items():
        if only == "qwen" and repo != QWEN:
            continue
        if only == "dragflow" and repo == QWEN:
            continue
        if only == "support" and repo in (QWEN, FLUX):
            continue
        print(f"Downloading {repo}@{revision}", flush=True)
        folder = snapshot_download(
            repo,
            revision=revision,
            allow_patterns=PATTERNS[repo],
            cache_dir=str(project / "models/hub"),
            max_workers=4,
        )
        saved["models"][repo] = {"revision": revision, "path": folder}
        atomic_json(destination, saved)
    if set(saved["models"]) >= set(lock["models"]):
        write_local_config(project, saved, lock)
    return saved


def sam3_folder(project: Path, entry: dict) -> Path:
    return project / "models" / entry["model_id"].replace("/", "__") / entry["revision"]


def prepare_sam3(project: Path, lock: dict, saved: dict, destination: Path) -> dict:
    """Fetch only the pinned SAM 3 files that are missing; every file, old or new, must match its SHA256."""
    entry = lock["region_proposal"]["sam3"]
    folder = sam3_folder(project, entry)
    for name, expected in entry["files"].items():
        path = folder / name
        if path.is_file():
            print(f"Verifying {path}", flush=True)
        else:
            from huggingface_hub import hf_hub_download

            print(f"Downloading {entry['model_id']}@{entry['revision']}/{name}", flush=True)
            hf_hub_download(entry["model_id"], name, revision=entry["revision"], local_dir=str(folder))
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            # Never delete model files here; the user decides what to do with a mismatching file.
            raise ConfigError(f"{path} does not match the pinned SHA256 in configs/upstream.lock.json")
    saved["models"][entry["model_id"]] = {
        "revision": entry["revision"],
        "path": str(folder),
        "files": entry["files"],
    }
    atomic_json(destination, saved)
    return saved


def verify_sam3(cfg: dict) -> dict:
    """Preflight: the run's SAM 3 settings match weights.lock.json and the checkpoint files are complete."""
    settings = cfg["region_proposal"]
    path = cfg["generator"].get("weights_lock")
    if not path or not Path(path).is_file():
        raise ConfigError("Missing weights lock. Run advv prepare --only sam3.")
    entry = read_json(Path(path)).get("models", {}).get(settings["model_id"])
    if not entry or len(entry.get("revision", "")) != 40 or not entry.get("files"):
        raise ConfigError(f"Missing pinned local weights: {settings['model_id']} (advv prepare --only sam3)")
    if (
        not settings.get("model_path")
        or Path(settings["model_path"]).resolve() != Path(entry["path"]).resolve()
        or settings.get("model_revision") != entry["revision"]
    ):
        raise ConfigError("region_proposal.model_path/model_revision and the weights lock disagree")
    for name, expected in entry["files"].items():
        file = Path(entry["path"]) / name
        if not file.is_file() or file.stat().st_size != expected["bytes"]:
            raise ConfigError(f"Missing or incomplete SAM 3 file: {file}")
    return entry


def write_local_config(project: Path, weights: dict, lock: dict) -> Path:
    path = project / "configs/advv.local.yaml"
    if path.exists():
        return path  # Existing user settings are never overwritten.
    cfg = yaml.safe_load((project / "configs/advv.example.yaml").read_text())
    cfg["execution"].update(
        generator_python="../.venv-dragflow/bin/python", verifier_python="../.venv-qwen/bin/python"
    )
    cfg["generator"].update(
        revision=lock["dragflow_commit"],
        upstream_config="../third_party/DragFlow/framework/config.yaml",
        weights_lock="../models/weights.lock.json",
    )
    cfg["verifier"].update(
        model_path=weights["models"][QWEN]["path"],
        model_revision=lock["models"][QWEN],
        processor_revision=lock["models"][QWEN],
    )
    sam3 = lock.get("region_proposal", {}).get("sam3")
    if sam3 and sam3["model_id"] in weights["models"]:
        cfg["execution"]["proposal_python"] = "../.venv-sam3/bin/python"
        cfg["region_proposal"].update(
            model_path=weights["models"][sam3["model_id"]]["path"],
            model_revision=sam3["revision"],
            repo_path="../third_party/sam3",
            code_revision=sam3["code_commit"],
        )
    atomic_bytes(path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).encode())
    return path


def git_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def verify_weights(cfg: dict) -> dict:
    path = cfg["generator"].get("weights_lock")
    if not path or not Path(path).is_file():
        raise ConfigError("Missing weights lock. Run advv prepare after Hugging Face authentication.")
    weights = read_json(Path(path))
    for repo in PATTERNS:
        entry = weights.get("models", {}).get(repo)
        if not entry or not Path(entry["path"]).is_dir() or len(entry["revision"]) != 40:
            raise ConfigError(f"Missing pinned local weights: {repo}")
    qwen = weights["models"][QWEN]
    if (
        Path(cfg["verifier"]["model_path"]).resolve() != Path(qwen["path"]).resolve()
        or cfg["verifier"]["model_revision"] != qwen["revision"]
        or cfg["verifier"]["processor_revision"] != qwen["revision"]
    ):
        raise ConfigError("Qwen model/processor and weights lock disagree")
    return weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--only", choices=ONLY, default="all")
    args = parser.parse_args()
    prepare(args.project, only=args.only)
