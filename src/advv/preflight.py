from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import yaml

from .backends.dragflow import ALLOWED_PARAMETERS
from .config import validate_config
from .errors import ConfigError
from .ingest import scan_sources
from .prepare import git_revision, verify_sam3, verify_weights
from .storage import digest, file_hash


def preflight(cfg: dict, *, check_inputs=True) -> dict:
    validate_config(cfg)
    if not cfg["run"]["target_count"]:
        raise ConfigError("Specify total N with 'advv run N' or --target-count N")
    gpus = cfg["execution"]["selected_gpu_ids"]
    if not gpus or len(gpus) < 2:
        raise ConfigError("Specify at least 2 allowed GPU IDs; official DragFlow uses two devices")
    gpu_info = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    available = {line.split(",")[0].strip(): line.strip() for line in gpu_info.splitlines()}
    if not set(gpus) <= set(available):
        raise ConfigError("One or more selected GPUs are unavailable")
    repo = Path(cfg["generator"]["repo_path"])
    if not cfg["generator"].get("revision") or git_revision(repo) != cfg["generator"]["revision"]:
        raise ConfigError("DragFlow checkout does not match the pinned revision")
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise ConfigError("DragFlow tracked files were modified; restore the pinned checkout")
    weights = verify_weights(cfg)
    upstream_path = cfg["generator"].get("upstream_config")
    if not upstream_path or not Path(upstream_path).is_file():
        raise ConfigError("Missing upstream_config path")
    upstream = yaml.safe_load(Path(upstream_path).read_text())
    if upstream["mode"] != "flux" or upstream["forward_diffusion_mode"] != "IN" or upstream["use_adap_scale"]:
        raise ConfigError("MVP uses FLUX inversion with the fixed upstream feature scale policy")
    params = cfg["generator"]["parameters"]
    if set(params) - ALLOWED_PARAMETERS:
        raise ConfigError(f"Unsupported generator parameters: {set(params) - ALLOWED_PARAMETERS}")
    for key, value in params.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ConfigError(f"Invalid numeric generator parameter: {key}")
        if key.endswith("_num") and (type(value) is not int or value < (0 if key == "skip_step_num" else 1)):
            raise ConfigError(f"{key} must be an integer with a valid step count")
    merged = upstream | params
    if merged["skip_step_num"] >= min(merged["inversion_step_num"], merged["sampling_step_num"]):
        raise ConfigError("skip_step_num must be below inversion and sampling steps")
    required = {
        "Qwen/Qwen3.5-4B": [
            "config.json",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "chat_template.jinja",
        ],
        "black-forest-labs/FLUX.1-dev": [
            "ae.safetensors",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "text_encoder_2/config.json",
            "tokenizer_2/tokenizer_config.json",
        ],
        "Tencent/InstantCharacter": ["instantcharacter_ip-adapter.bin"],
        "google/siglip-so400m-patch14-384": ["model.safetensors", "preprocessor_config.json"],
        "facebook/dinov2-giant": ["model.safetensors", "preprocessor_config.json"],
        "openai/clip-vit-large-patch14": ["model.safetensors", "tokenizer_config.json"],
    }
    for model, filenames in required.items():
        root = Path(weights["models"][model]["path"])
        for filename in filenames:
            path = root / filename
            if not path.is_file():
                raise ConfigError(f"Missing model asset: {model}/{filename}")
            if filename.endswith(".index.json"):
                for shard in set(json.loads(path.read_text())["weight_map"].values()):
                    if not (path.parent / shard).is_file():
                        raise ConfigError(f"Missing model shard: {path.parent / shard}")
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus[:2]), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )
    environments = {}
    probes = {
        "generator": 'import sys; sys.path.insert(0,"framework"); from dragger import Dragger; import torch; print(torch.__version__)',
        "verifier": "from transformers import Qwen3_5ForConditionalGeneration; import torch, transformers; print(torch.__version__, transformers.__version__)",
    }
    for kind, code in probes.items():
        interpreter = cfg["execution"].get(kind + "_python")
        if not interpreter or not Path(interpreter).is_file():
            raise ConfigError(f"Missing {kind} Python environment")
        result = subprocess.run(
            [interpreter, "-c", code], cwd=repo, env=env, capture_output=True, text=True, timeout=120
        )
        if result.returncode:
            raise ConfigError(f"{kind} import failed: {result.stderr[-3000:]}")
        packages = subprocess.run(
            [interpreter, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout
        environments[kind] = {"versions": result.stdout.strip(), "packages": packages}
    if cfg["sampler"]["version"] == "object_region_v2":
        environments["proposal"] = check_proposal(cfg, gpus)
    frozen = {
        "_upstream_config": upstream,
        "_upstream_config_sha256": file_hash(Path(upstream_path)),
        "_weights": weights,
        "_weights_lock_sha256": digest(weights),
        "_environments": environments,
        "_fireflow_revision": git_revision(repo / "dependencies/FireFlow"),
    }
    for key, value in frozen.items():
        if key in cfg and cfg[key] != value:
            raise ConfigError(f"Pinned runtime changed ({key}); start a new run or restore dependencies")
    cfg.update(frozen)
    result = {"status": "ready", "gpus": [available[x] for x in gpus], "environments": environments}
    if check_inputs:
        _, result["inputs"] = scan_sources(cfg)
    return result


PROPOSAL_PROBE = (
    "import torch, sam3, advv; from sam3.model_builder import build_sam3_image_model; "
    "from sam3.model.sam3_image_processor import Sam3Processor; "
    "print(torch.__version__, torch.version.cuda, sam3.__file__)"
)


def check_proposal(cfg: dict, gpus: list[str]) -> dict:
    """object_region_v2: pinned SAM 3 checkout, checkpoint and worker Python (probed on the first GPU)."""
    settings = cfg["region_proposal"]
    if settings["backend"] != "sam3":
        raise ConfigError(f"region_proposal.backend {settings['backend']} is not implemented; use sam3")
    verify_sam3(cfg)
    repo = settings.get("repo_path")
    if not repo or not Path(repo).is_dir():
        raise ConfigError("Missing region_proposal.repo_path (official facebookresearch/sam3 checkout)")
    repo = Path(repo)
    if not settings.get("code_revision") or git_revision(repo) != settings["code_revision"]:
        raise ConfigError("SAM 3 checkout does not match region_proposal.code_revision")
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise ConfigError("SAM 3 tracked files were modified; restore the pinned checkout")
    interpreter = cfg["execution"].get("proposal_python")
    if not interpreter or not Path(interpreter).is_file():
        raise ConfigError("Missing proposal Python environment (execution.proposal_python, .venv-sam3)")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [interpreter, "-c", PROPOSAL_PROBE],
        cwd=Path(interpreter).parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode:
        raise ConfigError(f"proposal import failed: {result.stderr[-3000:]}")
    package = Path(result.stdout.split()[-1]).resolve()
    if not package.is_relative_to(repo.resolve()):
        raise ConfigError(f"proposal Python imports sam3 from {package}, not {repo}")
    packages = subprocess.run(
        [interpreter, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    return {"versions": result.stdout.strip(), "packages": packages}
