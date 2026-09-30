"""Real local Qwen integration probe; creates no augmented/exported data."""

import argparse
from dataclasses import asdict
from pathlib import Path

from advv.backends.process import LocalBackend
from advv.config import load_config
from advv.storage import atomic_json, read_json
from advv.verifiers.parser import parse_response

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gpus", required=True)
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
cfg = load_config(
    project / "configs/advv.example.yaml", input_dir=args.image.parent, target_count=1, gpus=args.gpus
)
lock = read_json(project / "models/weights.lock.json")["models"]["Qwen/Qwen3.5-4B"]
cfg["verifier"].update(
    model_path=lock["path"], model_revision=lock["revision"], processor_revision=lock["revision"]
)
cfg["execution"]["verifier_python"] = str(project / ".venv-qwen/bin/python")
root = args.output.resolve().parent
root.mkdir(parents=True, exist_ok=True)
atomic_json(root / "config.json", cfg)
backend = LocalBackend(cfg, root)


def model(request):
    try:
        return asdict(
            backend.complete(
                [Path(p) for p in request["images"]], request["prompt"], request["max_new_tokens"]
            )
        )
    except BaseException:
        backend.close()
        raise


results = {}
for kind, images, prompt, schema, tokens in [
    (
        "profile",
        [args.image],
        cfg["_assets"]["profile_prompt"].replace("{{user_hint}}", "null"),
        cfg["_assets"]["profile_schema"],
        512,
    ),
    ("physical", [args.image], cfg["_assets"]["physical_prompt"], cfg["_assets"]["vqa_schema"], 256),
]:
    result = model({"images": [str(p.resolve()) for p in images], "prompt": prompt, "max_new_tokens": tokens})
    results[kind] = result
    try:
        result["parsed"] = parse_response(result["text"], schema)
    except Exception as exc:
        result["parse_error"] = str(exc)
    atomic_json(args.output, results)
    print(kind, result["text"], flush=True)
if results["profile"].get("parsed"):
    import json

    prompt = cfg["_assets"]["semantic_prompt"].replace(
        "{{preservation_context}}", json.dumps(results["profile"]["parsed"])
    )
    result = model({"images": [str(args.image.resolve())] * 2, "prompt": prompt, "max_new_tokens": 256})
    results["semantic_identity_control"] = result
    try:
        result["parsed"] = parse_response(result["text"], cfg["_assets"]["vqa_schema"])
    except Exception as exc:
        result["parse_error"] = str(exc)
    atomic_json(args.output, results)
    print("semantic_identity_control", result["text"], flush=True)
backend.close()
