"""Render sampled requested edits without invoking any generation or VQA model."""

import argparse
from pathlib import Path

from advv.config import load_config
from advv.ingest import scan_sources, snapshot_sources
from advv.sampler import sample_edit
from advv.visualization import render

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
cfg = load_config(project / "configs/advv.example.yaml", input_dir=args.input_dir, target_count=1)
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
sources = snapshot_sources(scan_sources(cfg)[0], args.output)
for source in sources:
    for index, operation in enumerate(["relocation", "deformation", "rotation"]):
        cfg["sampler"]["operations"] = [operation]
        plan = sample_edit(
            source, {"summary": "Unverified source; geometry preview only."}, cfg, index, args.output
        )
        record = {"candidate_id": plan.edit_id, "plan": plan.to_dict(), "status": "planned", "checks": {}}
        result = render(args.output, source.to_dict(), record, cfg["visualization"])
        print(args.output / result["artifacts"]["drag_plan"])
