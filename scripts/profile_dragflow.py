"""Stage 0 profile of the pinned DragFlow (GPU; writes only to a new --out directory).

Runs a shortened inversion and the first N drag rounds of a stored candidate with one speedup arm and
saves the time split (no_grad/grad transformer forward, autograd backward, reclaim_memory, Python GC,
torch.profiler kernel categories incl. device copies) and peak memory as JSON. Timers synchronize CUDA
around each timed call, so totals are for attribution, not for end-to-end speed (use
check_dragflow_equivalence.py for that). profile.json is written even when the arm ends in oom/error;
the exit code is then 3 (0 = completed; 1 or 2 = bad inputs or arguments). Run with the DragFlow venv after the
GPU ownership check, with CUDA_VISIBLE_DEVICES set to the same two GPUs as --gpus:

    CUDA_VISIBLE_DEVICES=A,B .venv-dragflow/bin/python scripts/profile_dragflow.py --gpus A,B \
        --run-dir runs/v2_pilot_001 --candidate <id> --out _workspace/T018_profile/baseline
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import dragflow_harness as harness

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--gpus", required=True, help="exactly two GPU ids (no default)")
parser.add_argument("--run-dir", type=Path, required=True, help="existing run; read only")
parser.add_argument("--candidate", required=True, help="candidate id with backend_input/")
parser.add_argument("--seed", type=int, default=None, help="default: the candidate plan seed")
parser.add_argument(
    "--rounds", type=int, default=5, help="drag operations (K-loop) to run; 1..max_dragging_num (50 upstream)"
)
parser.add_argument(
    "--inversion-steps", type=int, default=3, help="executed inversion steps; 0 keeps the official count"
)
parser.add_argument(
    "--speedups", default="none", help="comma list of generator.speedups flags, 'exact_all', or 'none'"
)
parser.add_argument("--profile-round", type=int, default=1, help="drag round traced by torch.profiler")
parser.add_argument("--no-torch-profiler", action="store_true")
parser.add_argument("--out", type=Path, required=True, help="new directory under _workspace/")
parser.add_argument("--allow-external-out", action="store_true", help="allow --out outside the project")
args = parser.parse_args()
harness.expose_gpus(args.gpus)
flags = harness.parse_flags(args.speedups)
candidate = harness.load_candidate(args.run_dir, args.candidate, args.seed)
harness.check_rounds(candidate["cfg"], args.rounds, 1)
if not 0 <= args.profile_round < args.rounds:
    raise SystemExit("--profile-round must be within the rounds")
out = harness.new_output_dir(args.out, allow_external=args.allow_external_out, forbid=(Path(candidate["run_dir"]),))

import torch  # noqa: E402  (after CUDA_VISIBLE_DEVICES is set)

from advv.backends.dragflow_speedups import upstream_modules  # noqa: E402
from advv.storage import atomic_json  # noqa: E402

flow = harness.load_dragflow(candidate["cfg"], out, args.inversion_steps or None)
result = harness.run_arm(
    torch,
    flow,
    upstream_modules(),
    candidate,
    out,
    args.rounds,
    flags,
    sync_timers=True,
    profile_round=None if args.no_torch_profiler else args.profile_round,
)
result.pop("_latents")
result.pop("_z_inverted")


def summary(values):
    return {"count": len(values), "total": sum(values), "median": statistics.median(values) if values else None}


rounds = result["round_seconds"]
result["summary"] = {
    "forward_no_grad_seconds": summary(result["forward_seconds"]["no_grad"]),
    "forward_grad_seconds": summary(result["forward_seconds"]["grad"]),
    "backward_seconds": summary(result["backward_seconds"]),
    "round_seconds_excluding_first": summary(rounds[1:]),
    "reclaim_memory_seconds_per_round": (
        result["reclaim_memory"]["seconds"] / len(rounds) if rounds else None
    ),
}
atomic_json(
    out / "profile.json",
    {
        "kind": "dragflow_profile",
        "inputs": {k: str(v) for k, v in candidate.items() if k != "cfg"},
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "effective_upstream_steps": {
            k: flow.conf[k] for k in ("inversion_step_num", "sampling_step_num", "skip_step_num")
        },
        "environment": harness.environment(torch),
        **result,
    },
)
print(out / "profile.json")
print(f"status {result['status']}")
raise SystemExit(0 if result["status"] == harness.COMPLETED else 3)
