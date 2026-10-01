"""Stage 1/2 check: do generator.speedups patches reproduce the official DragFlow drag rounds? (GPU)

Loads DragFlow once, runs the official path (every flag off) and then each arm on the same candidate,
seed and inputs for N drag rounds, and compares per-round L1 loss, the per-round updated latent and the
inversion latent with the baseline. A repeated baseline (control_baseline) is not judged: its difference
from the baseline is the GPU noise floor, and an exact arm passes when every relative difference is within
max(--rtol, --noise-multiplier x noise floor) of the same metric. Arms with tf32 get separate, looser
diagnostic tolerances and never decide the exit code, since TF32 is a different recipe whose quality
needs full-image comparison. equivalence.json is rewritten after every arm (finished: false until the
end), and arms that hit OOM or another error are listed apart from numerical differences.

Exit codes: 0 every arm completed and every exact arm is within tolerance; 2 at least one completed exact
arm is outside tolerance; 3 no completed exact arm is outside tolerance but some arm ended in oom/error;
4 the baseline did not complete (nothing could be judged); 1 bad arguments, inputs or a crash.

    CUDA_VISIBLE_DEVICES=A,B .venv-dragflow/bin/python scripts/check_dragflow_equivalence.py --gpus A,B \\
        --run-dir runs/v2_pilot_001 --candidate <id> --out _workspace/T018_equivalence
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import dragflow_harness as harness

DEFAULT_ARMS = (
    "skip_unused_single_blocks",
    "skip_unused_single_blocks,disable_gradient_checkpointing",
    "skip_unused_single_blocks,disable_gradient_checkpointing,ip_adapter_on_transformer_device",
    "exact_all",
    "exact_all,tf32",
)
BASELINE, CONTROL = "baseline", "control_baseline"
EXIT_OK, EXIT_ERROR, EXIT_OUTSIDE, EXIT_INCOMPLETE, EXIT_NO_BASELINE = 0, 1, 2, 3, 4
EXIT_MEANING = {
    EXIT_OK: "every arm completed; every exact arm within tolerance",
    EXIT_OUTSIDE: "a completed exact arm is outside tolerance (numerical difference)",
    EXIT_INCOMPLETE: "no completed exact arm outside tolerance, but some arm ended in oom/error",
    EXIT_NO_BASELINE: "the baseline did not complete; no arm was judged",
}


class Parser(argparse.ArgumentParser):
    """Usage errors exit 1, so exit code 2 keeps meaning 'exact arm outside tolerance'."""

    def error(self, message):
        self.print_usage(sys.stderr)
        raise SystemExit(f"{self.prog}: error: {message}")


def build_parser() -> Parser:
    parser = Parser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpus", required=True, help="exactly two GPU ids (no default); CUDA_VISIBLE_DEVICES must match")
    parser.add_argument("--run-dir", type=Path, required=True, help="existing run; read only")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=None, help="default: the candidate plan seed")
    parser.add_argument(
        "--rounds", type=int, default=5, help="drag operations (K-loop) to run; 2..max_dragging_num (50 upstream)"
    )
    parser.add_argument(
        "--inversion-steps", type=int, default=0, help="executed inversion steps; 0 keeps the official count"
    )
    parser.add_argument(
        "--arm", action="append", default=None, help="comma list of flags (repeatable); default: cumulative arms"
    )
    parser.add_argument("--no-control", action="store_true", help="skip the repeated baseline (no noise floor)")
    parser.add_argument("--rtol", type=float, default=1e-5, help="exact arms: minimum relative tolerance")
    parser.add_argument(
        "--noise-multiplier", type=float, default=10.0, help="exact tolerance = max(rtol, k x control noise floor)"
    )
    parser.add_argument("--tf32-loss-rtol", type=float, default=1e-2)
    parser.add_argument("--tf32-latent-rtol", type=float, default=5e-2)
    parser.add_argument("--out", type=Path, required=True, help="new directory under _workspace/")
    parser.add_argument("--allow-external-out", action="store_true", help="allow --out outside the project")
    return parser


def relative(diff, scale):
    return diff / scale if scale > 0 else (0.0 if diff == 0 else float("inf"))


def compare(base: dict, other: dict) -> dict:
    """Per-round and inversion differences of two completed arms (torch or numpy arrays)."""
    rounds = []
    for i, (z0, z1) in enumerate(zip(base["_latents"], other["_latents"])):
        l0, l1 = base["losses"][i], other["losses"][i]
        loss_rel = None if l0 is None or l1 is None else relative(abs(l0 - l1), abs(l0))
        diff = float(abs(z0 - z1).max())
        rounds.append(
            {
                "round": i,
                "loss": [l0, l1],
                "loss_relative_diff": harness.finite(loss_rel),
                "latent_max_abs_diff": diff,
                "latent_relative_diff": harness.finite(relative(diff, float(abs(z0).max()))),
                "latent_mean_abs_diff": float(abs(z0 - z1).mean()),
            }
        )
    diff = float(abs(base["_z_inverted"] - other["_z_inverted"]).max())
    inversion = {
        "max_abs_diff": diff,
        "relative_diff": harness.finite(relative(diff, float(abs(base["_z_inverted"]).max()))),
    }
    return {"inversion": inversion, "rounds": rounds}


def worst(comparison: dict) -> dict:
    """Largest relative loss and latent difference (inversion counts as a latent); None if non-finite."""
    losses = [r["loss_relative_diff"] for r in comparison["rounds"]]
    latents = [r["latent_relative_diff"] for r in comparison["rounds"]] + [comparison["inversion"]["relative_diff"]]
    return {
        "loss_relative": None if None in losses else max(losses, default=0.0),
        "latent_relative": None if None in latents else max(latents),
    }


def within(comparison: dict, tolerance: dict) -> bool:
    largest = worst(comparison)
    return all(
        largest[key] is not None and largest[key] <= tolerance[key] for key in ("loss_relative", "latent_relative")
    )


def exact_tolerance(rtol: float, multiplier: float, floor: dict | None) -> dict:
    if floor is None:
        return {"loss_relative": rtol, "latent_relative": rtol}
    return {key: max(rtol, multiplier * floor[key]) for key in ("loss_relative", "latent_relative")}


def speed(result: dict) -> dict:
    later = result.get("round_seconds", [])[1:]
    return {
        "inversion_seconds": result.get("inversion_seconds"),
        "round_seconds": result.get("round_seconds", []),
        "median_round_seconds_excluding_first": statistics.median(later) if later else None,
        "wall_seconds": result.get("wall_seconds"),
        "peak_allocated_bytes": result.get("peak_allocated_bytes"),
        "peak_reserved_bytes": result.get("peak_reserved_bytes"),
    }


def build_report(results: dict, planned: list[str], settings: dict) -> dict:
    """Judge the arms run so far. planned names every arm in run order (baseline first)."""
    completed = {name for name, result in results.items() if result["status"] == harness.COMPLETED}
    base = results.get(BASELINE) if BASELINE in completed else None
    floor, floor_note = None, "no control_baseline arm (--no-control)" if CONTROL not in planned else None
    if base is not None and CONTROL in completed:
        floor = worst(compare(base, results[CONTROL]))
        if None in floor.values():
            floor, floor_note = None, "control_baseline difference is not finite"
    elif CONTROL in planned:
        floor_note = "control_baseline or baseline did not complete" if CONTROL in results else "not run yet"
    tolerance = exact_tolerance(settings["rtol"], settings["noise_multiplier"], floor)
    variant_tolerance = {"loss_relative": settings["tf32_loss_rtol"], "latent_relative": settings["tf32_latent_rtol"]}
    base_round = speed(base)["median_round_seconds_excluding_first"] if base else None
    arms, outside, inside, variant_outside = {}, [], [], []
    for name, result in results.items():
        entry = {"status": result["status"], "speedups": result["speedups"], "speed": speed(result)}
        for key in ("error", "traceback"):
            if key in result:
                entry[key] = result[key]
        arm_round = entry["speed"]["median_round_seconds_excluding_first"]
        entry["round_speedup_vs_baseline"] = (
            base_round / arm_round if base_round and arm_round and name in completed else None
        )
        if name == CONTROL:
            entry["judged_as"] = "noise_floor"
        elif name != BASELINE:
            variant = bool(result["speedups"]["precision_variants"])
            entry["judged_as"] = "precision_variant" if variant else "exact"
            if base is not None and name in completed:
                comparison = compare(base, result)
                limit = variant_tolerance if variant else tolerance
                ok = within(comparison, limit)
                entry["comparison"] = {"tolerance": limit, "worst": worst(comparison), "within_tolerance": ok, **comparison}
                if variant:
                    variant_outside += [] if ok else [name]
                else:
                    (inside if ok else outside).append(name)
        if name == CONTROL and base is not None and name in completed:
            entry["comparison"] = compare(base, result)
        arms[name] = entry
    incomplete = {name: results[name]["status"] for name in results if name not in completed}
    pending = [name for name in planned if name not in results]
    if pending:
        code = None
    elif base is None:
        code = EXIT_NO_BASELINE
    elif outside:
        code = EXIT_OUTSIDE
    elif incomplete:
        code = EXIT_INCOMPLETE
    else:
        code = EXIT_OK
    return {
        "finished": not pending,
        "exit_code": code,
        "exit_meaning": EXIT_MEANING.get(code),
        "pending_arms": pending,
        "incomplete_arms": incomplete,
        "exact_arms_outside_tolerance": outside,
        "exact_arms_within_tolerance": inside,
        "precision_variant_arms_outside_diagnostic_tolerance": variant_outside,
        "noise_floor": None if floor is None else {"source": CONTROL, **floor},
        "noise_floor_note": floor_note,
        "exact_tolerance": {
            "rule": "max(rtol, noise_multiplier * noise_floor) per metric; rtol alone without a noise floor",
            "rtol": settings["rtol"],
            "noise_multiplier": settings["noise_multiplier"],
            **tolerance,
        },
        "precision_variant_tolerance": variant_tolerance,
        "arms": arms,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    harness.expose_gpus(args.gpus)
    arms = [(BASELINE, harness.parse_flags("none"))]
    if not args.no_control:
        arms.append((CONTROL, harness.parse_flags("none")))
    arms += [(text, harness.parse_flags(text)) for text in (args.arm or DEFAULT_ARMS)]
    names = [name for name, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"Arm names must be unique: {names}")
    candidate = harness.load_candidate(args.run_dir, args.candidate, args.seed)
    harness.check_rounds(candidate["cfg"], args.rounds, 2)
    out = harness.new_output_dir(args.out, allow_external=args.allow_external_out, forbid=(Path(candidate["run_dir"]),))

    import torch

    from advv.backends.dragflow_speedups import upstream_modules
    from advv.storage import atomic_json

    settings = {k: getattr(args, k) for k in ("rtol", "noise_multiplier", "tf32_loss_rtol", "tf32_latent_rtol")}
    header = {
        "kind": "dragflow_equivalence",
        "inputs": {k: str(v) for k, v in candidate.items() if k != "cfg"},
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "environment": harness.environment(torch),
    }
    results: dict = {}

    def write(**extra) -> dict:
        report = {**header, **extra, **build_report(results, names, settings)}
        atomic_json(out / "equivalence.json", report)
        return report

    try:
        flow = harness.load_dragflow(candidate["cfg"], out, args.inversion_steps or None)
        modules = upstream_modules()
    except Exception as exc:
        write(load_error=f"{type(exc).__name__}: {exc}"[:2000])
        raise
    header["effective_upstream_steps"] = {
        k: flow.conf[k] for k in ("inversion_step_num", "sampling_step_num", "skip_step_num")
    }
    report = write()
    for name, flags in arms:
        print(f"== arm {name}", flush=True)
        results[name] = harness.run_arm(torch, flow, modules, candidate, out, args.rounds, flags)
        print(f"   {results[name]['status']}", flush=True)
        report = write()
    print(out / "equivalence.json")
    print(f"exit {report['exit_code']}: {report['exit_meaning']}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
