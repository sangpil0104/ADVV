from __future__ import annotations

import argparse
import json
import logging
import signal
from pathlib import Path

from .backends.process import LocalBackend
from .config import load_config, parse_gpus
from .errors import ADVVError
from .human_review import review_run
from .pipeline import Pipeline, create_run
from .preflight import preflight
from .prepare import ONLY, prepare
from .reporting import export_run, report_run
from .storage import read_json


def positive_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("N must be a positive integer") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("N must be a positive integer")
    return count


def main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(
        prog="advv", description="DragFlow augmentation with local Qwen verification"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "preflight"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=Path("configs/advv.local.yaml"))
        p.add_argument("--input-dir", type=Path)
        p.add_argument(
            "--target-count",
            type=positive_count,
            metavar="N",
            help="Total new, unique images that pass both VQA checks",
        )
        p.add_argument("--gpus", required=True, help="Allowed physical IDs, e.g. 0,1,2,3,4,5,6,7")
        if name == "run":
            p.add_argument(
                "count",
                type=positive_count,
                nargs="?",
                metavar="N",
                help="Total accepted images to collect across all inputs (e.g. advv run 10)",
            )
            p.add_argument(
                "--human-review",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Add the final human pass/fail stage (advv review); default comes from the config",
            )
            p.add_argument("--run-id")
            p.add_argument("--run-dir", type=Path)
            p.add_argument("--resume", action="store_true")
    p = sub.add_parser("prepare", help="Download the pinned models; inference remains offline")
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument(
        "--only", choices=ONLY, default="all", help="all = DragFlow + Qwen; sam3 (gated) only on request"
    )
    for name in ("export", "report", "review"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.count is not None and args.target_count is not None:
            parser.error("Specify N or --target-count N, not both")
        if args.count is not None:
            args.target_count = args.count
        if args.resume and args.target_count is not None:
            parser.error("Resume uses the saved target count; omit N and --target-count")
        if args.resume and args.human_review is not None:
            parser.error("Human review is fixed when a run is created; start a new run to change it")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "prepare":
            prepare(args.project, only=args.only)
            return 0
        if args.command == "review":
            from .storage import run_lock

            root = args.run_dir.resolve()
            with run_lock(root):
                try:
                    review_run(root)
                except KeyboardInterrupt:
                    print("\nReview paused; run the same command to continue.")
                print(export_run(root))
                print(report_run(root))
            return 0
        if args.command in ("report", "export"):
            from .storage import run_lock

            root = args.run_dir.resolve()
            with run_lock(root):
                print((report_run if args.command == "report" else export_run)(root))
            return 0
        if args.command == "run" and args.resume:
            if not args.run_dir or args.run_id or args.input_dir or args.target_count is not None:
                raise ADVVError("Resume requires --run-dir and --gpus; input/N changes require a new run")
            root = args.run_dir.resolve()
            cfg = read_json(root / "config.json")
            cfg["execution"]["selected_gpu_ids"] = parse_gpus(args.gpus)
            preflight(cfg, check_inputs=False)
        else:
            cfg = load_config(
                args.config,
                input_dir=args.input_dir,
                target_count=args.target_count,
                gpus=args.gpus,
                human_review=getattr(args, "human_review", None),
            )
            checks = preflight(cfg)
            if args.command == "preflight":
                print(
                    json.dumps(
                        {"status": checks["status"], "gpus": checks["gpus"], "inputs": checks["inputs"]},
                        indent=2,
                    )
                )
                return 0
            if not args.run_id or args.run_dir:
                raise ADVVError("New run requires --run-id; use --run-dir only with --resume")
            root = create_run(cfg, args.run_id, provenance=LocalBackend.provenance)
        backend = LocalBackend(cfg, root)
        state = Pipeline(root, backend, gpu_ids=cfg["execution"]["selected_gpu_ids"]).run()
        print(
            json.dumps(
                {
                    "run_dir": str(root),
                    "status": state["status"],
                    "accepted_count": state["accepted_count"],
                    "last_error": state.get("last_error"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if state["status"] == "completed" and cfg.get("human_review", {}).get("enabled"):
            print(f"Next: advv review --run-dir {root}")
        return 0 if state["status"] == "completed" else 130 if state["status"] == "interrupted" else 1
    except (ADVVError, OSError, ValueError, KeyError) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
