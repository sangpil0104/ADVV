from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

from ..errors import BackendError, FatalBackendError
from ..storage import atomic_json, read_json, within


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["generator", "verifier"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_json(args.run_dir / "config.json")
    backend = None
    for line in sys.stdin:
        request = json.loads(line)
        started = time.monotonic()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if backend is None:
                    if args.kind == "verifier":
                        from .qwen import Qwen

                        backend = Qwen(cfg)
                    else:
                        from .dragflow import DragFlow

                        backend = DragFlow(cfg, args.run_dir)
                result = backend(request)
            result.setdefault("info", {})["elapsed_seconds"] = time.monotonic() - started
            response = {"ok": True, "result": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            # Only explicitly categorized image-specific errors can continue.
            response = {
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "fatal": isinstance(exc, FatalBackendError) or not isinstance(exc, BackendError),
                    "elapsed_seconds": time.monotonic() - started,
                },
            }
        if request.get("receipt"):
            atomic_json(within(args.run_dir, request["receipt"]), response)
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
