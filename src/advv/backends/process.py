from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

from ..contracts import Generated, RawResponse
from ..errors import BackendError, FatalBackendError
from ..storage import read_json, within


class Worker:
    def __init__(self, kind: str, cfg: dict, run_dir: Path):
        selected = cfg["execution"]["selected_gpu_ids"]
        gpus = selected[:2] if kind == "generator" else selected[:1]
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=",".join(gpus),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            MPLBACKEND="Agg",
        )
        logfile = run_dir / f"logs/{kind}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        self.log = logfile.open("a", encoding="utf-8")
        self.proc = subprocess.Popen(
            [
                cfg["execution"][kind + "_python"],
                "-m",
                "advv.backends.worker",
                "--kind",
                kind,
                "--run-dir",
                str(run_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            cwd=str(run_dir),
            env=env,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.responses = queue.Queue()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        self.timeout = cfg["execution"]["worker_timeout_seconds"]

    def _read(self):
        for line in self.proc.stdout:
            self.responses.put(line)
        self.responses.put(None)

    def request(self, request: dict) -> dict:
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise FatalBackendError("Worker pipe closed; inspect logs") from exc
        end = time.monotonic() + self.timeout
        while True:
            try:
                line = self.responses.get(timeout=min(0.2, max(0.001, end - time.monotonic())))
                break
            except queue.Empty:
                if time.monotonic() >= end:
                    self.close()
                    raise BackendError(f"Worker timeout after {self.timeout}s")
        if line is None:
            raise FatalBackendError(f"Worker exited unexpectedly ({self.proc.poll()}); inspect logs")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FatalBackendError("Invalid worker protocol; inspect logs") from exc
        if not response["ok"]:
            error = response["error"]
            cls = FatalBackendError if error["fatal"] else BackendError
            raise cls(f"{error['type']}: {error['message']}")
        return response["result"]

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        for pipe in (self.proc.stdin, self.proc.stdout):
            pipe.close()
        self.log.close()


class LocalBackend:
    provenance = "dragflow+qwen_local"

    def __init__(self, cfg: dict, run_dir: Path):
        self.cfg, self.run_dir = cfg, run_dir
        self.worker = None

    def complete(self, images, prompt, max_new_tokens, *, receipt=None) -> RawResponse:
        if self.worker is None:
            self.worker = Worker("verifier", self.cfg, self.run_dir)
        try:
            result = self.worker.request(
                {
                    "action": "complete",
                    "images": [str(p) for p in images],
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "receipt": receipt,
                }
            )
        except BackendError:
            self.close()
            raise
        return RawResponse(result["text"], result["info"])

    def generate(self, source, plan, run_dir) -> Generated:
        self.close()  # Release Qwen before DragFlow occupies the selected GPUs.
        receipt = f"candidates/{plan.edit_id}/generation_receipt.json"
        saved = within(run_dir, receipt)
        if saved.exists():
            response = read_json(saved)
            if response["ok"]:
                result = response["result"]
                return Generated(within(run_dir, result["path"]), result["effective"], result["info"])
        worker = Worker("generator", self.cfg, self.run_dir)
        try:
            result = worker.request(
                {"action": "generate", "source": source.to_dict(), "plan": plan.to_dict(), "receipt": receipt}
            )
            return Generated(within(run_dir, result["path"]), result["effective"], result["info"])
        finally:
            worker.close()

    def close(self):
        if self.worker:
            self.worker.close()
            self.worker = None
