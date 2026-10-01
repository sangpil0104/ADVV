"""Shared measurement harness for profile_dragflow.py and check_dragflow_equivalence.py (GPU only).

Loads the pinned DragFlow through the ADVV wrapper once, then runs ``Dragger.__call__`` on a stored
candidate ``backend_input`` up to N drag rounds and stops before sampling. Results go to a new directory
under ``_workspace/`` (or outside the project with --allow-external-out), never into the source run.
CUDA_VISIBLE_DEVICES must be set by the caller; torch is imported only after it is checked.
"""

from __future__ import annotations

import copy
import gc
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT / "_workspace"
# run_arm statuses. Only "completed" arms reached every requested round and can be compared.
COMPLETED, OOM, ERROR = "completed", "oom", "error"


class Stop(Exception):
    """Raised after the requested drag round so sampling is skipped."""


def expose_gpus(value: str) -> list[str]:
    """Exactly two GPU ids, no default, and CUDA_VISIBLE_DEVICES must already name the same GPUs.

    The script never sets CUDA_VISIBLE_DEVICES itself, so the GPU choice is visible on the command line
    (and to the shared-server gpu-guard hook) before anything runs.
    """
    ids = [item.strip() for item in value.split(",")]
    if len(ids) != 2 or len(set(ids)) != 2 or not all(item.isdigit() for item in ids):
        raise SystemExit("--gpus needs exactly two distinct GPU ids, e.g. --gpus 8,9 (DragFlow uses cuda:0/1)")
    wanted = ",".join(ids)
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current is None:
        raise SystemExit(f"Set CUDA_VISIBLE_DEVICES={wanted} on the command line together with --gpus {wanted}")
    if current.replace(" ", "") != wanted:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={current} disagrees with --gpus {wanted}")
    if "torch" in sys.modules:
        raise SystemExit("torch was imported before the GPUs were exposed")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return ids


def new_output_dir(path: Path, *, allow_external: bool = False, forbid: tuple[Path, ...] = ()) -> Path:
    """Create a fresh output directory: under _workspace/, or outside the project with allow_external.

    Call it after the inputs are validated, so a failed check leaves no empty directory behind.
    """
    path = path.resolve()
    inside_project = path.is_relative_to(PROJECT)
    if inside_project and not (path.is_relative_to(WORKSPACE) and path != WORKSPACE):
        raise SystemExit(f"Output inside the project must be under _workspace/: {path}")
    if not inside_project and not allow_external:
        raise SystemExit(f"Output outside the project needs --allow-external-out: {path}")
    for other in forbid:
        if path.is_relative_to(Path(other).resolve()):
            raise SystemExit(f"Output must not be inside {other}: {path}")
    if path.exists():
        raise SystemExit(f"Output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def drag_round_limit(cfg: dict) -> int:
    """Rounds the harness may run: the TRANSPORT operations of the first drag step.

    A harness "round" is one K-loop operation (dragger.py:332-334). From operation max_dragging_num on,
    upstream enters INTENSIFY and sets conf["lr"] = 1000.0 for good (dragger.py:455-456); past
    max_dragging_num + max_intensify_num the drag step ends and sampling would be recorded instead.
    """
    merged = {**cfg["_upstream_config"], **cfg["generator"]["parameters"]}
    return int(merged["max_dragging_num"])


def check_rounds(cfg: dict, rounds: int, minimum: int) -> None:
    merged = {**cfg["_upstream_config"], **cfg["generator"]["parameters"]}
    if not merged.get("use_grad_mask"):
        raise SystemExit("The harness stops at _combine_latents, which upstream calls only with use_grad_mask")
    limit = drag_round_limit(cfg)
    if not minimum <= rounds <= limit:
        raise SystemExit(f"--rounds must be in [{minimum}, {limit}] (max_dragging_num, before INTENSIFY)")


def parse_flags(text: str) -> dict[str, bool]:
    from advv.backends.dragflow_speedups import SPEEDUP_KEYS, parse_speedups

    names = [] if text in ("", "none") else [name.strip() for name in text.split(",")]
    if "exact_all" in names:
        names.remove("exact_all")
        names += [key for key in SPEEDUP_KEYS if key != "tf32"]
    return parse_speedups({name: True for name in names})


def load_candidate(run_dir: Path, candidate: str, seed: int | None) -> dict:
    from advv.storage import read_json, within

    run_dir = run_dir.resolve()
    cfg = read_json(run_dir / "config.json")
    folder = within(run_dir, f"candidates/{candidate}/backend_input")
    for name in ("original_image.png", "operation.png", "instruction.json"):
        if not (folder / name).is_file():
            raise SystemExit(f"Missing {name} in {folder}")
    if seed is None:
        seed = read_json(within(run_dir, f"records/{candidate}.json", must_exist=True))["plan"]["seed"]
    return {"cfg": cfg, "input_folder": folder, "seed": seed, "run_dir": str(run_dir), "candidate": candidate}


def load_dragflow(cfg: dict, out: Path, inversion_steps: int | None):
    """The ADVV wrapper with every speedup off; arms install and undo their own patches."""
    from advv.backends.dragflow import DragFlow

    cfg = copy.deepcopy(cfg)
    cfg["generator"]["speedups"] = {}
    if inversion_steps is not None:
        if inversion_steps < 1:
            raise SystemExit("--inversion-steps must be >= 1")
        merged = {**cfg["_upstream_config"], **cfg["generator"]["parameters"]}
        # KVHookHub requires as many inversion as sampling steps (hookhub.py:233-234).
        total = merged["skip_step_num"] + inversion_steps
        cfg["generator"]["parameters"] = {
            **cfg["generator"]["parameters"],
            "inversion_step_num": total,
            "sampling_step_num": total,
        }
    return DragFlow(cfg, out)


def seed_all(torch, flow, seed: int) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    flow.conf["seed"] = seed


def sync(torch) -> None:
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def finite(value):
    return value if value is None or math.isfinite(value) else None


class Probe:
    """Instance-level wrappers around one Dragger call; removed again by ``close``.

    Records the inversion output, the per-round L1 losses (dragger.py:419), the per-round updated latent
    (the output of _combine_latents in dragger.py:458-463) and wall time per round. With ``sync_timers``
    it also times transformer forwards, autograd.grad, reclaim_memory and Python GC with synchronized
    CUDA clocks, which serializes GPU work and is meant for profiling only.
    """

    def __init__(self, torch, flow, modules, rounds: int, *, sync_timers=False, profile_round=None):
        self.torch, self.flow, self.dragger, self.modules = torch, flow, flow.dragger, modules
        self.rounds, self.sync_timers, self.profile_round = rounds, sync_timers, profile_round
        self.data = {
            "inversion_seconds": None,
            "losses": [],
            "round_seconds": [],
            "forward_seconds": {"no_grad": [], "grad": []},
            "backward_seconds": [],
            "reclaim_memory": {"calls": 0, "seconds": 0.0},
            "gc": {"collections": 0, "seconds": 0.0},
            "profiles": {},
        }
        self.latents, self.z_inverted = [], None
        self._undo, self._in_drag, self._grad_calls, self._mark, self._prof = [], False, 0, None, None

    def _set(self, owner, name, value):
        had = name in vars(owner)
        old = vars(owner).get(name)
        setattr(owner, name, value)
        self._undo.append(lambda: setattr(owner, name, old) if had else delattr(owner, name))

    def install(self):
        torch, dragger = self.torch, self.dragger
        inversion, extract = dragger.process_inversion, dragger._extract_latent_features
        combine, step = dragger._combine_latents, dragger.dragger_step

        def process_inversion(source_inputs):
            started = time.perf_counter()
            z = inversion(source_inputs=source_inputs)
            sync(torch)
            self.data["inversion_seconds"] = time.perf_counter() - started
            self.z_inverted = z.detach().float().cpu()
            return z

        def dragger_step(*args, **kwargs):
            self._in_drag, self._mark = True, time.perf_counter()
            return step(*args, **kwargs)

        def extract_features(timestep, z, target_inputs):
            grad = torch.is_grad_enabled() and getattr(z, "requires_grad", False)
            profiling = self.profile_round is not None and self._in_drag and (
                (not grad and "f_orig_forward" not in self.data["profiles"])
                or (grad and self._grad_calls == self.profile_round)
            )
            if profiling and self._prof is None:
                self._start_profile()
            result = extract(timestep=timestep, z=z, target_inputs=target_inputs)
            if grad:
                self._grad_calls += 1
            elif profiling:
                self._stop_profile("f_orig_forward")
            return result

        def combine_latents(*args, **kwargs):
            out = combine(*args, **kwargs)
            if not self._in_drag:
                return out
            sync(torch)
            now = time.perf_counter()
            self.data["round_seconds"].append(now - self._mark)
            self._mark = now
            self.latents.append(out.detach().float().cpu())
            if self._prof is not None and self._grad_calls == self.profile_round + 1:
                self._stop_profile("drag_round")
            if len(self.latents) >= self.rounds:
                raise Stop()
            return out

        self._set(dragger, "process_inversion", process_inversion)
        self._set(dragger, "dragger_step", dragger_step)
        self._set(dragger, "_extract_latent_features", extract_features)
        self._set(dragger, "_combine_latents", combine_latents)
        self._set(self.modules["dragger"], "F", LossRecorder(self.modules["dragger"].F, self.data["losses"]))
        if self.sync_timers:
            self._install_timers()

    def _install_timers(self):
        torch, data = self.torch, self.data
        transformer = self.dragger.pipeline.transformer
        state = {}

        def before(module, args, kwargs):
            z = kwargs.get("hidden_states", args[0] if args else None)
            state["grad"] = torch.is_grad_enabled() and getattr(z, "requires_grad", False)
            sync(torch)
            state["t"] = time.perf_counter()

        def after(module, args, kwargs, output):
            sync(torch)
            data["forward_seconds"]["grad" if state["grad"] else "no_grad"].append(time.perf_counter() - state["t"])

        handles = [
            transformer.register_forward_pre_hook(before, with_kwargs=True),
            transformer.register_forward_hook(after, with_kwargs=True),
        ]
        self._undo.append(lambda: [handle.remove() for handle in handles])
        grad = torch.autograd.grad

        def timed_grad(*args, **kwargs):
            sync(torch)
            started = time.perf_counter()
            result = grad(*args, **kwargs)
            sync(torch)
            data["backward_seconds"].append(time.perf_counter() - started)
            return result

        self._set(torch.autograd, "grad", timed_grad)
        for name in ("dashboard_utils", "dragger", "pipeline_flux", "overrider_DiT"):
            module = self.modules[name]
            self._set(module, "reclaim_memory", self._timed_reclaim(module.reclaim_memory))
        gc_state = {}

        def on_gc(phase, info):
            if phase == "start":
                gc_state["t"] = time.perf_counter()
            elif "t" in gc_state:
                data["gc"]["collections"] += 1
                data["gc"]["seconds"] += time.perf_counter() - gc_state.pop("t")

        gc.callbacks.append(on_gc)
        self._undo.append(lambda: gc.callbacks.remove(on_gc))

    def _timed_reclaim(self, reclaim):
        torch, record = self.torch, self.data["reclaim_memory"]

        def reclaim_memory():
            sync(torch)
            started = time.perf_counter()
            reclaim()
            record["calls"] += 1
            record["seconds"] += time.perf_counter() - started

        return reclaim_memory

    def _start_profile(self):
        torch = self.torch
        sync(torch)
        self._prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        )
        self._prof.__enter__()
        self._prof_started = time.perf_counter()

    def _stop_profile(self, name):
        sync(self.torch)
        wall = time.perf_counter() - self._prof_started
        self._prof.__exit__(None, None, None)
        self.data["profiles"][name] = summarize_profile(self._prof, wall)
        self._prof = None

    def close(self):
        if self._prof is not None:
            self._prof.__exit__(None, None, None)
            self._prof = None
        while self._undo:
            self._undo.pop()()


class LossRecorder:
    """Stands in for torch.nn.functional inside the upstream dragger module and records l1_loss values."""

    def __init__(self, functional, sink: list):
        self._functional, self._sink = functional, sink

    def __getattr__(self, name):
        return getattr(self._functional, name)

    def l1_loss(self, *args, **kwargs):
        value = self._functional.l1_loss(*args, **kwargs)
        self._sink.append(finite(float(value.detach())))
        return value


CATEGORIES = (
    ("memcpy_peer", ("Memcpy PtoP",)),
    ("memcpy_device", ("Memcpy DtoD",)),
    ("memcpy_host_to_device", ("Memcpy HtoD",)),
    ("memcpy_device_to_host", ("Memcpy DtoH",)),
    ("memset", ("Memset",)),
    ("attention", ("fmha", "flash", "attention", "efficient")),
    ("gemm", ("gemm", "cutlass", "xmma", "ampere_", "sm80_", "sm86_")),
)


def category(name: str) -> str:
    for label, needles in CATEGORIES:
        if any(needle.lower() in name.lower() for needle in needles):
            return label
    return "other_kernels"


def summarize_profile(prof, wall_seconds: float, top: int = 40) -> dict:
    device_rows, sync_rows = [], {}
    for event in prof.key_averages():
        device = str(getattr(event, "device_type", "")).endswith("CUDA")
        if device:
            micros = getattr(event, "self_device_time_total", None)
            if micros is None:
                micros = getattr(event, "self_cuda_time_total", 0)
            device_rows.append({"name": event.key, "count": event.count, "device_us": float(micros)})
        elif "Synchronize" in event.key or event.key in ("cudaFree", "cudaMalloc", "cudaMemcpy"):
            sync_rows[event.key] = {"count": event.count, "cpu_us": float(event.self_cpu_time_total)}
    totals: dict[str, float] = {}
    for row in device_rows:
        totals[category(row["name"])] = totals.get(category(row["name"]), 0.0) + row["device_us"]
    device_rows.sort(key=lambda row: -row["device_us"])
    return {
        "wall_seconds": wall_seconds,
        "device_seconds_by_category": {k: v / 1e6 for k, v in sorted(totals.items(), key=lambda kv: -kv[1])},
        "device_seconds_total_all_gpus": sum(totals.values()) / 1e6,
        "cuda_runtime_cpu_seconds": {k: v["cpu_us"] / 1e6 for k, v in sync_rows.items()},
        "top_kernels": device_rows[:top],
        "note": "Kernel time is summed over both GPUs; categories are name-based heuristics.",
    }


def run_arm(torch, flow, modules, candidate: dict, out: Path, rounds: int, flags: dict, **probe_kwargs) -> dict:
    """Install the arm's patches, run to the requested drag round, undo everything.

    Never raises for a failed arm: status is "completed", "oom" or "error" (with the exception). The
    upstream conf (which the drag loop may change, e.g. lr in INTENSIFY) is restored after every arm.
    """
    from advv.backends.dragflow_speedups import Installed, install_speedups, speedup_record

    if not flow.conf["use_grad_mask"]:
        raise SystemExit("The harness stops at _combine_latents, which upstream calls only with use_grad_mask")
    arm_dir = out / "load_data"
    arm_dir.mkdir(exist_ok=True)
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    saved_conf = dict(flow.conf)
    probe = Probe(torch, flow, modules, rounds, **probe_kwargs)
    installed = Installed()
    result = {"speedups": speedup_record(flags), "applied": [], "status": ERROR}
    started = time.perf_counter()
    try:
        installed = install_speedups(flags, torch=torch, dragger=flow.dragger, modules=modules)
        result.update(speedups=speedup_record(flags, torch), applied=list(installed.applied))
        seed_all(torch, flow, candidate["seed"])
        raw, instruction = flow.load_data(
            str(candidate["input_folder"]), str(arm_dir), device="cuda:0", dtype=torch.float32
        )
        probe.install()
        flow.dragger(raw, instruction, image_name=candidate["candidate"])
        result["error"] = f"Dragger finished after {len(probe.latents)} of {rounds} rounds"
    except Stop:
        result["status"] = COMPLETED
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status=OOM, error=str(exc)[:2000])
    except Exception as exc:  # recorded per arm; the next arm and the JSON still run
        result.update(error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-4000:])
    finally:
        probe.close()
        installed.undo()
        flow.conf.clear()
        flow.conf.update(saved_conf)
    if result["status"] == COMPLETED and (len(probe.latents) != rounds or probe.z_inverted is None):
        result.update(status=ERROR, error=f"Recorded {len(probe.latents)} of {rounds} rounds")
    result["wall_seconds"] = time.perf_counter() - started
    result["peak_allocated_bytes"] = [torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())]
    result["peak_reserved_bytes"] = [torch.cuda.max_memory_reserved(d) for d in range(torch.cuda.device_count())]
    result.update(probe.data)
    result["_latents"], result["_z_inverted"] = probe.latents, probe.z_inverted
    gc.collect()
    torch.cuda.empty_cache()
    return result


def environment(torch) -> dict:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32_defaults": {
            "cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "python": sys.version.split()[0],
    }
