"""Opt-in execution patches for the pinned DragFlow b3a8fa7, applied in-process by the ADVV wrapper.

Every flag is off by default, which leaves the official upstream execution path untouched. Upstream
files are never edited. The first four patches keep the same math (upstream line evidence in
docs/ARCHITECTURE.md); ``tf32`` changes matmul precision and is recorded as a recipe variant. Nothing
here imports torch at module level, so config validation and CPU tests run without it.
"""

from __future__ import annotations

import types
from typing import Callable

from ..errors import ConfigError

EXACT_SPEEDUPS = (
    "skip_unused_single_blocks",
    "disable_gradient_checkpointing",
    "ip_adapter_on_transformer_device",
    "light_reclaim_memory",
)
PRECISION_SPEEDUPS = ("tf32",)
SPEEDUP_KEYS = EXACT_SPEEDUPS + PRECISION_SPEEDUPS
# Upstream modules that bind reclaim_memory as a global (dashboard_utils defines it; the others
# receive it through `from dashboard_utils import *` or an explicit import).
RECLAIM_MODULES = ("dashboard_utils", "dragger", "pipeline_flux", "overrider_DiT")


def parse_speedups(value) -> dict[str, bool]:
    """generator.speedups: missing or null means all off (the official execution path)."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ConfigError("generator.speedups must be a mapping of flag: true|false")
    unknown = set(value) - set(SPEEDUP_KEYS)
    if unknown:
        raise ConfigError(f"Unknown generator.speedups flags: {sorted(unknown)}; allowed {list(SPEEDUP_KEYS)}")
    for key, flag in value.items():
        if type(flag) is not bool:
            raise ConfigError(f"generator.speedups.{key} must be true or false")
    return {key: value.get(key, False) for key in SPEEDUP_KEYS}


def speedup_record(flags: dict[str, bool], torch=None) -> dict:
    """What generation info and reports store, so a patched run never reads as the official path."""
    enabled = [key for key in SPEEDUP_KEYS if flags[key]]
    record = {
        "official_execution_path": not enabled,
        "enabled": enabled,
        "numerically_equivalent_patches": [key for key in EXACT_SPEEDUPS if flags[key]],
        "precision_variants": [key for key in PRECISION_SPEEDUPS if flags[key]],
    }
    if torch is not None:
        record["tf32_allowed"] = {
            "cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        }
    return record


class Installed:
    """Applied patches in order, with the undo steps the measurement scripts use to compare arms."""

    def __init__(self):
        self.applied: list[str] = []
        self._undo: list[Callable[[], None]] = []

    def add(self, name: str, undo: Callable[[], None]) -> None:
        self.applied.append(name)
        self._undo.append(undo)

    def undo(self) -> None:
        while self._undo:
            self._undo.pop()()
            self.applied.pop()


def install_speedups(flags: dict[str, bool], *, torch, dragger, modules: dict) -> Installed:
    """Apply the enabled patches after ``dragger.load_pipeline()``. All flags off changes nothing.

    ``modules`` maps upstream module names (RECLAIM_MODULES and ``attn_processor``) to the imported
    module objects. Preconditions are checked before anything is patched.
    """
    installed = Installed()
    if not any(flags.values()):
        return installed
    transformer = dragger.pipeline.transformer
    if flags["skip_unused_single_blocks"]:
        check_single_block_skip(dragger.conf, modules["overrider_DiT"].config)
    if flags["disable_gradient_checkpointing"] and transformer.gradient_checkpointing is not True:
        raise ConfigError("disable_gradient_checkpointing expects upstream to enable checkpointing")
    if flags["ip_adapter_on_transformer_device"] and not dragger.conf["use_adapter"]:
        raise ConfigError("ip_adapter_on_transformer_device needs use_adapter")
    try:
        if flags["skip_unused_single_blocks"]:
            dragger._extract_latent_features = skip_single_blocks(
                dragger._extract_latent_features, dragger, torch.is_grad_enabled
            )
            installed.add("skip_unused_single_blocks", lambda: delattr(dragger, "_extract_latent_features"))
        if flags["disable_gradient_checkpointing"]:
            transformer.disable_gradient_checkpointing()
            installed.add("disable_gradient_checkpointing", transformer.enable_gradient_checkpointing)
        if flags["ip_adapter_on_transformer_device"]:
            undo = move_ip_processors(
                transformer, modules["attn_processor"].FluxIPAttnProcessor, dragger.device_0, dragger.device_1
            )
            installed.add("ip_adapter_on_transformer_device", undo)
        if flags["light_reclaim_memory"]:
            installed.add("light_reclaim_memory", patch_reclaim_memory(modules, light_reclaim_memory(torch)))
        if flags["tf32"]:
            installed.add("tf32", enable_tf32(torch))
    except Exception:
        installed.undo()
        raise
    return installed


def check_single_block_skip(conf: dict, overrider_config: dict) -> None:
    """The loss reads only the configured intermediate features (dragger.py:163-169); the 38 single
    blocks run after all double blocks (overrider_DiT.py:346-420), so skipping them is exact only when
    every feature is a DOUBLE-* capture and the optimizer path is the default autograd one."""
    if conf.get("use_optimizer"):
        raise ConfigError("skip_unused_single_blocks supports only use_optimizer: false (upstream default)")
    for source in (conf, overrider_config):
        ids = source.get("target_block_feature_ids_flux") or []
        if not ids or not all(str(name).startswith("DOUBLE-") for name in ids):
            raise ConfigError(f"skip_unused_single_blocks needs only DOUBLE-* feature ids, got {ids}")


def skip_single_blocks(original, dragger, grad_enabled):
    """Wrap Dragger._extract_latent_features for the F_drag forward only.

    The F_drag call (dragger.py:353) is the only one with grad enabled and a latent that requires grad;
    its noise prediction is deleted unused (dragger.py:355). The F_orig call (dragger.py:320-326, no_grad)
    needs the noise prediction and keeps the full forward. The skipped call returns None for the noise
    prediction so any future use fails instead of reading a truncated forward.
    """

    def extract(timestep, z, target_inputs):
        if not (grad_enabled() and getattr(z, "requires_grad", False)):
            return original(timestep=timestep, z=z, target_inputs=target_inputs)
        transformer = dragger.pipeline.transformer
        blocks = transformer.single_transformer_blocks
        transformer.single_transformer_blocks = type(blocks)()
        try:
            _, features = original(timestep=timestep, z=z, target_inputs=target_inputs)
        finally:
            transformer.single_transformer_blocks = blocks
        return None, features

    return extract


def with_defaults(func, **values):
    """Copy of ``func`` with the same code and parameter names but different default values."""
    code = func.__code__
    defaults = list(func.__defaults__ or ())
    params = code.co_varnames[: code.co_argcount]
    names = params[len(params) - len(defaults) :]
    missing = set(values) - set(names)
    if missing:
        raise ConfigError(f"{func.__qualname__} has no defaulted parameters {sorted(missing)}")
    for i, name in enumerate(names):
        if name in values:
            defaults[i] = values[name]
    copy = types.FunctionType(code, func.__globals__, func.__name__, tuple(defaults), func.__closure__)
    copy.__kwdefaults__ = dict(func.__kwdefaults__) if func.__kwdefaults__ else None
    copy.__qualname__, copy.__doc__, copy.__module__ = func.__qualname__, func.__doc__, func.__module__
    return copy


def same_device_processor_class(cls, device: str):
    """FluxIPAttnProcessor hard-codes device_1="cuda:1" as default arguments (attn_processor.py:33-34,
    142-143) and every caller relies on the defaults. The subclass runs the same code with both devices
    set to the transformer's device; parameter names stay the same, so diffusers' kwarg filtering by
    signature (attention_processor.py:577-586) passes the same kwargs."""
    return type(
        "SameDevice" + cls.__name__,
        (cls,),
        {
            "__call__": with_defaults(cls.__call__, device_0=device, device_1=device),
            "_get_ip_hidden_states": with_defaults(cls._get_ip_hidden_states, device_0=device, device_1=device),
            "__module__": __name__,
        },
    )


def move_ip_processors(transformer, cls, device: str, home: str) -> Callable[[], None]:
    processors = list(transformer.attn_processors.values())
    if not processors or any(type(p) is not cls for p in processors):
        raise ConfigError("ip_adapter_on_transformer_device expects upstream FluxIPAttnProcessor on every attention")
    same = same_device_processor_class(cls, device)
    for processor in processors:
        processor.to(device)
        processor.__class__ = same

    def undo():
        for processor in processors:
            processor.__class__ = cls
            processor.to(home)

    return undo


def light_reclaim_memory(torch):
    """Upstream reclaim_memory (dashboard_utils.py:66-85) synchronizes, empties the cache and runs
    gc.collect on every GPU, many times per drag round; none of that changes values. Its one lasting
    side effect is the current CUDA device, left at the last GPU by the set_device loop, so that is kept."""

    def reclaim_memory():
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.device_count() - 1)

    return reclaim_memory


def patch_reclaim_memory(modules: dict, replacement) -> Callable[[], None]:
    originals = {}
    for name in RECLAIM_MODULES:
        module = modules[name]
        if not hasattr(module, "reclaim_memory"):
            raise ConfigError(f"Upstream module {name} has no reclaim_memory to replace")
        originals[name] = module.reclaim_memory
    for name in RECLAIM_MODULES:
        modules[name].reclaim_memory = replacement

    def undo():
        for name, original in originals.items():
            modules[name].reclaim_memory = original

    return undo


def enable_tf32(torch) -> Callable[[], None]:
    """Precision variant: fp32 tensors, TF32 tensor-core matmul/conv. PyTorch defaults are matmul off and
    cuDNN on, so the official path already runs the VAE convolutions with TF32."""
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def undo():
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous

    return undo


def upstream_modules() -> dict:
    """The already-imported pinned upstream modules (call after importing dragger)."""
    import sys

    names = RECLAIM_MODULES + ("adapter.attn_processor",)
    missing = [name for name in names if name not in sys.modules]
    if missing:
        raise ConfigError(f"Upstream DragFlow modules not imported: {missing}")
    modules = {name: sys.modules[name] for name in RECLAIM_MODULES}
    modules["attn_processor"] = sys.modules["adapter.attn_processor"]
    return modules
