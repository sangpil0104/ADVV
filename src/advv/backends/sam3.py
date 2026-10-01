"""Adapter for official facebookresearch/sam3 (pinned commit) inside the .venv-sam3 worker.

One set_image per source feeds both the text (entity) and point (part) prompts; the local checkpoint is
passed explicitly because the builder otherwise downloads without a revision.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

from ..errors import ConfigError, FatalBackendError

SAM3_CODE = "facebookresearch/sam3"
CHECKPOINT = "sam3.pt"
# Upstream keeps `scores > confidence_threshold` with the Python float cast to the score tensor's dtype, so
# a threshold just below an ADVV minimum rounds back onto it. Give SAM 3 a margin wider than a bf16 step
# near 1 (2**-8) and let ADVV's own `>=` filters decide; the few extra low-score masks stay in the receipt.
THRESHOLD_MARGIN = 0.01


def processor_threshold(settings: dict) -> float:
    lowest = settings["min_entity_score"]
    if settings["text_parts"]:
        lowest = min(lowest, settings["min_text_part_score"])
    return max(0.0, lowest - THRESHOLD_MARGIN)


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


class Sam3Predictor:
    def __init__(self, cfg: dict):
        settings = cfg["region_proposal"]
        for key in ("model_path", "model_revision", "code_revision"):
            if not settings.get(key):
                raise ConfigError(f"region_proposal.{key} must be set for the sam3 backend")
        checkpoint = Path(settings["model_path"]) / CHECKPOINT
        if not checkpoint.is_file():
            raise ConfigError(f"Missing SAM 3 checkpoint: {checkpoint}")
        import sam3
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not torch.cuda.is_available():
            raise FatalBackendError("SAM 3 needs one visible CUDA device")
        self.torch = torch
        # Official notebook precision: TF32 matmuls and a bf16 autocast session over fp32 weights.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        bpe = Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.model = build_sam3_image_model(
                bpe_path=str(bpe),
                device="cuda",
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
        # Upstream loads with strict=False and only prints missing keys.
        if "missing" in log.getvalue():
            raise FatalBackendError(f"SAM 3 checkpoint did not load cleanly: {log.getvalue()[-2000:]}")
        if getattr(self.model, "inst_interactive_predictor", None) is None:
            raise FatalBackendError("SAM 3 point prompts need enable_inst_interactivity=True")
        self.processor = Sam3Processor(
            self.model, device="cuda", confidence_threshold=processor_threshold(settings)
        )
        # A test double of the package marks itself so its output can never pass as real SAM 3.
        self.backend = "fixture" if getattr(sam3, "__fixture__", False) else "sam3"
        self.models = {settings["model_id"]: settings["model_revision"], SAM3_CODE: settings["code_revision"]}
        self.load_stdout = log.getvalue()
        self.state = None

    @contextlib.contextmanager
    def session(self, image):
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.state = self.processor.set_image(image)
            try:
                yield self
            finally:
                self.state = None

    def text(self, phrase: str):
        self.processor.reset_all_prompts(self.state)
        out = self.processor.set_text_prompt(prompt=phrase, state=self.state)
        masks = _numpy(out["masks"])[:, 0] > 0.5
        scores, boxes = _numpy(out["scores"]), _numpy(out["boxes"])
        return [(masks[i], float(scores[i]), [float(v) for v in boxes[i]]) for i in range(len(scores))]

    def point(self, point):
        masks, scores, _ = self.model.predict_inst(
            self.state,
            point_coords=np.array([point], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        masks, scores = _numpy(masks) > 0.5, _numpy(scores)
        return [(masks[k], float(scores[k])) for k in range(len(scores))]

    def info(self) -> dict:
        torch = self.torch
        return {
            "backend": self.backend,
            "torch": torch.__version__,
            "precision": "fp32 weights, bf16 autocast, tf32",
            "confidence_threshold": self.processor.confidence_threshold,
            "load_stdout": self.load_stdout,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
