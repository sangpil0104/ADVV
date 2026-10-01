"""Region proposal worker: raw model masks for one source per request, written as a receipt.

Cleanup, filtering and proposals.json are done by the coordinator (advv.proposals) on CPU.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..contracts import Source
from ..errors import ConfigError, DataError
from ..proposals import collect_raw, part_seed, subject_parts_error, write_raw
from ..storage import load_rgb, within


def make_proposer(cfg: dict, run_dir: Path) -> Proposer:
    backend = cfg["region_proposal"]["backend"]
    if backend == "sam3":
        from .sam3 import Sam3Predictor

        return Proposer(cfg, run_dir, Sam3Predictor(cfg))
    if backend == "grounding_dino_sam2.1":
        raise ConfigError(
            "region_proposal.backend grounding_dino_sam2.1 is a planned comparison path and is not "
            "implemented; use sam3"
        )
    raise ConfigError(f"Unsupported region_proposal backend: {backend}")


class Proposer:
    def __init__(self, cfg: dict, run_dir: Path, predictor):
        self.cfg, self.run_dir, self.predictor = cfg, run_dir, predictor

    def __call__(self, request: dict) -> dict:
        if request.get("action") != "propose":
            raise DataError(f"Unsupported proposal action: {request.get('action')!r}")
        source = Source(**request["source"])
        subjects, parts = request["subjects"], request["parts"]
        if not subjects or not all(isinstance(s, str) and s for s in subjects):
            raise DataError(f"No grounding phrases for {source.source_id}")
        problem = subject_parts_error(parts, subjects)
        if problem:
            raise DataError(f"Part names for {source.source_id}: {problem}")
        image = load_rgb(within(self.run_dir, source.image_path, must_exist=True))
        settings = self.cfg["region_proposal"]
        seed = self.cfg["run"]["seed"]
        started = time.monotonic()
        with self.predictor.session(image):
            raw = collect_raw(self.predictor, subjects, parts, settings, lambda i: part_seed(seed, source, i))
        meta = {
            "source_id": source.source_id,
            "source_pixel_sha256": source.pixel_sha256,
            "backend": self.predictor.backend,
            "models": self.predictor.models,
            "info": {**self.predictor.info(), "inference_seconds": time.monotonic() - started},
        }
        raw_sha256 = write_raw(self.run_dir, request["raw_dir"], raw, image.size, meta)
        return {
            "raw_dir": request["raw_dir"],
            "raw_sha256": raw_sha256,
            "subjects": list(subjects),
            "parts": list(parts),
            "info": meta["info"],
        }
