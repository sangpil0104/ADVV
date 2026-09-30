import json
from pathlib import Path

import pytest
from PIL import Image

from advv.config import load_config
from advv.contracts import Generated, RawResponse
from advv.storage import atomic_image, load_rgb, pixel_hash, within

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    images = tmp_path / "input"
    images.mkdir()
    Image.new("RGB", (120, 80), (40, 60, 80)).save(images / "a.png")
    config = load_config(PROJECT / "configs/advv.example.yaml", input_dir=images, target_count=1, gpus="0,1")
    config["run"]["output_root"] = str(tmp_path / "runs")
    config["report"]["terminal_progress"] = False
    return config


class FakeBackend:
    """Test-only fixture. Production CLI has no fake mode."""

    provenance = "fake"

    def __init__(self, answers=None, colors=None, *, uncertain_source=False):
        self.answers = list(answers or ["YES", "YES"])
        self.colors = list(colors or [(200, 10, 30)])
        self.uncertain_source = uncertain_source
        self.generations = []
        self.calls = []
        self.closed = False

    def complete(self, images, prompt, max_new_tokens, *, receipt=None):
        self.calls.append(
            {"images": list(images), "hashes": [pixel_hash(p) for p in images], "prompt": prompt}
        )
        if prompt.startswith("Inspect this original"):
            return RawResponse(
                json.dumps(
                    {
                        "summary": "A visibly cracked part.",
                        "must_preserve": ["visible crack"],
                        "uncertain": self.uncertain_source,
                    }
                ),
                {"backend": "fake"},
            )
        value = self.answers.pop(0)
        if isinstance(value, BaseException):
            raise value
        if value.startswith("{") or value == "invalid":
            return RawResponse(value, {"backend": "fake"})
        return RawResponse(
            json.dumps({"answer": value, "reason": "Fixture judgment, not a real visual assessment."}),
            {"backend": "fake"},
        )

    def generate(self, source, plan, run_dir):
        self.generations.append((source.source_id, plan.attempt_index, plan.seed))
        color = self.colors.pop(0)
        if isinstance(color, BaseException):
            raise color
        image = (
            load_rgb(within(run_dir, source.image_path))
            if color == "source"
            else Image.new("RGB", (source.width, source.height), color)
        )
        output = within(run_dir, f"candidates/{plan.edit_id}/generated.png")
        atomic_image(output, image)
        effective = {
            "source_point": plan.source_point,
            "target_point": plan.target_point,
            "anchor_point": plan.anchor_point,
            "region_mask_path": plan.region_mask_path,
            "backend_input_size": [source.width, source.height],
            "backend": "fake",
        }
        return Generated(output, effective, {"backend": "fake"})

    def close(self):
        self.closed = True
