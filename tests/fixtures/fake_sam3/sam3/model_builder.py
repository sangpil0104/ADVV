import os
import time

import numpy as np


class FakeModel:
    inst_interactive_predictor = object()

    def predict_inst(self, state, point_coords, point_labels, multimask_output):
        """Three nested squares around the point, like whole/part/subpart; predicted IoUs unsorted."""
        h, w = state["original_height"], state["original_width"]
        x, y = (int(v) for v in point_coords[0])
        masks = np.zeros((3, h, w), bool)
        for k, r in enumerate((2, 12, 6)):
            masks[k, max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = True
        return masks, np.array([0.5, 0.95, 0.9], np.float32), np.zeros((3, 288, 288), np.float32)


def build_sam3_image_model(bpe_path, device, checkpoint_path, load_from_HF, enable_inst_interactivity):
    assert load_from_HF is False and enable_inst_interactivity is True and os.path.isfile(checkpoint_path)
    mode = os.environ.get("ADVV_FAKE_SAM3")
    if mode == "missing_keys":
        print(f"loaded {checkpoint_path} and found missing and/or unexpected keys:\nmissing_keys=['x']")
    if mode == "hang":
        time.sleep(60)
    return FakeModel()
