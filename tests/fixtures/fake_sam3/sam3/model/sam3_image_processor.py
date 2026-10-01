import os

import numpy as np

# phrase -> list of (x0, y0, x1, y1, score); anything else finds nothing.
# Part phrases: "kitten tail" inside the kitten; "tail" finds the same tail (lower score) and one tail
# outside every entity; "kitten head" scores between the processor threshold and min_text_part_score.
INSTANCES = {
    "kitten": [(20, 20, 59, 59, 0.9)],
    "box": [(80, 10, 109, 39, 0.7), (0, 60, 30, 79, 0.2)],
    "kitten tail": [(20, 40, 35, 59, 0.8)],
    "tail": [(20, 40, 35, 59, 0.6), (88, 60, 110, 75, 0.9)],
    "kitten head": [(40, 20, 59, 35, 0.495)],
}


class Sam3Processor:
    def __init__(self, model, device, confidence_threshold):
        self.model, self.confidence_threshold = model, confidence_threshold

    def set_image(self, image):
        return {"original_width": image.size[0], "original_height": image.size[1]}

    def reset_all_prompts(self, state):
        for key in ("masks", "scores", "boxes"):
            state.pop(key, None)

    def set_text_prompt(self, prompt, state):
        if os.environ.get("ADVV_FAKE_SAM3") == "text_error":
            raise RuntimeError("fake SAM 3 text failure")
        if "masks" in state:
            # Upstream keeps prompts in the state; a second text prompt must follow reset_all_prompts.
            raise RuntimeError("fake SAM 3: text prompt without reset_all_prompts")
        h, w = state["original_height"], state["original_width"]
        found = [i for i in INSTANCES.get(prompt, []) if i[4] > self.confidence_threshold]
        masks = np.zeros((len(found), 1, h, w), bool)
        for n, (x0, y0, x1, y1, _) in enumerate(found):
            masks[n, 0, y0 : y1 + 1, x0 : x1 + 1] = True
        state.update(
            masks=masks,
            scores=np.array([i[4] for i in found], np.float32),
            boxes=np.array([i[:4] for i in found], np.float32).reshape(-1, 4),
        )
        return state
