from __future__ import annotations

from pathlib import Path

from ..storage import digest, load_rgb


class Qwen:
    def __init__(self, cfg):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.torch, self.cfg = torch, cfg["verifier"]
        self.version = transformers.__version__
        self.processor = AutoProcessor.from_pretrained(self.cfg["model_path"], local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.cfg["model_path"],
            dtype=getattr(torch, self.cfg["dtype"]),
            device_map={"": "cuda:0"},
            local_files_only=True,
        ).eval()
        self.template_hash = digest(self.processor.chat_template)

    def __call__(self, request):
        torch = self.torch
        images = [load_rgb(Path(p)) for p in request["images"]]
        content = []
        for index, img in enumerate(images, 1):
            content.extend([{"type": "text", "text": f"Image {index}:"}, {"type": "image", "image": img}])
        content.append({"type": "text", "text": request["prompt"]})
        messages = [{"role": "user", "content": content}]
        template = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # Official Qwen3.5 template closes the thinking block for non-thinking mode.
        if "<think>\n\n</think>" not in template:
            raise RuntimeError("Pinned Qwen template did not disable thinking as expected")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=request["max_new_tokens"])
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return {
            "text": text,
            "info": {
                "backend": "qwen_multimodal_local",
                "model_revision": self.cfg["model_revision"],
                "processor_revision": self.cfg["processor_revision"],
                "dtype": self.cfg["dtype"],
                "transformers": self.version,
                "torch": torch.__version__,
                "chat_template_sha256": self.template_hash,
                "enable_thinking": False,
                "do_sample": False,
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "output_tokens": len(new_tokens),
                "image_sizes": [list(im.size) for im in images],
                "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            },
        }
