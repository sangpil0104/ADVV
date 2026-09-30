"""Adapter for official DragFlow b3a8fa7; no replacement image generator."""

from __future__ import annotations

import io
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ..contracts import EditRequest, Source
from ..errors import BackendError, ConfigError
from ..sampler import validate_plan
from ..storage import atomic_bytes, atomic_image, atomic_json, file_hash, load_rgb, read_json, within

ALLOWED_PARAMETERS = {
    "lr",
    "inversion_step_num",
    "sampling_step_num",
    "skip_step_num",
    "inversion_guidance_scale",
    "sampling_guidance_scale",
    "max_operation_num",
    "max_dragging_num",
    "max_intensify_num",
    "adapter_subject_scale",
}


def upstream_instruction(plan: dict) -> dict:
    return {
        "region_operations": {
            "0": {
                "task": {
                    "relocation": "transformation",
                    "deformation": "deformation",
                    "rotation": "rotation",
                }[plan["operation"]],
                "centroids": [plan["source_point"], plan["target_point"]],
                "anchors": plan.get("anchor_point"),
            }
        },
        "point_operations": {"begin_points": [], "target_points": []},
        "background_prompt": plan["source_prompt"],
        "editing_prompt": " " + plan["target_prompt"],
    }


def to_original(point, grid_size, original_size):
    return [float(point[i]) * original_size[i] / grid_size[i] for i in (0, 1)]


class DragFlow:
    def __init__(self, cfg, run_dir):
        import torch

        self.torch, self.cfg, self.run_dir = torch, cfg, run_dir
        if torch.cuda.device_count() < 2:
            raise ConfigError("Official DragFlow requires two visible GPUs")
        sys.path.insert(0, str(Path(cfg["generator"]["repo_path"]) / "framework"))
        # The pinned upstream reads ./framework/config.yaml during import.
        os.chdir(cfg["generator"]["repo_path"])
        import dashboard_utils
        from dragger import Dragger

        weights = read_json(Path(cfg["generator"]["weights_lock"]))["models"]

        def path(repo):
            return weights[repo]["path"]

        def local_download(repo_id, filename, **kwargs):
            local = Path(path(repo_id)) / filename
            if not local.is_file():
                raise ConfigError(f"Missing locked model file: {local}")
            return str(local)

        original_embedder = dashboard_utils.HFEmbedder

        def locked_embedder(version, *args, **kwargs):
            return original_embedder(path(version) if version in weights else version, *args, **kwargs)

        dashboard_utils.hf_hub_download = local_download
        dashboard_utils.HFEmbedder = locked_embedder
        conf = dict(cfg["_upstream_config"])
        params = cfg["generator"]["parameters"]
        if set(params) - ALLOWED_PARAMETERS:
            raise ConfigError(f"Unsupported DragFlow parameters: {set(params) - ALLOWED_PARAMETERS}")
        conf.update(params)
        conf.update(
            model_path_flux=path("black-forest-labs/FLUX.1-dev"),
            encoder_path_1=path("google/siglip-so400m-patch14-384"),
            encoder_path_2=path("facebook/dinov2-giant"),
            adapter_path=str(Path(path("Tencent/InstantCharacter")) / "instantcharacter_ip-adapter.bin"),
            use_mask_visualization=False,
            use_affine_visualization=False,
            show_step_images=None,
        )
        self.conf = conf
        self.dragger = Dragger(conf, dtype=torch.float32)
        self.dragger.load_pipeline()
        self.load_data = dashboard_utils.load_data

    def __call__(self, request):
        torch, root = self.torch, self.run_dir
        source, plan = Source(**request["source"]), EditRequest(**request["plan"])
        validate_plan(plan, source, root)
        seed = plan.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.conf["seed"] = seed
        folder = within(root, f"candidates/{plan.edit_id}")
        input_folder = folder / "backend_input"
        atomic_image(input_folder / "original_image.png", load_rgb(within(root, source.image_path)))
        with Image.open(within(root, plan.region_mask_path)) as image:
            atomic_image(input_folder / "operation.png", image.copy())
        instruction_dict = upstream_instruction(plan.to_dict())
        atomic_json(input_folder / "instruction.json", instruction_dict)
        atomic_bytes(folder / "effective_config.yaml", yaml.safe_dump(self.conf).encode())
        raw, instruction = self.load_data(
            str(input_folder), str(folder), device="cuda:0", dtype=torch.float32
        )
        for gpu in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(gpu)
        # Upstream optimizes latents with autograd. Do not wrap in inference_mode.
        _, generated, _ = self.dragger(raw, instruction, image_name=plan.edit_id)
        if generated is None:
            raise BackendError("DragFlow returned no generated image")
        op = instruction["region_operations"]["0"]
        if "region_init" not in op:
            raise BackendError("DragFlow did not record an executed edit region")
        region = op["region_init"].detach().cpu().float().squeeze().numpy()
        grid_size = [region.shape[1], region.shape[0]]
        original_size = [source.width, source.height]
        raw_mask_buffer = io.BytesIO()
        np.save(raw_mask_buffer, region, allow_pickle=False)
        native_path = f"candidates/{plan.edit_id}/effective_region.npy"
        atomic_bytes(within(root, native_path), raw_mask_buffer.getvalue())
        mask = Image.fromarray((region > 0).astype(np.uint8) * 255).resize(raw.size, Image.Resampling.NEAREST)
        mask_path = f"candidates/{plan.edit_id}/effective_area.png"
        atomic_image(within(root, mask_path), mask)
        points = [point.detach().cpu().tolist() for point in (op["points_fit"][0], op["points_fit"][2])]
        anchor = op.get("anchors_fit")
        anchor = anchor[0].detach().cpu().tolist() if anchor else None
        effective = {
            "source_point": to_original(points[0], grid_size, original_size),
            "target_point": to_original(points[1], grid_size, original_size),
            "anchor_point": to_original(anchor, grid_size, original_size) if anchor else None,
            "region_mask_path": mask_path,
            "mask_sha256": file_hash(within(root, mask_path)),
            "backend_input_size": list(reversed(self.dragger.full_shape)),
            "grid_size": grid_size,
            "grid_points": points,
            "grid_anchor": anchor,
            "native_region_path": native_path,
            "native_region_sha256": file_hash(within(root, native_path)),
            "transform": {
                "resize_image": "bicubic_to_floor_multiple_16",
                "padding": [0, 0, 0, 0],
                "upstream_region_interpolation": "bilinear_align_corners_false",
                "display_mask": "nonzero_native_region_nearest_to_original",
                "start_point": "upstream_replaced_with_rounded_region_centroid",
                "target_point": "upstream_rounded_original_to_feature_grid",
                "inverse_xy_scale": [original_size[i] / grid_size[i] for i in (0, 1)],
            },
            "upstream_instruction": instruction_dict,
        }
        output = f"candidates/{plan.edit_id}/generated.png"
        # This is the same final resize performed by upstream solve_outcomes.
        atomic_image(
            within(root, output), generated.convert("RGB").resize(raw.size, Image.Resampling.LANCZOS)
        )
        atomic_json(folder / "effective.json", effective)
        return {
            "path": output,
            "effective": effective,
            "info": {
                "backend": "dragflow",
                "revision": self.cfg["generator"]["revision"],
                "dtype": "float32",
                "transformer_quantization": "quanto_qint8",
                "torch": torch.__version__,
                "peak_vram_bytes": [
                    torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
                ],
            },
        }
