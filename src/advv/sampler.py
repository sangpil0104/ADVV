from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .contracts import EditRequest, Source
from .errors import DataError
from .storage import atomic_image, digest, file_hash, within


def feature_grid(width: int, height: int) -> tuple[int, int]:
    full_w, full_h = width // 16 * 16, height // 16 * 16
    scale = 4 if (full_w + full_h) / 2 > 1000 else 3 if (full_w + full_h) / 2 > 600 else 2
    return full_w // scale, full_h // scale


def grid_point(point, width, height):
    grid = feature_grid(width, height)
    return [round(point[i] / (width, height)[i] * grid[i]) for i in (0, 1)]


def sample_edit(source: Source, profile: dict, cfg: dict, attempt_index: int, run_dir: Path) -> EditRequest:
    settings = cfg["sampler"]
    seed = int(digest([cfg["run"]["seed"], source.source_id, source.pixel_sha256, attempt_index])[:8], 16)
    rng = random.Random(seed)
    w, h = source.width, source.height
    for _ in range(512):  # Geometry validation budget, not a generation/quota budget.
        shape = rng.choice(settings["mask_shapes"])
        fraction = rng.uniform(*settings["region_area_fraction"])
        area = w * h * fraction / (math.pi / 4 if shape == "ellipse" else 1)
        ratio = math.exp(rng.uniform(math.log(0.5), math.log(2))) * w / h
        rw, rh = max(3, round(math.sqrt(area * ratio))), max(3, round(math.sqrt(area / ratio)))
        if rw >= w - 2 or rh >= h - 2:
            continue
        left, top = rng.randrange(1, w - rw), rng.randrange(1, h - rh)
        box = [left, top, left + rw - 1, top + rh - 1]
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        getattr(draw, shape)(box, fill=255)
        ys, xs = np.nonzero(np.asarray(mask))
        fraction_actual = len(xs) / (w * h)
        if not settings["region_area_fraction"][0] <= fraction_actual <= settings["region_area_fraction"][1]:
            continue
        start = [int(round(float(xs.mean()))), int(round(float(ys.mean())))]
        op = rng.choice(settings["operations"])
        anchor, params = None, {}
        if op == "rotation":
            anchor = [left, top]  # Distinct point gives a nonzero rotation radius.
            degrees = rng.uniform(*settings["rotation_degrees"])
            if abs(degrees) < 2:
                continue
            radians = math.radians(degrees)
            dx, dy = start[0] - anchor[0], start[1] - anchor[1]
            end = [
                round(anchor[0] + dx * math.cos(radians) - dy * math.sin(radians)),
                round(anchor[1] + dx * math.sin(radians) + dy * math.cos(radians)),
            ]
            params = {"requested_rotation_degrees": degrees}
        else:
            distance = rng.uniform(*settings["displacement_diagonal_fraction"]) * math.hypot(w, h)
            angle = rng.uniform(0, 2 * math.pi)
            end = [round(start[0] + distance * math.cos(angle)), round(start[1] + distance * math.sin(angle))]
            params = {"displacement_pixels": [end[0] - start[0], end[1] - start[1]]}
        if not (0 <= end[0] < w and 0 <= end[1] < h) or end == start:
            continue
        # Reject moves which collapse to the same point on upstream's feature grid.
        grid = feature_grid(w, h)
        fitted = [grid_point(p, w, h) for p in [start, end] + ([anchor] if anchor else [])]
        if fitted[0] == fitted[1] or any(not 0 <= p[i] < grid[i] for p in fitted for i in (0, 1)):
            continue
        if anchor and fitted[2] == fitted[0]:
            continue
        break
    else:
        raise DataError("Sampler could not form valid geometry; inspect image dimensions and sampler ranges")
    eid = f"{source.source_id}_{attempt_index:08d}_{digest([seed, settings, cfg['generator']])[:10]}"
    mask_path = f"edit_regions/{eid}.png"
    atomic_image(within(run_dir, mask_path), mask)
    summary = profile["summary"]
    edit_text = (
        f"Apply a local {op} to the selected region from {start} toward {end}. "
        "Preserve the event, any damage or defect type, and the identifying traits of the subject. "
        "Do not repair damage or remove an accident."
    )
    return EditRequest(
        source.source_id,
        eid,
        attempt_index,
        seed,
        settings["version"],
        op,
        params,
        mask_path,
        start,
        end,
        anchor,
        summary,
        edit_text,
        file_hash(within(run_dir, mask_path)),
    )


def validate_plan(plan: EditRequest, source: Source, run_dir: Path) -> None:
    if plan.source_id != source.source_id or plan.operation not in {"relocation", "deformation", "rotation"}:
        raise DataError("Invalid source or operation in edit plan")
    for point in [plan.source_point, plan.target_point] + ([plan.anchor_point] if plan.anchor_point else []):
        if len(point) != 2 or not all(type(x) in (int, float) and math.isfinite(x) for x in point):
            raise DataError("Invalid point")
        if not 0 <= point[0] < source.width or not 0 <= point[1] < source.height:
            raise DataError("Point outside image")
    if plan.source_point == plan.target_point:
        raise DataError("Identity drag is not an augmentation")
    grid = feature_grid(source.width, source.height)
    fitted = [
        grid_point(p, source.width, source.height)
        for p in [plan.source_point, plan.target_point] + ([plan.anchor_point] if plan.anchor_point else [])
    ]
    if fitted[0] == fitted[1] or any(not 0 <= p[i] < grid[i] for p in fitted for i in (0, 1)):
        raise DataError("Edit becomes empty or out-of-bounds on upstream feature grid")
    if plan.operation == "rotation" and plan.anchor_point in (None, plan.source_point):
        raise DataError("Rotation needs a distinct anchor")
    path = within(run_dir, plan.region_mask_path, must_exist=True)
    if file_hash(path) != plan.mask_sha256:
        raise DataError("Edit mask changed")
    with Image.open(path) as mask:
        arr = np.asarray(mask)
        if (
            mask.mode != "L"
            or mask.size != (source.width, source.height)
            or not set(np.unique(arr)) <= {0, 255}
        ):
            raise DataError("Mask must be binary 0/255 at source resolution")
        x, y = map(round, plan.source_point)
        if x >= source.width or y >= source.height or not arr.any() or arr[y, x] != 255:
            raise DataError("Source point must lie in a nonempty mask")
