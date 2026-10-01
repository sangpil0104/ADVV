from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .contracts import EditRequest, Source
from .errors import DataError, SamplingSkipped
from .proposals import ProposalSet, centroid, dilate, inner_point, is_single_region
from .storage import atomic_bytes, atomic_image, digest, file_hash, within


def feature_grid(width: int, height: int) -> tuple[int, int]:
    full_w, full_h = width // 16 * 16, height // 16 * 16
    scale = 4 if (full_w + full_h) / 2 > 1000 else 3 if (full_w + full_h) / 2 > 600 else 2
    return full_w // scale, full_h // scale


def grid_point(point, width, height):
    grid = feature_grid(width, height)
    return [round(point[i] / (width, height)[i] * grid[i]) for i in (0, 1)]


def _linear_taps(in_size: int, out_size: int):
    """Source taps of PyTorch bilinear resize (align_corners=False, no antialias), in float32."""
    scale = np.float32(in_size) / np.float32(out_size)
    src = scale * (np.arange(out_size, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5)
    src = np.maximum(src, np.float32(0))
    i0 = src.astype(np.int64)
    weight = (src - i0.astype(np.float32)).astype(np.float32)
    return i0, np.minimum(i0 + 1, in_size - 1), np.float32(1) - weight, weight


def upstream_grid_region(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """The edit region on DragFlow's feature grid (dragger.py:362-367): bilinear resize, then > 0.5."""
    gw, gh = feature_grid(width, height)
    m = mask.astype(np.float32)
    y0, y1, wy0, wy1 = _linear_taps(height, gh)
    x0, x1, wx0, wx1 = _linear_taps(width, gw)
    top, bottom = m[y0], m[y1]
    top = top[:, x0] * wx0 + top[:, x1] * wx1
    bottom = bottom[:, x0] * wx0 + bottom[:, x1] * wx1
    return top * wy0[:, None] + bottom * wy1[:, None] > 0.5


def upstream_grid_start(mask: np.ndarray, width: int, height: int):
    """Upstream compute_centroid on the grid region; None when a thin mask vanishes there.

    Upstream would then warn and drag from grid (0, 0), so such a region cannot be planned.
    """
    region = upstream_grid_region(mask, width, height)
    if not region.any():
        return None
    ys, xs = np.nonzero(region)
    return [int(np.round(xs.mean())), int(np.round(ys.mean()))]


def grid_rotation_degrees(begin_cell, end, anchor, width: int, height: int) -> float:
    """The angle upstream rotates by (dragger_utils.py:109-121), from grid-rounded points."""
    a, e = grid_point(anchor, width, height), grid_point(end, width, height)
    raw = math.degrees(
        math.atan2(e[1] - a[1], e[0] - a[0]) - math.atan2(begin_cell[1] - a[1], begin_cell[0] - a[0])
    )
    return (raw + 180) % 360 - 180


def grid_ok(start, end, anchor, width, height) -> bool:
    """Reject moves which collapse to the same point on upstream's feature grid."""
    grid = feature_grid(width, height)
    fitted = [grid_point(p, width, height) for p in [start, end] + ([anchor] if anchor else [])]
    if fitted[0] == fitted[1] or any(not 0 <= p[i] < grid[i] for p in fitted for i in (0, 1)):
        return False
    return not (anchor and fitted[2] == fitted[0])


def attempt_seed(cfg: dict, source: Source, attempt_index: int) -> int:
    return int(digest([cfg["run"]["seed"], source.source_id, source.pixel_sha256, attempt_index])[:8], 16)


def sample_edit(
    source: Source, profile: dict, cfg: dict, attempt_index: int, run_dir: Path, *, proposals=None
) -> EditRequest:
    if cfg["sampler"]["version"] == "object_region_v2":
        if proposals is None:
            raise DataError(f"region_proposals_missing: {source.source_id}")
        return sample_object_region(source, profile, cfg, attempt_index, run_dir, proposals)
    settings = cfg["sampler"]
    seed = attempt_seed(cfg, source, attempt_index)
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
        if not grid_ok(start, end, anchor, w, h):
            continue
        break
    else:
        raise DataError("Sampler could not form valid geometry; inspect image dimensions and sampler ranges")
    eid = f"{source.source_id}_{attempt_index:08d}_{digest([seed, settings, cfg['generator']])[:10]}"
    mask_path = f"edit_regions/{eid}.png"
    atomic_image(within(run_dir, mask_path), mask)
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
        profile["summary"],
        _edit_text(op, "the selected region", start, end),
        file_hash(within(run_dir, mask_path)),
    )


def _edit_text(op, target, start, end) -> str:
    return (
        f"Apply a local {op} to {target} from {start} toward {end}. "
        "Preserve the event, any damage or defect type, and the identifying traits of the subject. "
        "Do not repair damage or remove an accident."
    )


ROTATION_SCAN_STEP_DEGREES = 0.5  # Static feasibility scan only; sampled angles stay continuous.
# Below this anchor distance on the feature grid, rounding distorts the executed angle (QA2 N2).
MIN_ROTATION_RADIUS_CELLS = 4.0


def _rotate(point, anchor, degrees: float) -> list[int]:
    radians = math.radians(degrees)
    dx, dy = point[0] - anchor[0], point[1] - anchor[1]
    return [
        round(anchor[0] + dx * math.cos(radians) - dy * math.sin(radians)),
        round(anchor[1] + dx * math.sin(radians) + dy * math.cos(radians)),
    ]


def _shift_box(op: str, option: dict, width: int, height: int):
    """Allowed integer shifts: relocation keeps the entity bbox in the image, deformation keeps the end point."""
    if op == "relocation":
        x0, y0, x1, y1 = option["bbox"]
    else:
        (x0, y0), (x1, y1) = option["start"], option["start"]
    return (-x0, width - 1 - x1), (-y0, height - 1 - y1)


def valid_shifts(start, box, distance, width: int, height: int, begin=None):
    """Every integer shift in box with |shift| in distance that moves start to another feature-grid cell.

    begin is upstream's grid start (the region centroid on the grid); the target must avoid that cell too.

    Returns (dx, dy, weight); weight 1/|shift| makes distance and direction approximately uniform
    within the allowed set, so sampling from it never has to retry.
    """
    (xlo, xhi), (ylo, yhi) = box
    reach = math.floor(distance[1])
    dx = np.arange(max(xlo, -reach), min(xhi, reach) + 1)
    dy = np.arange(max(ylo, -reach), min(yhi, reach) + 1)
    gx, gy = (a.ravel() for a in np.meshgrid(dx, dy))
    length = np.hypot(gx, gy)
    grid = feature_grid(width, height)
    fitted = grid_point(start, width, height)
    # Same arithmetic as grid_point, vectorized.
    fx = np.round((start[0] + gx) / width * grid[0])
    fy = np.round((start[1] + gy) / height * grid[1])
    ok = (
        (length >= distance[0])
        & (length <= distance[1])
        & ((fx != fitted[0]) | (fy != fitted[1]))
        & (fx >= 0)
        & (fx < grid[0])
        & (fy >= 0)
        & (fy < grid[1])
    )
    if begin is not None:
        ok &= (fx != begin[0]) | (fy != begin[1])
    if not (0 <= fitted[0] < grid[0] and 0 <= fitted[1] < grid[1]):
        ok[:] = False
    return gx[ok], gy[ok], 1 / length[ok]


def _grid_radius(option: dict, width: int, height: int) -> float:
    anchor = grid_point(option["anchor"], width, height)
    return math.hypot(option["grid_start"][0] - anchor[0], option["grid_start"][1] - anchor[1])


def _rotation_end_ok(option: dict, end, width: int, height: int) -> bool:
    return (
        0 <= end[0] < width
        and 0 <= end[1] < height
        and end != option["start"]
        and grid_ok(option["start"], end, option["anchor"], width, height)
        and grid_point(end, width, height) != option["grid_start"]
    )


def _rotation_feasible(option: dict, degrees_range, width: int, height: int) -> bool:
    if _grid_radius(option, width, height) < MIN_ROTATION_RADIUS_CELLS:
        return False
    lo, hi = degrees_range
    for degrees in np.append(np.arange(lo, hi, ROTATION_SCAN_STEP_DEGREES), hi).tolist():
        end = _rotate(option["start"], option["anchor"], degrees)
        if abs(degrees) >= 2 and _rotation_end_ok(option, end, width, height):
            return True
    return False


def displacement_range(cfg: dict, width: int, height: int) -> list[float]:
    return [f * math.hypot(width, height) for f in cfg["sampler"]["displacement_diagonal_fraction"]]


def region_options(proposals: ProposalSet, cfg: dict, width: int, height: int) -> dict:
    """Statically usable proposals per operation and level: {op: {level: [option, ...]}}."""
    return region_table(proposals, cfg, width, height)[0]


def region_table(proposals: ProposalSet, cfg: dict, width: int, height: int) -> tuple[dict, dict]:
    """(options, excluded): usable options per operation and level, and counts of proposals left out.

    An option is kept only if at least one valid geometry exists for it, so sampling cannot stall on it.
    A proposal is left out entirely when its region vanishes on DragFlow's feature grid or the grid centroid
    upstream drags from lies more than one cell from the planned start (thin masks, QA2 N1).
    """
    settings = cfg["sampler"]
    key = digest([settings, width, height])
    if key in proposals.cache:
        return proposals.cache[key]
    region = settings["object_region"]
    distance = displacement_range(cfg, width, height)
    options: dict = {}
    excluded: dict[str, int] = {}
    for item, level, entity in proposals.items():
        mask = proposals.masks[item["proposal_id"]]
        start = centroid(mask)  # DragFlow's actual drag start.
        begin = upstream_grid_start(mask, width, height)
        fitted = grid_point(start, width, height)
        reason = (
            "grid_empty"
            if begin is None
            else "grid_centroid_shift"
            if max(abs(begin[0] - fitted[0]), abs(begin[1] - fitted[1])) > 1
            else None
        )
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        option = {
            "proposal_id": item["proposal_id"],
            "level": level,
            "phrase": entity["phrase"],
            "bbox": item["bbox"],
            "start": start,
            "grid_start": begin,
            "select": inner_point(mask),  # Only picks the contour upstream.
        }
        for op in settings["operations"]:
            if level not in region["granularity"][op]:
                continue
            if op == "rotation":
                e_mask = proposals.masks[entity["proposal_id"]]
                contact = dilate(mask, region["contact_dilation_px"]) & e_mask & ~mask
                if not contact.any():
                    continue  # No joint with the rest of the entity: no physical pivot.
                ys, xs = np.nonzero(contact)
                candidate = {**option, "anchor": [round(float(xs.mean())), round(float(ys.mean()))]}
                if not _rotation_feasible(candidate, settings["rotation_degrees"], width, height):
                    continue
            else:
                candidate = option
                box = _shift_box(op, option, width, height)
                if not len(valid_shifts(option["start"], box, distance, width, height, begin)[0]):
                    continue
            options.setdefault(op, {}).setdefault(level, []).append(candidate)
    proposals.cache[key] = (options, excluded)
    return options, excluded


def sample_object_region(
    source: Source, profile: dict, cfg: dict, attempt_index: int, run_dir: Path, proposals: ProposalSet
) -> EditRequest:
    settings = cfg["sampler"]
    region = settings["object_region"]
    seed = attempt_seed(cfg, source, attempt_index)
    rng = random.Random(seed)
    w, h = source.width, source.height
    options = region_options(proposals, cfg, w, h)
    operations = [op for op in settings["operations"] if op in options]
    if not operations:
        raise DataError(f"source_no_region: {source.source_id} has no usable proposal")
    distance = displacement_range(cfg, w, h)
    for _ in range(512):  # Only rotation angles can be rejected; shifts are drawn from the valid set.
        op = rng.choice(operations)
        level = rng.choice([lv for lv in region["granularity"][op] if lv in options[op]])
        option = rng.choice(options[op][level])
        start, anchor = option["start"], option.get("anchor")
        if op == "rotation":
            degrees = rng.uniform(*settings["rotation_degrees"])
            if abs(degrees) < 2:
                continue
            end = _rotate(start, anchor, degrees)
            if not _rotation_end_ok(option, end, w, h):
                continue
            params = {
                "requested_rotation_degrees": degrees,
                # What upstream executes after rounding begin/target/anchor to the feature grid.
                "grid_rotation_degrees": grid_rotation_degrees(option["grid_start"], end, anchor, w, h),
                "grid_radius_cells": _grid_radius(option, w, h),
                "anchor_method": region["rotation_anchor"],
                "contact_dilation_px": region["contact_dilation_px"],
                "anchor_coordinates": "normalized_original_px",
            }
        else:
            box = _shift_box(op, option, w, h)
            dx, dy, weight = valid_shifts(start, box, distance, w, h, option["grid_start"])
            cumulative = np.cumsum(weight)
            i = min(
                int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")), len(dx) - 1
            )
            shift = [int(dx[i]), int(dy[i])]
            end = [start[0] + shift[0], start[1] + shift[1]]
            params = {"displacement_pixels": shift}
            if op == "relocation":
                x0, y0, x1, y1 = option["bbox"]
                params["entity_bbox_after"] = [x0 + shift[0], y0 + shift[1], x1 + shift[0], y1 + shift[1]]
        if not (0 <= end[0] < w and 0 <= end[1] < h) or end == start:
            continue
        if not grid_ok(start, end, anchor, w, h):
            continue
        params["upstream_grid_start"] = option["grid_start"]
        break
    else:
        raise SamplingSkipped(
            f"sampling_skipped: {source.source_id} attempt {attempt_index} found no valid geometry"
        )
    eid = f"{source.source_id}_{attempt_index:08d}_{digest([seed, settings, cfg['generator'], proposals.sha256])[:10]}"
    mask_path = f"edit_regions/{eid}.png"
    # Byte copy keeps the plan mask hash identical to the frozen proposal mask.
    proposal_mask = within(run_dir, f"{proposals.folder}/{option['proposal_id']}.png", must_exist=True)
    atomic_bytes(within(run_dir, mask_path), proposal_mask.read_bytes())
    target = f"the {option['phrase']}" if level == "entity" else f"a part of the {option['phrase']}"
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
        profile["summary"],
        _edit_text(op, target, start, end),
        file_hash(within(run_dir, mask_path)),
        region_proposal_id=option["proposal_id"],
        region_level=level,
        region_phrase=option["phrase"],
        region_select_point=option["select"],
    )


def validate_plan(plan: EditRequest, source: Source, run_dir: Path) -> None:
    if plan.source_id != source.source_id or plan.operation not in {"relocation", "deformation", "rotation"}:
        raise DataError("Invalid source or operation in edit plan")
    extra = [p for p in (plan.anchor_point, plan.region_select_point) if p is not None]
    for point in [plan.source_point, plan.target_point] + extra:
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
    region = (plan.region_proposal_id, plan.region_level, plan.region_phrase)
    v2 = plan.sampler_version == "object_region_v2"
    if v2:
        if (
            plan.region_level not in ("entity", "part")
            or not all(isinstance(x, str) and x for x in region)
            or plan.region_select_point is None
        ):
            raise DataError("object_region_v2 plan needs proposal id, level, phrase and select point")
    elif region != (None, None, None) or plan.region_select_point is not None:
        raise DataError("Region proposal fields belong to object_region_v2 plans only")
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
        # Upstream splits operation.png by contour and keeps the one containing centroids[0].
        x, y = map(round, plan.region_select_point if v2 else plan.source_point)
        if x >= source.width or y >= source.height or not arr.any() or arr[y, x] != 255:
            raise DataError(
                ("Region select point" if v2 else "Source point") + " must lie in a nonempty mask"
            )
        if v2:
            if not is_single_region(arr > 0):
                raise DataError("object_region_v2 mask must be one hole-free connected region")
            # Upstream drags from the region centroid; the plan's geometry must use the same start.
            if list(plan.source_point) != centroid(arr > 0):
                raise DataError("object_region_v2 source_point must be the mask centroid DragFlow drags from")
