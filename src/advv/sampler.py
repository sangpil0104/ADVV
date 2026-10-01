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


def grid_point(point, width, height, float32: bool = False):
    """The feature-grid point upstream scale_coordinates rounds point to.

    Upstream rounds torch.tensor(x / size * grid), a float32, half to even (dashboard_utils.py:209-228).
    float32=True reproduces that; object_region_v2 uses it. random_geometry_v1 keeps the float64 rounding
    its plans were made with; the two differ only where x / size * grid is within float32 error of k + 0.5.
    """
    grid = feature_grid(width, height)
    if float32:
        return [int(np.round(np.float32(point[i] / (width, height)[i] * grid[i]))) for i in (0, 1)]
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
    a, e = grid_point(anchor, width, height, True), grid_point(end, width, height, True)
    raw = math.degrees(
        math.atan2(e[1] - a[1], e[0] - a[0]) - math.atan2(begin_cell[1] - a[1], begin_cell[0] - a[0])
    )
    return (raw + 180) % 360 - 180


def grid_ok(start, end, anchor, width, height, float32: bool = False) -> bool:
    """Reject moves which collapse to the same point on upstream's feature grid."""
    grid = feature_grid(width, height)
    fitted = [grid_point(p, width, height, float32) for p in [start, end] + ([anchor] if anchor else [])]
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
    fitted = grid_point(start, width, height, True)
    # Same arithmetic as grid_point(float32=True), vectorized.
    fx = np.round(((start[0] + gx) / width * grid[0]).astype(np.float32))
    fy = np.round(((start[1] + gy) / height * grid[1]).astype(np.float32))
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
    anchor = grid_point(option["anchor"], width, height, True)
    return math.hypot(option["grid_start"][0] - anchor[0], option["grid_start"][1] - anchor[1])


def _rotated(point, anchor, radians: float) -> tuple[float, float]:
    dx, dy = point[0] - anchor[0], point[1] - anchor[1]
    return (
        anchor[0] + dx * math.cos(radians) - dy * math.sin(radians),
        anchor[1] + dx * math.sin(radians) + dy * math.cos(radians),
    )


def _cell_pixel(cell, near, width: int, height: int):
    """The pixel nearest to near whose feature-grid point is cell, or None when no pixel maps there."""
    grid = feature_grid(width, height)
    pixel = []
    for i, size in enumerate((width, height)):
        step = size / grid[i]
        lo, hi = max(0, math.floor((cell[i] - 1) * step)), min(size - 1, math.ceil((cell[i] + 1) * step))
        hits = [x for x in range(lo, hi + 1) if np.round(np.float32(x / size * grid[i])) == cell[i]]
        if not hits:
            return None
        pixel.append(min(hits, key=lambda x: (abs(x - near[i]), x)))
    return pixel


def rotation_target(option: dict, degrees: float, width: int, height: int, min_executed: float = 0.0):
    """(end, executed angle): the target whose grid cell makes upstream rotate closest to degrees.

    Upstream rotates by the angle between grid_start and the grid-rounded target about the grid-rounded
    anchor (dragger_utils.py:109-121). Rotating the pixel start and rounding the result can land on a cell
    a whole cell step away (T015: -10.5 deg became -17.7 deg at 7.2 cells), so the target is chosen among
    the cells around the rotated grid start instead, then placed on the pixel of that cell nearest to the
    rotated pixel start. A cell counts only if its executed angle has the sign of degrees and
    |executed| >= min_executed: with a tolerance above |degrees| a radial cell (0 deg) or a cell turning the
    other way could otherwise be the closest (T015 QA M1). None when no neighbouring cell is a valid target.
    """
    a = grid_point(option["anchor"], width, height, True)
    b = option["grid_start"]
    radians = math.radians(degrees)
    ideal = _rotated(b, a, radians)
    near = _rotated(option["start"], option["anchor"], radians)
    best = None
    for cy in range(round(ideal[1]) - 1, round(ideal[1]) + 2):
        for cx in range(round(ideal[0]) - 1, round(ideal[0]) + 2):
            end = _cell_pixel((cx, cy), near, width, height)
            if end is None or not _rotation_end_ok(option, end, width, height):
                continue
            executed = grid_rotation_degrees(b, end, option["anchor"], width, height)
            if executed * degrees <= 0 or abs(executed) < min_executed:
                continue
            error = abs(_angle_diff(executed, degrees))
            key = (error, math.hypot(cx - ideal[0], cy - ideal[1]), cy, cx)
            if best is None or key < best[0]:
                best = (key, end, executed)
    return None if best is None else (best[1], best[2])


def _angle_diff(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


def rotation_tolerance(region: dict, degrees: float) -> float:
    """Largest allowed |executed - requested| angle for a requested rotation."""
    return max(region["rotation_max_error_degrees"], region["rotation_max_error_fraction"] * abs(degrees))


def _rotation_fit(option: dict, degrees: float, region: dict, width: int, height: int):
    """rotation_target when its executed angle is within the tolerance, else None."""
    fit = rotation_target(option, degrees, width, height, region["rotation_min_executed_degrees"])
    if fit is None or abs(_angle_diff(fit[1], degrees)) > rotation_tolerance(region, degrees):
        return None
    return fit


def _rotation_end_ok(option: dict, end, width: int, height: int) -> bool:
    return (
        0 <= end[0] < width
        and 0 <= end[1] < height
        and end != option["start"]
        and grid_ok(option["start"], end, option["anchor"], width, height, True)
        and grid_point(end, width, height, True) != option["grid_start"]
    )


def _rotation_feasible(option: dict, degrees_range, region: dict, width: int, height: int) -> bool:
    if _grid_radius(option, width, height) < MIN_ROTATION_RADIUS_CELLS:
        return False
    lo, hi = degrees_range
    for degrees in np.append(np.arange(lo, hi, ROTATION_SCAN_STEP_DEGREES), hi).tolist():
        if abs(degrees) >= 2 and _rotation_fit(option, degrees, region, width, height):
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
    upstream drags from lies more than one cell from the planned start (thin masks, QA2 N1), or when the
    experimental object_region.region_phrase_filter does not match it (counted as phrase_filter).
    """
    settings = cfg["sampler"]
    key = digest([settings, width, height])
    if key in proposals.cache:
        return proposals.cache[key]
    region = settings["object_region"]
    phrase_filter = region.get("region_phrase_filter")
    distance = displacement_range(cfg, width, height)
    options: dict = {}
    excluded: dict[str, int] = {}
    for item, level, entity in proposals.items():
        if phrase_filter is not None and not phrase_matches(phrase_filter, item, level):
            excluded["phrase_filter"] = excluded.get("phrase_filter", 0) + 1
            continue
        mask = proposals.masks[item["proposal_id"]]
        start = centroid(mask)  # DragFlow's actual drag start.
        begin = upstream_grid_start(mask, width, height)
        fitted = grid_point(start, width, height, True)
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
            "proposal_phrase": item.get("phrase"),  # A part's own text phrase; None for point parts.
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
                if not _rotation_feasible(candidate, settings["rotation_degrees"], region, width, height):
                    continue
            else:
                candidate = option
                box = _shift_box(op, option, width, height)
                if not len(valid_shifts(option["start"], box, distance, width, height, begin)[0]):
                    continue
            options.setdefault(op, {}).setdefault(level, []).append(candidate)
    proposals.cache[key] = (options, excluded)
    return options, excluded


def phrase_matches(phrase_filter: dict, item: dict, level: str) -> bool:
    """Experimental region_phrase_filter: the proposal's own phrase (a point part has none) contains one of
    the listed strings, case-insensitively, at the filtered level (null: either level)."""
    if phrase_filter["level"] not in (None, level):
        return False
    phrase = (item.get("phrase") or "").lower()
    return any(text.lower() in phrase for text in phrase_filter["phrases_contain"])


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
            fit = _rotation_fit(option, degrees, region, w, h)
            if fit is None:
                continue  # No target cell executes this angle within the tolerance; draw again.
            end, executed = fit
            params = {
                "requested_rotation_degrees": degrees,
                # What upstream executes after rounding begin/target/anchor to the feature grid.
                "grid_rotation_degrees": executed,
                "rotation_tolerance_degrees": rotation_tolerance(region, degrees),
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
        if not grid_ok(start, end, anchor, w, h, True):
            continue
        params["upstream_grid_start"] = option["grid_start"]
        if region.get("region_phrase_filter") is not None:
            params["region_phrase_filter"] = region["region_phrase_filter"]
            params["region_proposal_phrase"] = option["proposal_phrase"]
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
        grid_point(p, source.width, source.height, plan.sampler_version == "object_region_v2")
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
