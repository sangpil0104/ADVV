"""CPU-only, deterministic previews from persisted requested/effective geometry."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from .storage import atomic_image, atomic_json, digest, file_hash, load_rgb, within
from .errors import DataError

RENDERER_VERSION = "pillow_v1"


def _font(size=16):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _status(record, name):
    check = record.get("checks", {}).get(name, {})
    return (
        check.get("response", {}).get("answer")
        if check.get("response")
        else check.get("status", "pending").upper()
    )


def render(run_dir: Path, source: dict, record: dict, settings: dict) -> dict:
    if not settings["enabled"]:
        return {"status": "disabled"}
    cid, plan = record["candidate_id"], record["plan"]
    folder = within(run_dir, f"visualizations/{cid}")
    original = load_rgb(within(run_dir, source["image_path"], must_exist=True))
    w, h = original.size
    effective = record.get("effective")
    geometry = effective or dict(
        source_point=plan["source_point"],
        target_point=plan["target_point"],
        anchor_point=plan.get("anchor_point"),
        region_mask_path=plan["region_mask_path"],
    )
    mask_path = within(run_dir, geometry["region_mask_path"], must_exist=True)
    if geometry.get("mask_sha256") and file_hash(mask_path) != geometry["mask_sha256"]:
        raise DataError("Effective visualization mask changed")
    with Image.open(mask_path) as image:
        mask = image.convert("L").resize((w, h), Image.Resampling.NEAREST)
    area = int(np.count_nonzero(np.asarray(mask)))
    factor = min(1.0, settings["max_panel_side_px"] / max(w, h))
    pw, ph = max(1, round(w * factor)), max(1, round(h * factor))
    # Actual rounded panel sizes determine transform, never a guessed scale.
    sx, sy = pw / w, ph / h
    overlay = original.resize((pw, ph), Image.Resampling.LANCZOS).convert("RGBA")
    preview_mask = mask.resize((pw, ph), Image.Resampling.NEAREST)
    blue = Image.new("RGBA", overlay.size, ImageColor.getrgb(settings["area_color"]) + (0,))
    blue.putalpha(preview_mask.point(lambda x: round(x * settings["area_alpha"])))
    overlay = Image.alpha_composite(overlay, blue)
    if settings["draw_area_outline"]:
        binary = preview_mask.point(lambda x: 255 if x else 0)
        border = np.asarray(binary) - np.asarray(binary.filter(ImageFilter.MinFilter(3)))
        outline = Image.new("RGBA", overlay.size, settings["area_color"])
        outline.putalpha(Image.fromarray(border))
        overlay = Image.alpha_composite(overlay, outline)
    draw = ImageDraw.Draw(overlay)
    start = tuple(geometry["source_point"][i] * (sx, sy)[i] for i in range(2))
    end = tuple(geometry["target_point"][i] * (sx, sy)[i] for i in range(2))
    draw.line([start, end], fill="black", width=6)
    draw.line([start, end], fill=settings["arrow_color"], width=3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = min(12, max(5, math.dist(start, end) * 0.4))
    wings = [
        (end[0] - length * math.cos(angle + offset), end[1] - length * math.sin(angle + offset))
        for offset in (-0.5, 0.5)
    ]
    draw.line([wings[0], end, wings[1]], fill="black", width=6)
    draw.line([wings[0], end, wings[1]], fill=settings["arrow_color"], width=3)
    radius = 6

    def label(x, y, text, direction):
        font = _font(15)
        x = max(2, min(pw - 21, x + 10 * direction))
        y = max(2, min(ph - 23, y - 25 if direction < 0 else y + 8))
        draw.text((x, y), text, fill="white", font=font, stroke_width=2, stroke_fill="black")

    x, y = start
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=settings["start_color"],
        outline="white",
        width=2,
    )
    label(x, y, "S", -1)
    x, y = end
    for a, b in (((x - radius, y), (x + radius, y)), ((x, y - radius), (x, y + radius))):
        draw.line((a, b), fill="black", width=6)
        draw.line((a, b), fill=settings["end_color"], width=3)
    label(x, y, "E", 1)
    anchor = geometry.get("anchor_point")
    if anchor and settings["show_anchor"]:
        x, y = anchor[0] * sx, anchor[1] * sy
        draw.polygon(
            [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)], fill=settings["anchor_color"], outline="white"
        )
        label(x, y, "A", -1)
    mode = "EFFECTIVE INPUT" if effective else "REQUESTED / NOT EXECUTED"

    def point(value):
        return "(" + ", ".join(f"{x:.2f}" for x in value) + ")"

    lines = [
        f"{cid} | {plan['operation']} | seed={plan['seed']}",
        f"{mode} | normalized original px | {w} x {h}",
        f"S start={point(geometry['source_point'])} | E end={point(geometry['target_point'])}",
        f"area={area} px ({area / (w * h):.2%}) | blue=selected area | arrow=intended movement",
        f"Physical: {_status(record, 'physical')} | Semantic: {_status(record, 'semantic')} | {record['status']}",
    ]
    if plan.get("region_level"):
        lines.append(
            f"region={plan['region_level']} | phrase={plan['region_phrase']} | proposal={plan['region_proposal_id']}"
        )
    if plan.get("region_select_point"):
        lines.append(
            f"S = mask centroid, the start DragFlow drags from | contour select={point(plan['region_select_point'])}"
        )
    if anchor:
        lines.append(f"A anchor={point(anchor)}")
    if effective:
        lines.append(
            f"Requested S={point(plan['source_point'])} E={point(plan['target_point'])}; backend grid/rounding recorded"
        )
    if plan["operation_params"]:
        lines.append(str(plan["operation_params"]))
    font = _font()
    artifacts = {}

    def panel_image(panels, headers):
        total_width = max(640, len(panels) * pw + (len(panels) + 1) * 16)
        wrapped = [
            part for line in lines for part in textwrap.wrap(line, width=max(30, (total_width - 32) // 9))
        ]
        canvas = Image.new("RGB", (total_width, ph + 64 + 24 * len(wrapped)), "#171B24")
        d = ImageDraw.Draw(canvas)
        for i, (panel, heading) in enumerate(zip(panels, headers)):
            canvas.paste(panel.convert("RGB"), (16 + i * (pw + 16), 36))
            d.text((16 + i * (pw + 16), 10), heading, fill="white", font=font)
        for i, line in enumerate(wrapped):
            d.text((16, ph + 48 + 24 * i), line, fill="#EEEEEE", font=font)
        return canvas

    if settings["save_drag_plan"]:
        atomic_image(
            folder / "drag_plan.png", panel_image([overlay], ["Original + drag settings (S / E / A)"])
        )
        artifacts["drag_plan"] = str((folder / "drag_plan.png").relative_to(run_dir))
    candidate_path = record.get("image_path")
    suppress = record.get("delete_image", False)
    if settings["save_comparison"] and candidate_path and not suppress:
        candidate = load_rgb(within(run_dir, candidate_path, must_exist=True))
        candidate.thumbnail((pw, ph), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (pw, ph), "#171B24")
        canvas.paste(candidate, ((pw - candidate.width) // 2, (ph - candidate.height) // 2))
        atomic_image(
            folder / "comparison.png",
            panel_image([overlay, canvas], ["Original + drag settings", "Generated image"]),
        )
        artifacts["comparison"] = str((folder / "comparison.png").relative_to(run_dir))
    metadata = {
        "source_id": source["source_id"],
        "candidate_id": cid,
        "plan_hash": digest(plan),
        "source_pixel_sha256": source["pixel_sha256"],
        "candidate_pixel_sha256": record.get("pixel_sha256"),
        "original_size": [w, h],
        "requested": plan,
        "effective": effective,
        "display_mode": mode,
        "display_geometry": geometry,
        "mask_sha256": file_hash(mask_path),
        "area_pixels": area,
        "area_fraction": area / (w * h),
        "area_coordinate_system": "normalized_original",
        "preview_transform": {"scale": [sx, sy], "offset": [16, 36]},
        "physical": _status(record, "physical"),
        "semantic": _status(record, "semantic"),
        "candidate_status": record["status"],
        "renderer_version": RENDERER_VERSION,
        "settings_hash": digest(settings),
        "artifacts": artifacts,
        "status": "ready",
    }
    if settings["save_metadata"]:
        atomic_json(folder / "metadata.json", metadata)
        artifacts["metadata"] = str((folder / "metadata.json").relative_to(run_dir))
    return {"status": "ready", "artifacts": artifacts, "renderer_version": RENDERER_VERSION}
