from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .contracts import Source
from .errors import DataError
from .storage import atomic_image, digest, file_hash, load_rgb, pixel_hash, safe_id, within

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def scan_sources(cfg: dict) -> tuple[list[Source], dict]:
    import json

    d = cfg["dataset"]
    root = Path(d["root"]).resolve()
    if d.get("input_dir"):
        folder = Path(d["input_dir"]).resolve()
        if not folder.is_dir():
            raise DataError(f"Input folder does not exist: {folder}")
        rows = []
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                if not path.resolve().is_relative_to(root):
                    raise DataError(f"Input file escapes dataset root: {path}")
                rel = path.relative_to(root).as_posix()
                sid = "src_" + digest(rel)[:16]
                rows.append(
                    dict(
                        source_id=sid,
                        image_path=rel,
                        split="source",
                        group_id=sid,
                        group_inferred=True,
                        domain="unspecified",
                        preserve_hint=None,
                        label=None,
                    )
                )
    else:
        rows = []
        for line in Path(d["manifest"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("schema_version") != "1.1":
                    raise DataError("Source manifest schema_version must be 1.1")
                rows.append(row)
    if not rows:
        raise DataError("No supported input images")
    sources, seen_ids, groups, hashes, duplicates = [], set(), {}, {}, []
    for row in rows:
        sid = safe_id(row["source_id"])
        if sid in seen_ids:
            raise DataError(f"Duplicate source_id: {sid}")
        seen_ids.add(sid)
        split = row.get("split", "source")
        if split not in {"source", "train", "val", "test"}:
            raise DataError(f"Invalid split: {split}")
        group = row.get("group_id") or sid
        if group in groups and groups[group] != split:
            raise DataError(f"Group leakage: {group}")
        groups[group] = split
        path = within(root, row["image_path"], must_exist=True)
        image = load_rgb(path)
        if min(image.size) < 32:
            raise DataError(f"DragFlow requires images at least 32 pixels per side: {path}")
        ph = pixel_hash(image)
        if ph in hashes:
            prev = hashes[ph]
            if prev["split"] != split:
                raise DataError(f"Pixel-identical images cross splits: {sid} and {prev['source_id']}")
            duplicates.append({"source_id": sid, "duplicate_of": prev["source_id"]})
            continue
        hashes[ph] = {"source_id": sid, "split": split}
        hint = row.get("preserve_hint")
        if hint is not None and (not isinstance(hint, str) or not hint.strip()):
            raise DataError("preserve_hint must be null or nonempty text")
        sources.append(
            Source(
                sid,
                str(path),
                str(path),
                file_hash(path),
                ph,
                *image.size,
                split,
                group,
                row.get("domain", "unspecified"),
                hint,
                row.get("label"),
                row.get("group_inferred", not bool(row.get("group_id"))),
            )
        )
    eligible = [s for s in sources if s.split in d["eligible_splits"]]
    if not eligible:
        raise DataError("No source/train images eligible for augmentation")
    return sources, {"duplicates": duplicates, "eligible": len(eligible), "total_unique": len(sources)}


def snapshot_sources(sources: list[Source], run_dir: Path) -> list[Source]:
    result = []
    for source in sources:
        image = load_rgb(Path(source.original_path))
        if (
            pixel_hash(image) != source.pixel_sha256
            or file_hash(Path(source.original_path)) != source.file_sha256
        ):
            raise DataError(f"Source changed during ingest: {source.source_id}")
        relative = f"inputs/images/{source.source_id}.png"
        atomic_image(within(run_dir, relative), image)
        result.append(replace(source, image_path=relative))
    return result
