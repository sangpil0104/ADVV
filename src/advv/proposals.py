"""Frozen entity/part region proposals for the object_region_v2 sampler.

Model workers only produce raw masks and scores; cleanup, filtering, storage and
validation live here so they run on CPU with fixtures.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .contracts import Source
from .errors import DataError
from .storage import (
    atomic_bytes,
    atomic_image,
    atomic_json,
    digest,
    file_hash,
    now,
    read_json,
    safe_id,
    within,
)

SCHEMA_VERSION = "1.2"
RAW_SCHEMA_VERSION = "1.2"
PROPOSAL_BACKENDS = {"sam3", "grounding_dino_sam2.1"}
FIXTURE_BACKEND = "fixture"
PART_SOURCES = ("text", "point")
TEXT_PART_FORMS = ("subject_part", "part")


def _runs(mask: np.ndarray, diagonal: bool):
    """Row runs of True pixels and their union-find component roots."""
    runs, parent = [], []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    previous = []
    for y, row in enumerate(mask):
        padded = np.concatenate(([False], row, [False])).astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        current = []
        for start, end in zip(edges[::2].tolist(), edges[1::2].tolist()):
            index = len(runs)
            runs.append((y, start, end))
            parent.append(index)
            reach = 1 if diagonal else 0
            for other in previous:
                _, s, e = runs[other]
                if s < end + reach and start < e + reach:
                    a, b = root(index), root(other)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
            current.append(index)
        previous = current
    return runs, [root(i) for i in range(len(runs))]


def component_count(mask: np.ndarray) -> int:
    return len(set(_runs(mask.astype(bool), True)[1]))


def largest_component(mask: np.ndarray) -> np.ndarray:
    runs, roots = _runs(mask.astype(bool), True)
    if not runs:
        return np.zeros(mask.shape, bool)
    sizes: dict[int, int] = {}
    for (_, s, e), r in zip(runs, roots):
        sizes[r] = sizes.get(r, 0) + e - s
    best = min(sizes, key=lambda r: (-sizes[r], r))
    result = np.zeros(mask.shape, bool)
    for (y, s, e), r in zip(runs, roots):
        if r == best:
            result[y, s:e] = True
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background not 4-connected to the border, like a filled external contour."""
    mask = mask.astype(bool)
    runs, roots = _runs(~mask, False)
    h, w = mask.shape
    outside = {r for (y, s, e), r in zip(runs, roots) if y in (0, h - 1) or s == 0 or e == w}
    result = mask.copy()
    for (y, s, e), r in zip(runs, roots):
        if r not in outside:
            result[y, s:e] = True
    return result


def clean_mask(mask: np.ndarray) -> np.ndarray:
    return fill_holes(largest_component(mask))


def is_single_region(mask: np.ndarray) -> bool:
    mask = mask.astype(bool)
    return bool(mask.any()) and component_count(mask) == 1 and np.array_equal(fill_holes(mask), mask)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Square (2k+1) kernel: every pixel within Chebyshev distance k of the mask."""
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * pixels + 1))) > 0


def centroid(mask: np.ndarray) -> list[int]:
    """Rounded pixel centroid: where DragFlow starts the drag, even when it lies outside a concave mask."""
    ys, xs = np.nonzero(mask)
    return [int(round(float(xs.mean()))), int(round(float(ys.mean())))]


def inner_point(mask: np.ndarray) -> list[int]:
    """Centroid if it lies inside the mask, else the nearest mask pixel (upstream picks the contour by point)."""
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    x, y = round(cx), round(cy)
    if mask[y, x]:
        return [int(x), int(y)]
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return [int(xs[i]), int(ys[i])]


def part_prompt_points(entity: np.ndarray, count: int, seed: int) -> list[list[int]]:
    """Seeded stratified points inside an entity for part (multimask) prompts."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bbox(entity)
    k = math.ceil(math.sqrt(count))
    xs = np.linspace(x0, x1 + 1, k + 1).astype(int)
    ys = np.linspace(y0, y1 + 1, k + 1).astype(int)
    points = []
    for j in range(k):
        for i in range(k):
            cy, cx = np.nonzero(entity[ys[j] : ys[j + 1], xs[i] : xs[i + 1]])
            if len(cx):
                n = int(rng.integers(len(cx)))
                points.append([int(cx[n] + xs[i]), int(cy[n] + ys[j])])
    if len(points) > count:
        points = [points[i] for i in sorted(rng.choice(len(points), count, replace=False).tolist())]
    return points


def part_seed(run_seed: int, source: Source, entity_index: int) -> int:
    """Seed of the part prompt points of one raw entity; fixed by run seed, source and worker order."""
    return int(digest([run_seed, source.source_id, source.pixel_sha256, "part_points", entity_index])[:8], 16)


def subject_parts_error(parts, subjects) -> str | None:
    """Why parts is not a v4 per-subject part list ([{"subject", "parts"}, ...]) for subjects, or None.

    Each entry names a distinct profile subject; a subject without an entry has no named parts.
    """
    if not isinstance(parts, list):
        return "parts must be a list of {subject, parts}"
    seen = set()
    for entry in parts:
        if not isinstance(entry, dict) or set(entry) != {"subject", "parts"}:
            return "each parts entry must be {subject, parts}"
        if not isinstance(entry["subject"], str) or entry["subject"] not in subjects or entry["subject"] in seen:
            return f"parts entry subject {entry['subject']!r} is not a distinct profile subject"
        names = entry["parts"]
        if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
            return f"parts of {entry['subject']!r} must be non-empty strings"
        seen.add(entry["subject"])
    return None


def subject_parts(parts, subject: str) -> list[str]:
    """Part names a v4 per-subject part list gives one subject (none without an entry)."""
    return next((list(entry["parts"]) for entry in parts if entry["subject"] == subject), [])


def text_part_phrases(subject: str, parts, forms) -> list[str]:
    """Text prompts for named parts of one entity, in parts order then form order, without repeats."""
    phrases = []
    for part in parts:
        for form in forms:
            phrase = f"{subject} {part}" if form == "subject_part" else part
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def containment(part: np.ndarray, entity: np.ndarray) -> float:
    """Fraction of the part mask that lies inside the entity mask."""
    area = np.count_nonzero(part)
    return np.count_nonzero(part & entity) / area if area else 0.0


def nested_duplicate(a: np.ndarray, b: np.ndarray, settings: dict) -> bool:
    """The smaller mask lies (almost) inside the larger and is nearly as large: the same part twice.

    A part that is much smaller than the one holding it (boom vs boom+arm) is a different part and stays.
    """
    small, large = sorted((np.count_nonzero(a), np.count_nonzero(b)))
    if not small:
        return False
    return (
        np.count_nonzero(a & b) / small >= settings["part_dedupe_containment"]
        and small / large >= settings["part_dedupe_area_ratio"]
    )


RAW_PART_KEYS = {"text": ("phrase", "score", "containment", "mask"), "point": ("point", "score", "mask")}


def _check_raw_entity(item) -> None:
    if not isinstance(item, dict) or any(key not in item for key in ("phrase", "score", "mask")):
        raise DataError("Raw entity must have phrase, score and mask")
    for part in item.get("parts", []):
        source = part.get("source") if isinstance(part, dict) else None
        if source not in PART_SOURCES:
            raise DataError(f"Raw part source {source!r} is not one of {PART_SOURCES}")
        missing = [key for key in RAW_PART_KEYS[source] if key not in part]
        if missing:
            raise DataError(f"Raw {source} part is missing {missing}")


def collect_raw(predictor, phrases, parts, settings: dict, seed_for) -> dict:
    """Query a proposal model: text prompt per phrase -> entities; then per entity named-part text prompts
    and seeded points inside it -> parts.

    predictor.text(phrase) yields (mask, score, box); predictor.point([x, y]) yields (mask, predicted_iou)
    per multimask output. parts is the per-subject part list: an entity is asked only for the parts of its
    own phrase. A text-part instance is offered to every entity that asked for the phrase, with its
    containment (share of the instance inside the cleaned entity mask). Masks that build_proposals rejects before reading them
    (score or containment below the setting) are dropped here; their scores stay in the receipt.
    """
    entities = []
    for phrase in phrases:
        for mask, score, box in predictor.text(phrase):
            entities.append({"phrase": phrase, "score": float(score), "box": box, "mask": mask, "parts": []})
    text_parts = list(parts) if settings["text_parts"] else []
    answers: dict[str, list] = {}
    for index, entity in enumerate(entities):
        if entity["score"] < settings["min_entity_score"]:
            continue
        cleaned = clean_mask(entity["mask"])
        if not cleaned.any():
            continue
        names = subject_parts(text_parts, entity["phrase"])
        for phrase in text_part_phrases(entity["phrase"], names, settings["text_part_forms"]):
            if phrase not in answers:
                answers[phrase] = predictor.text(phrase)
            for k, (mask, score, box) in enumerate(answers[phrase]):
                inside = containment(np.asarray(mask, bool), cleaned)
                keep = float(score) >= settings["min_text_part_score"] and (
                    inside >= settings["text_part_containment"]
                )
                entity["parts"].append(
                    {
                        "source": "text",
                        "phrase": phrase,
                        "instance_index": k,
                        "score": float(score),
                        "box": box,
                        "containment": inside,
                        "mask": mask if keep else None,
                    }
                )
        points = part_prompt_points(cleaned, settings["part_points_per_entity"], seed_for(index))
        for point in points:
            for k, (mask, score) in enumerate(predictor.point(point)):
                keep = float(score) >= settings["min_part_score"]
                entity["parts"].append(
                    {
                        "source": "point",
                        "point": point,
                        "multimask_index": k,
                        "score": float(score),
                        "mask": mask if keep else None,
                    }
                )
    return {"phrases": list(phrases), "parts": list(parts), "entities": entities}


def write_raw(run_dir: Path, folder: str, raw: dict, size: tuple[int, int], meta: dict) -> str:
    """Persist raw worker output (receipt): masks.npz first, then raw.json naming its hash."""
    w, h = size
    arrays, entities = {}, []

    def keep(key, mask):
        mask = np.asarray(mask, bool)
        if mask.shape != (h, w):
            raise DataError(f"Proposal mask shape {mask.shape} is not the source size {(h, w)}")
        arrays[key] = mask
        return key

    for i, entity in enumerate(raw["entities"]):
        parts = []
        for j, part in enumerate(entity["parts"]):
            mask = part["mask"]
            parts.append({**part, "mask": None if mask is None else keep(f"e{i:03d}_p{j:03d}", mask)})
        entities.append({**entity, "mask": keep(f"e{i:03d}", entity["mask"]), "parts": parts})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    masks = within(run_dir, f"{folder}/raw_masks.npz")
    atomic_bytes(masks, buffer.getvalue())
    record = {
        "schema_version": RAW_SCHEMA_VERSION,
        **meta,
        "size": [w, h],
        "phrases": list(raw["phrases"]),
        "parts": list(raw["parts"]),
        "entities": entities,
        "masks_file": "raw_masks.npz",
        "masks_sha256": file_hash(masks),
    }
    atomic_json(within(run_dir, f"{folder}/raw.json"), record)
    return file_hash(within(run_dir, f"{folder}/raw.json"))


def read_raw(run_dir: Path, folder: str) -> dict:
    """Load a raw receipt back into build_proposals input; refuses a changed or partial receipt."""
    try:
        record = read_json(within(run_dir, f"{folder}/raw.json", must_exist=True))
    except ValueError as exc:
        raise DataError(f"Malformed proposal receipt {folder}/raw.json") from exc
    if record.get("schema_version") != RAW_SCHEMA_VERSION:
        raise DataError(f"Unsupported proposal receipt in {folder}")
    path = within(run_dir, f"{folder}/{record['masks_file']}", must_exist=True)
    if file_hash(path) != record["masks_sha256"]:
        raise DataError(f"Proposal receipt masks changed in {folder}")
    w, h = record["size"]
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    def mask(key):
        if key not in arrays or arrays[key].shape != (h, w):
            raise DataError(f"Proposal receipt mask {key!r} missing or not {w}x{h}")
        return arrays[key].astype(bool)

    entities = [
        {
            **entity,
            "mask": mask(entity["mask"]),
            "parts": [{**p, "mask": None if p["mask"] is None else mask(p["mask"])} for p in entity["parts"]],
        }
        for entity in record["entities"]
    ]
    return {**record, "entities": entities}


def _in_range(value: float, bounds) -> bool:
    return bounds[0] <= value <= bounds[1]


def _part_order(parts: list) -> list[int]:
    """Named (text) parts first, then point parts; score descending within each, then worker order.

    The two scores are on different scales (detection confidence vs predicted IoU), so they are never
    compared: a named part wins a duplicate of a point mask.
    """
    rank = {source: n for n, source in enumerate(PART_SOURCES)}
    return sorted(range(len(parts)), key=lambda j: (rank[parts[j]["source"]], -parts[j]["score"], j))


def build_proposals(size: tuple[int, int], raw: dict, settings: dict) -> dict:
    """Clean, filter and deduplicate raw worker masks; deterministic for the same input.

    raw = {"phrases": [...], "parts": [{"subject", "parts"}, ...], "entities": [{"phrase", "score", "mask",
           "parts": [{"source": "text"|"point", "score", "mask", "phrase"?, "containment"?}]}]}
    Rejections of text parts are counted under "text_part_*", point parts under "part_*". Parts are kept
    greedily in _part_order, so of two duplicates (IoU or nested) the earlier one stays. With
    suppress_point_parts_with_text, an entity that keeps a text part uses no point part
    ("point_suppressed_by_text").
    """
    w, h = size
    rejected: dict[str, int] = {}

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    for item in raw["entities"]:
        _check_raw_entity(item)
    order = sorted(range(len(raw["entities"])), key=lambda i: (-raw["entities"][i]["score"], i))
    entities = []
    for i in order:
        item = raw["entities"][i]
        if item["score"] < settings["min_entity_score"]:
            reject("entity_score")
            continue
        mask = clean_mask(np.asarray(item["mask"], bool))
        if not mask.any() or not _in_range(
            np.count_nonzero(mask) / (w * h), settings["entity_area_fraction"]
        ):
            reject("entity_area")
            continue
        if any(iou(mask, kept["mask"]) >= settings["dedupe_iou"] for kept in entities):
            reject("entity_duplicate")
            continue
        entities.append({"phrase": item["phrase"], "score": float(item["score"]), "mask": mask, "raw": item})
    result = []
    for e_index, entity in enumerate(entities):
        eid = f"entity_{e_index:03d}"
        e_area = np.count_nonzero(entity["mask"])
        parts_raw = entity["raw"].get("parts", [])
        if not settings["text_parts"] and any(item["source"] == "text" for item in parts_raw):
            raise DataError("Raw text parts while region_proposal.text_parts is off")
        parts = []
        for j in _part_order(parts_raw):
            item = parts_raw[j]
            text = item["source"] == "text"
            prefix = "text_part" if text else "part"
            if (
                not text
                and settings["suppress_point_parts_with_text"]
                and any(kept["source"] == "text" for kept in parts)
            ):
                reject("point_suppressed_by_text")
                continue
            if item["score"] < settings["min_text_part_score" if text else "min_part_score"]:
                reject(f"{prefix}_score")
                continue
            if text and item["containment"] < settings["text_part_containment"]:
                reject("text_part_containment")
                continue
            if item["mask"] is None:
                raise DataError(f"A raw {item['source']} part that passes the score filter has no mask")
            mask = clean_mask(np.asarray(item["mask"], bool) & entity["mask"])
            if not mask.any() or not _in_range(
                np.count_nonzero(mask) / e_area, settings["part_area_fraction_of_entity"]
            ):
                reject(f"{prefix}_area")
                continue
            if any(iou(mask, kept["mask"]) >= settings["dedupe_iou"] for kept in parts):
                reject(f"{prefix}_duplicate")
                continue
            if any(nested_duplicate(mask, kept["mask"], settings) for kept in parts):
                reject(f"{prefix}_nested_duplicate")
                continue
            origin = {"phrase": item["phrase"]} if text else {"point": item["point"]}
            parts.append({"source": item["source"], **origin, "score": float(item["score"]), "mask": mask})
        result.append(
            {
                "proposal_id": eid,
                "phrase": entity["phrase"],
                "score": entity["score"],
                "mask": entity["mask"],
                "parts": [{"proposal_id": f"part_{e_index:03d}_{k:03d}", **p} for k, p in enumerate(parts)],
            }
        )
    return {
        "phrases": list(raw["phrases"]),
        "parts": list(raw.get("parts", [])),
        "entities": result,
        "rejected": rejected,
    }


def write_proposals(
    run_dir: Path, source: Source, built: dict, *, backend: str, models: dict, settings: dict, profile: dict
) -> dict:
    """Write masks first, then proposals.json as the commit marker.

    profile is the frozen source profile record whose subjects (and parts) were grounded.
    """
    subjects = profile["response"]["subjects"]
    if list(built["phrases"]) != list(subjects):
        raise DataError(
            f"Proposal phrases {built['phrases']!r} are not the frozen profile subjects {subjects!r}"
        )
    profile_parts = profile_part_names(profile, settings)
    if list(built["parts"]) != profile_parts:
        raise DataError(
            f"Proposal part names {built['parts']!r} are not the frozen profile parts {profile_parts!r}"
        )
    folder = f"proposals/{safe_id(source.source_id)}"
    size = source.width * source.height
    entities = []
    for entity in built["entities"]:
        parts = []
        for part in entity["parts"]:
            origin = {"phrase": part["phrase"]} if part["source"] == "text" else {"point": part["point"]}
            parts.append(
                {
                    "proposal_id": part["proposal_id"],
                    "source": part["source"],
                    **origin,
                    "score": part["score"],
                    "bbox": bbox(part["mask"]),
                    "area_fraction_of_entity": np.count_nonzero(part["mask"])
                    / np.count_nonzero(entity["mask"]),
                    **_save_mask(run_dir, folder, part),
                }
            )
        entities.append(
            {
                "proposal_id": entity["proposal_id"],
                "phrase": entity["phrase"],
                "score": entity["score"],
                "bbox": bbox(entity["mask"]),
                "area_fraction": np.count_nonzero(entity["mask"]) / size,
                **_save_mask(run_dir, folder, entity),
                "parts": parts,
            }
        )
    data = {
        "schema_version": SCHEMA_VERSION,
        "source_id": source.source_id,
        "source_pixel_sha256": source.pixel_sha256,
        "size": [source.width, source.height],
        "backend": backend,
        "models": models,
        "settings": settings,
        "source_profile_sha256": profile["frozen_sha256"],
        "subjects": list(subjects),
        "phrases": built["phrases"],
        "parts": list(built["parts"]),
        "status": "ready" if entities else "no_region",
        "entities": entities,
        "rejected": built.get("rejected", {}),
        "created_at": now(),
    }
    atomic_json(within(run_dir, f"{folder}/proposals.json"), data)
    return data


def profile_part_names(profile: dict, settings: dict) -> list:
    """Per-subject part names the worker was given: the frozen profile parts when text parts are on, else none."""
    return list(profile["response"].get("parts", [])) if settings["text_parts"] else []


def _save_mask(run_dir: Path, folder: str, item: dict) -> dict:
    name = f"{item['proposal_id']}.png"
    path = within(run_dir, f"{folder}/{name}")
    atomic_image(path, Image.fromarray(item["mask"].astype(np.uint8) * 255, "L"))
    return {"mask_path": name, "mask_sha256": file_hash(path)}


@dataclass(frozen=True)
class ProposalSet:
    source_id: str
    folder: str
    data: dict
    sha256: str
    masks: dict[str, np.ndarray]
    # Static option tables per sampler setting; derived data, not part of the proposal identity.
    cache: dict = field(default_factory=dict, compare=False, repr=False)

    def items(self):
        """(proposal, level, entity) for every stored entity and part."""
        for entity in self.data["entities"]:
            yield entity, "entity", entity
            for part in entity["parts"]:
                yield part, "part", entity


def proposals_path(source_id: str) -> str:
    return f"proposals/{safe_id(source_id)}/proposals.json"


def load_proposals(
    run_dir: Path, source: Source, settings: dict, *, profile: dict, allow_fixture: bool
) -> ProposalSet:
    folder = f"proposals/{safe_id(source.source_id)}"

    def require(condition, message):
        if not condition:
            raise DataError(f"Invalid region proposals for {source.source_id}: {message}")

    try:
        data = read_json(within(run_dir, proposals_path(source.source_id), must_exist=True))
    except ValueError as exc:
        raise DataError(f"Invalid region proposals for {source.source_id}: malformed JSON ({exc})") from exc
    require(isinstance(data, dict), "proposals.json must be an object")
    require(data.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    require(data.get("source_id") == source.source_id, "source_id mismatch")
    require(data.get("source_pixel_sha256") == source.pixel_sha256, "source pixel hash mismatch")
    require(data.get("size") == [source.width, source.height], "not at source resolution")
    backend = data.get("backend")
    require(
        backend in PROPOSAL_BACKENDS or (backend == FIXTURE_BACKEND and allow_fixture),
        f"backend {backend!r} is not allowed here",
    )
    require(data.get("settings") == settings, "proposal settings differ from this run's region_proposal")
    models = data.get("models")
    require(
        isinstance(models, dict)
        and models
        and all(isinstance(v, str) and v.strip() for v in models.values()),
        "model revisions must be recorded",
    )
    if backend != FIXTURE_BACKEND:
        revision = settings.get("model_revision")
        require(
            isinstance(revision, str) and revision.strip(),
            "region_proposal.model_revision must be pinned for a real proposal backend",
        )
        require(
            models.get(settings["model_id"]) == revision, "recorded model revision differs from the run's"
        )
    require(
        data.get("source_profile_sha256") == profile.get("frozen_sha256"),
        "made from a different frozen source profile",
    )
    subjects = profile.get("response", {}).get("subjects")
    require(data.get("subjects") == subjects, "subjects differ from the frozen source profile")
    phrases = data.get("phrases")
    require(phrases == subjects, "phrases must be the frozen profile subjects, in order")
    part_names = data.get("parts")
    require(
        isinstance(part_names, list) and part_names == profile_part_names(profile, settings),
        "parts differ from the frozen source profile",
    )
    problem = subject_parts_error(part_names, subjects)
    require(problem is None, str(problem))
    entities = data.get("entities")
    require(isinstance(entities, list), "entities must be a list")
    require(data.get("status") == ("ready" if entities else "no_region"), "status disagrees with entities")
    masks: dict[str, np.ndarray] = {}

    def mask_of(item, expected_id, min_score):
        require(isinstance(item, dict), f"{expected_id}: entry must be an object")
        require(item.get("proposal_id") == expected_id, f"unexpected proposal id {item.get('proposal_id')!r}")
        require(item.get("mask_path") == f"{expected_id}.png", f"{expected_id}: unexpected mask path")
        score = item.get("score")
        require(
            type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1, f"{expected_id}: score"
        )
        require(score >= min_score, f"{expected_id}: score below the configured minimum")
        path = within(run_dir, f"{folder}/{item['mask_path']}", must_exist=True)
        require(file_hash(path) == item.get("mask_sha256"), f"{expected_id}: mask file changed")
        with Image.open(path) as image:
            arr = np.asarray(image)
            require(
                image.mode == "L" and image.size == (source.width, source.height), f"{expected_id}: size/mode"
            )
        require(set(np.unique(arr).tolist()) <= {0, 255}, f"{expected_id}: mask must be binary 0/255")
        mask = arr > 0
        require(is_single_region(mask), f"{expected_id}: mask must be one hole-free connected region")
        require(item.get("bbox") == bbox(mask), f"{expected_id}: bbox mismatch")
        masks[expected_id] = mask
        return mask

    def area_ok(item, key, value, bounds, expected_id):
        stored = item.get(key)
        require(
            type(stored) in (int, float) and math.isclose(stored, value, rel_tol=1e-9, abs_tol=1e-12),
            f"{expected_id}: {key} does not match the mask",
        )
        require(_in_range(value, bounds), f"{expected_id}: {key} outside the configured range")

    def distinct(masks_so_far, mask, expected_id, nested=False):
        require(
            all(iou(mask, other) < settings["dedupe_iou"] for other in masks_so_far),
            f"{expected_id}: duplicate of a kept proposal (IoU >= dedupe_iou)",
        )
        require(
            not nested or not any(nested_duplicate(mask, other, settings) for other in masks_so_far),
            f"{expected_id}: nested duplicate of a kept part",
        )

    kept_entities = []
    for e_index, entity in enumerate(entities):
        eid = f"entity_{e_index:03d}"
        e_mask = mask_of(entity, eid, settings["min_entity_score"])
        e_area = np.count_nonzero(e_mask)
        area_ok(
            entity,
            "area_fraction",
            e_area / (source.width * source.height),
            settings["entity_area_fraction"],
            eid,
        )
        distinct(kept_entities, e_mask, eid)
        kept_entities.append(e_mask)
        require(entity.get("phrase") in phrases, f"{eid}: phrase not in phrases")
        require(isinstance(entity.get("parts"), list), f"{eid}: parts must be a list")
        kept_parts = []
        named = text_part_phrases(
            entity["phrase"], subject_parts(part_names, entity["phrase"]), settings["text_part_forms"]
        )
        sources = []
        for p_index, part in enumerate(entity["parts"]):
            pid = f"part_{e_index:03d}_{p_index:03d}"
            require(isinstance(part, dict) and part.get("source") in PART_SOURCES, f"{pid}: unknown source")
            text = part["source"] == "text"
            if text:
                require(part.get("phrase") in named, f"{pid}: phrase is not a named part of {eid}")
            else:
                point = part.get("point")
                require(
                    isinstance(point, list) and len(point) == 2 and all(type(v) is int for v in point),
                    f"{pid}: point prompt must be [x, y]",
                )
                x, y = point
                require(
                    0 <= x < source.width and 0 <= y < source.height and e_mask[y, x],
                    f"{pid}: point prompt must lie inside {eid}",
                )
            sources.append(part["source"])
            p_mask = mask_of(part, pid, settings["min_text_part_score" if text else "min_part_score"])
            require(not (p_mask & ~e_mask).any(), f"{pid}: part outside its entity")
            require(np.count_nonzero(p_mask) < e_area, f"{pid}: part equals entity")
            area_ok(
                part,
                "area_fraction_of_entity",
                np.count_nonzero(p_mask) / e_area,
                settings["part_area_fraction_of_entity"],
                pid,
            )
            distinct(kept_parts, p_mask, pid, nested=True)
            kept_parts.append(p_mask)
        require(sources == sorted(sources, key=PART_SOURCES.index), f"{eid}: text parts must precede points")
        require(
            not settings["suppress_point_parts_with_text"] or not ("text" in sources and "point" in sources),
            f"{eid}: point parts are suppressed when the entity has a text part",
        )
    # Content hash: the write timestamp is not part of the proposal's identity.
    content = {k: v for k, v in data.items() if k != "created_at"}
    return ProposalSet(source.source_id, folder, data, digest(content), masks)
