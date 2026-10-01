from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import yaml

from .errors import ConfigError
from .storage import digest

PATH_FIELDS = {
    "run": ["output_root"],
    "dataset": ["root", "input_dir", "manifest"],
    "source_profile": ["prompt_path", "response_schema"],
    "sampler": ["replay_plan"],
    "execution": ["generator_python", "verifier_python", "proposal_python"],
    "generator": ["repo_path", "upstream_config", "weights_lock"],
    "verifier": ["model_path", "physical_prompt_path", "semantic_prompt_path", "response_schema"],
}
OPTIONAL_PATH_FIELDS = {"region_proposal": ["model_path", "repo_path"]}
SAMPLERS = ("random_geometry_v1", "object_region_v2")
GRANULARITY = {"relocation": {"entity"}, "rotation": {"part"}, "deformation": {"part", "entity"}}


def load_config(path: Path, *, input_dir=None, target_count=None, gpus=None, human_review=None) -> dict:
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a YAML mapping")
    for section, fields in {**PATH_FIELDS, **OPTIONAL_PATH_FIELDS}.items():
        if section in OPTIONAL_PATH_FIELDS and cfg.get(section) is None:
            continue
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"Missing config section: {section}")
        for key in fields:
            value = cfg[section].get(key)
            if value is not None:
                candidate = path.parent / Path(value).expanduser()
                # Resolving a venv's Python symlink would bypass that venv.
                cfg[section][key] = (
                    os.path.abspath(candidate) if key.endswith("_python") else str(candidate.resolve())
                )
    if input_dir is not None:
        cfg["dataset"]["input_dir"] = str(Path(input_dir).resolve())
        cfg["dataset"]["root"] = cfg["dataset"]["input_dir"]
        cfg["dataset"]["manifest"] = None
    if target_count is not None:
        cfg["run"]["target_count"] = target_count
    if gpus is not None:
        cfg["execution"]["selected_gpu_ids"] = parse_gpus(gpus)
    if human_review is not None:
        cfg["human_review"] = {**(cfg.get("human_review") or {}), "enabled": human_review}
    cfg["_config_path"] = str(path)
    assets = {}
    for name, section, key in (
        ("profile_prompt", "source_profile", "prompt_path"),
        ("profile_schema", "source_profile", "response_schema"),
        ("physical_prompt", "verifier", "physical_prompt_path"),
        ("semantic_prompt", "verifier", "semantic_prompt_path"),
        ("vqa_schema", "verifier", "response_schema"),
    ):
        asset = Path(cfg[section][key])
        value = asset.read_text(encoding="utf-8")
        assets[name] = json.loads(value) if name.endswith("schema") else value
    cfg["_assets"] = assets
    validate_config(cfg)
    return cfg


def parse_gpus(value) -> list[str]:
    values = value.split(",") if isinstance(value, str) else value
    result = []
    for item in values:
        item = str(item).strip()
        if not item.isdigit():
            raise ConfigError("GPU IDs must be comma-separated nonnegative integers")
        result.append(str(int(item)))
    if not result or len(set(result)) != len(result):
        raise ConfigError("GPU IDs must be nonempty and unique")
    return result


def validate_config(cfg: dict) -> None:
    def require(condition, message):
        if not condition:
            raise ConfigError(message)

    require(cfg.get("schema_version") == "1.1", "Unsupported config schema_version")
    r, d, e = cfg["run"], cfg["dataset"], cfg["execution"]
    require(
        r["quota_scope"] == "total" and r["strategy"] == "random_until_quota",
        "Only total random quota supported",
    )
    require(
        r["max_attempts"] is None and r["max_wall_time_seconds"] is None,
        "Total attempt/time caps are disabled",
    )
    require(r["limit_policy"] == "until_target_or_user_stop", "Unsupported limit policy")
    require(
        r["source_schedule"] == "round_robin" and r["chain_augmentation"] is False,
        "Unsupported source schedule",
    )
    require(type(r["seed"]) is int and 0 <= r["seed"] < 2**32, "seed must be a 32-bit integer")
    require(
        r["target_count"] is None or (type(r["target_count"]) is int and r["target_count"] > 0),
        "N must be positive",
    )
    require(
        d["task"] == "image_only", "MVP exports image-only data; supervised exporters require reviewed labels"
    )
    require(bool(d.get("input_dir")) != bool(d.get("manifest")), "Choose input_dir or manifest, not both")
    require(set(d["eligible_splits"]).issubset({"source", "train"}), "Cannot augment val/test")
    require(
        d["enforce_group_disjoint"] and d["enforce_cross_split_hash_disjoint"], "Split checks are required"
    )
    require(
        e["mode"] == "shared_gpus_sequential" and e["candidate_concurrency"] == 1,
        "MVP uses one sequential candidate",
    )
    require(e["retry_oom"] is False and e["max_technical_retries"] in (0, 1), "Invalid retry policy")
    require(
        isinstance(e["worker_timeout_seconds"], (int, float)) and e["worker_timeout_seconds"] > 0,
        "Invalid worker timeout",
    )
    if e.get("selected_gpu_ids") is not None:
        e["selected_gpu_ids"] = parse_gpus(e["selected_gpu_ids"])
    require(
        type(e.get("max_consecutive_technical_failures")) is int
        and e["max_consecutive_technical_failures"] > 0,
        "A positive consecutive technical failure threshold is required",
    )
    s = cfg["sampler"]
    require(s.get("version") in SAMPLERS, "Unsupported sampler version")
    v1 = s["version"] == "random_geometry_v1"
    if v1:
        require(
            set(s["mask_shapes"]) <= {"ellipse", "rectangle"} and bool(s["mask_shapes"]),
            "Invalid mask shapes",
        )
    require(
        set(s["operations"]) <= {"relocation", "deformation", "rotation"} and bool(s["operations"]),
        "Invalid operations",
    )
    require(s["operation_distribution"] == "uniform", "Only uniform operation sampling supported")
    require(
        s.get("replay_plan") is None,
        "Use stored runs for resume; external replay plans are not an MVP run mode",
    )
    require(
        "deformation_scale" not in s,
        "Upstream deformation uses displacement, not scale; remove deformation_scale",
    )
    ranges = [("displacement_diagonal_fraction", 0, 1), ("rotation_degrees", -180, 180)]
    for key, lo, hi in ranges + ([("region_area_fraction", 0, 1)] if v1 else []):
        require_range(s, key, lo, hi, require)
    require(
        (not v1 or s["region_area_fraction"][0] > 0) and s["displacement_diagonal_fraction"][0] > 0,
        "Nonzero edits required",
    )
    if cfg.get("region_proposal") is not None or not v1:
        validate_region_proposal(cfg.get("region_proposal"), require, used=not v1)
    if s.get("object_region") is not None or not v1:
        validate_object_region(s, require, used=not v1)
    if not v1:
        schema = cfg.get("_assets", {}).get("profile_schema", {})
        require(
            "subjects" in schema.get("required", []),
            "object_region_v2 needs a source profile with subjects (source_profile_v4, or v3 with text_parts false)",
        )
        require(
            not cfg["region_proposal"]["text_parts"] or per_subject_parts(schema),
            "region_proposal.text_parts needs a source profile with per-subject parts (source_profile_v4)",
        )
    require(cfg["source_profile"]["mode"] == "auto_with_optional_hint", "Unsupported profile mode")
    require(
        cfg["source_profile"]["freeze_before_generation"]
        and cfg["source_profile"]["uncertain_action"] == "hold_source",
        "Profile must be fixed",
    )
    v = cfg["verifier"]
    require(
        v["backend"] == "qwen_multimodal_local" and cfg["generator"]["backend"] == "dragflow",
        "Real backend IDs required",
    )
    require(v["local_files_only"] and not v["fine_tune"], "Local pretrained inference required")
    require(
        v["enable_thinking"] is False and v["do_sample"] is False and v["batch_size"] == 1,
        "Unsupported decoding protocol",
    )
    require(
        v["check_order"] == ["physical", "semantic"] and v["semantic_only_after_physical_yes"],
        "Physical then semantic required",
    )
    require(
        v["dtype"] in ("bfloat16", "float16", "float32") and v["quantization"] is None,
        "Unsupported dtype/quantization",
    )
    require(v["preprocessing"]["policy"] == "pinned_processor_defaults", "Unsupported visual preprocessing")
    require(cfg["decision"]["require_all_yes"] == ["physical", "semantic"], "Both checks required")
    require(
        [cfg["decision"][k] for k in ("accept_answer", "reject_answer", "uncertain_answer")]
        == ["YES", "NO", "UNCERTAIN"],
        "Verdict enums are fixed",
    )
    require(not cfg["decision"]["retry_semantic_decisions"], "Do not retry semantic NO/UNCERTAIN")
    x = cfg["export"]
    require(x["format"] == "jsonl" and x["allowed_status"] == "accepted", "Invalid export policy")
    require(
        x["reject_fake_backend"] and x["deduplicate_exact_pixels"] and x["deduplicate_against_source"],
        "Export integrity checks required",
    )
    require(
        not x["near_duplicate_filter"] and not x["require_valid_labels"],
        "Unsupported image-only export option",
    )
    vis = cfg.get("visualization", {})
    require(vis.get("output_subdir") == "visualizations", "Visualization directory must be visualizations")
    require(
        vis.get("format") == "png" and vis.get("layout") == "source_overlay_and_generated",
        "Unsupported preview format/layout",
    )
    require(
        vis.get("rejected_comparison_policy") == "follow_image_retention",
        "Preview must follow NO deletion policy",
    )
    require(
        vis.get("start_marker") == "circle" and vis.get("end_marker") == "cross", "Unsupported marker shapes"
    )
    review = cfg.get("human_review", {"enabled": False})
    require(
        isinstance(review, dict) and set(review) == {"enabled"} and type(review["enabled"]) is bool,
        "human_review must be {enabled: true|false}",
    )
    storage = cfg.get("storage", {})
    require(
        storage.get("atomic_writes") is True and storage.get("persist_raw_vqa") is True,
        "Atomic writes and raw VQA persistence are required",
    )
    require(
        0 <= vis["area_alpha"] <= 1
        and type(vis["max_panel_side_px"]) is int
        and vis["max_panel_side_px"] >= 64,
        "Invalid preview dimensions/opacity",
    )


def per_subject_parts(schema: dict) -> bool:
    """Profile schema v4: a required parts list of {subject, parts} objects (v3 had one flat name list)."""
    items = schema.get("properties", {}).get("parts", {}).get("items", {})
    return (
        "parts" in schema.get("required", [])
        and items.get("type") == "object"
        and set(items.get("required", [])) == {"subject", "parts"}
    )


def require_range(section: dict, key: str, lo, hi, require) -> None:
    v = section.get(key)
    require(
        isinstance(v, list) and len(v) == 2 and all(type(x) in (float, int) and math.isfinite(x) for x in v),
        f"Invalid {key}",
    )
    require(lo <= v[0] <= v[1] <= hi, f"Out-of-range {key}")


def validate_object_region(s: dict, require, *, used: bool = True) -> None:
    o = s.get("object_region")
    require(isinstance(o, dict), "object_region_v2 needs sampler.object_region")
    g = o.get("granularity")
    require(isinstance(g, dict) and set(s["operations"]) <= set(g), "granularity needed for every operation")
    for op, levels in g.items():
        require(
            op in GRANULARITY
            and isinstance(levels, list)
            and levels
            and len(set(levels)) == len(levels)
            and set(levels) <= GRANULARITY[op],
            f"Invalid granularity for {op}; allowed {sorted(GRANULARITY.get(op, []))}",
        )
    require(o.get("rotation_anchor") == "part_entity_contact", "Unsupported rotation_anchor")
    k = o.get("contact_dilation_px")
    require(type(k) is int and 1 <= k <= 64, "contact_dilation_px must be an integer in [1, 64]")
    require(o.get("on_no_region") == "hold_source", "Regions never fall back to geometry; use hold_source")
    n = o.get("max_consecutive_sampling_skips")
    # Added after the first v2 example; a v1 run copied from an older config does not need it.
    require(
        (type(n) is int and 1 <= n <= 1000) or (not used and "max_consecutive_sampling_skips" not in o),
        "max_consecutive_sampling_skips must be an integer in [1, 1000]",
    )
    # Added in T015 (minimum in T015-fix); a v1 run copied from an older config does not need them.
    if used or any(key in o for key in ROTATION_ERROR_KEYS):
        d, f, m = (o.get(key) for key in ROTATION_ERROR_KEYS)
        require(
            type(d) in (int, float) and 0 < d <= 180,
            "object_region.rotation_max_error_degrees must be in (0, 180]",
        )
        require(
            type(f) in (int, float) and 0 <= f <= 1,
            "object_region.rotation_max_error_fraction must be in [0, 1]",
        )
        require(
            type(m) in (int, float) and 0 < m <= 180,
            "object_region.rotation_min_executed_degrees must be in (0, 180]",
        )


ROTATION_ERROR_KEYS = ("rotation_max_error_degrees", "rotation_max_error_fraction", "rotation_min_executed_degrees")


TEXT_PART_KEYS = ("text_parts", "text_part_forms", "min_text_part_score", "text_part_containment")
PART_FILTER_KEYS = ("part_dedupe_containment", "part_dedupe_area_ratio", "suppress_point_parts_with_text")


def validate_region_proposal(r, require, *, used: bool = True) -> None:
    require(isinstance(r, dict), "object_region_v2 needs a region_proposal section")
    require(r.get("backend") in ("sam3", "grounding_dino_sam2.1"), "Unsupported region_proposal backend")
    require(isinstance(r.get("model_id"), str) and r["model_id"], "region_proposal.model_id required")
    for key in ("model_path", "model_revision", "repo_path", "code_revision"):
        require(r.get(key) is None or isinstance(r[key], str), f"Invalid region_proposal.{key}")
    for key in ("min_entity_score", "min_part_score", "dedupe_iou"):
        v = r.get(key)
        require(type(v) in (int, float) and 0 <= v <= 1, f"region_proposal.{key} must be in [0, 1]")
    require(r["dedupe_iou"] > 0, "dedupe_iou must be positive")
    require_range(r, "entity_area_fraction", 0, 1, require)
    require_range(r, "part_area_fraction_of_entity", 0, 1, require)
    require(
        r["entity_area_fraction"][0] > 0
        and 0 < r["part_area_fraction_of_entity"][0]
        and r["part_area_fraction_of_entity"][1] < 1,
        "Area fractions must exclude empty masks and parts equal to the entity",
    )
    n = r.get("part_points_per_entity")
    require(type(n) is int and 1 <= n <= 256, "part_points_per_entity must be an integer in [1, 256]")
    # Added after the first v2 example; a v1 run copied from an older config does not need them.
    if used or any(key in r for key in TEXT_PART_KEYS):
        require(type(r.get("text_parts")) is bool, "region_proposal.text_parts must be true or false")
        forms = r.get("text_part_forms")
        require(
            isinstance(forms, list)
            and forms
            and len(set(forms)) == len(forms)
            and set(forms) <= {"subject_part", "part"},
            "region_proposal.text_part_forms must be a non-empty list of subject_part and/or part",
        )
        for key in ("min_text_part_score", "text_part_containment"):
            v = r.get(key)
            require(type(v) in (int, float) and 0 <= v <= 1, f"region_proposal.{key} must be in [0, 1]")
        require(r["text_part_containment"] > 0, "text_part_containment must be positive")
    if used or any(key in r for key in PART_FILTER_KEYS):
        for key in ("part_dedupe_containment", "part_dedupe_area_ratio"):
            v = r.get(key)
            require(type(v) in (int, float) and 0 < v <= 1, f"region_proposal.{key} must be in (0, 1]")
        require(
            type(r.get("suppress_point_parts_with_text")) is bool,
            "region_proposal.suppress_point_parts_with_text must be true or false",
        )
    gd = r.get("grounding_dino")
    if gd is not None or r["backend"] == "grounding_dino_sam2.1":
        require(isinstance(gd, dict), "grounding_dino thresholds required for grounding_dino_sam2.1")
        for key in ("box_threshold", "text_threshold"):
            v = gd.get(key)
            require(type(v) in (int, float) and 0 <= v <= 1, f"grounding_dino.{key} must be in [0, 1]")


def recipe_hash(cfg: dict) -> str:
    value = copy.deepcopy(cfg)
    value.pop("_config_path", None)
    value["execution"].pop("selected_gpu_ids", None)
    # Cosmetic rendering is independently versioned and does not alter inference.
    return digest(value)
