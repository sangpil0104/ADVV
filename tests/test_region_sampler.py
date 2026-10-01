import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from advv.config import validate_config
from advv.contracts import EditRequest, Source
from advv.errors import ConfigError, DataError, SamplingSkipped
from advv.ingest import scan_sources
from advv.pipeline import Pipeline, create_run
from advv.proposals import (
    ProposalSet,
    build_proposals,
    centroid,
    clean_mask,
    dilate,
    is_single_region,
    load_proposals,
    part_prompt_points,
    write_proposals,
)
from advv.reporting import records_for
from advv.sampler import (
    MIN_ROTATION_RADIUS_CELLS,
    feature_grid,
    grid_point,
    grid_rotation_degrees,
    region_options,
    region_table,
    sample_edit,
    upstream_grid_region,
    upstream_grid_start,
    validate_plan,
)
from advv.storage import atomic_json, digest, file_hash, read_json, within
from conftest import PROJECT, FakeBackend

W, H = 120, 80


def rect(x0, y0, x1, y1, size=(W, H)):
    mask = np.zeros((size[1], size[0]), bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def raw_fixture():
    kitten = rect(20, 20, 59, 59)
    return {
        "phrases": ["kitten", "box"],
        "entities": [
            {
                "phrase": "kitten",
                "score": 0.9,
                "mask": kitten,
                "parts": [
                    # left half (50%): 5 grid cells from its contact anchor, enough for rotation
                    point_part(0.95, rect(20, 20, 39, 59)),
                    point_part(0.99, kitten),  # whole entity is not a part
                    point_part(0.9, rect(50, 50, 51, 51)),  # too small
                ],
            },
            {"phrase": "box", "score": 0.7, "mask": rect(80, 10, 109, 39), "parts": []},
            {"phrase": "box", "score": 0.2, "mask": rect(0, 60, 30, 79), "parts": []},  # low score
            {"phrase": "kitten", "score": 0.8, "mask": rect(21, 20, 60, 59), "parts": []},  # duplicate
        ],
    }


def point_part(score, mask):
    return {"source": "point", "point": centroid(mask), "multimask_index": 0, "score": score, "mask": mask}


def profile_for(subjects, parts=()):
    response = {
        "summary": "s",
        "must_preserve": ["x"],
        "subjects": list(subjects),
        "parts": list(parts),
        "uncertain": False,
    }
    return {"response": response, "frozen_sha256": digest({"response": response, "hint": None})}


def fixture_proposals(root, source, cfg, raw=None, profile=None):
    raw = raw or raw_fixture()
    profile = profile or profile_for(raw["phrases"], raw.get("parts", ()))
    built = build_proposals((source.width, source.height), raw, cfg["region_proposal"])
    write_proposals(
        root,
        source,
        built,
        backend="fixture",
        models={"fixture": "test-only"},
        settings=cfg["region_proposal"],
        profile=profile,
    )
    return load_proposals(root, source, cfg["region_proposal"], profile=profile, allow_fixture=True)


@pytest.fixture
def source(cfg):
    return scan_sources(cfg)[0][0]


def test_clean_mask_components_and_holes():
    two = rect(0, 0, 9, 9) | rect(20, 0, 22, 2)
    assert not is_single_region(two)
    assert np.array_equal(clean_mask(two), rect(0, 0, 9, 9))
    ring = rect(0, 0, 9, 9) & ~rect(3, 3, 5, 5)
    assert not is_single_region(ring)
    assert is_single_region(clean_mask(ring)) and np.array_equal(clean_mask(ring), rect(0, 0, 9, 9))
    diagonal = rect(0, 0, 1, 1) | rect(2, 2, 3, 3)  # 8-connected, like a filled upstream contour
    assert is_single_region(diagonal)
    border_notch = rect(0, 0, 9, 9) & ~rect(0, 4, 2, 5)
    assert is_single_region(border_notch)


def test_filters_rank_dedupe_and_hierarchy(v2cfg):
    built = build_proposals((W, H), raw_fixture(), v2cfg["region_proposal"])
    assert [(e["proposal_id"], e["phrase"]) for e in built["entities"]] == [
        ("entity_000", "kitten"),
        ("entity_001", "box"),
    ]
    assert [p["proposal_id"] for p in built["entities"][0]["parts"]] == ["part_000_000"]
    assert built["rejected"] == {"entity_score": 1, "entity_duplicate": 1, "part_area": 2}


@pytest.mark.parametrize(
    "key,value,kept",
    [
        ("min_entity_score", 0.9, True),
        ("min_entity_score", 0.9001, False),
        ("entity_area_fraction", [1600 / 9600, 0.6], True),
        ("entity_area_fraction", [1600 / 9600 + 1e-6, 0.6], False),
        ("entity_area_fraction", [0.02, 1600 / 9600], True),
        ("entity_area_fraction", [0.02, 1600 / 9600 - 1e-6], False),
    ],
)
def test_entity_filter_boundaries(v2cfg, key, value, kept):
    settings = {**v2cfg["region_proposal"], key: value}
    raw = {
        "phrases": ["kitten"],
        "entities": [{"phrase": "kitten", "score": 0.9, "mask": rect(20, 20, 59, 59)}],
    }
    assert bool(build_proposals((W, H), raw, settings)["entities"]) is kept


@pytest.mark.parametrize(
    "key,value,kept",
    [
        ("min_part_score", 0.95, True),
        ("min_part_score", 0.9501, False),
        ("part_area_fraction_of_entity", [0.2, 0.7], True),
        ("part_area_fraction_of_entity", [0.2001, 0.7], False),
        ("part_area_fraction_of_entity", [0.05, 0.2], True),
        ("part_area_fraction_of_entity", [0.05, 0.1999], False),
    ],
)
def test_part_filter_boundaries(v2cfg, key, value, kept):
    settings = {**v2cfg["region_proposal"], key: value}
    raw = {
        "phrases": ["kitten"],
        "entities": [
            {
                "phrase": "kitten",
                "score": 0.9,
                "mask": rect(20, 20, 59, 59),
                "parts": [point_part(0.95, rect(20, 20, 39, 35))],
            }
        ],
    }
    assert bool(build_proposals((W, H), raw, settings)["entities"][0]["parts"]) is kept


def test_dedupe_boundary(v2cfg):
    a, b = rect(0, 0, 39, 39), rect(0, 0, 39, 31)  # IoU exactly 0.8
    raw = {"phrases": ["x"], "entities": [{"phrase": "x", "score": 0.9, "mask": m} for m in (a, b)]}
    for threshold, count in ((0.8, 1), (0.8001, 2)):
        settings = {**v2cfg["region_proposal"], "dedupe_iou": threshold}
        assert len(build_proposals((W, H), raw, settings)["entities"]) == count


def test_part_points_are_seeded_and_inside(v2cfg):
    entity = rect(20, 20, 59, 59)
    points = part_prompt_points(entity, 16, 7)
    assert points == part_prompt_points(entity, 16, 7) and len(points) == 16
    assert all(entity[y, x] for x, y in points)


def test_proposal_contract_round_trip_and_rejections(v2cfg, source, tmp_path):
    root = tmp_path / "run"
    proposals = fixture_proposals(root, source, v2cfg)
    folder = root / proposals.folder
    data = read_json(folder / "proposals.json")
    assert data["backend"] == "fixture" and data["status"] == "ready"
    assert sorted(p.name for p in folder.glob("*.png")) == [
        "entity_000.png",
        "entity_001.png",
        "part_000_000.png",
    ]
    settings = v2cfg["region_proposal"]
    profile = profile_for(["kitten", "box"])
    with pytest.raises(DataError, match="not allowed"):
        load_proposals(root, source, settings, profile=profile, allow_fixture=False)
    with pytest.raises(DataError, match="settings differ"):
        load_proposals(root, source, {**settings, "dedupe_iou": 0.5}, profile=profile, allow_fixture=True)
    with pytest.raises(DataError, match="pixel hash"):
        load_proposals(
            root, replace(source, pixel_sha256="0" * 64), settings, profile=profile, allow_fixture=True
        )

    def tamper(name, mask, **changes):
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(folder / name)
        edited = copy.deepcopy(data)
        edited["entities"][0].update(changes)
        edited["entities"][0]["parts"][0]["mask_sha256"] = file_hash(folder / "part_000_000.png")
        edited["entities"][0]["mask_sha256"] = file_hash(folder / "entity_000.png")
        atomic_json(folder / "proposals.json", edited)

    tamper("entity_000.png", rect(20, 20, 59, 59) | rect(0, 0, 3, 3))
    with pytest.raises(DataError, match="one hole-free connected region"):
        load_proposals(root, source, settings, profile=profile, allow_fixture=True)
    tamper("entity_000.png", rect(22, 22, 59, 59), bbox=[22, 22, 59, 59], area_fraction=38 * 38 / (W * H))
    with pytest.raises(DataError, match="part outside its entity"):
        load_proposals(root, source, settings, profile=profile, allow_fixture=True)
    (folder / "entity_000.png").write_bytes(b"changed")
    with pytest.raises(DataError, match="mask file changed"):
        load_proposals(root, source, settings, profile=profile, allow_fixture=True)


def test_v2_determinism_granularity_and_anchor(v2cfg, source, tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    a_set, b_set = (fixture_proposals(root, source, v2cfg) for root in (left, right))
    k = v2cfg["sampler"]["object_region"]["contact_dilation_px"]
    seen = set()
    for index in range(60):
        a = sample_edit(source, {"summary": "s"}, v2cfg, index, left, proposals=a_set)
        b = sample_edit(source, {"summary": "s"}, v2cfg, index, right, proposals=b_set)
        assert a.to_dict() == b.to_dict()
        validate_plan(a, source, left)
        assert a.schema_version == "1.3" and a.sampler_version == "object_region_v2"
        mask = a_set.masks[a.region_proposal_id]
        assert a.source_point == centroid(mask)
        assert mask[a.region_select_point[1], a.region_select_point[0]]
        assert a.mask_sha256 == file_hash(left / a_set.folder / f"{a.region_proposal_id}.png")
        seen.add((a.operation, a.region_level))
        if a.operation == "relocation":
            assert a.region_level == "entity"
            x0, y0, x1, y1 = a.operation_params["entity_bbox_after"]
            assert 0 <= x0 and 0 <= y0 and x1 < W and y1 < H
            shift = a.operation_params["displacement_pixels"]
            assert [a.target_point[i] - a.source_point[i] for i in (0, 1)] == shift
        if a.operation == "rotation":
            assert a.region_level == "part" and a.region_proposal_id == "part_000_000"
            entity = a_set.masks["entity_000"]
            contact = dilate(mask, k) & entity & ~mask
            ys, xs = np.nonzero(contact)
            ax, ay = a.anchor_point
            assert np.min(np.hypot(xs - ax, ys - ay)) <= k
            assert a.operation_params["anchor_method"] == "part_entity_contact"
        else:
            assert a.anchor_point is None
    assert seen == {
        ("relocation", "entity"),
        ("rotation", "part"),
        ("deformation", "part"),
        ("deformation", "entity"),
    }


def test_part_without_contact_is_not_rotated(v2cfg):
    entity, part = rect(0, 0, 9, 9) | rect(30, 0, 39, 9), rect(30, 0, 39, 9)
    data = {
        "entities": [
            {
                "proposal_id": "entity_000",
                "phrase": "x",
                "bbox": [0, 0, 39, 9],
                "parts": [{"proposal_id": "part_000_000", "bbox": [30, 0, 39, 9]}],
            }
        ]
    }
    fake = ProposalSet("a", "proposals/a", data, "h", {"entity_000": entity, "part_000_000": part})
    options = region_options(fake, v2cfg, W, H)
    assert "rotation" not in options
    assert [o["proposal_id"] for o in options["deformation"]["part"]] == ["part_000_000"]


def test_plan_region_fields_are_version_bound(v2cfg, cfg, source, tmp_path):
    v1 = sample_edit(source, {"summary": "s"}, cfg, 0, tmp_path)
    assert v1.region_proposal_id is None and v1.region_select_point is None and v1.schema_version == "1.3"
    for field, value in (("region_level", "entity"), ("region_select_point", v1.source_point)):
        with pytest.raises(DataError, match="object_region_v2 plans only"):
            validate_plan(replace(v1, **{field: value}), source, tmp_path)
    legacy = {k: v for k, v in v1.to_dict().items() if not k.startswith("region_") or k == "region_mask_path"}
    for version, extra in (("1.1", {}), ("1.2", {"region_proposal_id": None})):
        validate_plan(EditRequest(**{**legacy, **extra, "schema_version": version}), source, tmp_path)
    proposals = fixture_proposals(tmp_path, source, v2cfg)
    v2 = sample_edit(source, {"summary": "s"}, v2cfg, 0, tmp_path, proposals=proposals)
    for field in ("region_phrase", "region_select_point"):
        with pytest.raises(DataError, match="needs proposal id"):
            validate_plan(replace(v2, **{field: None}), source, tmp_path)
    moved = [v2.source_point[0] + 1, v2.source_point[1]]
    with pytest.raises(DataError, match="mask centroid"):
        validate_plan(replace(v2, source_point=moved), source, tmp_path)
    Image.fromarray((rect(20, 20, 59, 59) | rect(0, 0, 3, 3)).astype(np.uint8) * 255, "L").save(
        within(tmp_path, v2.region_mask_path)
    )
    broken = replace(v2, mask_sha256=file_hash(within(tmp_path, v2.region_mask_path)))
    with pytest.raises(DataError, match="one hole-free"):
        validate_plan(broken, source, tmp_path)
    with pytest.raises(DataError, match="region_proposals_missing"):
        sample_edit(source, {"summary": "s"}, v2cfg, 0, tmp_path)


def test_v1_plans_unchanged(cfg, source, tmp_path):
    golden = json.loads((PROJECT / "tests/fixtures/sampler_v1_golden.json").read_text())
    for index, expected in enumerate(golden):
        p = sample_edit(source, {"summary": "x"}, cfg, index, tmp_path).to_dict()
        keys = [
            "seed",
            "operation",
            "source_point",
            "target_point",
            "anchor_point",
            "operation_params",
            "mask_sha256",
        ]
        assert [p[k] for k in keys] == expected


def sid(root):
    return read_json(root / "sources.json")[0]["source_id"]


def start_v2(v2cfg, run_id, backend, raw=None):
    """Profile first (proposals are grounded on frozen subjects), then add proposals and run again."""
    root = create_run(v2cfg, run_id, provenance="fake")
    state = Pipeline(root, backend).run()
    if raw is False:
        return root, state
    assert "region_proposals_missing" in state["last_error"]["message"]
    source = Source(**read_json(root / "sources.json")[0])
    profile = read_json(root / f"profiles/{source.source_id}.json")
    fixture_proposals(root, source, v2cfg, raw, profile)
    return root, Pipeline(root, backend).run()


def test_v2_pipeline_records_region_and_footer(v2cfg):
    root, state = start_v2(v2cfg, "v2", FakeBackend(subjects=["kitten", "box"]))
    assert state["status"] == "completed" and state["accepted_count"] == 1
    record = records_for(root)[0]
    plan, sid = record["plan"], record["source_id"]
    assert plan["sampler_version"] == "object_region_v2" and plan["region_phrase"] in ("kitten", "box")
    assert record["region_proposals_sha256"] == state["region_proposals"][sid]["sha256"]
    metadata = read_json(root / f"visualizations/{record['candidate_id']}/metadata.json")
    assert metadata["requested"]["region_proposal_id"] == plan["region_proposal_id"]
    assert read_json(root / "report.json")["region_proposals"][sid]["status"] == "ready"


def test_v2_changed_proposals_block_resume(v2cfg):
    root, state = start_v2(
        v2cfg, "v2_resume", FakeBackend(["NO"], [(1, 2, 3), KeyboardInterrupt()], subjects=["kitten", "box"])
    )
    assert state["status"] == "interrupted"
    path = root / f"proposals/{sid(root)}/proposals.json"
    data = read_json(path)
    data["entities"][0]["score"] = 0.5
    atomic_json(path, data)
    backend = FakeBackend(["YES", "YES"], subjects=["kitten", "box"])
    state = Pipeline(root, backend).run()
    assert state["status"] == "failed" and "Region proposals changed" in state["last_error"]["message"]
    assert backend.generations == []


def test_v2_no_region_holds_source_without_fallback(v2cfg):
    raw = {"phrases": ["kitten"], "entities": [{"phrase": "kitten", "score": 0.1, "mask": rect(0, 0, 9, 9)}]}
    backend = FakeBackend(subjects=["kitten"])
    root, state = start_v2(v2cfg, "v2_none", backend, raw)
    assert read_json(root / f"proposals/{sid(root)}/proposals.json")["status"] == "no_region"
    assert state["status"] == "failed" and "no_eligible_sources" in state["last_error"]["message"]
    assert state["region_proposals"][sid(root)]["status"] == "source_no_region"
    assert backend.generations == [] and not (root / "edit_regions").exists()


def test_v2_missing_proposals_is_an_error_not_a_hold(v2cfg):
    backend = FakeBackend(subjects=["kitten"])
    root, state = start_v2(v2cfg, "v2_missing", backend, raw=False)
    assert state["status"] == "failed" and "region_proposals_missing" in state["last_error"]["message"]
    assert backend.generations == []


def test_v2_config_validation(v2cfg, cfg):
    bad = copy.deepcopy(v2cfg)
    bad["_assets"]["profile_schema"] = cfg["_assets"]["profile_schema"]
    with pytest.raises(ConfigError, match="subjects"):
        validate_config(bad)
    for path, value, message in (
        (("sampler", "object_region", "granularity", "relocation"), ["part"], "granularity"),
        (("sampler", "object_region", "on_no_region"), "fallback_geometry", "hold_source"),
        (("sampler", "object_region", "contact_dilation_px"), 0, "contact_dilation_px"),
        (("sampler", "object_region", "max_consecutive_sampling_skips"), 0, "max_consecutive_sampling_skips"),
        (("region_proposal", "backend"), "qwen_box", "backend"),
        (("region_proposal", "part_area_fraction_of_entity"), [0.05, 1.0], "equal to the entity"),
        (("region_proposal", "min_part_score"), 1.5, "min_part_score"),
    ):
        bad = copy.deepcopy(v2cfg)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ConfigError, match=message):
            validate_config(bad)
    bad = copy.deepcopy(v2cfg)
    del bad["region_proposal"]
    with pytest.raises(ConfigError, match="region_proposal"):
        validate_config(bad)
    legacy = copy.deepcopy(cfg)
    del legacy["region_proposal"], legacy["sampler"]["object_region"]
    validate_config(legacy)  # Pre-v2 configs stay valid for random_geometry_v1.


def test_v2_plans_match_golden_and_depend_on_seed(v2cfg, source, tmp_path):
    proposals = fixture_proposals(tmp_path, source, v2cfg)
    keys = [
        "seed",
        "operation",
        "region_proposal_id",
        "source_point",
        "target_point",
        "anchor_point",
        "region_select_point",
        "operation_params",
    ]
    plans = [
        sample_edit(source, {"summary": "s"}, v2cfg, i, tmp_path, proposals=proposals).to_dict()
        for i in range(16)
    ]
    golden = json.loads((PROJECT / "tests/fixtures/sampler_v2_golden.json").read_text())
    assert [[p[k] for k in keys] for p in plans] == golden
    assert {p["operation"] for p in plans} == {"relocation", "deformation", "rotation"}
    other = copy.deepcopy(v2cfg)
    other["run"]["seed"] += 1
    changed = [
        sample_edit(source, {"summary": "s"}, other, i, tmp_path, proposals=proposals).target_point
        != plans[i]["target_point"]
        for i in range(16)
    ]
    assert sum(changed) >= 14  # 16/16 observed; a seed that barely mattered would fail here


def test_concave_region_drags_from_centroid(v2cfg, source, tmp_path):
    from advv.backends.dragflow import upstream_instruction

    hook = rect(10, 10, 69, 69) & ~rect(25, 25, 69, 54)  # C shape: centroid lies outside the mask
    raw = {"phrases": ["hook"], "entities": [{"phrase": "hook", "score": 0.9, "mask": hook, "parts": []}]}
    proposals = fixture_proposals(tmp_path, source, v2cfg, raw)
    start = centroid(hook)
    assert not hook[start[1], start[0]]
    operations = set()
    for index in range(30):
        plan = sample_edit(source, {"summary": "s"}, v2cfg, index, tmp_path, proposals=proposals)
        validate_plan(plan, source, tmp_path)
        assert plan.source_point == start
        assert hook[plan.region_select_point[1], plan.region_select_point[0]]
        shift = plan.operation_params["displacement_pixels"]
        assert [plan.target_point[i] - start[i] for i in (0, 1)] == shift
        if plan.operation == "relocation":
            x0, y0, x1, y1 = plan.operation_params["entity_bbox_after"]
            assert [x0 - 10, y0 - 10] == shift and 0 <= x0 and 0 <= y0 and x1 < W and y1 < H
        # Upstream picks the contour with centroids[0], then drags the region centroid to centroids[1].
        centroids = upstream_instruction(plan.to_dict())["region_operations"]["0"]["centroids"]
        assert centroids == [plan.region_select_point, plan.target_point]
        operations.add(plan.operation)
    assert operations == {"relocation", "deformation"}


def wide_config(v2cfg, target):
    cfg = copy.deepcopy(v2cfg)
    cfg["sampler"]["operations"] = ["relocation"]
    cfg["region_proposal"]["entity_area_fraction"] = [0.02, 0.95]
    cfg["run"]["target_count"] = target
    validate_config(cfg)
    return cfg


WIDE = {
    "phrases": ["pipe"],
    "entities": [{"phrase": "pipe", "score": 0.9, "mask": rect(0, 5, 119, 74), "parts": []}],
}


def test_wide_entity_relocation_never_stalls(v2cfg, source, tmp_path):
    # Regression: only dx = 0 and a few dy keep this full-width entity in the image.
    cfg = wide_config(v2cfg, 40)
    proposals = fixture_proposals(tmp_path / "unit", source, cfg, WIDE)
    for index in range(200):
        plan = sample_edit(source, {"summary": "s"}, cfg, index, tmp_path / "unit", proposals=proposals)
        assert plan.operation_params["displacement_pixels"][0] == 0
    colors = [(i, i, i) for i in range(1, 60)]
    root, state = start_v2(cfg, "wide", FakeBackend(["YES"] * 80, colors, subjects=["pipe"]), WIDE)
    assert state["status"] == "completed" and state["accepted_count"] == 40
    assert "sampling_skipped" not in state
    for record in records_for(root):
        x0, y0, x1, y1 = record["plan"]["operation_params"]["entity_bbox_after"]
        assert (x0, x1) == (0, 119) and 0 <= y0 and y1 < H


def patch_sampler(monkeypatch, skip):
    import advv.pipeline as module

    real, calls = module.sample_edit, []

    def sampler(source, profile, cfg, attempt_index, run_dir, *, proposals=None):
        calls.append(attempt_index)
        if skip(attempt_index):
            raise SamplingSkipped(f"sampling_skipped: forced for attempt {attempt_index}")
        return real(source, profile, cfg, attempt_index, run_dir, proposals=proposals)

    monkeypatch.setattr(module, "sample_edit", sampler)
    return calls


def test_sampling_skip_advances_cursor_and_resumes(v2cfg, monkeypatch):
    calls = patch_sampler(monkeypatch, lambda i: i == 0)
    cfg = copy.deepcopy(v2cfg)
    cfg["run"]["target_count"] = 2
    subjects = ["kitten", "box"]
    root, state = start_v2(
        cfg, "skip", FakeBackend(["YES"] * 2, [(1, 2, 3), KeyboardInterrupt()], subjects=subjects)
    )
    assert state["status"] == "interrupted" and calls == [0, 1, 2]
    assert state["sampling_skipped"][sid(root)] == {"attempts": [0], "consecutive": 0}
    calls.clear()
    state = Pipeline(root, FakeBackend(["YES"] * 2, [(4, 5, 6)], subjects=subjects)).run()
    assert state["status"] == "completed" and state["accepted_count"] == 2 and calls == []
    assert [r["plan"]["attempt_index"] for r in records_for(root)] == [1, 2]
    assert read_json(root / "report.json")["sampling_skipped"][sid(root)]["attempts"] == [0]


def test_repeated_sampling_skips_hold_the_source(v2cfg, monkeypatch):
    calls = patch_sampler(monkeypatch, lambda i: True)
    subjects = ["kitten", "box"]
    root, state = start_v2(v2cfg, "hold", FakeBackend(subjects=subjects))
    limit = v2cfg["sampler"]["object_region"]["max_consecutive_sampling_skips"]
    assert state["status"] == "failed" and "no_eligible_sources" in state["last_error"]["message"]
    assert calls == list(range(limit)) and state["cursor"][sid(root)] == limit
    assert state["sampling_skipped"][sid(root)]["attempts"] == list(range(limit))
    assert state["region_proposals"][sid(root)]["held_reason"] == "sampling_exhausted"
    monkeypatch.undo()
    backend = FakeBackend(subjects=subjects)
    state = Pipeline(root, backend).run()
    assert "no_eligible_sources" in state["last_error"]["message"] and backend.generations == []


def test_proposals_are_bound_to_the_frozen_profile(v2cfg, source, tmp_path):
    profile = profile_for(["kitten", "box"])
    fixture_proposals(tmp_path, source, v2cfg, profile=profile)
    settings = v2cfg["region_proposal"]
    swapped = {**profile, "response": {**profile["response"], "subjects": ["box", "kitten"]}}
    for other, message in (
        (profile_for(["kitten"]), "different frozen source profile"),
        (swapped, "subjects differ"),
    ):
        with pytest.raises(DataError, match=message):
            load_proposals(tmp_path, source, settings, profile=other, allow_fixture=True)
    path = tmp_path / f"proposals/{source.source_id}/proposals.json"
    data = read_json(path)
    atomic_json(path, {**data, "phrases": ["box", "kitten"]})
    with pytest.raises(DataError, match="phrases must be the frozen profile subjects"):
        load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)
    with pytest.raises(DataError, match="not the frozen profile subjects"):
        fixture_proposals(tmp_path / "other", source, v2cfg, profile=profile_for(["dog"]))


def test_real_proposal_backend_needs_pinned_revision(v2cfg, source, tmp_path):
    profile = profile_for(["kitten", "box"])
    built = build_proposals((W, H), raw_fixture(), v2cfg["region_proposal"])
    for revision, recorded, message in (
        (None, "abc", "must be pinned"),
        ("abc", "def", "differs from the run"),
        ("abc", "abc", None),
    ):
        settings = {**v2cfg["region_proposal"], "model_revision": revision}
        models = {"facebook/sam3": recorded}
        write_proposals(
            tmp_path, source, built, backend="sam3", models=models, settings=settings, profile=profile
        )
        if message:
            with pytest.raises(DataError, match=message):
                load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=False)
        else:
            load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=False)


def test_loader_rechecks_filters_and_structure(v2cfg, source, tmp_path):
    profile = profile_for(["kitten", "box"])
    fixture_proposals(tmp_path, source, v2cfg, profile=profile)
    settings = v2cfg["region_proposal"]
    path = tmp_path / f"proposals/{source.source_id}/proposals.json"
    data = read_json(path)

    def expect(edited, message):
        atomic_json(path, edited)
        with pytest.raises(DataError, match=message):
            load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)

    low = copy.deepcopy(data)
    low["entities"][1]["score"] = 0.3
    expect(low, "entity_001: score below the configured minimum")
    part = copy.deepcopy(data)
    part["entities"][0]["parts"][0]["score"] = 0.79
    expect(part, "part_000_000: score below")
    area = copy.deepcopy(data)
    area["entities"][0]["area_fraction"] = 0.5
    expect(area, "area_fraction does not match")
    expect({**data, "entities": [1]}, "entry must be an object")
    expect([data], "must be an object")
    path.write_text("{", encoding="utf-8")
    with pytest.raises(DataError, match="malformed JSON"):
        load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)
    kitten = rect(20, 20, 59, 59)
    twins = {
        "phrases": ["kitten", "box"],
        "parts": [],
        "entities": [
            {"proposal_id": f"entity_00{i}", "phrase": "kitten", "score": 0.9, "mask": kitten, "parts": []}
            for i in (0, 1)
        ],
    }
    write_proposals(
        tmp_path, source, twins, backend="fixture", models={"f": "x"}, settings=settings, profile=profile
    )
    with pytest.raises(DataError, match="entity_001: duplicate"):
        load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)


def test_subject_pattern_matches_prompt():
    from advv.errors import ResponseError
    from advv.verifiers.parser import parse_response

    schema = json.loads((PROJECT / "schemas/source_profile_v3.schema.json").read_text())
    base = {"summary": "s", "must_preserve": ["x"], "parts": [], "uncertain": False}
    for subjects in (["kitten"], ["cracked pipe", "t-shirt"], ["a b c d e"], ["3d printer"]):
        parse_response(json.dumps({**base, "subjects": subjects}), schema)
    for subjects in (["Kitten"], ["pipe."], ["a b c d e f"], ["x-"], []):
        with pytest.raises(ResponseError):
            parse_response(json.dumps({**base, "subjects": subjects}), schema)
    base = {**base, "subjects": ["kitten"]}
    for parts in ([], ["tail"], ["front leg", "boom arm", "door", "head", "bucket"]):
        parse_response(json.dumps({**base, "parts": parts}), schema)
    for parts in (["Tail"], ["tail", "tail"], ["a", "b", "c", "d", "e", "f"], ["front leg."], [""]):
        with pytest.raises(ResponseError):
            parse_response(json.dumps({**base, "parts": parts}), schema)
    with pytest.raises(ResponseError, match="'parts' is a required property"):
        parse_response(json.dumps({k: v for k, v in base.items() if k != "parts"}), schema)
    prompt = (PROJECT / "prompts/source_profile_v3.txt").read_text()
    example = json.loads(next(line for line in prompt.splitlines() if line.startswith('{"summary"')))
    parse_response(json.dumps(example), schema)  # The prompt's own example obeys the schema.


def proposal_set(masks):
    """In-memory ProposalSet: {id: mask}; ids entity_xxx become entities, part_xxx_yyy parts of entity_000."""
    entities = [
        {"proposal_id": k, "phrase": "x", "bbox": bbox_of(m), "parts": []}
        for k, m in masks.items()
        if k.startswith("entity")
    ]
    for k, m in masks.items():
        if k.startswith("part"):
            entities[0]["parts"].append({"proposal_id": k, "bbox": bbox_of(m)})
    return ProposalSet("a", "proposals/a", {"entities": entities}, "h", masks)


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def test_grid_region_reproduces_upstream_bilinear_phase():
    # 64x64 -> 32x32 grid (scale 2): a 2 px bar aligned to a grid cell survives, shifted by 1 px it lands on
    # two half-covered cells (value exactly 0.5, not > 0.5) and vanishes, as in F.interpolate(bilinear).
    assert feature_grid(64, 64) == (32, 32)
    aligned, shifted = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    aligned[:, 10:12], shifted[:, 11:13] = True, True
    assert np.flatnonzero(upstream_grid_region(aligned, 64, 64).any(axis=0)).tolist() == [5]
    assert not upstream_grid_region(shifted, 64, 64).any()
    assert upstream_grid_start(shifted, 64, 64) is None


def test_thin_regions_are_excluded_before_sampling(v2cfg):
    # QA2 N1: 1600x1200 (scale 4). A 3 px bar vanishes on the grid (upstream would drag from (0, 0)); a blob
    # with a long thin tail keeps only the blob, so upstream's start is far from the planned centroid.
    w, h = 1600, 1200
    bar = np.zeros((h, w), bool)
    bar[400:800, 802:805] = True
    tail = np.zeros((h, w), bool)
    tail[300:340, 300:340] = True
    tail[318:321, 340:1300] = True
    body = np.zeros((h, w), bool)
    body[300:700, 300:900] = True
    assert upstream_grid_start(bar, w, h) is None
    begin, planned = upstream_grid_start(tail, w, h), grid_point(centroid(tail), w, h)
    assert max(abs(begin[0] - planned[0]), abs(begin[1] - planned[1])) > 1
    options, excluded = region_table(
        proposal_set({"entity_000": body, "entity_001": bar, "entity_002": tail}), v2cfg, w, h
    )
    assert excluded == {"grid_empty": 1, "grid_centroid_shift": 1}
    kept = {o["proposal_id"] for levels in options.values() for items in levels.values() for o in items}
    assert kept == {"entity_000"}
    option = options["relocation"]["entity"][0]
    assert option["grid_start"] == upstream_grid_start(body, w, h)
    planned = grid_point(option["start"], w, h)
    assert max(abs(option["grid_start"][i] - planned[i]) for i in (0, 1)) <= 1


def test_rotation_needs_grid_radius_and_records_effective_angle(v2cfg, source, tmp_path):
    entity = rect(20, 20, 59, 59)
    head = rect(20, 20, 39, 35)  # contact anchor only ~2.8 grid cells from the part centroid
    options = region_options(proposal_set({"entity_000": entity, "part_000_000": head}), v2cfg, W, H)
    assert "rotation" not in options and options["deformation"]["part"][0]["proposal_id"] == "part_000_000"
    proposals = fixture_proposals(tmp_path, source, v2cfg)
    rotations = [
        plan
        for plan in (
            sample_edit(source, {"summary": "s"}, v2cfg, i, tmp_path, proposals=proposals) for i in range(60)
        )
        if plan.operation == "rotation"
    ]
    assert rotations
    for plan in rotations:
        params = plan.operation_params
        assert params["grid_radius_cells"] >= MIN_ROTATION_RADIUS_CELLS
        expected = grid_rotation_degrees(
            params["upstream_grid_start"], plan.target_point, plan.anchor_point, W, H
        )
        assert params["grid_rotation_degrees"] == expected != 0
        assert grid_point(plan.target_point, W, H) != params["upstream_grid_start"]


def test_shift_targets_avoid_upstream_start_cell(v2cfg, source, tmp_path):
    proposals = fixture_proposals(tmp_path, source, v2cfg)
    for index in range(80):
        plan = sample_edit(source, {"summary": "s"}, v2cfg, index, tmp_path, proposals=proposals)
        begin = plan.operation_params["upstream_grid_start"]
        assert begin == upstream_grid_start(proposals.masks[plan.region_proposal_id], W, H)
        assert grid_point(plan.target_point, W, H) != begin


def test_v1_accepts_older_object_region_without_skip_limit(cfg, v2cfg):
    legacy = copy.deepcopy(cfg)
    del legacy["sampler"]["object_region"]["max_consecutive_sampling_skips"]
    validate_config(legacy)  # QA2 N3: v1 does not use object_region.
    legacy["sampler"]["object_region"]["max_consecutive_sampling_skips"] = 0
    with pytest.raises(ConfigError, match="max_consecutive_sampling_skips"):
        validate_config(legacy)
    v2 = copy.deepcopy(v2cfg)
    del v2["sampler"]["object_region"]["max_consecutive_sampling_skips"]
    with pytest.raises(ConfigError, match="max_consecutive_sampling_skips"):
        validate_config(v2)
