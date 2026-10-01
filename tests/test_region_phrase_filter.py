"""Experimental sampler.object_region.region_phrase_filter (CPU only)."""

import copy

import pytest
import yaml
from PIL import Image

from advv.config import PATH_FIELDS, load_config, recipe_hash, validate_config
from advv.errors import ConfigError, DataError
from advv.ingest import scan_sources
from advv.reporting import records_for
from advv.sampler import phrase_matches, region_table, sample_edit
from advv.storage import read_json
from conftest import PROJECT, FakeBackend
from test_region_sampler import H, W, fixture_proposals, profile_for, rect, sid, start_v2
from test_text_parts import KITTEN, kitten_parts, text_part

LEG = rect(20, 20, 39, 59)  # left half: far enough from its contact anchor for rotation
HEAD = rect(40, 20, 59, 39)
LEG_FILTER = {"level": "part", "phrases_contain": ["LEG"]}


def raw_leg_head():
    return {
        "phrases": ["kitten"],
        "parts": kitten_parts(("leg", "head")),
        "entities": [
            {
                "phrase": "kitten",
                "score": 0.9,
                "mask": KITTEN,
                "parts": [text_part("kitten leg", 0.9, LEG), text_part("kitten head", 0.9, HEAD)],
            }
        ],
    }


def with_filter(cfg, value):
    cfg = copy.deepcopy(cfg)
    cfg["sampler"]["object_region"]["region_phrase_filter"] = value
    validate_config(cfg)
    return cfg


def leg_proposals(v2cfg, source, root):
    raw = raw_leg_head()
    return fixture_proposals(root, source, v2cfg, raw, profile_for(raw["phrases"], raw["parts"]))


@pytest.fixture
def source(cfg):
    return scan_sources(cfg)[0][0]


def test_phrase_match_is_lowercase_substring_at_level():
    part = {"phrase": "cat front leg"}
    assert phrase_matches({"level": "part", "phrases_contain": ["Leg"]}, part, "part")
    assert not phrase_matches({"level": "entity", "phrases_contain": ["leg"]}, part, "part")
    assert phrase_matches({"level": None, "phrases_contain": ["tail", "front"]}, part, "part")
    assert not phrase_matches({"level": "part", "phrases_contain": ["legs"]}, part, "part")
    assert not phrase_matches({"level": None, "phrases_contain": ["leg"]}, {"point": [1, 2]}, "part")


def test_filter_restricts_options_to_matching_parts(v2cfg, source, tmp_path):
    proposals = leg_proposals(v2cfg, source, tmp_path)
    parts = {p["phrase"]: p["proposal_id"] for p in proposals.data["entities"][0]["parts"]}
    assert set(parts) == {"kitten leg", "kitten head"}
    full, _ = region_table(proposals, v2cfg, W, H)
    assert {o["proposal_id"] for o in full["deformation"]["part"]} == set(parts.values())
    cfg = with_filter(v2cfg, LEG_FILTER)
    options, excluded = region_table(proposals, cfg, W, H)
    assert excluded["phrase_filter"] == 2  # the entity and the head
    assert "relocation" not in options and set(options) == {"rotation", "deformation"}
    chosen = {o["proposal_id"] for levels in options.values() for items in levels.values() for o in items}
    assert chosen == {parts["kitten leg"]}
    plans = [sample_edit(source, {"summary": "s"}, cfg, i, tmp_path, proposals=proposals) for i in range(24)]
    assert {p.region_proposal_id for p in plans} == {parts["kitten leg"]}
    assert {p.region_level for p in plans} == {"part"}
    for plan in plans:
        assert plan.operation_params["region_phrase_filter"] == LEG_FILTER
        assert plan.operation_params["region_proposal_phrase"] == "kitten leg"
    again = [sample_edit(source, {"summary": "s"}, cfg, i, tmp_path, proposals=proposals) for i in range(24)]
    assert [p.to_dict() for p in again] == [p.to_dict() for p in plans]


def test_null_filter_keeps_unfiltered_plans(v2cfg, source, tmp_path):
    proposals = leg_proposals(v2cfg, source, tmp_path)
    explicit = copy.deepcopy(v2cfg)
    explicit["sampler"]["object_region"]["region_phrase_filter"] = None
    validate_config(explicit)
    for i in range(12):
        plan = sample_edit(source, {"summary": "s"}, v2cfg, i, tmp_path, proposals=proposals)
        assert "region_phrase_filter" not in plan.operation_params
    assert region_table(proposals, explicit, W, H) == region_table(proposals, v2cfg, W, H)


def test_no_matching_region_holds_without_fallback(v2cfg, source, tmp_path):
    proposals = leg_proposals(v2cfg, source, tmp_path)
    cfg = with_filter(v2cfg, {"level": "part", "phrases_contain": ["wing"]})
    assert region_table(proposals, cfg, W, H)[0] == {}
    with pytest.raises(DataError, match="source_no_region"):
        sample_edit(source, {"summary": "s"}, cfg, 0, tmp_path, proposals=proposals)
    # The entity matches by name but only at entity level, which the part filter excludes.
    cfg = with_filter(v2cfg, {"level": "part", "phrases_contain": ["kitten"]})
    options, _ = region_table(proposals, cfg, W, H)
    assert "relocation" not in options


def test_pipeline_holds_and_records_filter(v2cfg):
    cfg = with_filter(v2cfg, {"level": "part", "phrases_contain": ["wing"]})
    backend = FakeBackend(subjects=["kitten"], parts=kitten_parts(("leg", "head")))
    root, state = start_v2(cfg, "v2_filter_hold", backend, raw_leg_head())
    assert state["status"] == "failed" and "no_eligible_sources" in state["last_error"]["message"]
    entry = state["region_proposals"][sid(root)]
    assert entry["status"] == "source_no_region" and entry["region_phrase_filter"] == cfg["sampler"][
        "object_region"
    ]["region_phrase_filter"]
    assert backend.generations == []


def test_pipeline_records_filter_in_plan_and_report(v2cfg):
    cfg = with_filter(v2cfg, LEG_FILTER)
    backend = FakeBackend(subjects=["kitten"], parts=kitten_parts(("leg", "head")))
    root, state = start_v2(cfg, "v2_filter_leg", backend, raw_leg_head())
    assert state["status"] == "completed" and state["accepted_count"] == 1
    plan = records_for(root)[0]["plan"]
    assert plan["region_level"] == "part" and plan["operation_params"]["region_proposal_phrase"] == "kitten leg"
    report = read_json(root / "report.json")
    assert report["region_phrase_filter"] == LEG_FILTER
    assert report["region_proposals"][sid(root)]["region_phrase_filter"] == LEG_FILTER
    assert "Region phrase filter (experimental)" in (root / "report.md").read_text()
    assert read_json(root / "manifest.json")["recipe_sha256"] == recipe_hash(cfg)
    assert recipe_hash(cfg) != recipe_hash(v2cfg)


@pytest.mark.parametrize(
    "value, message",
    [
        ({"level": "leg", "phrases_contain": ["leg"]}, "level"),
        ({"level": "part"}, "region_phrase_filter must be"),
        ({"level": "part", "phrases_contain": ["leg"], "extra": 1}, "region_phrase_filter must be"),
        ({"level": "part", "phrases_contain": []}, "phrases_contain"),
        ({"level": "part", "phrases_contain": ["  "]}, "phrases_contain"),
        ({"level": "part", "phrases_contain": "leg"}, "phrases_contain"),
        (["leg"], "region_phrase_filter must be"),
    ],
)
def test_invalid_filter_is_rejected(v2cfg, value, message):
    with pytest.raises(ConfigError, match=message):
        with_filter(v2cfg, value)


def test_filter_is_v2_only(cfg):
    with pytest.raises(ConfigError, match="only for object_region_v2"):
        with_filter(cfg, LEG_FILTER)
    with_filter(cfg, None)


def test_explicit_null_loads_like_absent(tmp_path):
    raw = yaml.safe_load((PROJECT / "configs/advv.example.yaml").read_text())
    for section, keys in PATH_FIELDS.items():
        for key in keys:
            if raw[section].get(key) is not None:
                raw[section][key] = str((PROJECT / "configs" / raw[section][key]).resolve())
    images = tmp_path / "input"
    images.mkdir()
    Image.new("RGB", (120, 80)).save(images / "a.png")
    absent = tmp_path / "absent.yaml"
    absent.write_text(yaml.safe_dump(raw))
    raw["sampler"]["object_region"]["region_phrase_filter"] = None
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(yaml.safe_dump(raw))
    a, b = (load_config(p, input_dir=images, gpus="0") for p in (absent, explicit))
    assert "region_phrase_filter" not in b["sampler"]["object_region"]
    a.pop("_config_path"), b.pop("_config_path")
    assert recipe_hash(a) == recipe_hash(b)
