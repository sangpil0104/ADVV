"""T013 part rules: nested duplicates, the 0.02 area floor, point parts only without text parts,
malformed raw input, and the fake SAM 3 reset contract (CPU only)."""

import copy
import importlib.util

import numpy as np
import pytest

from advv.backends.sam3 import Sam3Predictor
from advv.config import validate_config
from advv.errors import ConfigError, DataError
from advv.proposals import build_proposals, load_proposals, nested_duplicate
from advv.storage import atomic_json, read_json
from conftest import PROJECT
from test_region_sampler import H, W, point_part, rect
from test_text_parts import TAIL, mixed_settings, raw_with, source, text_part, written  # noqa: F401

LARGE = rect(20, 20, 39, 59)  # 800 px inside the 1600 px kitten
SAME = rect(20, 20, 39, 51)  # 640 px inside LARGE: containment 1.0, area ratio 0.8, IoU 0.8
SHIFTED = rect(21, 20, 40, 51)  # 608 of 640 px inside LARGE: containment 0.95, area ratio 0.8
HALF = rect(20, 20, 39, 39)  # boom vs boom+arm: inside LARGE but half its size


def kept(settings, parts, names=("tail",)):
    built = build_proposals((W, H), raw_with(parts, names), settings)
    return built["entities"][0]["parts"], built["rejected"]


def test_example_config_values(cfg):
    r = cfg["region_proposal"]
    assert r["part_area_fraction_of_entity"] == [0.02, 0.70]
    assert (r["part_dedupe_containment"], r["part_dedupe_area_ratio"]) == (0.95, 0.80)
    assert r["suppress_point_parts_with_text"] is True


@pytest.mark.parametrize(
    "small,key,value,duplicate",
    [
        (SAME, "part_dedupe_area_ratio", 0.8, True),
        (SAME, "part_dedupe_area_ratio", 0.8001, False),
        (SHIFTED, "part_dedupe_containment", 0.95, True),
        (SHIFTED, "part_dedupe_containment", 0.9501, False),
        (HALF, "part_dedupe_containment", 0.95, False),  # a sub-part much smaller than its holder stays
    ],
)
def test_nested_duplicate_boundaries(v2cfg, small, key, value, duplicate):
    settings = {**v2cfg["region_proposal"], key: value}
    assert nested_duplicate(small, LARGE, settings) is duplicate
    assert nested_duplicate(LARGE, small, settings) is duplicate  # symmetric
    parts, rejected = kept(settings, [text_part("tail", 0.9, LARGE), text_part("kitten tail", 0.8, small)])
    assert len(parts) == (1 if duplicate else 2)
    assert rejected.get("text_part_nested_duplicate", 0) == int(duplicate)


def test_nested_duplicate_keeps_the_first_in_part_order(v2cfg):
    settings = v2cfg["region_proposal"]
    for scores, winner in (((0.9, 0.8), LARGE), ((0.7, 0.8), SAME)):
        parts, _ = kept(
            settings, [text_part("tail", scores[0], LARGE), text_part("kitten tail", scores[1], SAME)]
        )
        assert len(parts) == 1 and np.array_equal(parts[0]["mask"], winner)
    # Point parts use the same rule under their own reason.
    off = {**settings, "suppress_point_parts_with_text": False}
    parts, rejected = kept(off, [point_part(0.95, LARGE), point_part(0.9, SAME)])
    assert len(parts) == 1 and rejected == {"part_nested_duplicate": 1}


@pytest.mark.parametrize("lower,count", [(0.02, 1), (0.0201, 0)])
def test_part_area_floor_boundary(v2cfg, lower, count):
    settings = {**v2cfg["region_proposal"], "part_area_fraction_of_entity": [lower, 0.7]}
    small = rect(20, 20, 27, 23)  # 32 px = 0.02 of the kitten
    assert len(kept(settings, [text_part("tail", 0.9, small)])[0]) == count


POINTS = [point_part(0.9, rect(40, 20, 59, 39)), point_part(0.85, rect(40, 45, 59, 59))]


def test_points_are_suppressed_when_a_text_part_is_kept(v2cfg):
    parts, rejected = kept(v2cfg["region_proposal"], [*POINTS, text_part("tail", 0.6, TAIL)])
    assert [p["source"] for p in parts] == ["text"]
    assert rejected == {"point_suppressed_by_text": 2}


@pytest.mark.parametrize(
    "case",
    ["text_rejected", "text_parts_off", "no_part_names", "setting_off"],
)
def test_points_are_used_without_a_kept_text_part(v2cfg, case):
    settings = copy.deepcopy(v2cfg["region_proposal"])
    parts, names = [*POINTS, text_part("tail", 0.6, TAIL, inside=0.5)], ("tail",)
    if case == "text_parts_off":
        settings["text_parts"], parts = False, list(POINTS)
    elif case == "no_part_names":
        parts, names = list(POINTS), ()
    elif case == "setting_off":
        settings["suppress_point_parts_with_text"] = False
        parts = [*POINTS, text_part("tail", 0.6, TAIL)]
    built, rejected = kept(settings, parts, names)
    assert [p["source"] for p in built].count("point") == 2
    assert "point_suppressed_by_text" not in rejected


def test_suppression_is_per_entity(v2cfg):
    raw = raw_with([text_part("tail", 0.6, TAIL)])
    raw["phrases"] = ["kitten", "box"]
    box = rect(70, 10, 109, 49)
    raw["entities"].append(
        {"phrase": "box", "score": 0.8, "mask": box, "parts": [point_part(0.9, rect(70, 10, 89, 29))]}
    )
    built = build_proposals((W, H), raw, v2cfg["region_proposal"])
    assert [[p["source"] for p in e["parts"]] for e in built["entities"]] == [["text"], ["point"]]


def test_loader_refuses_nested_duplicates(v2cfg, source, tmp_path):  # noqa: F811
    loose = {**mixed_settings(v2cfg), "part_dedupe_area_ratio": 0.81}
    profile = written(
        loose, source, tmp_path, parts=[text_part("tail", 0.9, LARGE), text_part("tail", 0.8, SAME)]
    )
    path = tmp_path / f"proposals/{source.source_id}/proposals.json"
    data = read_json(path)
    assert len(data["entities"][0]["parts"]) == 2
    strict = mixed_settings(v2cfg)
    atomic_json(path, {**data, "settings": strict})
    with pytest.raises(DataError, match="part_000_001: nested duplicate"):
        load_proposals(tmp_path, source, strict, profile=profile, allow_fixture=True)


@pytest.mark.parametrize(
    "part,message",
    [
        (
            {k: v for k, v in text_part("tail", 0.9, TAIL).items() if k != "containment"},
            "missing \\['containment'\\]",
        ),
        ({k: v for k, v in point_part(0.9, TAIL).items() if k != "point"}, "missing \\['point'\\]"),
        ({"source": "point", "score": 0.9}, "missing \\['point', 'mask'\\]"),
    ],
)
def test_malformed_raw_parts_are_data_errors(v2cfg, part, message):
    with pytest.raises(DataError, match=message):
        build_proposals((W, H), raw_with([part]), v2cfg["region_proposal"])


def test_malformed_raw_entity_is_a_data_error(v2cfg):
    raw = raw_with([])
    del raw["entities"][0]["mask"]
    with pytest.raises(DataError, match="phrase, score and mask"):
        build_proposals((W, H), raw, v2cfg["region_proposal"])


def test_part_filter_config_validation(v2cfg, cfg):
    for key, value, message in (
        ("part_dedupe_containment", 0, "part_dedupe_containment must be in \\(0, 1\\]"),
        ("part_dedupe_containment", 1.01, "part_dedupe_containment"),
        ("part_dedupe_area_ratio", True, "part_dedupe_area_ratio"),
        ("suppress_point_parts_with_text", 1, "suppress_point_parts_with_text must be true or false"),
    ):
        bad = copy.deepcopy(v2cfg)
        bad["region_proposal"][key] = value
        with pytest.raises(ConfigError, match=message):
            validate_config(bad)
    missing = copy.deepcopy(v2cfg)
    del missing["region_proposal"]["part_dedupe_area_ratio"]
    with pytest.raises(ConfigError, match="part_dedupe_area_ratio"):
        validate_config(missing)
    legacy = copy.deepcopy(cfg)
    for key in ("part_dedupe_containment", "part_dedupe_area_ratio", "suppress_point_parts_with_text"):
        del legacy["region_proposal"][key]
    validate_config(legacy)  # An older region_proposal section stays valid for random_geometry_v1.
    legacy["region_proposal"]["suppress_point_parts_with_text"] = True
    with pytest.raises(ConfigError, match="part_dedupe_containment"):
        validate_config(legacy)  # One key of the group present: all of it is checked.


def fake_processor_module():
    path = PROJECT / "tests/fixtures/fake_sam3/sam3/model/sam3_image_processor.py"
    spec = importlib.util.spec_from_file_location("fake_sam3_image_processor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeImage:
    size = (W, H)


def test_fake_processor_needs_reset_between_text_prompts(monkeypatch):
    module = fake_processor_module()
    predictor = object.__new__(Sam3Predictor)
    predictor.processor = module.Sam3Processor(None, "cpu", 0.49)
    predictor.state = predictor.processor.set_image(FakeImage())
    assert len(predictor.text("kitten")) == 1 and len(predictor.text("tail")) == 2
    monkeypatch.setattr(predictor.processor, "reset_all_prompts", lambda state: None)
    with pytest.raises(RuntimeError, match="without reset_all_prompts"):
        predictor.text("kitten tail")
