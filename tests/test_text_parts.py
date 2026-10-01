"""Named-part text prompts next to point prompts: collection, attribution, filters, contract (CPU only)."""

import copy

import numpy as np
import pytest

from advv.backends.sam3 import THRESHOLD_MARGIN, processor_threshold
from advv.config import validate_config
from advv.errors import ConfigError, DataError
from advv.pipeline import Pipeline, create_run
from advv.proposals import build_proposals, collect_raw, load_proposals, text_part_phrases, write_proposals
from advv.storage import atomic_json, read_json
from conftest import FakeBackend
from test_region_sampler import H, W, point_part, profile_for, rect

KITTEN = rect(20, 20, 59, 59)
TAIL = rect(20, 40, 35, 59)


def text_part(phrase, score, mask, inside=1.0, index=0):
    return {
        "source": "text",
        "phrase": phrase,
        "instance_index": index,
        "score": score,
        "box": None,
        "containment": inside,
        "mask": mask,
    }


def raw_with(parts, names=("tail",)):
    return {
        "phrases": ["kitten"],
        "parts": list(names),
        "entities": [{"phrase": "kitten", "score": 0.9, "mask": KITTEN, "parts": parts}],
    }


class Predictor:
    """In-process stand-in for the SAM 3 adapter: phrase -> instances, any point -> one square."""

    def __init__(self, instances):
        self.instances, self.queries = instances, []

    def text(self, phrase):
        self.queries.append(phrase)
        return [(m, s, [0, 0, 1, 1]) for m, s in self.instances.get(phrase, [])]

    def point(self, point):
        x, y = point
        return [(rect(max(0, x - 3), max(0, y - 3), x + 3, y + 3), 0.9)]


def test_part_phrases_follow_parts_then_forms():
    assert text_part_phrases("kitten", ["tail", "front leg"], ["subject_part", "part"]) == [
        "kitten tail",
        "tail",
        "kitten front leg",
        "front leg",
    ]
    assert text_part_phrases("kitten", ["tail"], ["part"]) == ["tail"]
    assert text_part_phrases("kitten", [], ["subject_part", "part"]) == []


def test_collect_attributes_text_instances_by_containment(v2cfg):
    settings = {**v2cfg["region_proposal"], "part_points_per_entity": 1}
    other = rect(70, 10, 109, 49)
    stray = rect(50, 50, 65, 65)  # 100 of 256 pixels inside the kitten, none inside the cat
    predictor = Predictor(
        {
            "kitten": [(KITTEN, 0.9)],
            "cat": [(other, 0.8)],
            "kitten tail": [(TAIL, 0.7)],
            "tail": [(TAIL, 0.6), (stray, 0.95)],
            "cat tail": [(rect(70, 30, 80, 49), 0.4)],
        }
    )
    raw = collect_raw(predictor, ["kitten", "cat"], ["tail"], settings, lambda i: i)
    # "tail" is asked once and offered to both entities.
    assert predictor.queries == ["kitten", "cat", "kitten tail", "tail", "cat tail"]
    assert raw["parts"] == ["tail"]
    kitten, cat = ([p for p in e["parts"] if p["source"] == "text"] for e in raw["entities"])
    assert [(p["phrase"], p["instance_index"]) for p in kitten] == [
        ("kitten tail", 0),
        ("tail", 0),
        ("tail", 1),
    ]
    assert [p["containment"] for p in kitten] == [1.0, 1.0, pytest.approx(100 / 256)]
    assert kitten[2]["mask"] is None  # high score, but mostly outside the kitten
    assert [p["mask"] is not None for p in kitten[:2]] == [True, True]
    # The cat keeps nothing: its own tail is below min_text_part_score, the others lie outside it.
    assert [(p["phrase"], p["mask"] is None) for p in cat] == [
        ("cat tail", True),
        ("tail", True),
        ("tail", True),
    ]
    assert all(p["source"] == "point" for e in raw["entities"] for p in e["parts"][-1:])


def test_text_parts_off_asks_no_part_phrases(v2cfg):
    settings = {**v2cfg["region_proposal"], "text_parts": False, "part_points_per_entity": 1}
    predictor = Predictor({"kitten": [(KITTEN, 0.9)], "kitten tail": [(TAIL, 0.9)]})
    raw = collect_raw(predictor, ["kitten"], ["tail"], settings, lambda i: i)
    assert predictor.queries == ["kitten"]
    assert {p["source"] for p in raw["entities"][0]["parts"]} == {"point"}


def test_low_entities_get_no_part_queries(v2cfg):
    predictor = Predictor({"kitten": [(KITTEN, 0.3)], "kitten tail": [(TAIL, 0.9)]})
    raw = collect_raw(predictor, ["kitten"], ["tail"], v2cfg["region_proposal"], lambda i: i)
    assert predictor.queries == ["kitten"] and raw["entities"][0]["parts"] == []


def test_build_orders_text_first_and_counts_rejections_by_source(v2cfg):
    # Point suppression off so both sources stay side by side (the suppressed case is tested below).
    settings = {**v2cfg["region_proposal"], "suppress_point_parts_with_text": False}
    # Just below the configured cut-offs, so the test follows the example yaml values.
    low_score = settings["min_text_part_score"] - 0.01
    outside = settings["text_part_containment"] - 0.01
    parts = [
        point_part(0.99, rect(20, 40, 35, 59)),  # same mask as the named tail: the named part wins
        point_part(0.9, rect(40, 20, 59, 39)),
        text_part("tail", 0.55, TAIL),
        text_part("kitten tail", 0.6, TAIL),
        text_part("tail", low_score, None),  # below min_text_part_score
        text_part("tail", 0.9, None, inside=outside),  # mostly outside the entity
        text_part("kitten tail", 0.8, rect(50, 50, 51, 51)),  # too small
    ]
    built = build_proposals((W, H), raw_with(parts), settings)
    kept = built["entities"][0]["parts"]
    assert [(p["proposal_id"], p["source"], p.get("phrase"), p["score"]) for p in kept] == [
        ("part_000_000", "text", "kitten tail", 0.6),
        ("part_000_001", "point", None, 0.9),
    ]
    assert kept[1]["point"] == [50, 30] and built["parts"] == ["tail"]
    assert built["rejected"] == {
        "text_part_score": 1,
        "text_part_containment": 1,
        "text_part_area": 1,
        "text_part_duplicate": 1,
        "part_duplicate": 1,
    }


@pytest.mark.parametrize(
    "key,value,kept",
    [
        ("min_text_part_score", 0.6, True),
        ("min_text_part_score", 0.6001, False),
        ("text_part_containment", 0.9, True),
        ("text_part_containment", 0.9001, False),
    ],
)
def test_text_part_filter_boundaries(v2cfg, key, value, kept):
    settings = {**v2cfg["region_proposal"], key: value}
    built = build_proposals((W, H), raw_with([text_part("tail", 0.6, TAIL, inside=0.9)]), settings)
    assert bool(built["entities"][0]["parts"]) is kept


def test_build_refuses_unlabelled_or_disabled_text_parts(v2cfg):
    settings = v2cfg["region_proposal"]
    with pytest.raises(DataError, match="source None"):
        build_proposals((W, H), raw_with([{"score": 0.9, "mask": TAIL}]), settings)
    with pytest.raises(DataError, match="text_parts is off"):
        build_proposals((W, H), raw_with([text_part("tail", 0.9, TAIL)]), {**settings, "text_parts": False})


def mixed_settings(v2cfg):
    """Both part sources kept side by side, so one file exercises the text and the point contract."""
    return {**v2cfg["region_proposal"], "suppress_point_parts_with_text": False}


def written(settings, source, root, parts=None, names=("tail",)):
    if parts is None:
        parts = [text_part("kitten tail", 0.6, TAIL), point_part(0.9, rect(40, 20, 59, 39))]
    profile = profile_for(["kitten"], names)
    built = build_proposals((W, H), raw_with(parts, names), settings)
    write_proposals(
        root,
        source,
        built,
        backend="fixture",
        models={"fixture": "test-only"},
        settings=settings,
        profile=profile,
    )
    return profile


@pytest.fixture
def source(cfg):
    from advv.ingest import scan_sources

    return scan_sources(cfg)[0][0]


def test_text_part_contract_round_trip(v2cfg, source, tmp_path):
    settings = mixed_settings(v2cfg)
    profile = written(settings, source, tmp_path)
    loaded = load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)
    data = loaded.data
    assert data["schema_version"] == "1.1" and data["parts"] == ["tail"]
    first, second = data["entities"][0]["parts"]
    assert first["source"] == "text" and first["phrase"] == "kitten tail" and "point" not in first
    assert second["source"] == "point" and second["point"] == [50, 30] and "phrase" not in second
    assert set(loaded.masks) == {"entity_000", "part_000_000", "part_000_001"}


def test_loader_rechecks_text_part_provenance(v2cfg, source, tmp_path):
    settings = mixed_settings(v2cfg)
    profile = written(settings, source, tmp_path)
    path = tmp_path / f"proposals/{source.source_id}/proposals.json"
    data = read_json(path)

    def expect(edited, message, profile=profile, settings=settings):
        atomic_json(path, edited)
        with pytest.raises(DataError, match=message):
            load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)

    def part_edit(**changes):
        edited = copy.deepcopy(data)
        edited["entities"][0]["parts"][0].update(changes)
        return edited

    expect(part_edit(phrase="kitten head"), "part_000_000: phrase is not a named part")
    expect(part_edit(source="guess"), "part_000_000: unknown source")
    expect(part_edit(score=0.49), "part_000_000: score below")
    point = copy.deepcopy(data)
    point["entities"][0]["parts"][1]["point"] = [50.0, 30.0]
    expect(point, "point prompt must be")
    for outside in ([10, 10], [W, 30], [-1, 30]):  # outside the entity / the image
        point["entities"][0]["parts"][1]["point"] = outside
        expect(point, "point prompt must lie inside entity_000")
    suppressing = v2cfg["region_proposal"]
    expect({**data, "settings": suppressing}, "point parts are suppressed", settings=suppressing)
    expect({**data, "parts": ["tail", "head"]}, "parts differ from the frozen source profile")
    other = {**profile, "response": {**profile["response"], "parts": ["head"]}}
    expect(data, "parts differ", profile=other)
    expect(data, "settings differ", settings={**settings, "text_parts": False})


def test_loader_needs_text_parts_before_point_parts(v2cfg, source, tmp_path):
    settings = v2cfg["region_proposal"]
    built = build_proposals((W, H), raw_with([text_part("kitten tail", 0.6, TAIL)]), settings)
    text = built["entities"][0]["parts"][0]
    point = {"proposal_id": "part_000_000", "source": "point", "point": [50, 30], "score": 0.9}
    point["mask"] = rect(40, 20, 59, 39)
    built["entities"][0]["parts"] = [point, {**text, "proposal_id": "part_000_001"}]
    profile = profile_for(["kitten"], ["tail"])
    write_proposals(
        tmp_path, source, built, backend="fixture", models={"f": "x"}, settings=settings, profile=profile
    )
    with pytest.raises(DataError, match="entity_000: text parts must precede points"):
        load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)


def test_write_refuses_parts_other_than_the_profile(v2cfg, source, tmp_path):
    built = build_proposals((W, H), raw_with([], ["tail"]), v2cfg["region_proposal"])
    with pytest.raises(DataError, match="not the frozen profile parts"):
        write_proposals(
            tmp_path,
            source,
            built,
            backend="fixture",
            models={"f": "x"},
            settings=v2cfg["region_proposal"],
            profile=profile_for(["kitten"], ["head"]),
        )


@pytest.mark.parametrize("text_parts, expected", [(True, ["tail", "front leg"]), (False, [])])
def test_pipeline_sends_frozen_profile_parts(v2cfg, text_parts, expected):
    v2cfg["region_proposal"]["text_parts"] = text_parts
    raw = {
        "phrases": ["kitten"],
        "entities": [{"phrase": "kitten", "score": 0.9, "mask": KITTEN, "parts": []}],
    }
    backend = FakeBackend(subjects=["kitten"], parts=["tail", "front leg"], proposals=raw)
    root = create_run(v2cfg, f"parts_{text_parts}", provenance="fake")
    state = Pipeline(root, backend).run()
    assert state["status"] == "completed"
    sid = read_json(root / "sources.json")[0]["source_id"]
    assert backend.proposal_calls == [(sid, ["kitten"], expected)]
    assert read_json(root / f"proposals/{sid}/proposals.json")["parts"] == expected


def test_text_part_config_validation(v2cfg, cfg):
    for key, value, message in (
        ("text_parts", "yes", "text_parts must be true or false"),
        ("text_part_forms", [], "text_part_forms"),
        ("text_part_forms", ["part", "part"], "text_part_forms"),
        ("text_part_forms", ["whole"], "text_part_forms"),
        ("min_text_part_score", 1.5, "min_text_part_score"),
        ("text_part_containment", 0, "text_part_containment must be positive"),
    ):
        bad = copy.deepcopy(v2cfg)
        bad["region_proposal"][key] = value
        with pytest.raises(ConfigError, match=message):
            validate_config(bad)
    bad = copy.deepcopy(v2cfg)
    del bad["region_proposal"]["min_text_part_score"]
    with pytest.raises(ConfigError, match="min_text_part_score"):
        validate_config(bad)
    no_parts = copy.deepcopy(v2cfg)
    schema = no_parts["_assets"]["profile_schema"]
    schema["required"] = [k for k in schema["required"] if k != "parts"]
    with pytest.raises(ConfigError, match="profile with parts"):
        validate_config(no_parts)
    no_parts["region_proposal"]["text_parts"] = False
    validate_config(no_parts)  # Point parts alone do not need profile parts.
    legacy = copy.deepcopy(cfg)
    for key in ("text_parts", "text_part_forms", "min_text_part_score", "text_part_containment"):
        del legacy["region_proposal"][key]
    validate_config(legacy)  # An older region_proposal section stays valid for random_geometry_v1.


def test_processor_threshold_survives_low_precision_scores(v2cfg):
    settings = v2cfg["region_proposal"]
    assert processor_threshold(settings) == pytest.approx(0.49)
    assert processor_threshold({**settings, "min_text_part_score": 0.3}) == pytest.approx(0.29)
    assert processor_threshold(
        {**settings, "text_parts": False, "min_text_part_score": 0.3}
    ) == pytest.approx(0.49)
    assert processor_threshold({**settings, "min_entity_score": 0.0}) == 0.0
    # Upstream compares `scores > threshold` in the score dtype; a score equal to an ADVV minimum must
    # survive that cast for every minimum, which a nextafter(-inf) threshold did not (T009b QA L1).
    bf16_step = 2.0**-8
    assert THRESHOLD_MARGIN > bf16_step
    for minimum in np.linspace(0.01, 1.0, 100):
        threshold = processor_threshold(
            {**settings, "min_entity_score": minimum, "min_text_part_score": minimum}
        )
        assert np.float32(minimum) > np.float32(threshold)
        assert minimum - threshold > bf16_step
