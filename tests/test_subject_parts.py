"""T015: source_profile_v4 parts belong to one subject each; entities are asked only for their own parts."""

import copy
import json

import pytest

from advv.backends.region import Proposer
from advv.errors import DataError, ResponseError
from advv.pipeline import Pipeline, create_run, profile_parts_check
from advv.proposals import (
    build_proposals,
    collect_raw,
    load_proposals,
    subject_parts,
    subject_parts_error,
    write_proposals,
)
from advv.storage import atomic_json, read_json
from advv.verifiers.parser import parse_response
from conftest import PROJECT, FakeBackend
from test_region_sampler import H, W, profile_for, rect
from test_text_parts import Predictor, source, text_part  # noqa: F401

CAR, TREE = rect(10, 30, 59, 69), rect(70, 5, 109, 74)
BUMPER = rect(10, 55, 25, 69)
CAR_PARTS = [{"subject": "damaged car", "parts": ["front bumper"]}, {"subject": "tree", "parts": []}]


def schema():
    return json.loads((PROJECT / "schemas/source_profile_v4.schema.json").read_text())


def test_entity_is_asked_only_for_its_own_subject_parts(v2cfg):
    # v2_pilot_001: one flat list made SAM 3 search "tree front bumper" and reject 39 instances.
    settings = {**v2cfg["region_proposal"], "part_points_per_entity": 1}
    predictor = Predictor(
        {
            "damaged car": [(CAR, 0.9)],
            "tree": [(TREE, 0.9)],
            "damaged car front bumper": [(BUMPER, 0.8)],
            "front bumper": [(BUMPER, 0.7)],
        }
    )
    raw = collect_raw(predictor, ["damaged car", "tree"], CAR_PARTS, settings, lambda i: i)
    assert predictor.queries == ["damaged car", "tree", "damaged car front bumper", "front bumper"]
    car, tree = ([p for p in e["parts"] if p["source"] == "text"] for e in raw["entities"])
    assert [p["phrase"] for p in car] == ["damaged car front bumper", "front bumper"]
    assert tree == []
    assert raw["parts"] == CAR_PARTS
    # A subject without an entry has no named parts either.
    predictor.queries.clear()
    collect_raw(predictor, ["damaged car", "tree"], CAR_PARTS[:1], settings, lambda i: i)
    assert not any(q.startswith("tree ") for q in predictor.queries)


def test_shared_part_name_is_asked_once_for_subjects_that_list_it(v2cfg):
    settings = {**v2cfg["region_proposal"], "part_points_per_entity": 1}
    predictor = Predictor({"car": [(CAR, 0.9)], "truck": [(TREE, 0.9)], "wheel": [(BUMPER, 0.8)]})
    parts = [{"subject": "car", "parts": ["wheel"]}, {"subject": "truck", "parts": ["wheel"]}]
    raw = collect_raw(predictor, ["car", "truck"], parts, settings, lambda i: i)
    assert predictor.queries.count("wheel") == 1
    assert [p["containment"] for e in raw["entities"] for p in e["parts"] if p.get("phrase") == "wheel"] == [
        1.0,
        0.0,
    ]


def test_subject_parts_lookup():
    assert subject_parts(CAR_PARTS, "damaged car") == ["front bumper"]
    assert subject_parts(CAR_PARTS, "tree") == []
    assert subject_parts(CAR_PARTS[:1], "tree") == []
    assert subject_parts([], "tree") == []


def test_v4_schema_boundaries():
    v4 = schema()
    base = {"summary": "s", "must_preserve": ["x"], "subjects": ["damaged car", "tree"], "uncertain": False}
    for parts in (
        [],
        CAR_PARTS,
        [{"subject": "tree", "parts": []}],
        [{"subject": "damaged car", "parts": ["a", "b", "c", "d", "e"]}],
        [{"subject": name, "parts": []} for name in ("a", "b", "c")],
    ):
        parse_response(json.dumps({**base, "parts": parts}), v4)
    for parts in (
        ["front bumper"],  # v3 flat list
        [{"subject": "tree"}],
        [{"parts": ["door"]}],
        [{"subject": "tree", "parts": [], "extra": 1}],
        [{"subject": "Tree", "parts": []}],
        [{"subject": "tree", "parts": ["Door"]}],
        [{"subject": "tree", "parts": ["door", "door"]}],
        [{"subject": "tree", "parts": ["a", "b", "c", "d", "e", "f"]}],
        [{"subject": name, "parts": []} for name in ("a", "b", "c", "d")],
        [{"subject": "tree", "parts": [""]}],
    ):
        with pytest.raises(ResponseError):
            parse_response(json.dumps({**base, "parts": parts}), v4)
    with pytest.raises(ResponseError, match="'parts' is a required property"):
        parse_response(json.dumps(base), v4)
    prompt = (PROJECT / "prompts/source_profile_v4.txt").read_text()
    example = json.loads(next(line for line in prompt.splitlines() if line.startswith('{"summary"')))
    parse_response(json.dumps(example), v4)  # The prompt's own example obeys the schema ...
    profile_parts_check(example)  # ... and the cross-field rule.
    assert "{{user_hint}}" in prompt


def test_profile_parts_must_name_distinct_profile_subjects():
    base = {"subjects": ["damaged car", "tree"]}
    profile_parts_check({**base, "parts": CAR_PARTS})
    profile_parts_check({**base, "parts": []})
    for parts, message in (
        ([{"subject": "car", "parts": ["door"]}], "'car' is not a distinct profile subject"),
        ([CAR_PARTS[1], CAR_PARTS[1]], "'tree' is not a distinct profile subject"),
    ):
        with pytest.raises(ResponseError, match=message):
            profile_parts_check({**base, "parts": parts})


def test_unhashable_part_subject_is_reported_not_raised():
    # T015 QA L2: a list subject equal to a (corrupt) subjects element used to hit `in set` -> TypeError.
    problem = subject_parts_error([{"subject": ["tree"], "parts": []}], [["tree"]])
    assert problem == "parts entry subject ['tree'] is not a distinct profile subject"
    assert subject_parts_error([{"subject": {"a": 1}, "parts": []}], ["tree"]) is not None


def test_profile_with_unknown_part_subject_is_a_profile_error(v2cfg):
    bad = [{"subject": "dog", "parts": ["tail"]}]
    backend = FakeBackend(subjects=["kitten"], parts=bad, proposals={"phrases": ["kitten"], "entities": []})
    root = create_run(v2cfg, "unknown_part_subject", provenance="fake")
    state = Pipeline(root, backend).run()
    assert "no_eligible_sources" in state["last_error"]["message"] and backend.proposal_calls == []
    profile = read_json(next((root / "profiles").glob("*.json")))
    assert profile["status"] == "source_error"
    assert "'dog' is not a distinct profile subject" in profile["checks"]["profile"]["attempts"][-1]["error"]


def test_pipeline_sends_and_freezes_per_subject_parts(v2cfg):
    raw = {
        "phrases": ["damaged car", "tree"],
        "entities": [
            {"phrase": "damaged car", "score": 0.9, "mask": CAR, "parts": []},
            {"phrase": "tree", "score": 0.9, "mask": TREE, "parts": []},
        ],
    }
    backend = FakeBackend(subjects=["damaged car", "tree"], parts=CAR_PARTS, proposals=raw)
    root = create_run(v2cfg, "per_subject", provenance="fake")
    assert Pipeline(root, backend).run()["status"] == "completed"
    sid = read_json(root / "sources.json")[0]["source_id"]
    assert backend.proposal_calls == [(sid, ["damaged car", "tree"], CAR_PARTS)]
    assert read_json(root / f"proposals/{sid}/proposals.json")["parts"] == CAR_PARTS


def test_loader_refuses_a_part_phrase_of_another_subject(v2cfg, source, tmp_path):  # noqa: F811
    settings = v2cfg["region_proposal"]
    profile = profile_for(["damaged car", "tree"], CAR_PARTS)
    raw = {
        "phrases": ["damaged car", "tree"],
        "parts": CAR_PARTS,
        "entities": [
            {"phrase": "damaged car", "score": 0.9, "mask": CAR, "parts": []},
            {"phrase": "tree", "score": 0.8, "mask": TREE, "parts": []},
        ],
    }
    built = build_proposals((W, H), raw, settings)
    write_proposals(
        tmp_path, source, built, backend="fixture", models={"f": "x"}, settings=settings, profile=profile
    )
    path = tmp_path / f"proposals/{source.source_id}/proposals.json"
    data = read_json(path)
    load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)
    # A bumper-shaped part under the tree, named with the car's part: the tree has no named parts.
    bumper = rect(75, 60, 90, 74)
    raw["entities"][1]["parts"] = [text_part("front bumper", 0.9, bumper)]
    built = build_proposals((W, H), raw, settings)
    assert [p["phrase"] for p in built["entities"][1]["parts"]] == ["front bumper"]
    write_proposals(
        tmp_path, source, built, backend="fixture", models={"f": "x"}, settings=settings, profile=profile
    )
    with pytest.raises(DataError, match="part_001_000: phrase is not a named part of entity_001"):
        load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)
    edited = copy.deepcopy(data)
    edited["parts"] = edited["parts"] + [{"subject": "tree", "parts": []}]
    atomic_json(path, edited)
    with pytest.raises(DataError, match="parts differ from the frozen source profile"):
        load_proposals(tmp_path, source, settings, profile=profile, allow_fixture=True)
    twice = profile_for(["damaged car", "tree"], CAR_PARTS + CAR_PARTS[1:])
    twice_data = {**data, "parts": twice["response"]["parts"], "source_profile_sha256": twice["frozen_sha256"]}
    atomic_json(path, twice_data)
    with pytest.raises(DataError, match="'tree' is not a distinct profile subject"):
        load_proposals(tmp_path, source, settings, profile=twice, allow_fixture=True)


def test_worker_refuses_flat_or_foreign_part_lists(v2cfg, source, tmp_path):  # noqa: F811
    proposer = Proposer(v2cfg, tmp_path, predictor=None)
    request = {"action": "propose", "source": source.to_dict(), "subjects": ["kitten"]}
    for parts in (
        ["tail"],
        [{"subject": "dog", "parts": ["tail"]}],
        [{"subject": "kitten", "parts": [""]}],
        [{"subject": ["kitten"], "parts": ["tail"]}],  # unhashable subject: DataError, not TypeError
    ):
        with pytest.raises(DataError, match="Part names for"):
            proposer({**request, "parts": parts})
