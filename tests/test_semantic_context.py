"""What the semantic-preservation VQA sees of the frozen source profile (CPU only)."""

import json

from advv.pipeline import SEMANTIC_PROFILE_KEYS, Pipeline, create_run, semantic_context
from conftest import FakeBackend
from test_text_parts import KITTEN

V2_PROFILE = {"summary": "A visibly cracked part.", "must_preserve": ["visible crack"], "uncertain": False}
V3_PROFILE = {
    "summary": "A kitten in grass.",
    "must_preserve": ["a kitten"],
    "subjects": ["kitten"],
    "parts": ["head", "tail"],
    "uncertain": False,
}


def semantic_prompts(backend):
    return [c["prompt"] for c in backend.calls if c["prompt"].startswith("You are comparing two images")]


def test_context_holds_only_the_preservation_criteria():
    context = json.loads(semantic_context(V3_PROFILE, "keep the kitten"))
    assert set(context) == {"source_profile", "preserve_hint"}
    assert (
        set(context["source_profile"])
        == set(SEMANTIC_PROFILE_KEYS)
        == {"summary", "must_preserve", "uncertain"}
    )
    assert context["source_profile"] == {k: V3_PROFILE[k] for k in SEMANTIC_PROFILE_KEYS}
    assert context["preserve_hint"] == "keep the kitten"


def test_profile_without_region_keys_renders_as_before():
    # Profile v2 (sampler v1) has no subjects/parts: the context string, and so the semantic prompt hash
    # and cache identity, are byte-for-byte what the full-profile context produced before T013.
    for hint in (None, "a cracked part"):
        before = json.dumps({"source_profile": V2_PROFILE, "preserve_hint": hint}, ensure_ascii=False)
        assert semantic_context(V2_PROFILE, hint) == before


def test_v1_run_semantic_prompt_is_unchanged(cfg):
    backend = FakeBackend()
    root = create_run(cfg, "semantic_v1", provenance="fake")
    assert Pipeline(root, backend).run()["status"] == "completed"
    template = cfg["_assets"]["semantic_prompt"]
    profile = json.loads(json.dumps(V2_PROFILE))
    before = json.dumps({"source_profile": profile, "preserve_hint": None}, ensure_ascii=False)
    assert semantic_prompts(backend) == [template.replace("{{preservation_context}}", before)]


def test_v2_run_semantic_prompt_leaves_out_subjects_and_parts(v2cfg):
    raw = {
        "phrases": ["kitten"],
        "entities": [{"phrase": "kitten", "score": 0.9, "mask": KITTEN, "parts": []}],
    }
    backend = FakeBackend(subjects=["kitten"], parts=["tail"], proposals=raw)
    root = create_run(v2cfg, "semantic_v2", provenance="fake")
    assert Pipeline(root, backend).run()["status"] == "completed"
    (prompt,) = semantic_prompts(backend)
    expected = json.dumps({"source_profile": V2_PROFILE, "preserve_hint": None}, ensure_ascii=False)
    assert prompt == v2cfg["_assets"]["semantic_prompt"].replace("{{preservation_context}}", expected)
    assert '"subjects"' not in prompt and '"parts"' not in prompt
