"""Tests for the shared episode core (agent/episode.py).

The Episode is the train/eval-parity surface: agent/harness.py and
grpo/agent_loop.py both delegate every model-observable computation here.
These tests pin the semantics both loops rely on — especially the ones that
used to diverge between the two hand-synced copies (response budgets, grep
citability, id_map bookkeeping, tool-call budget feedback).
"""

from __future__ import annotations

import json

from agent.episode import Episode, EpisodeConfig
from agent.prompts import render_grep_results, render_search_results_json


def _doc(i: int, text: str = "some passage text") -> dict:
    return {
        "doc_id": f"doc{i}",
        "title": f"Title {i}",
        "text": text,
        "score": 1.0 - i / 100,
    }


# --------------------------------------------------------------------- search


def test_search_dedup_and_seen_commit():
    ep = Episode(EpisodeConfig(top_k_default=2))
    rendered, summary = ep.process_search_results([_doc(0), _doc(1), _doc(2)], 2, 1)
    # Walk stops after top_k=2 NEW docs.
    assert summary["top_doc_ids"] == ["doc0", "doc1"]
    assert set(ep.seen) == {"doc0", "doc1"}
    assert ep.seen["doc0"].first_seen_turn == 1

    # Second search: doc1 already seen -> id-only stub, doc2 is new.
    rendered, summary = ep.process_search_results([_doc(1), _doc(2)], 2, 2)
    assert summary["top_doc_ids"] == ["doc2"]
    assert summary["num_seen_refs"] == 1
    entries = json.loads(rendered)
    assert entries[0] == {"doc_id": "doc1", "seen": True}
    assert entries[1]["doc_id"] == "doc2"
    # Metadata of the already-seen doc was refreshed.
    assert ep.seen["doc1"].num_times_retrieved == 2


def test_search_fetch_k_overfetches_by_seen():
    ep = Episode(EpisodeConfig(top_k_default=5, top_k_max=50))
    assert ep.search_fetch_k(5) == 5
    ep.process_search_results([_doc(0), _doc(1)], 5, 1)
    assert ep.search_fetch_k(5) == 7


def test_search_budget_fitting_gates_citability():
    long_text = "x" * 300
    ep = Episode(
        EpisodeConfig(max_tool_response_length=400, tool_response_truncate_side="left")
    )
    results = [_doc(0, long_text), _doc(1, long_text), _doc(2, long_text)]
    rendered, summary = ep.process_search_results(results, 3, 1)
    assert len(rendered) <= 400
    # Docs whose full passage was truncated out must NOT become citable.
    assert summary["top_doc_ids"] == list(ep.seen)
    assert 0 < len(ep.seen) < 3
    assert summary["num_new"] == len(ep.seen)
    # And they are dropped from the answer if the model cites them anyway.
    assert ep.validate_answer(["doc0", "doc1", "doc2"]) == list(ep.seen)


def test_search_no_budget_renders_everything():
    ep = Episode(EpisodeConfig())  # max_tool_response_length=0 -> unlimited
    results = [_doc(i, "y" * 5000) for i in range(3)]
    rendered, summary = ep.process_search_results(results, 3, 1)
    assert rendered == render_search_results_json(results)
    assert summary["top_doc_ids"] == ["doc0", "doc1", "doc2"]


# ----------------------------------------------------------------------- grep


def test_grep_budget_fitting_gates_citability():
    long_text = "z" * 300
    ep = Episode(
        EpisodeConfig(max_tool_response_length=400, tool_response_truncate_side="left")
    )
    results = [_doc(0, long_text), _doc(1, long_text), _doc(2, long_text)]
    rendered, summary = ep.process_grep_results(results, 10, 1)
    assert len(rendered) <= 400
    assert 0 < len(ep.seen) < 3
    assert summary["top_doc_ids"] == list(ep.seen)
    assert summary["total_matches"] == 10
    # Grep hits carry the 1.0 exact-match sentinel.
    assert all(sp.best_score == 1.0 for sp in ep.seen.values())


def test_grep_no_budget_marks_all_seen():
    ep = Episode(EpisodeConfig())
    results = [_doc(0), _doc(1)]
    rendered, summary = ep.process_grep_results(results, 2, 1)
    assert rendered == render_grep_results(results, 2)
    assert set(ep.seen) == {"doc0", "doc1"}


# ----------------------------------------------------------------- neighbours


def test_neighbours_register_all_results():
    ep = Episode(EpisodeConfig())
    rendered, summary = ep.process_neighbours_results(
        [_doc(0), _doc(1)], "ok", "doc0", 1, 3
    )
    assert set(ep.seen) == {"doc0", "doc1"}
    assert all(sp.best_score == 0.0 for sp in ep.seen.values())
    assert summary["status"] == "ok"
    assert ep.seen["doc0"].first_seen_turn == 3


def test_neighbours_char_truncation_when_over_budget():
    ep = Episode(
        EpisodeConfig(max_tool_response_length=100, tool_response_truncate_side="left")
    )
    rendered, _ = ep.process_neighbours_results(
        [_doc(0, "n" * 500)], "ok", "doc0", 1, 1
    )
    assert len(rendered) <= 100 + len("...(truncated)")
    assert rendered.endswith("...(truncated)")


# --------------------------------------------------------------------- id_map


def test_id_map_serialization_roundtrip():
    ep = Episode(EpisodeConfig(use_id_map=True))
    rendered, summary = ep.process_search_results([_doc(0), _doc(1)], 2, 1)
    entries = json.loads(rendered)
    # The model sees serial ints, assigned by first appearance.
    assert [e["doc_id"] for e in entries] == [0, 1]
    # Summaries and seen stay in real doc_id space.
    assert summary["top_doc_ids"] == ["doc0", "doc1"]
    assert set(ep.seen) == {"doc0", "doc1"}

    # get_neighbours arg: serial int decodes to the real doc_id, the display
    # string stays as the model wrote it.
    real, display = ep.decode_doc_id_arg("1")
    assert (real, display) == ("doc1", "1")
    real, display = ep.decode_doc_id_arg("99")
    assert (real, display) == ("99", "99")

    # <answer> with serial ints translates back to real doc_ids.
    assert ep.validate_answer(["1", "0", "1", "7", "junk"]) == ["doc1", "doc0"]

    # Same doc keeps the same serial across turns/tools.
    rendered, _ = ep.process_search_results([_doc(1), _doc(2)], 2, 2)
    entries = json.loads(rendered)
    assert entries[0] == {"doc_id": 1, "seen": True}
    assert entries[1]["doc_id"] == 2


# --------------------------------------------------------------------- answer


def test_validate_answer_drops_hallucinated_and_duplicates():
    ep = Episode(EpisodeConfig())
    ep.process_search_results([_doc(0), _doc(1)], 2, 1)
    assert ep.validate_answer(["doc1", "ghost", "doc1", "doc0"]) == ["doc1", "doc0"]


# --------------------------------------------------------------------- budget


def test_tool_call_budget_feedback_flow():
    ep = Episode(EpisodeConfig(max_tool_calls=2, tool_budget_feedback=True))
    assert not ep.budget_spent
    assert ep.register_tool_call("resp").endswith("[calls used: 1/2]")
    assert ep.register_tool_call("resp").endswith("[calls used: 2/2]")
    assert ep.budget_spent
    stub = ep.over_budget_stub()
    assert "tool_budget_exhausted" in stub
    assert stub.endswith("[calls used: 2/2]")
    assert ep.num_over_budget_calls == 1
    # The post-budget final-answer turn is granted exactly once.
    assert ep.grant_final_answer_turn(True) is True
    assert ep.grant_final_answer_turn(True) is False


def test_no_feedback_no_suffix():
    ep = Episode(EpisodeConfig(max_tool_calls=2, tool_budget_feedback=False))
    assert ep.register_tool_call("resp") == "resp"
    assert ep.grant_final_answer_turn(True) is False


# ------------------------------------------------------------------ arguments


def test_argument_parsing_is_tolerant():
    ep = Episode(EpisodeConfig(top_k_default=7, top_k_max=10))
    assert ep.parse_top_k({}) == 7
    assert ep.parse_top_k({"top_k": "3"}) == 3
    assert ep.parse_top_k({"top_k": "lots"}) == 7
    assert ep.parse_top_k({"top_k": 999}) == 10
    assert ep.parse_top_k({"top_k": -5}) == 1
    assert Episode.parse_window({"window": "bad"}) == 1
    assert Episode.parse_window({"window": 99}) == 10


def test_model_source_pinned_by_default():
    ep = Episode(EpisodeConfig())
    assert ep.resolve_source({"source": "web"}, "musique") == "musique"
    ep = Episode(EpisodeConfig(allow_model_source=True))
    assert ep.resolve_source({"source": "web"}, "musique") == "web"
    assert ep.resolve_source({}, "musique") == "musique"


# --------------------------------------------------------- trajectory doc ids


def test_trajectory_doc_ids_encounter_order_dedup():
    ep = Episode(EpisodeConfig(use_id_map=True))
    ep.process_search_results([_doc(0), _doc(1)], 2, 1)
    ep.process_grep_results([_doc(1), _doc(2)], 2, 2)
    ep.process_neighbours_results([_doc(3)], "ok", "2", 1, 3)
    assert ep.trajectory_doc_ids == ["doc0", "doc1", "doc2", "doc3"]
