"""End-to-end smoke test of AgentHarness over the shared episode core.

Uses a fake OpenAI-shaped model client and a fake tool client, so it exercises
the real run loop (budget feedback, tool execution, answer validation) without
a vLLM or tool server.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.episode import Episode, EpisodeConfig
from agent.harness import AgentHarness
from agent.schemas import AgentInput, Message


def _tool_call(name: str, args: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _completion(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class FakeModelClient:
    """Replays a scripted list of assistant turns."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        return self._turns.pop(0)


class FakeToolClient:
    def __init__(self, docs):
        self.docs = docs

    def local_search(self, query, top_k, source=None, extra_params=None):
        return {"results": self.docs[:top_k], "status": "ok", "total_matches": top_k}

    def grep(self, pattern, top_k, source=None, extra_params=None):
        return {
            "results": self.docs[:top_k],
            "status": "ok",
            "total_matches": len(self.docs),
        }

    def get_neighbours(self, doc_id, window, source=None, extra_params=None):
        return {"results": [], "status": "independent_chunk", "total_matches": 0}


def _docs(n, text_len=50):
    return [
        {
            "doc_id": f"doc{i}",
            "title": f"Title {i}",
            "text": "t" * text_len,
            "score": 0.9 - i / 100,
        }
        for i in range(n)
    ]


def _harness(turns, docs, **kwargs):
    return AgentHarness(
        model="fake",
        tool_client=FakeToolClient(docs),
        model_client=FakeModelClient(turns),
        prompt_version="v2",
        **kwargs,
    )


def test_search_then_answer():
    turns = [
        _completion(tool_calls=[_tool_call("search", {"query": "q", "top_k": 3})]),
        _completion(content="<answer>doc1, doc0, ghost</answer>"),
    ]
    h = _harness(turns, _docs(3))
    out = h.run(AgentInput(messages=[Message(role="user", content="q?")]))
    assert out.stopped_reason == "answer"
    # Hallucinated id dropped, order preserved.
    assert out.ranked_doc_ids == ["doc1", "doc0"]
    assert [p.doc_id for p in out.ranked_passages] == ["doc1", "doc0"]
    # The tool message the model saw is exactly the shared-core render.
    ref = Episode(EpisodeConfig())
    expected, _ = ref.process_search_results(_docs(3), 3, 1)
    tool_msgs = [m for m in out.trajectory.messages_full if m.role == "tool"]
    assert tool_msgs[0].content == expected


def test_response_budget_applies_at_inference():
    # Long passages + small budget: trailing hits are dropped from the render
    # and must not be citable — the divergence that motivated agent/episode.py.
    turns = [
        _completion(tool_calls=[_tool_call("search", {"query": "q", "top_k": 3})]),
        _completion(content="<answer>doc0, doc1, doc2</answer>"),
    ]
    h = _harness(
        turns,
        _docs(3, text_len=300),
        max_tool_response_length=400,
        tool_response_truncate_side="left",
    )
    out = h.run(AgentInput(messages=[Message(role="user", content="q?")]))
    tool_msgs = [m for m in out.trajectory.messages_full if m.role == "tool"]
    assert len(tool_msgs[0].content) <= 400
    assert 0 < len(out.ranked_doc_ids) < 3


def test_budget_feedback_grants_one_final_turn():
    search = {"query": "q", "top_k": 2}
    turns = [
        _completion(tool_calls=[_tool_call("search", search)]),
        # Budget (1) is spent; this call gets the stub nudge...
        _completion(tool_calls=[_tool_call("search", search, call_id="call_2")]),
        # ...and the model answers on its one extra turn.
        _completion(content="<answer>doc0</answer>"),
    ]
    h = _harness(turns, _docs(2), tool_budget_feedback=True)
    out = h.run(
        AgentInput(messages=[Message(role="user", content="q?")], max_tool_calls=1)
    )
    assert out.stopped_reason == "answer"
    assert out.ranked_doc_ids == ["doc0"]
    assert out.trajectory.num_tool_calls == 1
    assert out.trajectory.num_over_budget_calls == 1
    tool_msgs = [m for m in out.trajectory.messages_full if m.role == "tool"]
    assert tool_msgs[0].content.endswith("[calls used: 1/1]")
    assert "tool_budget_exhausted" in tool_msgs[1].content


def test_id_map_answer_translation():
    turns = [
        _completion(tool_calls=[_tool_call("search", {"query": "q", "top_k": 2})]),
        _completion(content="<answer>1, 0, 5</answer>"),
    ]
    h = _harness(turns, _docs(2), use_id_map=True)
    out = h.run(AgentInput(messages=[Message(role="user", content="q?")]))
    assert out.ranked_doc_ids == ["doc1", "doc0"]
    tool_msgs = [m for m in out.trajectory.messages_full if m.role == "tool"]
    assert [e["doc_id"] for e in json.loads(tool_msgs[0].content)] == [0, 1]
