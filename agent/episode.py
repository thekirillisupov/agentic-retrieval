"""Shared per-episode core of the ReAct retrieval loop.

``agent/harness.py`` (eval / inference over the OpenAI API) and
``grpo/agent_loop.py`` (veRL training rollout over raw token ids) execute the
*same* episode: identical tool semantics, seen/citability bookkeeping, doc_id
serialization, tool-response budgets, and tool-call budget feedback. They used
to each carry a hand-synced copy of that logic and drifted — the harness
rendered tool responses without the training-time response budget and marked
budget-truncated grep hits citable, so at inference the policy saw contexts it
was never trained on.

This module is now the single source of truth for everything the *model
observes* inside an episode and for which doc_ids become citable. The two
loops keep only their transport (sync OpenAI chat vs async token-id
generation, HTTP client, retries / circuit breaker) and token bookkeeping
local. Any change to episode semantics belongs here, so train and eval cannot
diverge again.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agent.prompts import (
    apply_tool_response_truncation,
    fit_grep_results_to_budget,
    fit_search_display_to_budget,
    render_grep_results,
    render_neighbours_results,
    render_search_results_json,
)
from agent.schemas import SeenPassage, assign_serial_id

log = logging.getLogger(__name__)

# `search` and `local_search` are the same tool; prompt profiles advertise one
# of the two names.
SEARCH_TOOL_NAMES = frozenset({"local_search", "search"})
KNOWN_TOOL_NAMES = SEARCH_TOOL_NAMES | {"grep", "get_neighbours"}

_OVER_BUDGET_STUB = json.dumps(
    {
        "error": "tool_budget_exhausted",
        "instruction": (
            "You have used all tool calls. Do not "
            "call any more tools. Return your final "
            "<answer> now."
        ),
    }
)


@dataclass(frozen=True)
class EpisodeConfig:
    """Knobs that shape what the model observes during an episode.

    Training (``grpo/agent_loop.py``) and eval/inference
    (``agent/harness.py``) each build this from their own config, but the
    *values* must match between a training run and any eval/serving of the
    resulting weights — the policy only ever saw episodes shaped by them.
    """

    max_tool_calls: int = 10
    top_k_default: int = 10
    top_k_max: int = 50
    # Surface the tool-call budget to the policy: suffix every tool response
    # with `[calls used: N/max_tool_calls]` and, once the budget is spent,
    # grant one final turn (with an explicit nudge) to emit the <answer>
    # instead of cutting the episode off.
    tool_budget_feedback: bool = False
    # Replace doc_ids with small per-episode integers (0,1,2… by first
    # appearance) in every tool output; the model answers with those ints and
    # they are translated back to doc_ids for scoring.
    use_id_map: bool = False
    # Whether a model-emitted `source` tool arg may override the pinned
    # per-episode corpus. False = the pinned corpus always wins (one question
    # == one base), matching how the model is trained; `source` is currently
    # not exposed in any tool schema anyway.
    allow_model_source: bool = False
    # Total-response character budget: whole hits are dropped from a tool
    # response until its render fits (see fit_*_to_budget). Docs whose full
    # passage was truncated out do NOT become citable. 0 = unlimited.
    max_tool_response_length: int = 0
    tool_response_truncate_side: str = "left"
    # Per-passage token cap applied to each hit's `text` BEFORE the
    # total-response budget, so one verbose chunk cannot crowd out the rest of
    # top_k. Requires a tokenizer; 0 disables.
    max_passage_tokens: int = 0


class Episode:
    """Mutable state + shared semantics for one ReAct episode/rollout.

    Owns the `seen` registry (which doc_ids are citable, with metadata), the
    doc_id<->serial-int maps, tool-call budget counters, and the full
    tool-response pipeline: per-passage truncation → dedup/serialization →
    response-budget fitting → seen/trajectory bookkeeping → summary.

    The caller (harness or agent_loop) parses tool arguments with the helpers
    here, performs the actual tool-server request itself, then feeds the raw
    response into the matching ``process_*`` method.
    """

    def __init__(self, cfg: EpisodeConfig, tokenizer: Any = None) -> None:
        self.cfg = cfg
        # Used only for max_passage_tokens; None disables that cap.
        self.tokenizer = tokenizer
        # doc_id -> metadata for every document fully shown to the model.
        # Membership == citable in the final <answer>.
        self.seen: dict[str, SeenPassage] = {}
        # id_map_inv[i] == real doc_id for serial int i; id_map is the reverse.
        # Populated lazily; only consulted when cfg.use_id_map.
        self.id_map: dict[str, int] = {}
        self.id_map_inv: list[str] = []
        # Every doc_id shown in a tool response, in encounter order, deduped
        # (training diagnostics: trajectory recall).
        self.trajectory_doc_ids: list[str] = []
        self._trajectory_doc_ids_set: set[str] = set()
        self.num_tool_calls = 0
        # Tool calls emitted after the budget was already spent (each one gets
        # a stub nudge). High values mean the model ignores the nudge.
        self.num_over_budget_calls = 0
        # Ensures the post-budget "final answer" turn is granted exactly once,
        # so a model that keeps calling tools can't spin.
        self._final_answer_chance_used = False

    # ------------------------------------------------------------ budget

    @property
    def budget_spent(self) -> bool:
        return self.num_tool_calls >= self.cfg.max_tool_calls

    def over_budget_stub(self) -> str:
        """Tool-response content for a call emitted after the budget was spent.

        Only meaningful in tool_budget_feedback mode: every emitted tool_call
        must still get a tool response or the chat template (and downstream
        trace) is malformed, so over-budget calls are answered with an explicit
        stub that nudges the model to emit its <answer> on the next turn.
        """
        self.num_over_budget_calls += 1
        return (
            _OVER_BUDGET_STUB
            + f"\n[calls used: {self.num_tool_calls}/{self.cfg.max_tool_calls}]"
        )

    def register_tool_call(self, rendered: str) -> str:
        """Count an executed tool call; in feedback mode, suffix its response
        with the running `[calls used: N/max]` marker."""
        self.num_tool_calls += 1
        if self.cfg.tool_budget_feedback:
            rendered = (
                f"{rendered}\n[calls used: "
                f"{self.num_tool_calls}/{self.cfg.max_tool_calls}]"
            )
        return rendered

    def grant_final_answer_turn(self, budget_exhausted: bool) -> bool:
        """One-shot post-budget "final answer" turn (feedback mode only).

        The over-budget stubs already tell the model it is out of tool calls,
        so the caller's loop simply continues into one more assistant turn.
        """
        grant = (
            budget_exhausted
            and self.cfg.tool_budget_feedback
            and not self._final_answer_chance_used
        )
        if grant:
            self._final_answer_chance_used = True
        return grant

    # ------------------------------------------------------- argument parsing

    def resolve_source(
        self, args: dict[str, Any], pinned_source: str | None
    ) -> str | None:
        if self.cfg.allow_model_source:
            return str(args.get("source") or "").strip() or pinned_source
        return pinned_source

    def parse_top_k(self, args: dict[str, Any]) -> int:
        try:
            top_k = int(args.get("top_k", self.cfg.top_k_default))
        except (TypeError, ValueError):
            top_k = self.cfg.top_k_default
        return max(1, min(top_k, self.cfg.top_k_max))

    def search_fetch_k(self, top_k: int) -> int:
        """Over-fetch so that after deduplication against `seen` the response
        still shows ~top_k NEW docs."""
        return min(top_k + len(self.seen), self.cfg.top_k_max)

    @staticmethod
    def parse_window(args: dict[str, Any]) -> int:
        try:
            window = int(args.get("window", 1))
        except (TypeError, ValueError):
            window = 1
        return max(1, min(window, 10))

    def decode_doc_id_arg(self, raw: str) -> tuple[str, str]:
        """Resolve a model-passed doc_id argument (get_neighbours).

        Returns ``(real_doc_id, anchor_display)``: the doc_id to send to the
        tool server and the string to echo back in error/no-op messages. When
        id_map is active the model passes a serial integer; decode it, but keep
        the original string for messages shown to the model.
        """
        if not self.cfg.use_id_map:
            return raw, raw
        try:
            idx = int(raw)
            real = self.id_map_inv[idx] if 0 <= idx < len(self.id_map_inv) else raw
        except ValueError:
            real = raw
        return real, raw

    # ----------------------------------------------------- error/stub replies

    @staticmethod
    def unknown_tool_response(name: str) -> tuple[str, dict[str, Any]]:
        return json.dumps({"error": f"unknown tool: {name}"}), {
            "error": "unknown_tool"
        }

    @staticmethod
    def tool_error_response(exc: Exception) -> tuple[str, dict[str, Any]]:
        return json.dumps({"error": str(exc), "results": []}), {"error": str(exc)}

    # ------------------------------------------------------- result pipelines

    def process_search_results(
        self, results: list[dict[str, Any]], top_k: int, turn: int
    ) -> tuple[str, dict[str, Any]]:
        """Consume a search response: render + bookkeeping.

        Renders the ranked hits in order, preserving the position of every
        passage but optimising context: NEW docs are shown as full passages,
        already-seen docs as id-only placeholders ({"doc_id": ..., "seen":
        true}). The caller over-fetched (see search_fetch_k); we walk in rank
        order until top_k NEW docs have been shown, then stop (any trailing
        seen ids below the last new doc are dropped). The display is then
        fitted to the response budget, and ONLY new docs whose full passage
        survived fitting become citable (added to `seen`). Already-seen docs
        still get their score metadata refreshed.
        """
        results = self._truncate_passage_texts(results)
        display: list[dict[str, Any]] = []
        new_doc_ids: list[str] = []
        pending: dict[str, dict[str, Any]] = {}
        scored: list[dict[str, Any]] = []
        for r in results:
            did = r["doc_id"]
            if did in self.seen:
                self.seen[did].update(float(r.get("score", 0.0)))
                display.append({"doc_id": self._display_id(did)})
            else:
                new_doc_ids.append(did)
                pending[did] = r
                if "score" in r:
                    scored.append({"doc_id": did, "score": float(r["score"])})
                if self.cfg.use_id_map:
                    display.append({**r, "doc_id": self._display_id(did)})
                else:
                    display.append(r)
                if len(new_doc_ids) >= top_k:
                    break

        if self.cfg.max_tool_response_length > 0:
            fitted, rendered = fit_search_display_to_budget(
                display,
                self.cfg.max_tool_response_length,
                self.cfg.tool_response_truncate_side,
            )
        else:
            fitted, rendered = display, render_search_results_json(display)
        self._record_trajectory_doc_ids(fitted)

        fitted_new = self._fitted_new_doc_ids(new_doc_ids, fitted)
        for did in fitted_new:
            r = pending[did]
            self.seen[did] = SeenPassage(
                doc_id=did,
                title=r.get("title", ""),
                text=r.get("text", ""),
                best_score=float(r.get("score", 0.0)),
                first_seen_turn=turn,
            )
        fitted_new_set = set(fitted_new)
        scored = [s for s in scored if s["doc_id"] in fitted_new_set]

        summary: dict[str, Any] = {
            "num_results": len(fitted),
            "num_new": len(fitted_new),
            "num_seen_refs": len(fitted) - len(fitted_new),
            # Only the newly-shown docs (full passages). Seen placeholders
            # were already reported when first surfaced.
            "top_doc_ids": fitted_new,
            "doc_ids_with_scores": scored,
        }
        return rendered, summary

    def process_grep_results(
        self, results: list[dict[str, Any]], total_matches: int, turn: int
    ) -> tuple[str, dict[str, Any]]:
        """Consume a grep response: render + bookkeeping.

        Fits to the response budget BEFORE marking docs seen: a hit truncated
        out of the response must not become citable.
        """
        results = self._truncate_passage_texts(results)
        if self.cfg.use_id_map:
            display = [
                {**r, "doc_id": self._display_id(r["doc_id"])} for r in results
            ]
        else:
            display = results
        # Map each rendered entry back to its raw result (display may hold
        # serial ints). Identity-keyed so it survives front/back drops.
        raw_by_obj = {id(d): r for d, r in zip(display, results)}

        if self.cfg.max_tool_response_length > 0:
            fitted, rendered = fit_grep_results_to_budget(
                display,
                total_matches,
                self.cfg.max_tool_response_length,
                self.cfg.tool_response_truncate_side,
            )
        else:
            fitted, rendered = display, render_grep_results(display, total_matches)
        self._record_trajectory_doc_ids(fitted)

        fitted_doc_ids: list[str] = []
        for disp in fitted:
            r = raw_by_obj[id(disp)]
            did = r["doc_id"]
            fitted_doc_ids.append(did)
            # Grep hits have no relevance score; 1.0 marks an exact match.
            if did not in self.seen:
                self.seen[did] = SeenPassage(
                    doc_id=did,
                    title=r.get("title", ""),
                    text=r.get("text", ""),
                    best_score=1.0,
                    first_seen_turn=turn,
                )
        summary = {
            "num_results": len(fitted),
            "total_matches": total_matches,
            "top_doc_ids": fitted_doc_ids,
        }
        return rendered, summary

    def process_neighbours_results(
        self,
        results: list[dict[str, Any]],
        status: str,
        anchor_display: str,
        window: int,
        turn: int,
    ) -> tuple[str, dict[str, Any]]:
        """Consume a get_neighbours response: render + bookkeeping.

        Neighbours are structural (no relevance score): every returned chunk
        is registered as citable with a 0.0 score sentinel. The render is
        char-truncated (not hit-dropped) when over budget — windows are small,
        so a whole-hit fit like search/grep has never been needed here.
        """
        results = self._truncate_passage_texts(results)
        if self.cfg.use_id_map:
            display = [
                {**r, "doc_id": self._display_id(r["doc_id"])} for r in results
            ]
        else:
            display = results
        self._record_trajectory_doc_ids(display)
        rendered = render_neighbours_results(display, status, anchor_display, window)
        if (
            self.cfg.max_tool_response_length > 0
            and len(rendered) > self.cfg.max_tool_response_length
        ):
            rendered = apply_tool_response_truncation(
                rendered,
                self.cfg.max_tool_response_length,
                self.cfg.tool_response_truncate_side,
            )
        for r in results:
            did = r["doc_id"]
            if did not in self.seen:
                self.seen[did] = SeenPassage(
                    doc_id=did,
                    title=r.get("title", ""),
                    text=r.get("text", ""),
                    best_score=0.0,
                    first_seen_turn=turn,
                )
        summary = {
            "num_results": len(results),
            "status": status,
            "top_doc_ids": [r["doc_id"] for r in results],
        }
        return rendered, summary

    # ------------------------------------------------------------- the answer

    def validate_answer(self, answer_tokens: list[str]) -> list[str]:
        """Translate the model's <answer> list into real, citable doc_ids.

        Drops hallucinated ids (never fully shown => not in `seen`), unknown
        serial ids, and duplicates — first occurrence wins, so the model
        cannot game NDCG/recall by repeating the same doc_id.
        """
        valid: list[str] = []
        emitted: set[str] = set()
        dropped: list[str] = []
        for token in answer_tokens:
            if self.cfg.use_id_map:
                try:
                    idx = int(token)
                    did = (
                        self.id_map_inv[idx]
                        if 0 <= idx < len(self.id_map_inv)
                        else None
                    )
                except ValueError:
                    did = None
                if did is None:
                    dropped.append(str(token))
                    continue
            else:
                did = token
            if did not in self.seen or did in emitted:
                dropped.append(str(token))
            else:
                valid.append(did)
                emitted.add(did)
        if dropped:
            # One line per answer (not per id): early-training policies can
            # hallucinate many ids per rollout and per-id logging floods logs.
            log.warning(
                "Dropped %d hallucinated/unknown/duplicate id(s) from answer "
                "(kept %d): %s",
                len(dropped),
                len(valid),
                dropped[:10],
            )
        return valid

    # -------------------------------------------------------------- internals

    def _display_id(self, doc_id: str) -> Any:
        """doc_id as shown to the model: the serial int when id_map is on."""
        if not self.cfg.use_id_map:
            return doc_id
        return assign_serial_id(doc_id, self.id_map, self.id_map_inv)

    def _truncate_passage_texts(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Cap each passage's ``text`` field at max_passage_tokens tokens.

        This is independent of ``max_tool_response_length``: that budget drops
        *whole hits* to fit a total-response character budget, so a handful of
        unusually long chunks can crowd out the rest of the requested top_k
        just by being verbose. Truncating per-passage first means every hit
        costs a bounded amount of context, and the total-response budget then
        only needs to trim the *count* of hits, not silently mangle which ones
        survive.

        No-op when the tokenizer is unavailable or max_passage_tokens <= 0.
        """
        max_tokens = self.cfg.max_passage_tokens
        if self.tokenizer is None or max_tokens <= 0:
            return results
        out: list[dict[str, Any]] = []
        for r in results:
            text = r.get("text")
            if not isinstance(text, str) or not text:
                out.append(r)
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                out.append(r)
                continue
            truncated = self.tokenizer.decode(
                ids[:max_tokens], skip_special_tokens=True
            )
            out.append({**r, "text": truncated + " ...(truncated)"})
        return out

    def _fitted_new_doc_ids(
        self, candidate_new_ids: list[str], fitted: list[dict[str, Any]]
    ) -> list[str]:
        """Keep only new doc_ids whose full passage survived budget fitting."""
        fitted_ids = {r["doc_id"] for r in fitted if "text" in r}
        out: list[str] = []
        for did in candidate_new_ids:
            key: Any = (
                self.id_map[did]
                if self.cfg.use_id_map and did in self.id_map
                else did
            )
            if key in fitted_ids:
                out.append(did)
        return out

    def _record_trajectory_doc_ids(self, display: list[dict[str, Any]]) -> None:
        """Record doc_ids shown in a tool response, encounter order, deduped."""
        for item in display:
            raw = item.get("doc_id")
            if raw is None:
                continue
            if isinstance(raw, int):
                if not (0 <= raw < len(self.id_map_inv)):
                    continue
                did = self.id_map_inv[raw]
            else:
                did = str(raw)
                if not did:
                    continue
            if did not in self._trajectory_doc_ids_set:
                self.trajectory_doc_ids.append(did)
                self._trajectory_doc_ids_set.add(did)


def tool_budget_kwargs_from_cfg(
    agent_cfg: dict[str, Any], model_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Read the tool-response budget knobs shared by all harness entry points.

    Returns kwargs for ``AgentHarness`` (max_tool_response_length,
    tool_response_truncate_side, max_passage_tokens, tokenizer). The values
    must match the training run's ``rollout.multi_turn`` /
    ``agent_loop.retrieval_react`` settings. ``max_passage_tokens > 0``
    requires ``model.tokenizer_path`` (HF name or local dir) since the harness
    talks to the model over an API and has no tokenizer of its own.
    """
    max_passage_tokens = int(agent_cfg.get("max_passage_tokens", 0) or 0)
    tokenizer = None
    if max_passage_tokens > 0:
        tokenizer_path = model_cfg.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError(
                "agent.max_passage_tokens > 0 requires model.tokenizer_path "
                "(HF name or local dir) so the harness can tokenize passages"
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return {
        "max_tool_response_length": int(
            agent_cfg.get("max_tool_response_length", 0) or 0
        ),
        "tool_response_truncate_side": str(
            agent_cfg.get("tool_response_truncate_side", "left") or "left"
        ),
        "max_passage_tokens": max_passage_tokens,
        "tokenizer": tokenizer,
    }
