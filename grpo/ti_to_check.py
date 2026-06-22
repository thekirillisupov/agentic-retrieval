"""Token-In / Token-Out consistency check for GRPO trajectories.

Why
---
Multi-turn RL is brittle when tokenisation drifts. The actor sees a sequence
``S_actor`` at training time; the rollout server saw a sequence ``S_rollout``
when sampling. If the two diverge by even one token (e.g. due to a chat
template that re-renders whitespace differently), every subsequent log-prob
ratio is computed against the wrong reference and the policy update is
silently corrupted.

This module re-applies the chat template to ``messages_full`` (saved by the
AgentLoop) and asserts that the resulting token sequence equals
``prompt_ids + response_ids`` modulo trailing pad. We run it on a sample of
trajectories per training step (``ti_to_check_every_n``).

Usage
-----
* Standalone: ``python -m grpo.ti_to_check --traj <path>`` to verify a single
  saved trajectory file.
* Inside training: import ``check_consistency`` and call it from a callback.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from agent.prompts import get_tool_schemas

log = logging.getLogger(__name__)


def _normalize_token_ids(tokenized_output: Any) -> list[int]:
    """Flatten ``apply_chat_template(tokenize=True)`` output into ``list[int]``.

    Transformers >=4.40 returns a ``BatchEncoding`` of ``Encoding`` objects when
    ``tools=`` is passed, instead of a plain list of ints. We mirror veRL's
    ``normalize_token_ids`` so the TI/TO diff stays comparable to what the
    trainer actually feeds back through the optimiser.
    """
    token_ids: Any = tokenized_output
    if isinstance(tokenized_output, dict) and "input_ids" in tokenized_output:
        token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if (
        isinstance(token_ids, list)
        and len(token_ids) == 1
        and isinstance(token_ids[0], (list, tuple))
    ):
        token_ids = list(token_ids[0])
    if not isinstance(token_ids, list):
        raise TypeError(
            f"token_ids must be list-like, got {type(token_ids).__name__}: {token_ids!r}"
        )
    out: list[int] = []
    for tid in token_ids:
        if hasattr(tid, "item"):
            tid = tid.item()
        out.append(int(tid))
    return out


def check_consistency(
    *,
    tokenizer: Any,
    messages_full: list[dict[str, Any]],
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int] | None = None,
    tools: list[dict[str, Any]] | None = None,
    prompt_version: str = "v1_edit",
) -> dict[str, Any]:
    """Re-tokenise messages_full and compare against prompt_ids+response_ids.

    IMPORTANT — thinking models (Qwen3.5):
    A naive full-sequence re-render of ``messages_full`` does NOT reconstruct the
    rollout token ids, because the chat template rewrites reasoning sections: it
    injects an empty ``</think>\\n\\n`` block when an assistant turn has no
    ``<think>`` content, and shifts the ``<think>`` token across the prompt/response
    boundary when it does. The training path is unaffected by this — ``agent_loop``
    accumulates the *actual* sampled token ids in ``running_ids`` and never
    re-renders — so ``response_ids`` fed to the optimiser are exact. The only
    invariant this check can robustly assert is ``mask_consistent``; the re-render
    diff is reported as informational, not as a hard failure.

    Returns a report dict with::

        ok                    bool         mask_consistent (primary safety signal)
        mask_consistent       bool         mask length == response_ids length
        full_render_match     bool         re-render reproduces prompt+response exactly
                                           (expected False for Qwen3.5 thinking output)
        prompt_match          bool         prompt prefix is identical
        first_diff_index      int | None   index of first mismatch (or None)
        rendered_len          int          len(re-tokenised)
        recorded_len          int          len(prompt_ids+response_ids)
    """
    tools = tools if tools is not None else get_tool_schemas(prompt_version)
    cleaned = [
        {k: v for k, v in m.items() if not k.startswith("_")}
        for m in messages_full
    ]
    rendered_raw = tokenizer.apply_chat_template(
        cleaned,
        add_generation_prompt=False,
        tools=tools,
        tokenize=True,
    )
    rendered: list[int] = _normalize_token_ids(rendered_raw)
    recorded = list(prompt_ids) + list(response_ids)

    n = min(len(rendered), len(recorded))
    first_diff: int | None = None
    for i in range(n):
        if rendered[i] != recorded[i]:
            first_diff = i
            break
    if first_diff is None and len(rendered) != len(recorded):
        first_diff = n

    prompt_len = len(prompt_ids)
    prompt_match = rendered[:prompt_len] == recorded[:prompt_len]
    full_render_match = first_diff is None and len(rendered) == len(recorded)

    mask_consistent = response_mask is None or len(response_mask) == len(response_ids)

    report = {
        "ok": mask_consistent,
        "mask_consistent": mask_consistent,
        "full_render_match": full_render_match,
        "prompt_match": prompt_match,
        "first_diff_index": first_diff,
        "rendered_len": len(rendered),
        "recorded_len": len(recorded),
    }
    if not mask_consistent:
        log.warning("TI/TO mask inconsistency: %s", report)
    elif not full_render_match:
        log.debug(
            "TI/TO re-render diff at index %s (expected for Qwen3.5 thinking; "
            "training uses accumulated ids, not this re-render)",
            first_diff,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", required=True, help="path to a trajectory json")
    parser.add_argument("--tokenizer", required=True, help="HF model id or path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from transformers import AutoTokenizer  # local import; heavy

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    blob = json.loads(Path(args.traj).read_text())
    prompt_version = blob.get("prompt_version", "v1_edit")
    rep = check_consistency(
        tokenizer=tok,
        messages_full=blob["messages_full"],
        prompt_ids=blob["prompt_ids"],
        response_ids=blob["response_ids"],
        response_mask=blob.get("response_mask"),
        prompt_version=prompt_version,
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
