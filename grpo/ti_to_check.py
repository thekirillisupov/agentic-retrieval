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

from agent.prompts import LOCAL_SEARCH_TOOL_SCHEMA

log = logging.getLogger(__name__)


def check_consistency(
    *,
    tokenizer: Any,
    messages_full: list[dict[str, Any]],
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-tokenise messages_full and compare against prompt_ids+response_ids.

    Returns a report dict with::

        ok                    bool
        prompt_match          bool         prompt prefix is identical
        suffix_match          bool         response suffix is identical
        first_diff_index      int | None   index of first mismatch (or None)
        rendered_len          int          len(re-tokenised)
        recorded_len          int          len(prompt_ids+response_ids)
        mask_consistent       bool         mask length == response_ids length
    """
    tools = tools if tools is not None else [LOCAL_SEARCH_TOOL_SCHEMA]
    cleaned = [
        {k: v for k, v in m.items() if not k.startswith("_")}
        for m in messages_full
    ]
    rendered: list[int] = tokenizer.apply_chat_template(
        cleaned,
        add_generation_prompt=False,
        tools=tools,
        tokenize=True,
    )
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
    suffix_match = (
        len(rendered) >= prompt_len
        and rendered[prompt_len:] == recorded[prompt_len:]
    )

    mask_consistent = response_mask is None or len(response_mask) == len(response_ids)

    report = {
        "ok": first_diff is None and mask_consistent,
        "prompt_match": prompt_match,
        "suffix_match": suffix_match,
        "first_diff_index": first_diff,
        "rendered_len": len(rendered),
        "recorded_len": len(recorded),
        "mask_consistent": mask_consistent,
    }
    if not report["ok"]:
        log.warning("TI/TO mismatch: %s", report)
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
    rep = check_consistency(
        tokenizer=tok,
        messages_full=blob["messages_full"],
        prompt_ids=blob["prompt_ids"],
        response_ids=blob["response_ids"],
        response_mask=blob.get("response_mask"),
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
