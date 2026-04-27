"""TI/TO consistency check.

For RL-style training over our trajectories we need to be able to reconstruct
the exact token sequence the model saw at training time. This checker is a
placeholder hook — on MVP it logs a warning if `messages_full` is empty or
missing the system message, which would break re-tokenization. The real
template-roundtrip check requires a tokenizer + chat template, which is gated
behind the model being available locally.

Plug in the full check (apply_chat_template → re-tokenize → compare) as soon as
you start writing trajectories you intend to fine-tune from.
"""

from __future__ import annotations

import logging

from agent.schemas import Trajectory

log = logging.getLogger(__name__)


def maybe_check_consistency(t: Trajectory) -> bool:
    """Return True if the trajectory passes basic structural checks."""
    if not t.messages_full:
        log.warning("trajectory %s: empty messages_full", t.trajectory_id)
        return False
    if t.messages_full[0].role != "system":
        log.warning(
            "trajectory %s: first message is not system (got %s)",
            t.trajectory_id,
            t.messages_full[0].role,
        )
        return False

    # tool_call_id integrity: every tool message references a tool_call from a
    # preceding assistant message.
    open_tool_calls: set[str] = set()
    for m in t.messages_full:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    open_tool_calls.add(tc_id)
        elif m.role == "tool":
            if m.tool_call_id and m.tool_call_id not in open_tool_calls:
                log.warning(
                    "trajectory %s: tool message references unknown tool_call_id %s",
                    t.trajectory_id,
                    m.tool_call_id,
                )
                return False

    return True
