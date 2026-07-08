"""Trajectory writer: one JSON file per rollout.

Schema is the one specified in the technical doc. We keep `messages_full` exactly
as the harness saw it (system + user + assistant + tool messages, in order). That
makes the file directly consumable by veRL/Search-R1-style RL pipelines later
without a re-derivation step.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from agent.schemas import Trajectory

log = logging.getLogger(__name__)


def trajectory_to_dict(t: Trajectory) -> dict[str, Any]:
    return {
        "trajectory_id": t.trajectory_id,
        "timestamp": t.timestamp,
        "model": t.model,
        "prompt_version": t.prompt_version,
        "input": {
            "messages": [m.to_openai() for m in t.input_messages],
            "max_turns": t.max_turns,
            "max_tool_calls": t.max_tool_calls,
        },
        "messages_full": [m.to_openai() for m in t.messages_full],
        "tool_calls": [dataclasses.asdict(tc) for tc in t.tool_calls],
        "output": {
            "ranked_doc_ids": t.ranked_doc_ids,
            "gold_doc_ids": t.gold_doc_ids,
            "stopped_reason": t.stopped_reason,
            "num_turns": t.num_turns,
            "num_tool_calls": t.num_tool_calls,
        },
        "tokens": {
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.completion_tokens,
            "total_tokens": t.total_tokens,
        },
        "latency": {
            "llm_latencies_ms": t.llm_latencies_ms,
            "tool_latencies_ms": [tc.latency_ms for tc in t.tool_calls],
        },
    }


class TrajectoryWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _subfolder(self, trajectory: Trajectory) -> Path:
        date = trajectory.timestamp[:10] if trajectory.timestamp else "unknown"
        version = trajectory.prompt_version or "unknown"
        sub = self.out_dir / date / version
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def write(self, trajectory: Trajectory) -> Path:
        sub = self._subfolder(trajectory)
        path = sub / f"{trajectory.trajectory_id}.json"
        path.write_text(
            json.dumps(trajectory_to_dict(trajectory), ensure_ascii=False, indent=2)
        )
        return path
