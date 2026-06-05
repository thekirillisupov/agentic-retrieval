"""Convert MuSiQue train (jsonl with gold doc_ids) into a veRL-ready parquet.

Each row of the parquet is one prompt rolled out by the GRPO trainer. veRL
expects the columns below (see verl.utils.dataset.RLHFDataset):

    prompt          chat-format list of dicts, e.g.
                    [{"role": "system", ...}, {"role": "user", ...}]
    data_source     a string used to dispatch reward/score functions.
    ability         label, used for filtering / book-keeping only.
    reward_model    {"style": "...", "ground_truth": <json-encoded gold>}
    extra_info      free-form dict; we put gold_doc_ids and question_id here so
                    the reward function can read them back without parsing.
    agent_name      veRL's AgentLoop registry key. Must match the name registered
                    in agent_loop.py.

We intentionally do NOT pack the tool schema or the system prompt into this file:
the AgentLoop owns those (single source of truth = `agent.prompts`). That keeps
the parquet stable across prompt-version bumps — re-rendering the system prompt
does not require regenerating the data.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from agent.prompts import get_prompt

log = logging.getLogger(__name__)


DEFAULT_DATA_SOURCE = "musique_retrieval"
ABILITY = "agentic_retrieval"
AGENT_NAME = "retrieval_react"


def _row(
    record: dict[str, Any],
    prompt_version: str,
    *,
    data_source: str = DEFAULT_DATA_SOURCE,
) -> dict[str, Any]:
    system_prompt = get_prompt(prompt_version)
    gold = list(record["gold_doc_ids"])
    return {
        "data_source": data_source,
        "ability": ABILITY,
        "agent_name": AGENT_NAME,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["question"]},
        ],
        "reward_model": {
            "style": "ndcg_at_k",
            "ground_truth": json.dumps(gold, ensure_ascii=False),
        },
        "extra_info": {
            "question_id": record["question_id"],
            "gold_doc_ids": gold,
            "prompt_version": prompt_version,
            "answer": record.get("answer", ""),
        },
    }


def build(
    in_path: Path,
    out_train: Path,
    out_val: Path | None,
    *,
    prompt_version: str = "v1",
    data_source: str = DEFAULT_DATA_SOURCE,
    val_size: int = 200,
    val_frac: float | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> tuple[int, int]:
    rows: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_row(json.loads(line), prompt_version, data_source=data_source))
            if limit is not None and len(rows) >= limit:
                break

    rng = random.Random(seed)
    rng.shuffle(rows)

    if val_frac is not None:
        if not 0.0 < val_frac < 1.0:
            raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
        val_size = max(1, round(len(rows) * val_frac))

    if out_val is not None and val_size > 0:
        val = rows[:val_size]
        train = rows[val_size:]
    else:
        val = []
        train = rows

    out_train.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(out_train, index=False)
    if out_val is not None and val:
        out_val.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(val).to_parquet(out_val, index=False)

    log.info(
        "wrote %d train rows to %s, %d val rows to %s",
        len(train), out_train, len(val), out_val,
    )
    return len(train), len(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True,
                        help="jsonl produced by eval_.build_train")
    parser.add_argument("--out-train", default="data/processed/musique/grpo_train.parquet")
    parser.add_argument("--out-val", default="data/processed/musique/grpo_val.parquet")
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="If set, overrides --val-size with round(n * val_frac) validation rows.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build(
        Path(args.in_path),
        Path(args.out_train),
        Path(args.out_val) if args.out_val else None,
        prompt_version=args.prompt_version,
        data_source=args.data_source,
        val_size=args.val_size,
        val_frac=args.val_frac,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
