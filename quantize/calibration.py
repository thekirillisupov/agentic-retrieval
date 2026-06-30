"""Build a calibration dataset for weight-only INT8 (W8A16) quantization.

Calibration prompts MUST match the agentic format the model was trained and
served with, otherwise GPTQ minimises weight error on a distribution the model
never sees at inference. We render each generated question through the SAME path
the rollout/harness uses (agent/prompts.py):

    * the versioned system prompt          -> get_prompt(version)
    * the <client>-wrapped user turn        -> format_user_content(question, version)
    * the tool schemas for that version     -> get_tool_schemas(version)

and tokenise with the model's chat template using the exact kwargs the serving
stack uses (add_generation_prompt=True, enable_thinking=False — see
scripts/serve_vllm.sh and configs/gspo_qwen3_moe.yaml:data.apply_chat_template_kwargs).

The questions chosen by the user are single-turn (system + one user query). That
covers the system-prompt + tool-schema + query activation distribution, which is
what weight-only quantization needs; it does NOT include tool-result turns (use
recorded trajectories for that — see README).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from agent.prompts import format_user_content, get_prompt, get_tool_schemas


def load_questions(path: str | Path, field: str = "question") -> list[str]:
    """Load raw question strings from a .jsonl / .json / .txt / .parquet file.

    - .jsonl : one JSON object per line; ``field`` is pulled from each (falls back
               to a bare string line if it is not JSON).
    - .json  : a list of strings, or a list of objects carrying ``field``.
    - .txt   : one question per non-empty line.
    - .parquet: a column named ``field`` (or a ``question`` column).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        col = field if field in df.columns else "question"
        if col not in df.columns:
            raise ValueError(
                f"{path} has no '{field}' or 'question' column; columns={list(df.columns)}"
            )
        return [str(q) for q in df[col].tolist() if str(q).strip()]

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        out: list[str] = []
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(str(item[field]))
            else:
                raise ValueError(f"unexpected item type in {path}: {type(item)}")
        return [q for q in out if q.strip()]

    # .jsonl / .txt (and anything else line-oriented)
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if suffix == ".jsonl":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                if isinstance(obj, str):
                    out.append(obj)
                elif isinstance(obj, dict):
                    out.append(str(obj.get(field, obj.get("question", ""))))
                else:
                    raise ValueError(f"unexpected JSONL row in {path}: {type(obj)}")
            else:
                out.append(line)
    return [q for q in out if q.strip()]


def build_messages(question: str, prompt_version: str) -> list[dict[str, str]]:
    """Render one question into the trained [system, user] chat format."""
    return [
        {"role": "system", "content": get_prompt(prompt_version)},
        {"role": "user", "content": format_user_content(question, prompt_version)},
    ]


def build_calibration_dataset(
    questions: list[str],
    tokenizer: Any,
    *,
    prompt_version: str = "v2_search_only",
    num_samples: int = 512,
    max_seq_len: int = 2048,
    enable_thinking: bool = False,
    seed: int = 0,
):
    """Tokenise rendered agentic prompts into a HF Dataset of input_ids.

    Mirrors the harness/rollout call exactly: apply_chat_template(messages,
    tools=get_tool_schemas(version), add_generation_prompt=True,
    enable_thinking=False). Returns a datasets.Dataset with ``input_ids`` and
    ``attention_mask`` columns, ready to hand to llmcompressor.oneshot.
    """
    from datasets import Dataset

    tools = get_tool_schemas(prompt_version)

    rng = random.Random(seed)
    pool = list(questions)
    rng.shuffle(pool)
    if num_samples and len(pool) > num_samples:
        pool = pool[:num_samples]

    rows: list[dict[str, list[int]]] = []
    skipped = 0
    for q in pool:
        ids = tokenizer.apply_chat_template(
            build_messages(q, prompt_version),
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        if not ids:
            skipped += 1
            continue
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})

    if not rows:
        raise ValueError(
            "no calibration samples produced — check the questions file and field name"
        )
    if skipped:
        print(f"[calibration] skipped {skipped} empty renderings")
    print(
        f"[calibration] {len(rows)} samples | prompt_version={prompt_version} "
        f"| max_seq_len={max_seq_len} | tools={[t['function']['name'] for t in tools]}"
    )
    return Dataset.from_list(rows)
