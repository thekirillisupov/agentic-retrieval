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

Single-turn mode (default): each sample is [system, user(query)].  Covers the
system-prompt + tool-schema + query activation distribution; sufficient for
weight-only quant.

Trajectory mode (higher fidelity): each sample is a full rollout conversation,
including tool-call and tool-result turns.  Activates the MoE experts that are
only visited during multi-turn reasoning; important for W8A8.  Use
``load_messages_from_trajectory_parquet`` (grpo_train.parquet) or
``load_messages_from_trajectory_jsonl`` (trajectories_data/*.jsonl) + then
``build_calibration_from_messages`` instead of the single-turn helpers.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from agent.prompts import format_user_content, get_prompt, get_tool_schemas

_CLIENT_TAG_RE = re.compile(r"^<client>(.*)</client>$", re.DOTALL)
_USER_TAG_RE = re.compile(r"^<user>(.*)</user>$", re.DOTALL)


def extract_question_from_prompt(prompt_messages: Any) -> str:
    """Pull the raw question text out of a stored prompt messages array."""
    for msg in prompt_messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = str(msg.get("content", ""))
        else:
            role = str(getattr(msg, "role", ""))
            content = str(getattr(msg, "content", ""))
        if role != "user":
            continue
        stripped = content.strip()
        for pat in (_CLIENT_TAG_RE, _USER_TAG_RE):
            m = pat.match(stripped)
            if m:
                return m.group(1).strip()
        return stripped
    return ""


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
        # Newer transformers may return a BatchEncoding (dict-like) instead of
        # a plain list of token ids. Extract the ids in that case.
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
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


# ---------------------------------------------------------------------------
# Trajectory-based calibration (higher fidelity: includes tool-result turns)
# ---------------------------------------------------------------------------

def load_messages_from_trajectory_parquet(
    path: str | Path,
    *,
    prompt_field: str = "prompt",
    default_prompt_version: str = "v2_search_only",
    max_rows: int | None = None,
) -> list[tuple[list[dict[str, str]], str]]:
    """Load pre-formatted message arrays from a trajectory-format parquet.

    Reads the ``prompt`` column of ``grpo_train.parquet`` (or any parquet that
    stores lists of ``{role, content}`` dicts in that column).  Returns a list
    of ``(messages, prompt_version)`` tuples so the caller can attach the right
    tool schema when tokenising.

    The ``prompt_version`` is read from ``extra_info['prompt_version']`` when
    present; ``default_prompt_version`` is used as a fallback.
    """
    import pandas as pd

    path = Path(path)
    df = pd.read_parquet(path)
    if prompt_field not in df.columns:
        raise ValueError(
            f"{path} has no '{prompt_field}' column; columns={list(df.columns)}"
        )

    has_extra = "extra_info" in df.columns
    out: list[tuple[list[dict[str, str]], str]] = []
    for i, row in df.iterrows():
        raw_msgs = row[prompt_field]
        msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in raw_msgs]
        if not msgs:
            continue

        pv = default_prompt_version
        if has_extra:
            ei = row["extra_info"]
            if isinstance(ei, dict):
                pv = str(ei.get("prompt_version", default_prompt_version))

        out.append((msgs, pv))
        if max_rows and len(out) >= max_rows:
            break

    print(
        f"[calibration] loaded {len(out)} trajectory prompts from {path.name} "
        f"(prompt_field='{prompt_field}')"
    )
    return out


def load_messages_from_trajectory_jsonl(
    path: str | Path,
    *,
    messages_field: str = "messages_full",
    default_prompt_version: str = "v2_search_only",
    max_rows: int | None = None,
    seed: int = 0,
) -> list[tuple[list[dict[str, str]], str]]:
    """Load full multi-turn trajectories from a trajectory JSONL file.

    Each line must be a JSON object with a ``messages_full`` field (list of
    ``{role, content}`` dicts covering the complete rollout: system, user,
    assistant tool-calls, tool results, final assistant answer).  This is the
    format produced by the GRPO trainer's ``val_trajectory_dir`` checkpoints
    (e.g. ``trajectories_data/gspo_qwen3_moe/65.jsonl``).

    When ``max_rows`` is set and the file has more non-empty lines, a
    deterministic random subset (``seed``) is drawn without replacement and
    only those lines are parsed.

    Returns ``(messages, prompt_version)`` tuples.  ``prompt_version`` is read
    from the row's top-level ``prompt_version`` key when present.
    """
    path = Path(path)

    def _parse_row(row: dict) -> tuple[list[dict[str, str]], str] | None:
        msgs_raw = row.get(messages_field) or []
        if not msgs_raw:
            return None
        msgs = [
            {"role": str(m["role"]), "content": str(m.get("content") or "")}
            for m in msgs_raw
        ]
        pv = str(row.get("prompt_version", default_prompt_version))
        return msgs, pv

    non_empty_line_indices: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                non_empty_line_indices.append(i)

    picked_indices = non_empty_line_indices
    if max_rows is not None and len(non_empty_line_indices) > max_rows:
        picked_indices = random.Random(seed).sample(non_empty_line_indices, max_rows)
        print(
            f"[calibration] randomly sampled {len(picked_indices)}/"
            f"{len(non_empty_line_indices)} lines from {path.name} (seed={seed})"
        )

    wanted = set(picked_indices)
    lines_by_index: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted:
                lines_by_index[i] = line
            if len(lines_by_index) == len(wanted):
                break

    out: list[tuple[list[dict[str, str]], str]] = []
    skipped = 0
    for line_no in picked_indices:
        line = lines_by_index[line_no].strip()
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[calibration] warning: skip line {line_no + 1}: {exc}")
            skipped += 1
            continue

        parsed = _parse_row(row)
        if parsed is None:
            skipped += 1
            continue
        out.append(parsed)

    print(
        f"[calibration] loaded {len(out)} trajectories from {path.name} "
        f"(skipped={skipped}, field='{messages_field}')"
    )
    return out


def build_calibration_from_messages(
    trajectory_messages: list[tuple[list[dict[str, str]], str]],
    tokenizer: Any,
    *,
    num_samples: int = 512,
    max_seq_len: int = 4096,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    seed: int = 0,
    shuffle: bool = True,
):
    """Tokenise pre-built message arrays into a HF Dataset for llmcompressor.

    Unlike ``build_calibration_dataset`` (which re-renders from raw questions),
    this function accepts already-formatted conversations so you can use:

    * Single-turn prompts verbatim from ``grpo_train.parquet`` (same quality as
      the question-based path, but sourced directly from training data).
    * Full multi-turn rollouts from trajectory JSONL (tool calls + tool results
      included — highest fidelity for W8A8 calibration).

    ``trajectory_messages`` is a list of ``(messages, prompt_version)`` pairs as
    returned by ``load_messages_from_trajectory_parquet`` or
    ``load_messages_from_trajectory_jsonl``.

    For single-turn prompts sourced from the parquet, pass
    ``add_generation_prompt=True`` to match the inference template.  For full
    trajectories (complete rollout), leave it ``False``.
    """
    from datasets import Dataset

    pool = list(trajectory_messages)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pool)
        if num_samples and len(pool) > num_samples:
            pool = rng.sample(pool, num_samples)
            print(
                f"[calibration] randomly sampled {len(pool)}/{len(trajectory_messages)} "
                f"trajectories (seed={seed})"
            )
    elif num_samples and len(pool) > num_samples:
        pool = pool[:num_samples]

    rows: list[dict[str, list[int]]] = []
    skipped = 0
    pv_counts: dict[str, int] = {}
    for msgs, pv in pool:
        tools = get_tool_schemas(pv)
        pv_counts[pv] = pv_counts.get(pv, 0) + 1
        try:
            ids = tokenizer.apply_chat_template(
                msgs,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            print(f"[calibration] warning: skipping sample ({exc})")
            skipped += 1
            continue

        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        if not ids:
            skipped += 1
            continue
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})

    if not rows:
        raise ValueError(
            "no calibration samples produced — check trajectory file and field names"
        )
    if skipped:
        print(f"[calibration] skipped {skipped} samples")
    print(
        f"[calibration] {len(rows)} trajectory samples | max_seq_len={max_seq_len} "
        f"| add_generation_prompt={add_generation_prompt} | pv_counts={pv_counts}"
    )
    return Dataset.from_list(rows)
