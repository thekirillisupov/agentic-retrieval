# GRPO training for the agentic retrieval ReAct loop

End-to-end Group-Relative Policy Optimisation over the multi-turn ReAct rollout.
The reward is **NDCG@10 against gold supporting passages** with a one-sided
length penalty layered on top. Framework is veRL (`verl.trainer.main_ppo`)
driven by a custom `AgentLoopBase` subclass; see the technical spec for the
design rationale.

## What lives where

| Path | Role |
|---|---|
| `eval_/build_train.py` | MuSiQue train → `musique_train.jsonl` with `gold_doc_ids`. |
| `grpo/data_prep.py` | jsonl → veRL parquet (one row per question). |
| `grpo/agent_loop.py` | `RetrievalReActAgentLoop` (registered as `retrieval_react`). Drives the ReAct rollout against veRL's internal vLLM rollout server, tracks `response_mask` (1 = policy, 0 = template/tool framing) by re-templating after every tool turn. |
| `grpo/reward.py` | `compute_score(...)` → NDCG@10 + length penalty + parse-error penalty. Reuses `eval_.metrics.ndcg_at_k` so train- and eval-time numbers are directly comparable. |
| `grpo/ti_to_check.py` | Token-In / Token-Out consistency: re-templates `messages_full` and diffs against `prompt_ids + response_ids`. |
| `configs/grpo_qwen3_14b.yaml` | All training knobs (single source of truth). |
| `configs/agent_loop_registry.yaml` | Registers `retrieval_react` for the trainer. |
| `scripts/build_grpo_data.sh` | parse → `build_train` → `data_prep`. |
| `scripts/train_grpo.sh` | `python -m verl.trainer.main_ppo --config-path configs --config-name grpo_qwen3_14b`. |

## Pipeline

```
data/raw/musique/musique_ans_v1.0_train.jsonl
        │  (indexing.parse_musique)
        ▼
data/processed/musique/{corpus.jsonl, source_to_doc_id.json}
        │  (eval_.build_train)
        ▼
data/processed/musique/musique_train.jsonl   ← gold_doc_ids per question
        │  (grpo.data_prep)
        ▼
data/processed/musique/grpo_{train,val}.parquet
        │  (verl.trainer.main_ppo + RetrievalReActAgentLoop + grpo.reward.compute_score)
        ▼
checkpoints/grpo_qwen3_14b/
```

## Quickstart

```bash
# 0. Install veRL extras (pulls in ray, hydra, vllm, ...)
pip install -e .[verl]

# 1. Build parquets
#    SKIP_PARSE=1 if data/processed/musique/{corpus.jsonl,source_to_doc_id.json} exist
SKIP_PARSE=1 bash scripts/build_grpo_data.sh

# 2. Tool server (FAISS + E5 embedder; the rollout hits this for /local_search)
GPUS=3 CONFIG=configs/musique.yaml bash scripts/serve_tool.sh

# 3. Train. Internal vLLM rollout server is spun up by veRL — do NOT also run scripts/serve_vllm.sh.
bash scripts/train_grpo.sh
```

## Spec → implementation map

The reviewer flagged a small set of GRPO knobs that must be wired correctly.
Each row links a knob to where it lives in the code so the audit is mechanical.

| Concern | Where | Setting |
|---|---|---|
| **Reward = NDCG@k against gold ids** | `grpo/reward.py::compute_score` reuses `eval_.metrics.ndcg_at_k` | `reward_kwargs.ndcg_k=10` |
| **Length normalisation in the loss** | actor PG loss aggregation | `actor_rollout_ref.actor.loss_agg_mode=token-mean` |
| **Length penalty in the reward** | one-sided per-token penalty in `reward.py` = number of tokens in the response_mask (including tool tokens) | `reward_kwargs.length_alpha=0.05`, `length_target_tokens=2048` |
| **GRPO group size G** | rollout fan-out | `actor_rollout_ref.rollout.n=8` |
| **Group-relative advantages** | advantage estimator | `algorithm.adv_estimator=grpo`, `norm_adv_by_std_in_grpo=true` |
| **Zero-variance group filtering** | drop groups with no reward signal before update | `algorithm.filter_groups.enable=true`, `metric=score`, `max_num_gen_batches=4` |
| **KL anchor against frozen ref** | actor loss term, not reward shaping | `actor.use_kl_loss=true`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl` |
| **Initial model** | actor + ref + rollout | `actor_rollout_ref.model.path=Qwen/Qwen3-14B` |
| **TI/TO consistency** | re-render `messages_full`, diff against `prompt_ids+response_ids` | `grpo/ti_to_check.py`; sampled every `ti_to_check.every_n_steps=50` |
| **Multi-turn rollout (ReAct)** | async rollout w/ AgentLoop | `actor_rollout_ref.rollout.mode=async`, `rollout.agent.agent_loop_config_path=configs/agent_loop_registry.yaml` |
| **Tool-message framing not rewarded** | per-token mask | `response_mask=0` for every token added by re-templating, see `agent_loop.py` |

## How the AgentLoop bookkeeps tokens

Multi-turn RL is brittle when tokenisation drifts. To avoid silent corruption
of log-prob ratios we never hand-stitch tool wraps — the chat template is the
single source of truth.

Per turn:

1. `generate(prompt_ids + response_ids)` → `new_ids` (mask = 1).
2. Append the assistant message; if it has tool calls, hit the tool server and
   append `role: tool` messages.
3. Re-render the full message list with `apply_chat_template(...,
   add_generation_prompt=True)` → `next_full_ids`.
4. Assert `next_full_ids[: len(prompt_ids) + len(response_ids)] ==
   prompt_ids + response_ids` (chat template should be prefix-extending).
5. Append the tail delta to `response_ids` with mask = 0 (it is template
   framing + the tool messages, not anything the policy generated).

If the prefix-stable property doesn't hold (rare; some tokenisers merge across
boundaries), the loop falls back to a conservative re-tokenise + best-effort
mask reconstruction with a `WARNING`. `ti_to_check.py` audits this property
on a sample of trajectories per training step.

## Reward semantics in one screenful

```python
gold     = parquet_row.extra_info.gold_doc_ids
ranked   = agent_loop_output.extra_fields.ranked_doc_ids   # filtered to seen ids
stopped  = agent_loop_output.extra_fields.stopped_reason
respn    = agent_loop_output.extra_fields.response_len

ndcg     = ndcg_at_k(ranked, set(gold), k=10) if stopped != "parse_error" else 0.0
overrun  = max(0, respn - target)
len_pen  = min(1.0, alpha * overrun / target)              # one-sided

score    = ndcg - len_pen + format_penalty                  # format_penalty default 0
```

Everything else (`ndcg`, `length_penalty`, `format_penalty`, `response_len`,
`stopped_reason`, `num_tool_calls`, `answered`) is returned alongside `score`
for logging and is folded into `extra_info` so `algorithm.filter_groups` can
see the per-rollout signal when computing zero-variance flags.

## Key invariants worth preserving on edits

- **`agent.prompts` is the single source of truth** for the system prompt and
  for tool-result rendering. The parquet does NOT store the system prompt —
  changing it does not require regenerating data. The AgentLoop and the
  trainer both read `agent.prompts`.
- **`reward_model.ground_truth`** is JSON-encoded in the parquet (a list of
  doc_ids); `extra_info.gold_doc_ids` is the same list as a Python list. The
  reward function tries `extra_info` first and falls back to parsing the
  ground-truth string. Both must stay in sync.
- **`agent_name=retrieval_react`** in every parquet row must match
  `configs/agent_loop_registry.yaml` and the `@register("retrieval_react")`
  decorator on `RetrievalReActAgentLoop`.
- **The data flow `prompt_ids + response_ids` is what the trainer sees**; the
  AgentLoop returns *exactly* that, plus `response_mask` of equal length to
  `response_ids`. Don't pad `response_mask` separately — the trainer expects
  `len(response_mask) == len(response_ids)`.

## verl 0.7.1 — what we actually wired against

`grpo/agent_loop.py` targets the **verl 0.7.1** API:

| Surface | Used as |
|---|---|
| `verl.experimental.agent_loop.AgentLoopBase` | base class; `__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kwargs)` |
| `AgentLoopBase.apply_chat_template(messages, tools=, remove_system_prompt=)` | async helper; `add_generation_prompt=True` is hardcoded |
| `self.server_manager.generate(request_id, *, prompt_ids, sampling_params)` | returns `TokenOutput` with `.token_ids` |
| `verl.experimental.agent_loop.AgentLoopOutput` | pydantic model: `prompt_ids`, `response_ids`, `response_mask`, `metrics: AgentLoopMetrics`, `extra_fields`, `num_turns` |
| `verl.experimental.agent_loop.AgentLoopMetrics` | typed metrics: `generate_sequences: float`, `tool_calls: float`, `num_preempted: int` |
| `verl.utils.profiler.simple_timer` | timing context-manager that writes into a metrics dict |
| `verl.utils.rollout_trace.rollout_trace_op` | decorator; opt-in tracing via `rollout_config.trace` |
| `@register("retrieval_react")` | populates `_agent_loop_registry`; veRL also reads `configs/agent_loop_registry.yaml` and overlays its entries |
| `MultiTurnConfig` | dataclass; `multi_turn.{enable, max_assistant_turns, max_user_turns, max_tool_response_length, tool_response_truncate_side}` are read by us |
| `RLHFDataset` w/ `data.return_raw_chat=true` | passes the chat-format prompt to `run(...)` as `kwargs["raw_prompt"]` |

Earlier versions (≤0.5) used a `init_class` classmethod and a different
`server_manager` interface; if you upgrade/downgrade verl, the changes
needed are usually localised to the constructor and the `generate(...)` call.

### Compatibility patch for `torch < 2.6`

verl 0.7.1's `verl/utils/fsdp_utils.py` references `DTensorSpec` in a
function-annotation that's only imported on the `torch >= 2.6` branch — so
under `torch < 2.6` the module fails to import with
`NameError: name 'DTensorSpec' is not defined`. The fix is to add a single
line at the top of `fsdp_utils.py`:

```python
from __future__ import annotations
```

That defers the annotation evaluation and the rest of verl works as-is. We
keep this patch local to the venv (don't commit it into a fork unless the
upstream regression bites in your CI). The repo's `pyproject.toml` pins
`torch<2.6` because the deployment box's CUDA driver (12.3) doesn't support
torch 2.6+ wheels.

## Tunables you'll touch first

| Knob | Default | When to change |
|---|---|---|
| `data.train_batch_size` | 64 | OOM during rollout → halve. |
| `actor_rollout_ref.rollout.n` | 8 | Smaller G if reward-variance is already non-zero on most groups (usually start at 8). |
| `algorithm.filter_groups.max_num_gen_batches` | 4 | If too many batches are rejected, raise the cap or weaken the filter. |
| `reward_kwargs.length_alpha` | 0.05 | Set to 0 for pure NDCG. Raise (e.g. 0.1) if rollouts run away on tool calls. |
| `agent_loop.retrieval_react.max_turns` | 8 | Hard cap on assistant turns. Lower → faster but may truncate good trajectories. |
| `agent_loop.retrieval_react.tool_server_url` | `http://localhost:8100` | Pointing at a remote tool server / load balancer. |
| `trainer.test_freq` | 50 | Validation cadence in steps. |
| `ti_to_check.every_n_steps` | 50 | Audit cadence. Cheap; leave low for the first ~500 steps. |

## Outputs

- `checkpoints/grpo_qwen3_14b/global_step_*` — actor weights (FSDP shards) +
  optimiser state. Use the veRL conversion utilities to merge into a HF
  checkpoint for downstream eval (`scripts/run_eval.sh`).
- `wandb` (or stdout if `logger=['console']`) — per-step `score`, `ndcg`,
  `length_penalty`, `response_len`, `stopped_reason`, `answered`, plus the
  standard veRL diagnostics (KL, clip frac, advantages).

## Sanity-checking the pipeline without burning a real run

The cheapest end-to-end smoke (≈30 s on a warm box) covers imports, reward,
parquet round-trip, tool server reachability, and TI/TO consistency for both
clean and corrupted trajectories. Useful when you change anything in
`grpo/`, the parquet schema, or the chat template.

```bash
python - <<'PY'
import json, pandas as pd, httpx
from agent.prompts import get_tool_schemas, get_prompt
from grpo.reward import compute_score
from grpo.ti_to_check import _normalize_token_ids, check_consistency
from transformers import AutoTokenizer

# (1) reward on stub data
extra = {"ranked_doc_ids":["a","b","c"], "gold_doc_ids":["b","d"],
         "stopped_reason":"answer", "response_len":1500, "num_tool_calls":2}
print(compute_score("musique_retrieval", "",
                    json.dumps(extra["gold_doc_ids"]), extra)["score"])

# (2) parquet round-trip
t = pd.read_parquet("data/processed/musique/grpo_train.parquet")
assert {"data_source","agent_name","prompt","reward_model","extra_info"} <= set(t.columns)
assert t.iloc[0]["agent_name"] == "retrieval_react"

# (3) tool server reachability
print(httpx.get("http://localhost:8100/healthz", timeout=5).json())

# (4) TI/TO clean + injected mismatch
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
msgs = [{"role":"system","content":get_prompt("v2")},
        {"role":"user","content":"hi"},
        {"role":"assistant","content":"<answer>x</answer>"}]
full = _normalize_token_ids(tok.apply_chat_template(
    msgs, add_generation_prompt=False, tools=get_tool_schemas("v2"), tokenize=True))
prompt = _normalize_token_ids(tok.apply_chat_template(
    msgs[:2], add_generation_prompt=True, tools=get_tool_schemas("v2"), tokenize=True))
resp = full[len(prompt):]
ok = check_consistency(tokenizer=tok, messages_full=msgs,
                       prompt_ids=prompt, response_ids=resp,
                       response_mask=[1]*len(resp))
assert ok["ok"], ok
PY
```

You can also run TI/TO on a saved harness trajectory if it includes
`prompt_ids` and `response_ids`:

```bash
python -m grpo.ti_to_check \
    --traj trajectories_data/2026-04-29/v1/<uuid>.json \
    --tokenizer Qwen/Qwen3-14B
```

## Related docs

- Repo overview, single-shot baseline, agent eval: `README.md`.
- Architectural rationale and SID-1 mapping: `retrieval_agent_spec-3.md`.
