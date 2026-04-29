# Retrieval Agent MVP

SID-1-like retrieval agent over multi-hop corpora (MuSiQue, SBOL FAQ). Returns a
ranked list of `doc_id` for the user's last turn, given the full conversation as context.

Three independent processes:

1. **vLLM** — serves Qwen3.5-35B-A3B (OpenAI-compatible API).
2. **Tool server** — FastAPI wrapper around faiss + E5 embedder.
3. **Agent harness** — Python client, runs the ReAct loop, talks to vLLM and the tool server.

See the technical specification for the full architectural rationale.

## Layout

```
agent/           ReAct loop, prompts, schemas, parser
tool_server/     FastAPI app: /local_search, /healthz, /stats
indexing/        corpus parsers (MuSiQue, SBOL) + faiss index builder
eval_/           NDCG@k, Recall@k; single-shot baseline; agent eval; train-set builder
trajectories/    JSON trace writer + TI/TO consistency check
grpo/            GRPO training: data_prep, AgentLoop, reward, TI/TO check
configs/         per-dataset configs (default.yaml, sbol.yaml) + grpo_qwen3_14b.yaml
scripts/         shell wrappers for vLLM, tool server, build, eval, GRPO training
indexes/         FAISS indexes per dataset (musique/, sbol/)
data/processed/  processed corpora and eval sets per dataset (musique/, sbol/)
```

`eval_` rather than `eval` to avoid shadowing the Python builtin.

## Quickstart

```bash
# 1. Install
pip install -e .

# 2. Drop raw data into data/raw/<dataset>/

# 3a. Build MuSiQue (parse → index → eval dataset)
bash scripts/build_all.sh        # writes to data/processed/musique/ and indexes/musique/

# 3b. Build SBOL FAQ index
bash scripts/build_sbol.sh       # writes to data/processed/sbol/ and indexes/sbol/

# 4. Serve vLLM (terminal 1) and tool server (terminal 2)
bash scripts/serve_vllm.sh                          # using venv: source ../matreshka_reranker/.venv/bin/activate
bash scripts/serve_tool.sh                          # MuSiQue (default)
CONFIG=configs/sbol.yaml bash scripts/serve_tool.sh # SBOL

# 5. Run baseline + agent eval
bash scripts/run_eval.sh                                              # MuSiQue
CONFIG=configs/sbol.yaml OUT_DIR=data/processed/sbol bash scripts/run_eval.sh  # SBOL
```

## Configuration

One config file per dataset under `configs/`. All paths, model names, and hyperparams
live there. The agent harness, tool server, and indexing scripts all accept `--config`
and the shell scripts expose a `CONFIG=` env var to switch datasets.

| Config | Dataset | Index dir | Embedder |
|--------|---------|-----------|----------|
| `configs/default.yaml` | MuSiQue | `indexes/musique/` | `intfloat/e5-large-v2` |
| `configs/sbol.yaml` | SBOL FAQ | `indexes/sbol/` | `intfloat/multilingual-e5-large` |

## Trajectories

Every rollout writes a JSON file under `trajectories_data/` with the full message
history, tool calls, and token counts. Format is designed to drop straight into
veRL/Search-R1-style RL pipelines later — see `trajectories/writer.py`.

## GRPO training

End-to-end GRPO over the ReAct loop on MuSiQue train. Reward = NDCG@10 against
the gold supporting passages, with a one-sided length penalty layered on top.

```bash
# 1. Install the verl extras (pulls in ray, hydra, vllm, ...)
pip install -e .[verl]

# 2. Build the parquet (parses MuSiQue train + writes grpo_{train,val}.parquet)
bash scripts/build_grpo_data.sh

# 3. Start the tool server (FAISS + E5 embedder)
bash scripts/serve_tool.sh

# 4. Launch training (8x GPU). The actor's rollout server is internal to veRL —
#    don't start scripts/serve_vllm.sh for training.
bash scripts/train_grpo.sh
```

Knobs called out by the spec:

| Concern | Where |
|---|---|
| **Reward = NDCG@k** | `grpo/reward.py` reuses `eval_.metrics.ndcg_at_k`. |
| **Length normalisation** | `actor.loss_agg_mode=token-mean` + per-token reward penalty (`length_alpha`). |
| **Group size G** | `actor_rollout_ref.rollout.n=8`. |
| **Zero-variance filtering** | `algorithm.filter_groups.enable=true, metric=score`. |
| **TI/TO consistency** | `grpo/ti_to_check.py` re-templates `messages_full` and diffs against `prompt_ids+response_ids`; sampled every `ti_to_check.every_n_steps`. |
| **Initial model** | `Qwen/Qwen3-14B` at `actor_rollout_ref.model.path`. |
| **Framework** | veRL (`verl.trainer.main_ppo`) with custom AgentLoop. |

The custom AgentLoop (`grpo/agent_loop.py`, `name=retrieval_react`) drives the
ReAct rollout: each turn it (a) generates with the in-process rollout server,
(b) parses Qwen3-XML tool calls, (c) hits the tool server, (d) re-templates the
new tool messages and extends `response_mask` with 0s so the optimiser only
takes credit for tokens the actor itself produced.

## Experiments

enable_thinking=false agent n=200 ndcg@10=0.4750 recall@10=0.4275 avg_calls=2.88
enable_thinking=true agent n=200  ndcg@10=0.3960  recall@10=0.3800  avg_calls=2.13
enable_thinking=false and <think> tag
