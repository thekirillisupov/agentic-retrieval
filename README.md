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
pip install -e .[verl]
SKIP_PARSE=1 bash scripts/build_grpo_data.sh   # writes grpo_{train,val}.parquet
bash scripts/serve_tool.sh                     # FAISS + E5 (:8100)
bash scripts/train_grpo.sh                     # veRL spins up its own rollout vLLM
```

Full design + spec→implementation map + open caveats live in
[`GRPO_README.md`](./GRPO_README.md).

## Inference image

Self-contained image that serves the quantized actor + the ReAct harness as a
single HTTP endpoint. Embedder / reranker / index are **not** included — the
harness calls an external `search` service (`SEARCH_URL`).

```
[vLLM :8000 OpenAI API] <--local--> [harness /retrieve :8080] --> external SEARCH_URL
```

```bash
# Build (lean; weights are NOT baked in)
docker build -f Dockerfile.inference -t agentic-retrieval:infer .

# Run — mount the W8A8 weights, point at your external search service
docker run --gpus all -p 8080:8080 \
  -v "$PWD/checkpoints/quantized/qwen3_5_35b_a3b_w8a8:/models/qwen3_5_35b_a3b_w8a8:ro" \
  -e SEARCH_URL=http://your-search-host:8100 \
  agentic-retrieval:infer

# Call the agent: pass the dialogue + retrieval params (index routing, filters,
# … — everything except the query). The model only decides the query text for
# each search / get_neighbours call.
curl -s localhost:8080/retrieve -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "who directed the sequel to Alien?"}],
  "search_params": {},
  "prompt_version": "v2"
}'
```

`prompt_version` is per-request (falls back to the config default): `v2`
(search + get_neighbours) or `v2_search_only` (search only). Each profile fixes
its own tool set, so this also switches which tools the model is offered. `GET
/prompts` lists the available versions.

Config lives in [`configs/inference.yaml`](./configs/inference.yaml); `VLLM_URL`,
`MODEL_NAME`, `SEARCH_URL` override it at runtime. `agent.prompt_version`,
`use_id_map`, `tool_budget_feedback` and `max_tool_calls` must match the config
the weights were trained/quantized with. Route names on the external service are
configurable under `search.endpoints` (default `/local_search`). For a
harness-only deployment (separate vLLM) use `scripts/serve_agent.sh`.

## Experiments

enable_thinking=false agent n=200 ndcg@10=0.4750 recall@10=0.4275 avg_calls=2.88
enable_thinking=true agent n=200  ndcg@10=0.3960  recall@10=0.3800  avg_calls=2.13
enable_thinking=false and <think> tag
