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
eval_/           NDCG@k, Recall@k; single-shot baseline; agent eval
trajectories/    JSON trace writer + TI/TO consistency check
configs/         per-dataset configs (default.yaml = MuSiQue, sbol.yaml = SBOL FAQ)
scripts/         shell wrappers for vLLM, tool server, build, eval
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

## Experiments

enable_thinking=false agent n=200 ndcg@10=0.4750 recall@10=0.4275 avg_calls=2.88
enable_thinking=true agent n=200  ndcg@10=0.3960  recall@10=0.3800  avg_calls=2.13
enable_thinking=false and <think> tag
