# Retrieval Agent MVP

SID-1-like retrieval agent over MuSiQue. Returns a ranked list of `doc_id` for the
user's last turn, given the full conversation as context.

Three independent processes:

1. **vLLM** — serves Qwen3.5-35B-A3B (OpenAI-compatible API).
2. **Tool server** — FastAPI wrapper around faiss + E5 embedder.
3. **Agent harness** — Python client, runs the ReAct loop, talks to vLLM and the tool server.

See the technical specification for the full architectural rationale.

## Layout

```
agent/           ReAct loop, prompts, schemas, parser
tool_server/     FastAPI app: /local_search, /healthz, /stats
indexing/        MuSiQue → corpus.jsonl → faiss index
eval_/           NDCG@k, Recall@k; single-shot baseline; agent eval
trajectories/    JSON trace writer + TI/TO consistency check
configs/         default.yaml — single source of truth for paths and hyperparams
scripts/         shell wrappers for vLLM, tool server, build, eval
```

`eval_` rather than `eval` to avoid shadowing the Python builtin.

## Quickstart

```bash
# 1. Install
pip install -e .

# 2. Drop raw MuSiQue jsonl into data/raw/musique/
#    (musique_ans_v1.0_train.jsonl, musique_ans_v1.0_dev.jsonl)

# 3. Build everything (parse → index → eval dataset)
bash scripts/build_all.sh

# 4. Serve vLLM (terminal 1) and tool server (terminal 2)
bash scripts/serve_vllm.sh # using venv: source ../matreshka_reranker/.venv/bin/activate
bash scripts/serve_tool.sh # using home venv 

# 5. Run baseline + agent eval
bash scripts/run_eval.sh
```

## Configuration

Single config at `configs/default.yaml`. All paths, model names, and hyperparams live
there. The agent harness, tool server, and indexing scripts all read it.

## Trajectories

Every rollout writes a JSON file under `trajectories_data/` with the full message
history, tool calls, and token counts. Format is designed to drop straight into
veRL/Search-R1-style RL pipelines later — see `trajectories/writer.py`.

## Experiments

enable_thinking=false agent n=200 ndcg@10=0.4750 recall@10=0.4275 avg_calls=2.88
enable_thinking=true agent n=200  ndcg@10=0.3960  recall@10=0.3800  avg_calls=2.13
enable_thinking=false and <think> tag
