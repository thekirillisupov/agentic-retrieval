#!/usr/bin/env bash
# Serve google/gemma-4-26B-A4B-it over vLLM for the ReAct eval harness.
#
# Gemma-4 tool calling in vLLM is a three-part contract — get any part wrong and
# tool calls silently fail to parse or you get a train/inference token mismatch:
#   1. --tool-call-parser gemma4
#   2. --reasoning-parser gemma4   (thought channel <|channel>thought…<channel|>
#      and tool calls share one stream; the tool parser trips over thought blocks
#      without this — the same Qwen thinking+tools footgun, solved natively here)
#   3. --chat-template tool_chat_template_gemma4.jinja  shipped in vLLM's examples/.
#      This is NOT the model's HF-repo default template — it must byte-match the
#      one veRL tokenizes with, or the special tokens <|tool_call> / <tool_call|>
#      diverge between train and inference.
#
# Thinking is a conversation-level control token (<|think|> consolidated into the
# system turn), toggled per-request via chat_template_kwargs. We default it off to
# mirror the Qwen enable_thinking=false baseline; the gemma4 reasoning parser still
# absorbs any thought channel the 26B model leaks even when off, keeping tool-call
# parsing and the final answer clean.
#
# The chat template lives in the vLLM source tree (examples/), which pip does not
# always install. Point CHAT_TEMPLATE at it explicitly if auto-resolution fails:
#   CHAT_TEMPLATE=/path/to/vllm/examples/tool_chat_template_gemma4.jinja \
#     bash scripts/serve_vllm_gemma.sh
#
# Usage: bash scripts/serve_vllm_gemma.sh
set -euo pipefail

export MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-gemma4}"
export REASONING_PARSER="${REASONING_PARSER:-gemma4}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"

# Locate examples/tool_chat_template_gemma4.jinja unless CHAT_TEMPLATE is given.
if [[ -z "${CHAT_TEMPLATE:-}" ]]; then
  CHAT_TEMPLATE="$(python - <<'PY' 2>/dev/null || true
import importlib.util, os
spec = importlib.util.find_spec("vllm")
if spec and spec.origin:
    pkg = os.path.dirname(spec.origin)                 # .../site-packages/vllm
    root = os.path.dirname(pkg)                          # .../site-packages
    for base in (pkg, root, os.path.dirname(root)):      # installed or source layout
        cand = os.path.join(base, "examples", "tool_chat_template_gemma4.jinja")
        if os.path.exists(cand):
            print(cand)
            break
PY
)"
fi
if [[ -z "${CHAT_TEMPLATE}" ]]; then
  echo "ERROR: could not locate tool_chat_template_gemma4.jinja under the vLLM" >&2
  echo "       install. Set CHAT_TEMPLATE=/path/to/examples/tool_chat_template_gemma4.jinja" >&2
  echo "       (from the vLLM source tree that matches your installed version)." >&2
  exit 1
fi
export CHAT_TEMPLATE
echo "[serve_vllm_gemma] MODEL=${MODEL}  CHAT_TEMPLATE=${CHAT_TEMPLATE}"

exec bash "$(dirname "$0")/serve_vllm.sh"
