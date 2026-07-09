#!/usr/bin/env bash
# Fetch examples/tool_chat_template_gemma4.jinja for the *installed* vLLM version.
#
# The gemma4 tool-call parser is not self-sufficient: it expects this exact
# template (special tokens <|tool_call> / <tool_call|>), NOT the model's HF-repo
# default. It must also byte-match what veRL tokenizes with on the training side,
# so we pin the download to the installed vLLM version (falling back to main).
#
# Resolution order:
#   1. an already-present copy inside the installed vLLM (examples/ dir), copied
#      into the cache as-is;
#   2. download from the vLLM GitHub at tag v<installed-version>;
#   3. download from main (last resort — verify it matches your train stack).
#
# Prints the resolved path on stdout; serve_vllm_gemma.sh picks it up from the
# default cache location automatically.
#
# Usage: bash scripts/fetch_gemma_chat_template.sh
set -euo pipefail

NAME="tool_chat_template_gemma4.jinja"
DEST_DIR="${DEST_DIR:-$(cd "$(dirname "$0")/.." && pwd)/scripts/chat_templates}"
DEST="${DEST_DIR}/${NAME}"
mkdir -p "${DEST_DIR}"

# 1) Already shipped inside the installed vLLM?
LOCAL="$(python - <<'PY' 2>/dev/null || true
import importlib.util, os, glob
spec = importlib.util.find_spec("vllm")
if spec and spec.origin:
    pkg = os.path.dirname(spec.origin)
    for base in (pkg, os.path.dirname(pkg), os.path.dirname(os.path.dirname(pkg))):
        hits = glob.glob(os.path.join(base, "**", "tool_chat_template_gemma4.jinja"),
                         recursive=True)
        if hits:
            print(hits[0]); break
PY
)"
if [[ -n "${LOCAL}" && -f "${LOCAL}" ]]; then
  cp "${LOCAL}" "${DEST}"
  echo "copied from vLLM install: ${LOCAL}" >&2
  echo "${DEST}"
  exit 0
fi

# 2)/3) Download, version-pinned then main.
VER="$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)"
BASE="https://raw.githubusercontent.com/vllm-project/vllm"
REFS=()
[[ -n "${VER}" ]] && REFS+=("v${VER}")
REFS+=("main")

for ref in "${REFS[@]}"; do
  url="${BASE}/${ref}/examples/${NAME}"
  if curl -fsSL "${url}" -o "${DEST}.tmp" 2>/dev/null && [[ -s "${DEST}.tmp" ]]; then
    mv "${DEST}.tmp" "${DEST}"
    echo "downloaded from ${ref}: ${url}" >&2
    [[ "${ref}" == "main" && -n "${VER}" ]] && \
      echo "WARN: tag v${VER} not found; used main — verify it matches your train stack." >&2
    echo "${DEST}"
    exit 0
  fi
  rm -f "${DEST}.tmp"
done

echo "ERROR: could not obtain ${NAME} (installed vLLM version: ${VER:-unknown})." >&2
echo "       Download it manually from the vLLM source tree matching your version" >&2
echo "       and place it at ${DEST}." >&2
exit 1
