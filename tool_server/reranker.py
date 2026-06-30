"""Cross-encoder reranker client.

FAISS gives a fast but coarse first-stage ranking (cosine similarity over a
single dense vector). A cross-encoder reranker (Qwen3-Reranker-4B) jointly reads
(query, document) and scores relevance directly, which is much more accurate but
too expensive to run over the whole corpus. So the pipeline is:

    query -> embed -> FAISS top-N candidates -> rerank -> top_k

The reranker model is served separately by vLLM (scripts/serve_reranker.sh,
``--task score`` with the Qwen3ForSequenceClassification hf_overrides). This
client just POSTs to its OpenAI-compatible ``/v1/rerank`` endpoint, which scores
each document via the model's "yes"/"no" logits and returns them sorted.

The client is intentionally fail-soft: any error (timeout, 5xx, malformed body)
raises ``RerankError`` so the caller can fall back to the FAISS order instead of
failing the whole search. Under a 128-worker rollout a reranker brownout must
not take down retrieval.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


class RerankError(Exception):
    """Raised when the reranker server cannot return a usable ranking."""


class Reranker:
    """Async client for a vLLM ``/v1/rerank`` endpoint."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        timeout_s: float = 30.0,
    ) -> None:
        # ``url`` is the full rerank endpoint (e.g. http://localhost:8200/v1/rerank).
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        # Reused across requests; one keep-alive pool avoids per-call TCP/TLS
        # setup under high rollout concurrency.
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Return ``[(orig_index, score), ...]`` sorted best-first, len <= top_n.

        ``orig_index`` indexes into the ``documents`` list passed in, so the
        caller can map back to its candidate metadata. Raises ``RerankError``
        on any failure so the caller can fall back to the input order.
        """
        if not documents:
            return []

        # vLLM applies Qwen3-Reranker's built-in "Judge whether the Document
        # meets the requirements based on the Query" prompt; the relevance score
        # is the model's softmaxed "yes" logit.
        payload: dict = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        try:
            resp = await self._client.post(self.url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RerankError(f"rerank request failed: {exc}") from exc

        results = body.get("results")
        if not isinstance(results, list):
            raise RerankError(f"rerank response missing 'results': {body!r}")

        ranked: list[tuple[int, float]] = []
        for r in results:
            try:
                idx = int(r["index"])
                score = float(r["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RerankError(f"malformed rerank result {r!r}: {exc}") from exc
            if 0 <= idx < len(documents):
                ranked.append((idx, score))
        # vLLM already sorts by score, but sort defensively so callers can rely
        # on it regardless of server version.
        ranked.sort(key=lambda t: t[1], reverse=True)
        return ranked[:top_n]

    async def aclose(self) -> None:
        await self._client.aclose()
