"""Parse <answer>...</answer> from the assistant's final message."""

from __future__ import annotations

import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_answer(text: str | None) -> list[str] | None:
    """Return the list of doc_ids inside <answer>...</answer>, or None.

    Splits on commas, strips whitespace, drops empties. Order is preserved
    because the model's ranking is the output we care about.
    """
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return ids or None
