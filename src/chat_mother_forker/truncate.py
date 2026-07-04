"""Middle-out truncation helpers.

The guiding idea (per design discussion): the most important parts of a turn
or a conversation are the beginning (intent) and the end (conclusion), so
when something is too long we drop the *middle* and say how much we dropped.
This is deliberately simple -- lengths are measured in Python characters, not
bytes, and the exact truncated length is allowed to drift a little around the
target as long as it's in the right ballpark.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

MAX_TURN_CHARS = 2000
MAX_PREVIEW_CHARS = 256
MAX_TURNS = 50

T = TypeVar("T")


def truncate_middle_text(text: str, max_chars: int = MAX_TURN_CHARS) -> str:
    """Drop the middle of `text` if it's longer than `max_chars`, replacing
    it with a `[N characters truncated]` marker line.
    """
    if len(text) <= max_chars:
        return text

    keep = max_chars // 2
    head = text[:keep]
    tail = text[len(text) - keep :]
    dropped = len(text) - len(head) - len(tail)

    return f"{head}\n[{dropped} characters truncated]\n{tail}"


def truncate_preview(text: str, max_chars: int = MAX_PREVIEW_CHARS) -> str:
    """Truncate `text` to its first `max_chars` characters, for display in a
    search result list. Truncation is from the end only (no middle-drop --
    there's nothing useful to show "after" a preview).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def truncate_middle_list(items: Sequence[T], max_items: int, marker_factory) -> list[T]:
    """Drop items from the middle of a sequence if it has more than
    `max_items`, keeping the first and last halves. `marker_factory(n)` is
    called with the number of dropped items and must return a single
    placeholder item (of the same type as the sequence) to splice in.
    """
    if len(items) <= max_items:
        return list(items)

    keep = max_items // 2
    head = list(items[:keep])
    tail = list(items[len(items) - keep :])
    dropped = len(items) - len(head) - len(tail)

    return [*head, marker_factory(dropped), *tail]
