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
MAX_PREVIEW_CHARS = 128
MAX_CONTEXT_CHARS = 128
MAX_TURNS = 50

T = TypeVar("T")


def _collapse_newlines(text: str) -> str:
    """Replace CR/LF with single spaces so a match window renders on one
    line. Each replacement is exactly one character for one character
    (including "\\r\\n" -> two spaces), so string length -- and therefore
    any previously computed character offsets -- is preserved.
    """
    return text.replace("\r", " ").replace("\n", " ")


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
    there's nothing useful to show "after" a preview). Newlines are
    collapsed to spaces first, so the preview always renders on one line.
    """
    text = _collapse_newlines(text.strip())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def context_window(text: str, match_start: int, match_len: int, width: int = MAX_CONTEXT_CHARS) -> str:
    """Extract up to `width` characters of `text` centered on a match,
    wrapping the matched substring in `**bold**` markdown.

    `match_start`/`match_len` describe the match's position in `text` (e.g.
    from `text.lower().find(needle)` against the original, non-lowered
    `text`). Newlines are collapsed to spaces first (so the result always
    renders on a single line) -- this is a 1:1 character replacement, so
    `match_start`/`match_len` stay valid against the collapsed text. Ellipsis
    markers are added on whichever side(s) got clipped.
    """
    text = _collapse_newlines(text)
    text_len = len(text)
    match_start = max(0, min(match_start, text_len))
    match_end = max(match_start, min(match_start + match_len, text_len))

    half = max(width - (match_end - match_start), 0) // 2
    start = max(0, match_start - half)
    end = min(text_len, match_end + half)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < text_len else ""

    return (
        f"{prefix}{text[start:match_start]}"
        f"**{text[match_start:match_end]}**"
        f"{text[match_end:end]}{suffix}"
    )


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
