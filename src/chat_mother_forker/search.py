"""chat_search: list recent conversations across all providers, optionally
filtered by a substring.

Performance strategy (per design): each provider's cheap `list_candidates()`
is sorted by recency and capped to `CANDIDATES_PER_PROVIDER` *before* any
full parsing happens. Only those capped-and-merged candidates get loaded to
check the search filter / extract checkpoints. This means a search string
that only matches something older than the newest 100 conversations in a
given provider will not be found -- that's an accepted trade-off for keeping
this fast on machines with huge amounts of history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from chat_mother_forker.checkpoint import Checkpoint, find_checkpoints
from chat_mother_forker.models import ConversationRef
from chat_mother_forker.providers.base import ChatProvider
from chat_mother_forker.truncate import MAX_PREVIEW_CHARS, MAX_TURNS, truncate_preview

CANDIDATES_PER_PROVIDER = 100


@dataclass
class SearchResult:
    provider: str
    conversation_id: str
    mtime: float
    preview: str
    checkpoints: list[Checkpoint]

    @property
    def date(self) -> str:
        return (
            datetime.fromtimestamp(self.mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        )


def gather_sorted_candidates(
    providers: Sequence[ChatProvider], per_provider: int
) -> list[ConversationRef]:
    merged: list[ConversationRef] = []
    for provider in providers:
        candidates = sorted(provider.list_candidates(), key=lambda r: r.mtime, reverse=True)
        merged.extend(candidates[:per_provider])
    merged.sort(key=lambda r: r.mtime, reverse=True)
    return merged


def search_conversations(
    providers: Sequence[ChatProvider],
    search: Optional[str] = None,
    max_results: int = MAX_TURNS,
    candidates_per_provider: int = CANDIDATES_PER_PROVIDER,
) -> list[SearchResult]:
    by_name = {p.name: p for p in providers}
    candidates = gather_sorted_candidates(providers, candidates_per_provider)

    needle = search.strip().lower() if search else None
    results: list[SearchResult] = []

    for ref in candidates:
        provider = by_name[ref.provider]
        conversation = provider.load(ref)
        checkpoints = find_checkpoints(conversation)

        if needle:
            composite_id = f"{ref.provider}:{ref.conversation_id}".lower()
            matches = (
                needle in ref.conversation_id.lower()
                or needle in composite_id
                or any(needle in cp.slug.lower() or needle in cp.uuid.lower() for cp in checkpoints)
                or any(needle in m.text.lower() for m in conversation.messages)
            )
            if not matches:
                continue

        results.append(
            SearchResult(
                provider=ref.provider,
                conversation_id=ref.conversation_id,
                mtime=ref.mtime,
                preview=truncate_preview(conversation.first_user_text(), MAX_PREVIEW_CHARS),
                checkpoints=checkpoints,
            )
        )
        if len(results) >= max_results:
            break

    return results


def render_search_results(results: list[SearchResult], search: Optional[str]) -> str:
    if not results:
        if search:
            return f'No conversations found matching "{search}".'
        return "No conversations found."

    lines = []
    for r in results:
        lines.append(f"- {r.date} | {r.provider}:{r.conversation_id}")
        lines.append(f"  prompt: {r.preview or '(empty)'}")
        if r.checkpoints:
            slugs = ", ".join(f"{cp.slug} (UUID={cp.uuid})" for cp in r.checkpoints)
            lines.append(f"  slugs: {slugs}")

    header = f'{len(results)} conversation(s)' + (f' matching "{search}"' if search else "")
    return header + ":\n" + "\n".join(lines)
