"""chat_search: list recent conversations across all providers, optionally
filtered by a substring.

The substring filter matches against conversation id, checkpoint
slug/uuid, and user/assistant message text -- tool call/result text is
excluded (noisy, and it keeps a nested chat_fork transcript inside a
TOOL_RESULT from polluting matches with someone else's conversation).

Performance strategy (per design): each provider's cheap `list_candidates()`
is sorted by recency and capped to `CANDIDATES_PER_PROVIDER` *before* any
full parsing happens. Only those capped-and-merged candidates get loaded to
check the search filter / extract checkpoints. This means a search string
that only matches something older than the newest 100 conversations in a
given provider will not be found -- that's an accepted trade-off for keeping
this fast on machines with huge amounts of history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

from chat_mother_forker.checkpoint import Checkpoint, find_checkpoints
from chat_mother_forker.models import ConversationRef, Message, Role
from chat_mother_forker.providers.base import ChatProvider
from chat_mother_forker.truncate import MAX_PREVIEW_CHARS, context_window, truncate_preview

CANDIDATES_PER_PROVIDER = 100
MAX_SEARCH_RESULTS = 50


@dataclass
class SearchResult:
    provider: str
    conversation_id: str
    mtime: float
    preview: str
    checkpoints: list[Checkpoint]
    # Best-effort project/workspace directory name, e.g. "chat-mother-forker".
    # None when the provider couldn't determine it.
    project: Optional[str] = None
    # Raw filesystem path to the conversation file, for direct grep/inspection.
    # This is the provider's `locator` field (file path for file-based providers).
    file_path: Optional[str] = None
    # Total number of parsed messages (user + assistant + tool_call + tool_result).
    # Useful for judging conversation depth/richness before forking.
    message_count: int = 0
    # Subset of "conversation_id", "checkpoint_slug", "checkpoint_uuid",
    # "transcript" -- which of the searchable fields the needle matched.
    # Empty when `search` was None.
    matched_in: list[str] = field(default_factory=list)
    transcript_hit_count: int = 0
    first_context: str = ""
    last_context: str = ""

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


@dataclass
class _TranscriptMatches:
    hit_count: int = 0
    first_context: str = ""
    last_context: str = ""


def _find_transcript_matches(messages: list[Message], needle: str) -> _TranscriptMatches:
    """Scan user/assistant messages (in conversation order) for `needle`
    (already lowercased), returning the total occurrence count plus
    bolded context windows around the first and last occurrence.

    Occurrences are counted per-message via `str.count`, and the
    first/last context window is extracted from whichever message holds
    that occurrence -- never spanning multiple messages.
    """
    hit_count = 0
    first_context = ""
    last_context = ""

    for m in messages:
        if m.role not in (Role.USER, Role.ASSISTANT):
            continue
        lowered = m.text.lower()
        count_in_message = lowered.count(needle)
        if count_in_message == 0:
            continue

        hit_count += count_in_message
        if not first_context:
            first_idx = lowered.find(needle)
            first_context = context_window(m.text, first_idx, len(needle))
        last_idx = lowered.rfind(needle)
        last_context = context_window(m.text, last_idx, len(needle))

    if hit_count <= 1:
        last_context = ""

    return _TranscriptMatches(hit_count, first_context, last_context)


def search_conversations(
    providers: Sequence[ChatProvider],
    search: Optional[str] = None,
    max_results: int = MAX_SEARCH_RESULTS,
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

        matched_in: list[str] = []
        transcript_matches = _TranscriptMatches()

        if needle:
            composite_id = f"{ref.provider}:{ref.conversation_id}".lower()
            if needle in ref.conversation_id.lower() or needle in composite_id:
                matched_in.append("conversation_id")
            if any(needle in cp.slug.lower() for cp in checkpoints):
                matched_in.append("checkpoint_slug")
            if any(needle in cp.uuid.lower() for cp in checkpoints):
                matched_in.append("checkpoint_uuid")

            transcript_matches = _find_transcript_matches(conversation.messages, needle)
            if transcript_matches.hit_count > 0:
                matched_in.append("transcript")

            if not matched_in:
                continue

        results.append(
            SearchResult(
                provider=ref.provider,
                conversation_id=ref.conversation_id,
                mtime=ref.mtime,
                preview=truncate_preview(conversation.first_user_text(), MAX_PREVIEW_CHARS),
                checkpoints=checkpoints,
                project=conversation.project,
                file_path=ref.locator,
                message_count=len(conversation.messages),
                matched_in=matched_in,
                transcript_hit_count=transcript_matches.hit_count,
                first_context=transcript_matches.first_context,
                last_context=transcript_matches.last_context,
            )
        )
        if len(results) >= max_results:
            break

    return results


_MATCHED_IN_LABELS = {
    "conversation_id": "conversation id",
    "checkpoint_slug": "checkpoint slug",
    "checkpoint_uuid": "checkpoint uuid",
}


def _render_matched_in(r: SearchResult) -> str:
    parts = []
    for reason in r.matched_in:
        if reason == "transcript":
            parts.append(f"transcript ({r.transcript_hit_count} hit{'s' if r.transcript_hit_count != 1 else ''})")
        else:
            parts.append(_MATCHED_IN_LABELS[reason])
    return ", ".join(parts)


def render_search_results(results: list[SearchResult], search: Optional[str]) -> str:
    if not results:
        if search:
            return f'No conversations found matching "{search}".'
        return "No conversations found."

    lines = []
    for r in results:
        project_suffix = f" | {r.project}" if r.project else ""
        lines.append(f"- {r.date} | {r.provider}:{r.conversation_id}{project_suffix}")
        lines.append(f"  prompt: {r.preview or '(empty)'}")
        lines.append(f"  messages: {r.message_count}")
        if r.file_path:
            lines.append(f"  file: {r.file_path}")
        if r.checkpoints:
            slugs = ", ".join(f"{cp.slug} (UUID={cp.uuid})" for cp in r.checkpoints)
            lines.append(f"  slugs: {slugs}")
        if r.matched_in:
            lines.append(f"  matched: {_render_matched_in(r)}")
        if r.first_context:
            lines.append(f"  first: {r.first_context}")
        if r.last_context:
            lines.append(f"  last: {r.last_context}")

    header = f'{len(results)} conversation(s)' + (f' matching "{search}"' if search else "")
    return header + ":\n" + "\n".join(lines)
