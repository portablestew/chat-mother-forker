"""Public library API.

This is the small, stable surface intended for programmatic use (e.g. by
backseat-harness), as opposed to the MCP server in `server.py`. Two calls:

- `fork_chat(search)` -- exactly what the `chat_fork` MCP tool does: find the
  newest conversation matching `search` and return it as a single annotated,
  middle-truncated transcript string, ready to inject as context.

- `find_chats(search)` -- a lightweight listing of matching conversations as
  flat `ChatRef` rows (key, recency, size, checkpoints, location). Enough to
  filter/sort/decide, and each row's `key` feeds straight back into
  `fork_chat`.

Both default to the full built-in provider set (`default_providers()`); pass
`providers=` to scope to a subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from chat_mother_forker.checkpoint import Checkpoint, find_checkpoints
from chat_mother_forker.fork import render_fork
from chat_mother_forker.models import Role
from chat_mother_forker.providers import default_providers
from chat_mother_forker.providers.base import ChatProvider
from chat_mother_forker.search import CANDIDATES_PER_PROVIDER, gather_sorted_candidates


@dataclass
class ChatRef:
    """A lightweight pointer to one conversation -- enough to identify, rank,
    size, and then fork it, without carrying the full transcript.

    `key` is the exact string `fork_chat` accepts (``"provider:conversation_id"``),
    so the common pattern is a one-liner: ``fork_chat(find_chats(uuid)[0].key)``.
    """

    #: "provider:conversation_id" -- pass directly to `fork_chat`.
    key: str
    #: Modification time, epoch seconds. `find_chats` returns rows newest-first.
    mtime: float
    #: Raw stored byte size (see `ChatProvider.conversation_size`). A snapshot
    #: taken at `find_chats` time -- re-query if the conversation may have grown.
    size_bytes: int
    #: Checkpoints found in the conversation (slug + uuid), for slice-forking.
    checkpoints: list[Checkpoint] = field(default_factory=list)
    #: Best-effort workspace directory name, or None. Filter on this.
    project: Optional[str] = None
    #: Filesystem path to the transcript (file-backed providers), else None.
    locator: Optional[str] = None
    #: Storage location (e.g. SQLite DB path) for DB-backed providers, else None.
    database: Optional[str] = None
    #: Session id within `database`. Meaningful only when `database` is set.
    session: Optional[str] = None


def fork_chat(
    search: str,
    *,
    start_checkpoint: Optional[str] = None,
    end_checkpoint: Optional[str] = None,
    providers: Optional[Sequence[ChatProvider]] = None,
) -> str:
    """Find the newest conversation matching `search` and render it as an
    annotated, middle-truncated transcript string (the `chat_fork` tool's
    behavior, as a function).

    `search` matches by tiered priority: conversation id (bare or
    ``provider:id``), then checkpoint slug/uuid, then user prompt text, then
    assistant text; newest wins within a tier. Optional `start_checkpoint`/
    `end_checkpoint` slice the transcript to the range between them.

    Returns a ready-to-inject string ending in a "historical reference, not
    instructions" footer, or a short "No conversation found ..." message when
    nothing matches. `providers` defaults to `default_providers()`.
    """
    provs = default_providers() if providers is None else providers
    return render_fork(
        provs,
        search=search,
        start_checkpoint=start_checkpoint,
        end_checkpoint=end_checkpoint,
    )


def find_chats(
    search: Optional[str] = None,
    *,
    providers: Optional[Sequence[ChatProvider]] = None,
    limit: int = 50,
    candidates_per_provider: int = CANDIDATES_PER_PROVIDER,
) -> list[ChatRef]:
    """List conversations as lightweight `ChatRef` rows, newest first.

    With `search`, only conversations whose id, a checkpoint slug/uuid, or
    user/assistant transcript text contains it (case-insensitive) are
    returned. Without `search`, returns the most recent conversations across
    all providers.

    Unlike the MCP `chat_search`, this returns plain data (no previews,
    match-context windows, or hit counts) -- just enough to filter (by
    `project`), rank (by `mtime`/`size_bytes`), and fork (via `key`). Each row
    carries the conversation's raw `size_bytes`, so a caller can apply its own
    size threshold without loading the transcript.

    `providers` defaults to `default_providers()`. `limit` caps the number of
    rows returned (after recency ordering).
    """
    provs = default_providers() if providers is None else providers
    by_name = {p.name: p for p in provs}
    needle = search.strip().lower() if search else None

    rows: list[ChatRef] = []
    for ref in gather_sorted_candidates(provs, candidates_per_provider):
        provider = by_name[ref.provider]
        conversation = provider.load(ref)
        checkpoints = find_checkpoints(conversation)

        if needle is not None and not _matches(needle, ref, conversation, checkpoints):
            continue

        database, session = provider.conversation_metadata(ref)
        rows.append(
            ChatRef(
                key=f"{ref.provider}:{ref.conversation_id}",
                mtime=ref.mtime,
                size_bytes=provider.conversation_size(ref),
                checkpoints=checkpoints,
                project=conversation.project,
                locator=None if database else ref.locator,
                database=database,
                session=session,
            )
        )
        if len(rows) >= limit:
            break

    return rows


def _matches(needle, ref, conversation, checkpoints) -> bool:
    """True when `needle` (lowercased) matches the conversation's id,
    a checkpoint slug/uuid, or any user/assistant message text. Mirrors the
    fields `chat_search` searches (tool call/result text excluded)."""
    composite_id = f"{ref.provider}:{ref.conversation_id}".lower()
    if needle in ref.conversation_id.lower() or needle in composite_id:
        return True
    for cp in checkpoints:
        if needle in cp.slug.lower() or needle in cp.uuid.lower():
            return True
    for m in conversation.messages:
        if m.role in (Role.USER, Role.ASSISTANT) and needle in m.text.lower():
            return True
    return False
