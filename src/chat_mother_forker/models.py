"""Core data model shared by every provider.

Providers translate whatever on-disk format a tool uses (flat JSON/JSONL files,
SQLite, ...) into these plain structures. Everything downstream (turn grouping,
truncation, search, fork) only ever deals with these types, never with a
provider's native format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def basename_from_path(path: str) -> Optional[str]:
    """Return the last path segment of `path`, e.g. "chat-mother-forker" from
    "C:\\Dev\\github\\chat-mother-forker".

    Handles both Windows ('\\') and POSIX ('/') separators regardless of the
    host OS parsing it -- a chat history file can embed a path written on a
    different platform than the one now reading it, so this deliberately
    avoids os.path (which only recognizes the separator of the OS it runs
    on).
    """
    if not path:
        return None
    normalized = path.replace("\\", "/").rstrip("/")
    if not normalized:
        return None
    return normalized.rsplit("/", 1)[-1] or None


class Role(str, Enum):
    """Who/what produced a message.

    Turn grouping only cares about USER vs. everything else. The other values
    exist so rendering can label things distinctly (e.g. "TOOL_CALL: grep")
    and so checkpoint scraping can look specifically at TOOL_RESULT text,
    where the chat_checkpoint tool's literal output lives.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OTHER = "other"

    @property
    def is_user(self) -> bool:
        return self is Role.USER


@dataclass
class Message:
    """A single message/event within a conversation, already normalized."""

    role: Role
    text: str
    # Optional short label for rendering, e.g. a tool name ("grep", "bash").
    label: Optional[str] = None
    # Raw ISO-8601 timestamp string if the provider has one. Not required for
    # any current logic (ordering comes from file position); kept for
    # potential future use and debugging.
    timestamp: Optional[str] = None


@dataclass
class ConversationRef:
    """Cheap, sortable pointer to a conversation, obtained without fully
    parsing it. Used to pick the top-N candidates by recency before doing the
    more expensive full load.
    """

    provider: str
    conversation_id: str
    # Opaque token the owning provider uses to load the full conversation.
    # Never interpreted outside the provider that produced it.
    locator: str
    # Modification time as epoch seconds (works the same on Windows and
    # Linux, avoids timezone parsing entirely).
    mtime: float


@dataclass
class Conversation:
    """A fully loaded conversation."""

    ref: ConversationRef
    messages: list[Message] = field(default_factory=list)
    # Best-effort project/workspace directory name (e.g. "chat-mother-forker"),
    # derived by the provider from whatever cwd/workspace info its native
    # storage format exposes. None when a provider can't determine it.
    project: Optional[str] = None

    def first_user_text(self) -> str:
        for m in self.messages:
            if m.role.is_user and m.text.strip():
                return m.text
        return ""
