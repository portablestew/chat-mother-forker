"""Provider abstraction.

A provider knows how one specific tool (Kiro CLI, Kiro IDE, Claude Code CLI,
...) stores its chat history on disk and how to turn that into the shared
`Message`/`Conversation` model. Nothing outside a provider module is allowed
to know whether the underlying storage is flat JSON/JSONL files, SQLite, or
anything else -- that is exactly the point of this interface.

Every provider must support two operations, split for performance reasons:

- `list_candidates()`: a *cheap* scan that returns a `ConversationRef` per
  conversation with just enough information to sort by recency (currently:
  file/row modification time). This must NOT parse message bodies. On a
  machine with thousands of stored conversations, this is what keeps
  `chat_search`/`chat_fork` fast.

- `load(ref)`: the expensive full parse of one conversation, given a
  `ConversationRef` previously returned by `list_candidates()` from the same
  provider. The `ref.locator` field is opaque outside the provider that
  created it -- a provider is free to put a file path, a SQLite rowid, or
  anything else in there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from chat_mother_forker.models import Conversation, ConversationRef


class ChatProvider(ABC):
    """Base class for a single chat-history source."""

    #: Short, stable identifier used in `ConversationRef.provider` and in
    #: any user-facing output that needs to say where a conversation came
    #: from. Subclasses must set this.
    name: str = ""

    @abstractmethod
    def list_candidates(self) -> Iterable[ConversationRef]:
        """Return a ConversationRef for every conversation this provider can
        see, without fully parsing any of them. Order is not guaranteed;
        callers sort by `mtime` themselves.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, ref: ConversationRef) -> Conversation:
        """Fully parse the conversation referenced by `ref` (which must have
        been produced by this same provider's `list_candidates()`), and
        return it as a normalized `Conversation`.
        """
        raise NotImplementedError
