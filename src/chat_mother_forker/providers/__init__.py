"""Provider registry.

Add a new tool by writing a `ChatProvider` subclass and listing an instance
of it in `ALL_PROVIDERS` below. Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from chat_mother_forker.providers.base import ChatProvider
from chat_mother_forker.providers.claude_code import ClaudeCodeProvider
from chat_mother_forker.providers.kiro_cli import KiroCliProvider
from chat_mother_forker.providers.kiro_ide import KiroIdeProvider
from chat_mother_forker.search import CANDIDATES_PER_PROVIDER

ALL_PROVIDERS: list[ChatProvider] = [
    KiroCliProvider(),
    # `max_sessions` lets the (potentially large) execution-log index scan
    # stop early once it has enough sessions to satisfy chat_search/chat_fork's
    # per-provider cap -- see KiroIdeProvider.__init__ for the full rationale.
    KiroIdeProvider(max_sessions=CANDIDATES_PER_PROVIDER),
    ClaudeCodeProvider(),
]

__all__ = ["ChatProvider", "ALL_PROVIDERS"]
