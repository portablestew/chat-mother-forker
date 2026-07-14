"""Provider registry.

Add a new tool by writing a `ChatProvider` subclass and listing an instance
of it in `ALL_PROVIDERS` below. Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from chat_mother_forker.providers.base import ChatProvider
from chat_mother_forker.providers.claude_code import ClaudeCodeProvider
from chat_mother_forker.providers.kiro_cli import KiroCliProvider
from chat_mother_forker.providers.kiro_ide import KiroIdeProvider
from chat_mother_forker.providers.kiro_ide_v2 import KiroIdeV2Provider
from chat_mother_forker.search import CANDIDATES_PER_PROVIDER

ALL_PROVIDERS: list[ChatProvider] = [
    KiroCliProvider(),
    # `max_sessions` lets the (potentially large) execution-log index scan
    # stop early once it has enough sessions to satisfy chat_search/chat_fork's
    # per-provider cap -- see KiroIdeProvider.__init__ for the full rationale.
    KiroIdeProvider(max_sessions=CANDIDATES_PER_PROVIDER),
    # Newer Kiro IDE builds write sessions in a different, simpler layout
    # (one directory per session under ~/.kiro/sessions/<hash>/sess_<uuid>/)
    # -- see kiro_ide_v2.py. Kept as a separate provider rather than folded
    # into KiroIdeProvider since the on-disk shape and event schema are
    # unrelated; older, still-legacy sessions remain reachable via
    # KiroIdeProvider above.
    KiroIdeV2Provider(),
    ClaudeCodeProvider(),
]

__all__ = ["ChatProvider", "ALL_PROVIDERS"]
