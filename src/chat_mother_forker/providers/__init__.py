"""Provider registry.

Add a new tool by writing a `ChatProvider` subclass and listing an instance
of it in `ALL_PROVIDERS` below. Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from chat_mother_forker.providers.base import ChatProvider
from chat_mother_forker.providers.claude_code import ClaudeCodeProvider
from chat_mother_forker.providers.cline import ClineProvider
from chat_mother_forker.providers.kilo import KiloProvider
from chat_mother_forker.providers.kiro_cli import KiroCliProvider
from chat_mother_forker.providers.kiro_ide import KiroIdeProvider
from chat_mother_forker.providers.kiro_ide_v2 import KiroIdeV2Provider
from chat_mother_forker.providers.opencode import OpenCodeProvider
from chat_mother_forker.search import CANDIDATES_PER_PROVIDER

def default_providers() -> list[ChatProvider]:
    """Build a fresh list of every built-in provider, configured the way the
    MCP server and library callers want by default.

    Returns a new list of new provider instances on each call, so callers
    (e.g. backseat-harness) can hold, filter, or mutate the list without
    affecting anyone else. Pass the result (or a subset) to
    `search_conversations`, `find_chats`, `render_fork`/`fork_chat`.
    """
    return [
        KiroCliProvider(),
        # `max_sessions` lets the (potentially large) execution-log index
        # scan stop early once it has enough sessions to satisfy
        # chat_search/chat_fork's per-provider cap -- see
        # KiroIdeProvider.__init__ for the full rationale.
        KiroIdeProvider(max_sessions=CANDIDATES_PER_PROVIDER),
        # Newer Kiro IDE builds write sessions in a different, simpler layout
        # (one directory per session under ~/.kiro/sessions/<hash>/sess_<uuid>/)
        # -- see kiro_ide_v2.py. Kept as a separate provider rather than folded
        # into KiroIdeProvider since the on-disk shape and event schema are
        # unrelated; older, still-legacy sessions remain reachable via
        # KiroIdeProvider above.
        KiroIdeV2Provider(),
        ClaudeCodeProvider(),
        ClineProvider(),
        KiloProvider(),
        # SQLite DB: push the recency cut into SQL (exact -- same ordering key
        # callers use) so list_candidates never materializes every session row.
        OpenCodeProvider(max_sessions=CANDIDATES_PER_PROVIDER),
    ]


#: Default provider set for the MCP server. Library callers should prefer
#: `default_providers()` (a fresh, independent list) over sharing this global.
ALL_PROVIDERS: list[ChatProvider] = default_providers()

__all__ = ["ChatProvider", "ALL_PROVIDERS", "default_providers"]
