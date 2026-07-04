"""Provider registry.

Add a new tool by writing a `ChatProvider` subclass and listing an instance
of it in `ALL_PROVIDERS` below. Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from chat_mother_forker.providers.base import ChatProvider
from chat_mother_forker.providers.kiro_cli import KiroCliProvider

ALL_PROVIDERS: list[ChatProvider] = [
    KiroCliProvider(),
    # Drop in more providers here as they're implemented, e.g.:
    # KiroIdeProvider(),
    # ClaudeCodeCliProvider(),
]

__all__ = ["ChatProvider", "ALL_PROVIDERS"]
