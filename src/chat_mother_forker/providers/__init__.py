"""Provider registry.

Add a new tool by writing a `ChatProvider` subclass and listing an instance
of it in `ALL_PROVIDERS` below. Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from chat_mother_forker.providers.base import ChatProvider
from chat_mother_forker.providers.claude_code import ClaudeCodeProvider
from chat_mother_forker.providers.kiro_cli import KiroCliProvider
from chat_mother_forker.providers.kiro_ide import KiroIdeProvider

ALL_PROVIDERS: list[ChatProvider] = [
    KiroCliProvider(),
    KiroIdeProvider(),
    ClaudeCodeProvider(),
]

__all__ = ["ChatProvider", "ALL_PROVIDERS"]
