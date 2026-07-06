"""Shared test fixtures and helpers.

`FakeProvider` lets search/fork tests exercise the merge/filter/rank logic
without touching the filesystem at all -- it's just an in-memory
implementation of the same `ChatProvider` interface a real provider (Kiro
CLI, or anything dropped in later) has to satisfy.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pytest

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role
from chat_mother_forker.providers.base import ChatProvider


def msg(role: Role, text: str, label: Optional[str] = None) -> Message:
    return Message(role=role, text=text, label=label)


def user(text: str) -> Message:
    return msg(Role.USER, text)


def assistant(text: str) -> Message:
    return msg(Role.ASSISTANT, text)


def tool_call(name: str, text: str = "{}") -> Message:
    return msg(Role.TOOL_CALL, text, label=name)


def tool_result(text: str) -> Message:
    return msg(Role.TOOL_RESULT, text)


class FakeProvider(ChatProvider):
    """An in-memory `ChatProvider` for testing, backed by a plain dict of
    conversation_id -> (mtime, messages) rather than any real storage.
    """

    def __init__(self, name: str = "fake"):
        self.name = name
        self._conversations: dict[str, tuple[float, list[Message], Optional[str]]] = {}

    def add(
        self,
        conversation_id: str,
        mtime: float,
        messages: list[Message],
        project: Optional[str] = None,
    ) -> None:
        self._conversations[conversation_id] = (mtime, messages, project)

    def list_candidates(self) -> Iterable[ConversationRef]:
        return [
            ConversationRef(
                provider=self.name,
                conversation_id=cid,
                locator=cid,
                mtime=mtime,
            )
            for cid, (mtime, _messages, _project) in self._conversations.items()
        ]

    def load(self, ref: ConversationRef) -> Conversation:
        _mtime, messages, project = self._conversations[ref.conversation_id]
        return Conversation(ref=ref, messages=list(messages), project=project)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
