"""Provider for Kiro CLI.

Layout on disk (same on Windows and Linux, rooted at KIRO_HOME which defaults
to ~/.kiro):

    <KIRO_HOME>/sessions/<workspace-hash>/<session-uuid>/session.json
    <KIRO_HOME>/sessions/<workspace-hash>/<session-uuid>/messages.jsonl

`messages.jsonl` is a JSON-lines event log. The fields we care about are
`payload.type` and `payload.content`:

- "user"            -> a user message, `payload.content` is plain text.
- "assistant"       -> an assistant message, `payload.content` is plain text.
- "tool_call"       -> a tool invocation, `payload.toolName` + `payload.args`.
- "tool_result"     -> a tool's output, `payload.content` is the result text.
- anything else (e.g. "turn_start", "session_metadata") is bookkeeping with
  no user-facing text and is skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role
from chat_mother_forker.providers.base import ChatProvider

_MESSAGES_FILENAME = "messages.jsonl"


def _kiro_home() -> Path:
    override = os.environ.get("KIRO_HOME")
    if override:
        return Path(override)
    return Path.home() / ".kiro"


class KiroCliProvider(ChatProvider):
    name = "kiro_cli"

    def __init__(self, kiro_home: Optional[Path] = None):
        self._kiro_home = kiro_home or _kiro_home()

    def list_candidates(self) -> Iterable[ConversationRef]:
        sessions_root = self._kiro_home / "sessions"
        if not sessions_root.is_dir():
            return []

        refs = []
        # <sessions_root>/<workspace-hash>/<session-uuid>/messages.jsonl
        for messages_path in sessions_root.glob("*/*/" + _MESSAGES_FILENAME):
            session_dir = messages_path.parent
            try:
                mtime = messages_path.stat().st_mtime
            except OSError:
                continue
            refs.append(
                ConversationRef(
                    provider=self.name,
                    conversation_id=session_dir.name,
                    locator=str(messages_path),
                    mtime=mtime,
                )
            )
        return refs

    def load(self, ref: ConversationRef) -> Conversation:
        messages_path = Path(ref.locator)
        messages: list[Message] = []

        with messages_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = self._to_message(event)
                if message is not None:
                    messages.append(message)

        return Conversation(ref=ref, messages=messages)

    @staticmethod
    def _to_message(event: dict) -> Optional[Message]:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return None

        kind = payload.get("type")
        timestamp = event.get("timestamp")

        if kind == "user":
            text = payload.get("content")
            if not isinstance(text, str) or not text.strip():
                return None
            return Message(role=Role.USER, text=text, timestamp=timestamp)

        if kind == "assistant":
            text = payload.get("content")
            if not isinstance(text, str) or not text.strip():
                return None
            return Message(role=Role.ASSISTANT, text=text, timestamp=timestamp)

        if kind == "tool_call":
            tool_name = payload.get("toolName") or "tool"
            args = payload.get("args")
            try:
                args_text = json.dumps(args, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                args_text = str(args)
            return Message(
                role=Role.TOOL_CALL,
                text=args_text,
                label=tool_name,
                timestamp=timestamp,
            )

        if kind == "tool_result":
            content = payload.get("content")
            if content is None:
                return None
            text = content if isinstance(content, str) else json.dumps(content, default=str)
            if not text.strip():
                return None
            return Message(role=Role.TOOL_RESULT, text=text, timestamp=timestamp)

        # "turn_start", "session_metadata", and anything else we don't
        # recognize yet carry no user-facing text -- skip rather than guess.
        return None
