"""Provider for Kiro CLI.

Layout on disk (rooted at KIRO_HOME which defaults to ~/.kiro):

    <KIRO_HOME>/sessions/cli/<session-uuid>.jsonl
    <KIRO_HOME>/sessions/cli/<session-uuid>.json   (session metadata)

`<uuid>.jsonl` is a JSON-lines event log. Each line has the shape:

    {"version":"v1", "kind":"<Kind>", "data": {...}}

Supported `kind` values:

- "Prompt"           -> a user message; text in `data.content[*].data`
                        where `content[*].kind == "text"`.
- "AssistantMessage" -> an assistant message, same content layout.
- "ToolUse"         -> a tool invocation (data.name, data.input).
- "ToolResult"      -> a tool result (data.content).

Anything else is bookkeeping (system, turn_start, etc.) and is skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role
from chat_mother_forker.providers.base import ChatProvider

_JSONL_SUFFIX = ".jsonl"


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
        sessions_root = self._kiro_home / "sessions" / "cli"
        if not sessions_root.is_dir():
            return []

        refs = []
        for jsonl_path in sessions_root.glob("*" + _JSONL_SUFFIX):
            # Skip empty files (no messages to parse)
            try:
                stat = jsonl_path.stat()
            except OSError:
                continue
            if stat.st_size == 0:
                continue

            session_id = jsonl_path.stem  # UUID portion of filename
            refs.append(
                ConversationRef(
                    provider=self.name,
                    conversation_id=session_id,
                    locator=str(jsonl_path),
                    mtime=stat.st_mtime,
                )
            )
        return refs

    def load(self, ref: ConversationRef) -> Conversation:
        jsonl_path = Path(ref.locator)
        messages: list[Message] = []

        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
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
        kind = event.get("kind")
        data = event.get("data")
        if not isinstance(data, dict):
            return None

        timestamp = data.get("meta", {}).get("timestamp") if isinstance(data.get("meta"), dict) else None
        timestamp_str = str(timestamp) if timestamp is not None else None

        if kind == "Prompt":
            text = _extract_text_content(data)
            if not text or not text.strip():
                return None
            return Message(role=Role.USER, text=text, timestamp=timestamp_str)

        if kind == "AssistantMessage":
            text = _extract_text_content(data)
            if not text or not text.strip():
                return None
            return Message(role=Role.ASSISTANT, text=text, timestamp=timestamp_str)

        if kind == "ToolUse":
            tool_name = data.get("name") or "tool"
            tool_input = data.get("input")
            try:
                args_text = json.dumps(tool_input, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                args_text = str(tool_input)
            return Message(
                role=Role.TOOL_CALL,
                text=args_text,
                label=tool_name,
                timestamp=timestamp_str,
            )

        if kind == "ToolResult":
            content = data.get("content")
            if content is None:
                return None
            text = _extract_text_content(data) if isinstance(content, list) else str(content)
            if not text.strip():
                return None
            return Message(role=Role.TOOL_RESULT, text=text, timestamp=timestamp_str)

        # Anything else (system, turn_start, etc.) -- skip.
        return None


def _extract_text_content(data: dict) -> str:
    """Extract concatenated text from a content array like
    [{"kind":"text","data":"..."}].
    """
    content = data.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("kind") == "text":
            parts.append(item.get("data", ""))
    return "\n".join(parts)
