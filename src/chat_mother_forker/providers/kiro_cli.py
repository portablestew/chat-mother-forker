"""Provider for Kiro CLI.

Layout on disk (rooted at KIRO_HOME which defaults to ~/.kiro):

    <KIRO_HOME>/sessions/cli/<session-uuid>.jsonl
    <KIRO_HOME>/sessions/cli/<session-uuid>.json   (session metadata)

`<uuid>.jsonl` is a JSON-lines event log. Each line has the shape:

    {"version":"v1", "kind":"<Kind>", "data": {...}}

Supported `kind` values:

- "Prompt"           -> a user message; text in `data.content[*].data`
                        where `content[*].kind == "text"`.
- "AssistantMessage" -> an assistant message. Content is an array of items
                        that may include both `{"kind":"text","data":"..."}`
                        (text fragments) and `{"kind":"toolUse","data":{...}}`
                        (tool invocations). Each text fragment becomes an
                        ASSISTANT message and each toolUse becomes a
                        TOOL_CALL message.
- "ToolResults"      -> one or more tool results. Structure is:
                        data.content[*].kind == "toolResult"
                        data.content[*].data.content[*].kind == "json"|"text"
                        For "json" items, the actual MCP response lives at
                        data.content[*].data.content[*].data.content[*].text
                        (the standard MCP TextContent format).

Legacy format (kept for backward compatibility with older sessions):
- "ToolUse"         -> standalone tool invocation (data.name, data.input).
- "ToolResult"      -> standalone tool result (data.content).

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

                for message in self._to_messages(event):
                    messages.append(message)

        return Conversation(ref=ref, messages=messages)

    @staticmethod
    def _to_messages(event: dict) -> list[Message]:
        """Convert a single JSONL event into zero or more Messages.

        An AssistantMessage with embedded toolUse items may produce
        multiple messages (one ASSISTANT + one or more TOOL_CALL).
        """
        kind = event.get("kind")
        data = event.get("data")
        if not isinstance(data, dict):
            return []

        timestamp = data.get("meta", {}).get("timestamp") if isinstance(data.get("meta"), dict) else None
        timestamp_str = str(timestamp) if timestamp is not None else None

        if kind == "Prompt":
            text = _extract_text_content(data)
            if not text or not text.strip():
                return []
            return [Message(role=Role.USER, text=text, timestamp=timestamp_str)]

        if kind == "AssistantMessage":
            return _parse_assistant_message(data, timestamp_str)

        if kind == "ToolResults":
            return _parse_tool_results(data, timestamp_str)

        # Legacy: standalone ToolUse (older sessions)
        if kind == "ToolUse":
            tool_name = data.get("name") or "tool"
            tool_input = data.get("input")
            try:
                args_text = json.dumps(tool_input, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                args_text = str(tool_input)
            return [Message(
                role=Role.TOOL_CALL,
                text=args_text,
                label=tool_name,
                timestamp=timestamp_str,
            )]

        # Legacy: standalone ToolResult (older sessions)
        if kind == "ToolResult":
            content = data.get("content")
            if content is None:
                return []
            text = _extract_text_content(data) if isinstance(content, list) else str(content)
            if not text.strip():
                return []
            return [Message(role=Role.TOOL_RESULT, text=text, timestamp=timestamp_str)]

        # Anything else (system, turn_start, etc.) -- skip.
        return []


def _parse_assistant_message(data: dict, timestamp_str: Optional[str]) -> list[Message]:
    """Parse an AssistantMessage event.

    The content array may contain:
    - {"kind": "text", "data": "..."} — assistant prose
    - {"kind": "toolUse", "data": {"toolUseId":..., "name":..., "input":...}}

    We emit one ASSISTANT message per text fragment and one TOOL_CALL
    message per toolUse item.
    """
    content = data.get("content")
    if not isinstance(content, list):
        return []

    messages: list[Message] = []
    text_parts: list[str] = []

    for item in content:
        if not isinstance(item, dict):
            continue
        item_kind = item.get("kind")

        if item_kind == "text":
            text_data = item.get("data", "")
            if text_data and text_data.strip():
                text_parts.append(text_data)

        elif item_kind == "toolUse":
            tool_data = item.get("data", {})
            tool_name = tool_data.get("name") or "tool"
            tool_input = tool_data.get("input")
            try:
                args_text = json.dumps(tool_input, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                args_text = str(tool_input)
            messages.append(Message(
                role=Role.TOOL_CALL,
                text=args_text,
                label=tool_name,
                timestamp=timestamp_str,
            ))

    # Emit a single ASSISTANT message for all text parts combined
    if text_parts:
        combined_text = "\n".join(text_parts)
        messages.insert(0, Message(role=Role.ASSISTANT, text=combined_text, timestamp=timestamp_str))

    return messages


def _parse_tool_results(data: dict, timestamp_str: Optional[str]) -> list[Message]:
    """Parse a ToolResults (plural) event.

    Structure:
        data.content[] -> items with kind="toolResult"
        each toolResult.data.content[] -> items with kind="json" or "text"
        for "json": the MCP response is at .data.content[].text (TextContent)
        for "text": plain text at .data (string)
    """
    content = data.get("content")
    if not isinstance(content, list):
        return []

    messages: list[Message] = []

    for item in content:
        if not isinstance(item, dict) or item.get("kind") != "toolResult":
            continue
        tool_result_data = item.get("data", {})
        text = _extract_tool_result_text(tool_result_data)
        if text and text.strip():
            messages.append(Message(role=Role.TOOL_RESULT, text=text, timestamp=timestamp_str))

    return messages


def _extract_tool_result_text(tool_result_data: dict) -> str:
    """Extract text from a single toolResult item's content array.

    The content items may be:
    - {"kind": "json", "data": {"content": [{"type":"text","text":"..."}], ...}}
      (standard MCP TextContent response wrapped in JSON)
    - {"kind": "text", "data": "..."}
      (plain text)
    """
    inner_content = tool_result_data.get("content")
    if not isinstance(inner_content, list):
        return ""

    parts: list[str] = []
    for inner_item in inner_content:
        if not isinstance(inner_item, dict):
            continue
        inner_kind = inner_item.get("kind")

        if inner_kind == "json":
            # MCP response: data.content[*].text (TextContent items)
            json_data = inner_item.get("data", {})
            if isinstance(json_data, dict):
                mcp_content = json_data.get("content")
                if isinstance(mcp_content, list):
                    for tc in mcp_content:
                        if isinstance(tc, dict) and tc.get("type") == "text":
                            text_val = tc.get("text", "")
                            if text_val:
                                parts.append(text_val)
                # Also check for a top-level "result" field (structuredContent)
                elif "result" in json_data:
                    result_val = json_data["result"]
                    if isinstance(result_val, str):
                        parts.append(result_val)

        elif inner_kind == "text":
            text_data = inner_item.get("data", "")
            if text_data:
                parts.append(text_data)

    return "\n".join(parts)


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
