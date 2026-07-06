"""Provider for Claude Code (CLI).

Layout on disk (rooted at CLAUDE_HOME which defaults to ~/.claude):

    <CLAUDE_HOME>/projects/<encoded-project-path>/<session-uuid>.jsonl

The project directory name is the absolute workspace path with path separators
replaced by "--" (and the drive colon removed on Windows), e.g.:
    C:\\Dev\\github\\my-project -> C--Dev-github-my-project

Each `.jsonl` file is a JSON-lines event log. Each line has:

    {
      "type": "user"|"assistant"|"system"|"file-history-snapshot"|...,
      "message": {"role": "user"|"assistant", "content": ...},
      "sessionId": "...",
      "cwd": "C:\\Dev\\github\\my-project",
      "timestamp": "2026-07-04T23:34:16.737Z",
      "uuid": "...",
      ...
    }

`cwd` (present on every line) gives us the workspace directory directly;
its basename becomes `Conversation.project`.

Message content can be:
- A plain string (for simple messages).
- A list of content blocks: [{"type":"text","text":"..."}, {"type":"tool_use",...}, ...]

For assistant messages, content blocks may include:
- {"type":"text","text":"..."} -> assistant text
- {"type":"tool_use","name":"...","input":{...}} -> tool invocation
- {"type":"tool_result","content":"..."} -> tool result (appears in user messages)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role, basename_from_path
from chat_mother_forker.providers.base import ChatProvider

_JSONL_SUFFIX = ".jsonl"


def _unwrap_structured_result(text: str) -> str:
    """Normalize FastMCP's structured-content wrapper on MCP tool results.

    Unlike other clients (e.g. Kiro), Claude Code stores the *structured
    content* an MCP tool returns -- a single-key ``{"result": <value>}``
    object that FastMCP synthesizes for a scalar return -- as the literal
    text of the ``tool_result`` block. Left as-is, tool output reads as
    ``{"result":"..."}`` instead of the bare string every other provider
    yields. That prefix also breaks checkpoint discovery: ``find_checkpoints``
    anchors the ``CHAT CHECKPOINT ...`` line at position 0 of the message
    text, and ``{"result":"`` shifts it off the start.

    So when the text is exactly such a wrapper, unwrap it to the inner value.
    Anything else (plain text, multi-key JSON, malformed JSON) is returned
    untouched.
    """
    stripped = text.lstrip()
    if not stripped.startswith('{"result"') and not stripped.startswith('{ "result"'):
        return text
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return text
    if isinstance(obj, dict) and set(obj.keys()) == {"result"}:
        inner = obj["result"]
        return inner if isinstance(inner, str) else json.dumps(inner, default=str)
    return text


def _trim_before_first_user(messages: list[Message]) -> list[Message]:
    """Drop any messages before the first user message."""
    for i, m in enumerate(messages):
        if m.role is Role.USER:
            return messages[i:]
    return messages


def _claude_home() -> Path:
    override = os.environ.get("CLAUDE_HOME")
    if override:
        return Path(override)
    return Path.home() / ".claude"


class ClaudeCodeProvider(ChatProvider):
    name = "claude_code"

    def __init__(self, claude_home: Optional[Path] = None):
        self._claude_home = claude_home or _claude_home()

    def list_candidates(self) -> Iterable[ConversationRef]:
        projects_root = self._claude_home / "projects"
        if not projects_root.is_dir():
            return []

        refs = []
        for project_dir in projects_root.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl_path in project_dir.glob("*" + _JSONL_SUFFIX):
                try:
                    stat = jsonl_path.stat()
                except OSError:
                    continue
                if stat.st_size == 0:
                    continue

                session_id = jsonl_path.stem
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
        project: Optional[str] = None

        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if project is None:
                    cwd = event.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        project = basename_from_path(cwd)

                extracted = self._to_messages(event)
                messages.extend(extracted)

        messages = _trim_before_first_user(messages)
        return Conversation(ref=ref, messages=messages, project=project)

    @staticmethod
    def _to_messages(event: dict) -> list[Message]:
        """Convert one JSONL event into zero or more Message objects."""
        event_type = event.get("type")
        if event_type not in ("user", "assistant"):
            return []

        msg = event.get("message")
        if not isinstance(msg, dict):
            return []

        timestamp = event.get("timestamp")
        role_str = msg.get("role", event_type)
        content = msg.get("content", "")

        results: list[Message] = []

        # Simple string content
        if isinstance(content, str):
            if content.strip():
                role = Role.USER if role_str == "user" else Role.ASSISTANT
                results.append(Message(role=role, text=content, timestamp=timestamp))
            return results

        # Content block array
        if not isinstance(content, list):
            return results

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            if block_type == "text":
                text = block.get("text", "")
                if text.strip():
                    role = Role.USER if role_str == "user" else Role.ASSISTANT
                    results.append(Message(role=role, text=text, timestamp=timestamp))

            elif block_type == "tool_use":
                tool_name = block.get("name", "tool")
                tool_input = block.get("input", {})
                try:
                    args_text = json.dumps(tool_input, separators=(",", ":"), default=str)
                except (TypeError, ValueError):
                    args_text = str(tool_input)
                results.append(
                    Message(
                        role=Role.TOOL_CALL,
                        text=args_text,
                        label=tool_name,
                        timestamp=timestamp,
                    )
                )

            elif block_type == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    text = result_content
                elif isinstance(result_content, list):
                    # Can be [{"type":"text","text":"..."}]
                    text_parts = []
                    for part in result_content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    text = "\n".join(text_parts)
                else:
                    text = json.dumps(result_content, default=str)
                text = _unwrap_structured_result(text)
                if text.strip():
                    results.append(
                        Message(role=Role.TOOL_RESULT, text=text, timestamp=timestamp)
                    )

        return results
