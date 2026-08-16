"""Provider for Cline (VS Code extension).

Layout on disk (rooted at CLINE_HOME which defaults to ~/.cline):

    <CLINE_HOME>/data/sessions/<session-id>/
        <session-id>.json            # Metadata (title, model, cwd, tokens, etc.)
        <session-id>.messages.json  # Full conversation transcript

The session ID format is typically a timestamp + random suffix, e.g.:
    1786853509637_wzumu

Metadata file (<session-id>.json) contains:
    {
        "session_id": "1786853509637_wzumu",
        "source": "vscode",
        "started_at": "2026-08-16T04:11:49.644Z",
        "ended_at": "2026-08-16T04:55:55.394Z",
        "provider": "openrouter",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "cwd": "C:\\Dev\\github\\DemoFrameHook",
        "workspace_root": "C:\\Dev\\github\\DemoFrameHook",
        "prompt": "Hello are you able to run the build.ps1 script?",
        "title": "Hello are you able to run the build.ps1 script?",
        "tokensIn": 164681,
        "tokensOut": 5268,
        ...
    }

Messages file (<session-id>.messages.json) contains:
    {
        "version": 1,
        "updated_at": "2026-08-16T04:55:19.373Z",
        "agent": "lead",
        "sessionId": "1786853509637_wzumu",
        "origin": {"source": "vscode", "mode": "user", "sessionId": "...", "version": "4.1.10"},
        "messages": [
            {"id": "msg_...", "role": "user", "content": [{"type": "text", "text": "..."}], "ts": 1786853510060},
            {"id": "msg_...", "role": "assistant", "content": [{"type": "tool_use", "name": "pyddock__fs_stat", "input": {...}}], "ts": ...},
            {"id": "msg_...", "role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": [...]}], "ts": ...}
        ],
        "system_prompt": "You are Cline, an AI coding agent..."
    }

Role/content mapping:
- role: "user" -> Role.USER
- role: "assistant" -> Role.ASSISTANT
- content.type: "text" -> text message (role from parent)
- content.type: "tool_use" -> Role.TOOL_CALL (label=name, text=JSON(input))
- content.type: "tool_result" -> Role.TOOL_RESULT (text=content)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role, basename_from_path
from chat_mother_forker.providers.base import ChatProvider


def _cline_home() -> Path:
    override = os.environ.get("CLINE_HOME")
    if override:
        return Path(override)
    return Path.home() / ".cline"


class ClineProvider(ChatProvider):
    name = "cline"

    def __init__(self, cline_home: Optional[Path] = None):
        self._cline_home = cline_home or _cline_home()

    def list_candidates(self) -> Iterable[ConversationRef]:
        sessions_root = self._cline_home / "data" / "sessions"
        if not sessions_root.is_dir():
            return []

        refs = []
        for session_dir in sessions_root.iterdir():
            if not session_dir.is_dir():
                continue

            meta_file = session_dir / f"{session_dir.name}.json"
            if not meta_file.exists():
                continue

            try:
                stat = meta_file.stat()
            except OSError:
                continue

            if stat.st_size == 0:
                continue

            project = None
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                cwd = meta.get("workspace_root") or meta.get("cwd")
                if isinstance(cwd, str) and cwd:
                    project = basename_from_path(cwd)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                pass

            refs.append(
                ConversationRef(
                    provider=self.name,
                    conversation_id=session_dir.name,
                    locator=str(session_dir),
                    mtime=stat.st_mtime,
                )
            )
        return refs

    def load(self, ref: ConversationRef) -> Conversation:
        session_dir = Path(ref.locator)
        messages_file = session_dir / f"{session_dir.name}.messages.json"
        meta_file = session_dir / f"{session_dir.name}.json"

        messages: list[Message] = []
        project: Optional[str] = None

        # Read metadata for project info (cwd/workspace_root)
        try:
            with meta_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            cwd = meta.get("workspace_root") or meta.get("cwd")
            if isinstance(cwd, str) and cwd:
                project = basename_from_path(cwd)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            pass

        # Parse full conversation from messages file
        try:
            with messages_file.open("r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return Conversation(ref=ref, messages=[], project=project)

        raw_messages = data.get("messages", [])
        if not isinstance(raw_messages, list):
            return Conversation(ref=ref, messages=[], project=project)

        for raw_msg in raw_messages:
            extracted = self._to_messages(raw_msg)
            messages.extend(extracted)

        messages = _trim_before_first_user(messages)
        return Conversation(ref=ref, messages=messages, project=project)

    @staticmethod
    def _to_messages(raw_msg: dict) -> list[Message]:
        """Convert one Cline message into zero or more normalized Message objects."""
        if not isinstance(raw_msg, dict):
            return []

        role_str = raw_msg.get("role", "")
        content = raw_msg.get("content", [])
        timestamp = raw_msg.get("ts")

        results: list[Message] = []

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
                    # Can be [{"type": "text", "text": "..."}]
                    text_parts = []
                    for part in result_content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    text = "\n".join(text_parts)
                else:
                    text = json.dumps(result_content, default=str)
                if text.strip():
                    results.append(
                        Message(role=Role.TOOL_RESULT, text=text, timestamp=timestamp)
                    )

        return results


def _trim_before_first_user(messages: list[Message]) -> list[Message]:
    """Drop any messages before the first user message."""
    for i, m in enumerate(messages):
        if m.role is Role.USER:
            return messages[i:]
    return messages
