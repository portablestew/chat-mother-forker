"""Provider for Kiro IDE.

Kiro IDE stores its real conversation content in execution logs:

    <globalStorage>/<workspace-hash>/414d1636299d2b9e4ce7e17fb11f63e9/<execution-hash>

Each execution log is a JSON file containing:
- `chatSessionId`: the conversation this execution belongs to
- `startTime`: epoch millis, for ordering
- `context.messages`: prior conversation history (user, bot, tool entries)
- `actions[]`: the agent's work for this turn (say, readFiles, mcp, etc.)
- `input.data.messages`: user messages, with the last being the new prompt

Multiple executions share the same `chatSessionId` (one per user turn).
The latest execution for a session contains the most complete context.

Discovery and content both come from these execution logs — we scan once,
deduplicate by chatSessionId, and cache for the lifetime of the provider.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role
from chat_mother_forker.providers.base import ChatProvider

# Well-known subdirectory name where execution logs live within each workspace hash dir.
_EXEC_LOGS_SUBDIR = "414d1636299d2b9e4ce7e17fb11f63e9"


def _trim_before_first_user(messages: list[Message]) -> list[Message]:
    """Drop any messages before the first user message.

    Kiro IDE execution logs often start with boilerplate (system ack,
    initial file tree tool call) that has no user-facing value.
    """
    for i, m in enumerate(messages):
        if m.role is Role.USER:
            return messages[i:]
    return messages


def _global_storage_root() -> Optional[Path]:
    """Locate the Kiro IDE globalStorage directory for kiro.kiroagent."""
    override = os.environ.get("KIRO_IDE_STORAGE")
    if override:
        return Path(override)

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Kiro" / "User" / "globalStorage" / "kiro.kiroagent"
    elif sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Kiro"
            / "User"
            / "globalStorage"
            / "kiro.kiroagent"
        )
    else:
        config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return Path(config_home) / "Kiro" / "User" / "globalStorage" / "kiro.kiroagent"

    return None


class KiroIdeProvider(ChatProvider):
    name = "kiro_ide"

    def __init__(self, storage_root: Optional[Path] = None):
        self._storage_root = storage_root or _global_storage_root()
        # Lazily populated: chatSessionId -> (startTime, exec_file_path)
        self._session_index: Optional[dict[str, tuple[float, Path]]] = None

    def list_candidates(self) -> Iterable[ConversationRef]:
        index = self._ensure_index()
        refs = []
        for session_id, (start_time, exec_path) in index.items():
            # Convert startTime (epoch millis) to epoch seconds for mtime
            mtime = start_time / 1000.0
            refs.append(
                ConversationRef(
                    provider=self.name,
                    conversation_id=session_id,
                    locator=str(exec_path),
                    mtime=mtime,
                )
            )
        return refs

    def load(self, ref: ConversationRef) -> Conversation:
        exec_path = Path(ref.locator)
        try:
            data = json.loads(exec_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return Conversation(ref=ref, messages=[])

        messages = self._parse_execution_log(data)
        messages = _trim_before_first_user(messages)
        return Conversation(ref=ref, messages=messages)

    def _ensure_index(self) -> dict[str, tuple[float, Path]]:
        """Scan all execution logs once and build a session -> latest execution index."""
        if self._session_index is not None:
            return self._session_index

        self._session_index = {}
        if self._storage_root is None:
            return self._session_index

        for hash_dir in self._storage_root.iterdir():
            if not hash_dir.is_dir() or len(hash_dir.name) != 32:
                continue
            exec_dir = hash_dir / _EXEC_LOGS_SUBDIR
            if not exec_dir.is_dir():
                continue

            for exec_file in exec_dir.iterdir():
                if not exec_file.is_file():
                    continue
                self._index_execution_file(exec_file)

        return self._session_index

    def _index_execution_file(self, exec_file: Path) -> None:
        """Read just enough of an execution log to index it by session."""
        try:
            data = json.loads(exec_file.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return

        session_id = data.get("chatSessionId")
        start_time = data.get("startTime", 0)
        if not session_id or not start_time:
            return

        existing = self._session_index.get(session_id)  # type: ignore[union-attr]
        if existing is None or start_time > existing[0]:
            self._session_index[session_id] = (start_time, exec_file)  # type: ignore[union-attr]

    def _parse_execution_log(self, exec_data: dict) -> list[Message]:
        """Parse a full conversation from an execution log's context + actions."""
        messages: list[Message] = []

        # 1. Parse context.messages (the full history up to this turn)
        context_messages = exec_data.get("context", {}).get("messages", [])
        for ctx_msg in context_messages:
            messages.extend(self._parse_context_message(ctx_msg))

        # 2. Extract the new user prompt from input.data.messages (last entry)
        input_messages = exec_data.get("input", {}).get("data", {}).get("messages", [])
        if input_messages:
            last_user = input_messages[-1]
            text = self._extract_text_from_content_blocks(last_user.get("content", []))
            if text.strip():
                messages.append(Message(role=Role.USER, text=text))

        # 3. Parse actions from this turn (say, tool calls, etc.)
        for action in exec_data.get("actions", []):
            extracted = self._parse_action(action)
            if extracted:
                messages.append(extracted)

        return messages

    @staticmethod
    def _parse_context_message(ctx_msg: dict) -> list[Message]:
        """Parse one message from the context.messages array."""
        role = ctx_msg.get("role", "")
        entries = ctx_msg.get("entries", [])
        results: list[Message] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get("type", "")

            if entry_type == "text":
                text = entry.get("text", "")
                if not text.strip():
                    continue
                # Skip system prompts and environment context
                if text.startswith("<key_kiro_features>") or text.startswith("<EnvironmentContext>"):
                    continue
                if role == "human":
                    results.append(Message(role=Role.USER, text=text))
                elif role == "bot":
                    results.append(Message(role=Role.ASSISTANT, text=text))

            elif entry_type == "toolUse":
                tool_name = entry.get("name", "tool")
                args = entry.get("args", {})
                try:
                    args_text = json.dumps(args, separators=(",", ":"), default=str)
                except (TypeError, ValueError):
                    args_text = str(args)
                results.append(Message(role=Role.TOOL_CALL, text=args_text, label=tool_name))

            elif entry_type == "toolUseResponse":
                text = entry.get("message", "")
                if text.strip():
                    results.append(Message(role=Role.TOOL_RESULT, text=text))

        return results

    @staticmethod
    def _parse_action(action: dict) -> Optional[Message]:
        """Parse one action from the actions array into a Message, if relevant."""
        action_type = action.get("actionType", "")
        output = action.get("output", {})

        if action_type == "say":
            text = output.get("message", "")
            if text.strip():
                return Message(role=Role.ASSISTANT, text=text)

        elif action_type == "mcp":
            inp = action.get("input", {})
            tool_name = inp.get("toolName", "tool")
            args = inp.get("toolArgs", {})
            try:
                args_text = json.dumps(args, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                args_text = str(args)
            return Message(role=Role.TOOL_CALL, text=args_text, label=tool_name)

        elif action_type == "runCommand":
            inp = action.get("input", {})
            command = inp.get("command", "")
            if command:
                return Message(role=Role.TOOL_CALL, text=command, label="shell")

        return None

    @staticmethod
    def _extract_text_from_content_blocks(content) -> str:
        """Extract text from content blocks like [{"type":"text","text":"..."}]."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                # Skip environment context blocks
                if text.startswith("<EnvironmentContext>"):
                    continue
                parts.append(text)
        return "\n".join(parts)
