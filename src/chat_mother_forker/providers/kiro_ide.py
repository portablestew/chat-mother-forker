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

Discovery and content both come from these execution logs. The index is
rebuilt on every call rather than cached, since execution logs are written
during/after each turn and a long-running MCP server process would
otherwise see a stale view of what's on disk.

When the newest execution for a session is still `running` (i.e. the
current turn is in-flight), its `context.messages` is typically empty --
Kiro IDE only backfills it after the turn completes. In that case,
`load()` stitches together the full prior history from the most recent
*terminal-status* execution with the running file's own new-turn content
(`input.data.messages` + `actions[]`), giving a complete transcript up to
the current moment.

Known limitation: `chatSessionId` can rotate (e.g. around an IDE/MCP
restart), which starts a new session with an empty `context.messages` and
no prior terminal executions to fall back on. There's currently no
stitching across that rotation -- see the README's "Known gaps" section.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role
from chat_mother_forker.providers.base import ChatProvider

# Well-known subdirectory name where execution logs live within each workspace hash dir.
_EXEC_LOGS_SUBDIR = "414d1636299d2b9e4ce7e17fb11f63e9"

# Maximum age (in days) of execution log files to consider during indexing.
# Files with a filesystem mtime older than this are skipped entirely,
# avoiding expensive JSON parsing of stale data.
MAX_INDEX_AGE_DAYS = 30


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

        # When the newest execution is still running, its context.messages
        # is typically empty (Kiro IDE backfills it only after the turn
        # completes). In that case, stitch together the full prior history
        # from the previous terminal-status execution with the running
        # file's own new-turn content (input.data.messages + actions).
        if (
            data.get("status") == "running"
            and not data.get("context", {}).get("messages")
        ):
            messages = self._stitch_running_execution(data, ref.conversation_id)

        messages = _trim_before_first_user(messages)
        return Conversation(ref=ref, messages=messages)

    def _stitch_running_execution(self, running_data: dict, session_id: str) -> list[Message]:
        """Combine the previous terminal execution's full history with the
        running execution's own new-turn content.

        The previous terminal execution's ``_parse_execution_log`` output
        contains all prior turns (its context.messages + its own
        input/actions). The running execution contributes only its
        ``input.data.messages`` (the new user prompt) and ``actions[]``
        (the agent's in-progress work), since its ``context.messages`` is
        empty at this point.
        """
        start_time = running_data.get("startTime", float("inf"))
        previous = self._find_previous_terminal_execution(session_id, start_time)

        if previous is None:
            # No prior terminal execution to fall back on -- just parse
            # whatever the running file itself has (likely sparse).
            return self._parse_execution_log(running_data)

        # Full prior history from the completed execution
        messages = self._parse_execution_log(previous)

        # Append the running execution's new-turn content only
        input_messages = running_data.get("input", {}).get("data", {}).get("messages", [])
        if input_messages:
            last_user = input_messages[-1]
            text = self._extract_text_from_content_blocks(last_user.get("content", []))
            if text.strip():
                messages.append(Message(role=Role.USER, text=text))

        for action in running_data.get("actions", []):
            messages.extend(self._parse_action(action))

        return messages

    _TERMINAL_STATUSES = ("succeed", "aborted", "yielded")

    def _find_previous_terminal_execution(
        self, session_id: str, before_start_time: float
    ) -> Optional[dict]:
        """Scan every execution log for `session_id`, and return the raw dict
        of the one with the highest `startTime` that is still <
        `before_start_time` and has a terminal `status`. This is the
        candidate a "fall back to the last completed execution" fix would
        use in place of a `running` execution with an empty/incomplete
        context.
        """
        root = self._storage_root
        if root is None or not root.is_dir():
            return None

        best: Optional[dict] = None
        best_start_time = -1.0
        for hash_dir in root.iterdir():
            if not hash_dir.is_dir() or len(hash_dir.name) != 32:
                continue
            exec_dir = hash_dir / _EXEC_LOGS_SUBDIR
            if not exec_dir.is_dir():
                continue
            for f in exec_dir.iterdir():
                if not f.is_file():
                    continue
                try:
                    data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
                except (OSError, json.JSONDecodeError):
                    continue
                if data.get("chatSessionId") != session_id:
                    continue
                if data.get("status") not in self._TERMINAL_STATUSES:
                    continue
                st = data.get("startTime", 0)
                if not st or st >= before_start_time:
                    continue
                if st > best_start_time:
                    best, best_start_time = data, st

        return best

    def _ensure_index(self) -> dict[str, tuple[float, Path]]:
        """Scan all execution logs and build a session -> latest execution index.

        Re-scans on every call rather than caching, because execution logs
        are written during/after each turn -- a cached index goes stale
        within the lifetime of a long-running MCP server process, causing
        chat_fork to return incomplete transcripts.

        Performance optimization: uses filesystem mtime to pre-filter and
        sort files before JSON-parsing them. Files older than
        MAX_INDEX_AGE_DAYS are excluded entirely, and remaining files are
        processed in most-recent-first order.
        """
        session_index: dict[str, tuple[float, Path]] = {}
        if self._storage_root is None or not self._storage_root.is_dir():
            return session_index

        cutoff = time.time() - (MAX_INDEX_AGE_DAYS * 86400)

        # Collect all candidate files with their filesystem mtime.
        candidates: list[tuple[float, Path]] = []
        for hash_dir in self._storage_root.iterdir():
            if not hash_dir.is_dir() or len(hash_dir.name) != 32:
                continue
            exec_dir = hash_dir / _EXEC_LOGS_SUBDIR
            if not exec_dir.is_dir():
                continue

            for exec_file in exec_dir.iterdir():
                if not exec_file.is_file():
                    continue
                try:
                    mtime = exec_file.stat().st_mtime
                except OSError:
                    continue
                if mtime < cutoff:
                    continue
                candidates.append((mtime, exec_file))

        # Sort by mtime descending (most recent first).
        candidates.sort(key=lambda item: item[0], reverse=True)

        for _mtime, exec_file in candidates:
            self._index_execution_file(session_index, exec_file)

        return session_index

    def _index_execution_file(self, session_index: dict[str, tuple[float, Path]], exec_file: Path) -> None:
        """Read just enough of an execution log to index it by session."""
        try:
            data = json.loads(exec_file.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return

        session_id = data.get("chatSessionId")
        start_time = data.get("startTime", 0)
        if not session_id or not start_time:
            return

        existing = session_index.get(session_id)
        if existing is None or start_time > existing[0]:
            session_index[session_id] = (start_time, exec_file)

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
            messages.extend(self._parse_action(action))

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
    def _parse_action(action: dict) -> list[Message]:
        """Parse one action from the actions array into Messages, if relevant."""
        action_type = action.get("actionType", "")
        output = action.get("output", {})

        if action_type == "say":
            text = output.get("message", "")
            if text.strip():
                return [Message(role=Role.ASSISTANT, text=text)]

        elif action_type == "mcp":
            results: list[Message] = []
            inp = action.get("input", {})
            tool_name = inp.get("toolName", "tool")
            args = inp.get("toolArgs", {})
            try:
                args_text = json.dumps(args, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                args_text = str(args)
            results.append(Message(role=Role.TOOL_CALL, text=args_text, label=tool_name))
            # Also emit the tool's response so that e.g. chat_checkpoint
            # output is visible to find_checkpoints() even during stitching.
            response = output.get("response", "") if isinstance(output, dict) else ""
            if response and response.strip():
                results.append(Message(role=Role.TOOL_RESULT, text=response))
            return results

        elif action_type == "runCommand":
            inp = action.get("input", {})
            command = inp.get("command", "")
            if command:
                return [Message(role=Role.TOOL_CALL, text=command, label="shell")]

        return []

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
