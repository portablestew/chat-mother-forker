"""Provider for the new (v2) Kiro IDE chat storage format.

Newer Kiro IDE builds store sessions at:

    <KIRO_HOME>/sessions/<workspace-hash>/sess_<uuid>/
        session.json      -- metadata (title, workspacePaths, status, ...)
        messages.jsonl     -- append-only event log, one JSON object per line
        publish.cursor      -- internal sync/publish offset, not needed here

This supersedes the old execution-log format handled by `KiroIdeProvider`
(see `kiro_ide.py`) -- each session now lives in exactly one directory with
one append-only transcript file, instead of being scattered across many
per-turn execution-log files that had to be grouped by a `chatSessionId`
buried inside each file's JSON body. Concretely, this removes the need for:

- Cross-file session grouping/indexing (one dir == one session already).
- The "running execution has empty context, stitch in the prior terminal
  execution" workaround -- `messages.jsonl` is appended to as the turn
  progresses, so a conversation's own file already has everything, even
  mid-turn.
- Scraping the workspace path out of an embedded `<fileTree>` blob --
  `session.json` has an explicit `workspacePaths` field.

`messages.jsonl` line shape:

    {"id": "...", "timestamp": "...", "payload": {"type": "<kind>", ...}}

Relevant `payload.type` values:

- "user"       -> a user message; text in `payload.content`.
- "assistant"  -> an assistant message; text in `payload.content`.
- "tool_call"   -> a tool invocation; `payload.toolName` + `payload.args`.
- "tool_result" -> a tool's result, keyed by `payload.toolCallId` back to
                   the tool_call that produced it. `payload.content` is a
                   string that is one of:
                     - a JSON object with a "response" key (MCP tools,
                       e.g. `{"response": "...", "imageBase64Urls": []}`)
                       -- the "response" value is the tool's actual text.
                     - a JSON object with nothing useful (e.g. "{}", from
                       some built-in tools) -- no text to show.
                     - plain, non-JSON text (other built-in tools, e.g.
                       `read_file`'s rendered file content, or a
                       cancellation message) -- used as-is.

Everything else (`turn_start`, `turn_end`, `steering_inclusion`,
`session_metadata`, `usage_summary`, `session_event`,
`pending_interaction`, `interaction_resolved`, ...) is bookkeeping and is
skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role, basename_from_path
from chat_mother_forker.providers.base import ChatProvider

_MESSAGES_FILENAME = "messages.jsonl"
_SESSION_METADATA_FILENAME = "session.json"
_SESSION_DIR_PREFIX = "sess_"

# Directory name used for Kiro CLI sessions under the same sessions root --
# never a workspace-hash directory, so it's skipped during discovery.
_CLI_SESSIONS_DIRNAME = "cli"


def _kiro_home() -> Path:
    override = os.environ.get("KIRO_HOME")
    if override:
        return Path(override)
    return Path.home() / ".kiro"


class KiroIdeV2Provider(ChatProvider):
    name = "kiro_ide_v2"

    def __init__(self, kiro_home: Optional[Path] = None):
        self._kiro_home = kiro_home or _kiro_home()

    def _sessions_root(self) -> Path:
        return self._kiro_home / "sessions"

    def list_candidates(self) -> Iterable[ConversationRef]:
        sessions_root = self._sessions_root()
        if not sessions_root.is_dir():
            return []

        refs: list[ConversationRef] = []
        for workspace_dir in sessions_root.iterdir():
            if not workspace_dir.is_dir() or workspace_dir.name == _CLI_SESSIONS_DIRNAME:
                continue

            for session_dir in workspace_dir.iterdir():
                if not session_dir.is_dir() or not session_dir.name.startswith(_SESSION_DIR_PREFIX):
                    continue

                messages_path = session_dir / _MESSAGES_FILENAME
                try:
                    mtime = messages_path.stat().st_mtime
                except OSError:
                    continue

                conversation_id = session_dir.name[len(_SESSION_DIR_PREFIX):]
                refs.append(
                    ConversationRef(
                        provider=self.name,
                        conversation_id=conversation_id,
                        locator=str(session_dir),
                        mtime=mtime,
                    )
                )
        return refs

    def load(self, ref: ConversationRef) -> Conversation:
        session_dir = Path(ref.locator)
        messages = self._parse_messages(session_dir / _MESSAGES_FILENAME)
        project = self._load_project(session_dir / _SESSION_METADATA_FILENAME)
        return Conversation(ref=ref, messages=messages, project=project)

    @staticmethod
    def _load_project(session_metadata_path: Path) -> Optional[str]:
        """Best-effort project/workspace name from `session.json`'s
        `workspacePaths` field -- an explicit list of workspace root paths,
        unlike the old format which had none and had to scrape one out of
        an embedded file-tree blob.
        """
        try:
            meta = json.loads(session_metadata_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return None
        workspace_paths = meta.get("workspacePaths")
        if isinstance(workspace_paths, list) and workspace_paths:
            first = workspace_paths[0]
            if isinstance(first, str):
                return basename_from_path(first)
        return None

    def _parse_messages(self, messages_path: Path) -> list[Message]:
        messages: list[Message] = []
        try:
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
        except OSError:
            return []
        return messages

    @staticmethod
    def _to_message(event: dict) -> Optional[Message]:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return None
        payload_type = payload.get("type")
        timestamp = event.get("timestamp")

        if payload_type == "user":
            text = payload.get("content", "")
            if not isinstance(text, str) or not text.strip():
                return None
            return Message(role=Role.USER, text=text, timestamp=timestamp)

        if payload_type == "assistant":
            text = payload.get("content", "")
            if not isinstance(text, str) or not text.strip():
                return None
            return Message(role=Role.ASSISTANT, text=text, timestamp=timestamp)

        if payload_type == "tool_call":
            tool_name = payload.get("toolName") or "tool"
            args = payload.get("args")
            try:
                args_text = json.dumps(args, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                args_text = str(args)
            return Message(role=Role.TOOL_CALL, text=args_text, label=tool_name, timestamp=timestamp)

        if payload_type == "tool_result":
            text = _extract_tool_result_text(payload.get("content"))
            if not text.strip():
                return None
            return Message(role=Role.TOOL_RESULT, text=text, timestamp=timestamp)

        # Bookkeeping types (turn_start, turn_end, steering_inclusion,
        # session_metadata, usage_summary, session_event,
        # pending_interaction, interaction_resolved, ...) -- skip.
        return None


def _extract_tool_result_text(content) -> str:
    """Extract the human-relevant text out of a tool_result's raw `content`
    string.

    `content` is one of:
    - A JSON object with a "response" key (MCP tools) -- that value is the
      tool's actual text, e.g. the literal `chat_checkpoint` output line.
    - A JSON object with nothing useful in it (e.g. "{}") -- no text.
    - Plain, non-JSON text (most built-in tools) -- used as-is.
    """
    if not isinstance(content, str):
        return ""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    if isinstance(parsed, dict):
        response = parsed.get("response")
        if isinstance(response, str):
            return response
        return ""
    return content
