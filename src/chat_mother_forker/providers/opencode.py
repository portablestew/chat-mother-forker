"""Provider for OpenCode (https://opencode.ai).

OpenCode stores its history in a single SQLite database (not flat files).
Default location:

    ~/.local/share/opencode/opencode.db     (Windows, macOS, Linux)
    override via OPENCODE_DB env var, or XDG_DATA_HOME/opencode/opencode.db

Three tables hold the conversation content (alongside an `event` log we
never read -- the normalized tables are the source of truth for history):

    session   - one row per session (id `ses_...`, project_id, parent_id for
                subagent sessions, slug, directory, title, tokens, cost,
                time_created/time_updated, ...)
    message   - one row per top-level message (id `msg_...`, session_id,
                data=JSON); `data.role` is "user" or "assistant"
    part      - one row per message part (id `prt_...`, message_id,
                session_id, data=JSON); `data.type` discriminates the part shape

`part.data` types and their mapping:
- "text"        -> USER/ASSISTANT text (role taken from the parent message's
                   `data.role`); `data.text`
- "tool"        -> a combined tool call + result: `data.tool` is the tool name,
                   `data.state.input` the arguments (-> TOOL_CALL), and
                   `data.state.output` the result (-> TOOL_RESULT). Input and
                   output live on the same part (there is no separate
                   result row); a checkpoint line, when present, is in
                   `state.output`.
- "step-start", "step-finish", "reasoning" -> skipped (internal step
                   bookkeeping / thinking, not user-facing transcript)

`session.directory` holds the absolute workspace path, whose basename becomes
`Conversation.project`. `session.time_updated` (epoch millis) drives recency.
`session.parent_id` links subagent sessions (e.g. a `task` tool child) to
their parent; subagent sessions are listed like any other session.

A read-only connection is opened per call (``mode=ro`` URI) so a running
OpenCode instance can keep writing the database while history is read.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import (
    Conversation,
    ConversationRef,
    Message,
    Role,
    basename_from_path,
)
from chat_mother_forker.providers.base import ChatProvider


def _opencode_db_path() -> Path:
    override = os.environ.get("OPENCODE_DB")
    if override:
        return Path(override)
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home) / "opencode" / "opencode.db"
    return Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def _connect_readonly(db_path: Path) -> Optional[sqlite3.Connection]:
    """Open a read-only connection, or None when the DB doesn't exist yet."""
    if not db_path.is_file():
        return None
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _trim_before_first_user(messages: list[Message]) -> list[Message]:
    """Drop any messages before the first user message."""
    for i, m in enumerate(messages):
        if m.role is Role.USER:
            return messages[i:]
    return messages


def _dumps(value: object) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


class OpenCodeProvider(ChatProvider):
    name = "opencode"

    def __init__(self, db_path: Optional[Path] = None, *, max_sessions: Optional[int] = None):
        """
        `max_sessions`, if given, pushes the recency cut into SQL
        (``ORDER BY time_updated DESC LIMIT ...``) instead of materializing
        every session row. Safe (unlike Kiro IDE's scan window) because the
        SQL ordering key is the same one callers sort on, so the top-N is
        exact. Leave as None to return every session, matching
        `list_candidates()`'s documented contract.
        """
        self._db_path = db_path or _opencode_db_path()
        self._max_sessions = max_sessions

    def list_candidates(self) -> Iterable[ConversationRef]:
        conn = _connect_readonly(self._db_path)
        if conn is None:
            return []
        try:
            sql = "SELECT id, time_updated FROM session"
            params: tuple = ()
            if self._max_sessions is not None:
                sql += " ORDER BY time_updated DESC LIMIT ?"
                params = (self._max_sessions,)
            else:
                sql += " ORDER BY time_updated DESC"
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        refs = []
        for row in rows:
            refs.append(
                ConversationRef(
                    provider=self.name,
                    conversation_id=row["id"],
                    locator=row["id"],
                    # time_updated is epoch millis; mtime is epoch seconds.
                    mtime=row["time_updated"] / 1000.0,
                )
            )
        return refs

    def load(self, ref: ConversationRef) -> Conversation:
        session_id = ref.locator
        conn = _connect_readonly(self._db_path)
        if conn is None:
            return Conversation(ref=ref, messages=[])

        try:
            row = conn.execute(
                "SELECT directory FROM session WHERE id=?", (session_id,)
            ).fetchone()
            project = None
            if row and row["directory"]:
                project = basename_from_path(row["directory"])

            msg_rows = conn.execute(
                "SELECT id, data FROM message WHERE session_id=?"
                " ORDER BY time_created, id",
                (session_id,),
            ).fetchall()
            part_rows = conn.execute(
                "SELECT id, message_id, data FROM part WHERE session_id=?"
                " ORDER BY time_created, id",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()

        parts_by_message: dict[str, list[dict]] = {}
        for pr in part_rows:
            try:
                part = json.loads(pr["data"])
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(part, dict):
                parts_by_message.setdefault(pr["message_id"], []).append(part)

        messages: list[Message] = []
        for mr in msg_rows:
            try:
                mdata = json.loads(mr["data"])
            except (json.JSONDecodeError, ValueError):
                continue
            role_str = mdata.get("role") if isinstance(mdata, dict) else None
            for part in parts_by_message.get(mr["id"], ()):
                messages.extend(self._to_messages(part, role_str))

        messages = _trim_before_first_user(messages)
        return Conversation(ref=ref, messages=messages, project=project)

    def conversation_metadata(self, ref: ConversationRef) -> tuple[Optional[str], Optional[str]]:
        return str(self._db_path), ref.locator

    @staticmethod
    def _to_messages(part: dict, role_str: Optional[str]) -> list[Message]:
        """Convert one OpenCode part into zero or more normalized Messages."""
        part_type = part.get("type", "")
        results: list[Message] = []

        if part_type == "text":
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                role = Role.USER if role_str == "user" else Role.ASSISTANT
                results.append(Message(role=role, text=text))

        elif part_type == "tool":
            tool_name = part.get("tool", "tool")
            state = part.get("state") or {}
            tool_input = state.get("input", {})
            results.append(
                Message(role=Role.TOOL_CALL, text=_dumps(tool_input), label=tool_name)
            )

            output = state.get("output")
            if isinstance(output, str):
                if output.strip():
                    results.append(Message(role=Role.TOOL_RESULT, text=output))
            elif output is not None:
                results.append(Message(role=Role.TOOL_RESULT, text=_dumps(output)))

        return results
