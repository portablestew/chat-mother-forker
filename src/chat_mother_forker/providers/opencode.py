"""Provider for OpenCode (https://opencode.ai).

OpenCode stores its history in a single SQLite database (not flat files).
Default location:

    ~/.local/share/opencode/opencode.db     (Windows, macOS, Linux)
    override via OPENCODE_DB env var, or XDG_DATA_HOME/opencode/opencode.db

The database has been through two storage schemas. OpenCode <= 1.x wrote the
legacy tables; OpenCode 2.x writes a newer pair of tables. The 2.x tables are
a strict superset -- a 2.x database mirrors the legacy sessions into them and
appends new sessions there too -- so this provider *prefers* them and only
falls back to the legacy tables when they are absent (an un-upgraded 1.x DB).

New schema (OpenCode 2.x):

    session_v2       - one row per session (id `ses_...`, project_id,
                       parent_id for subagent sessions, slug, directory,
                       title, tokens, cost, time_created/time_updated, ...)
    session_message  - one row per message (`type`, `seq` for ordering,
                       `data`=JSON); `type` is "user", "assistant", or
                       bookkeeping ("system", "synthetic", "compaction",
                       "idle", ...)

`session_message.data` shapes by `type`:
- "user"      -> `data.text` is the user's text.
- "assistant" -> `data.content` is an ordered list of parts; "text" parts are
                 assistant output, "tool" parts are a combined tool call +
                 result (`data.name` is the tool, `state.input` the arguments
                 -> TOOL_CALL, and either `state.content` (a list of text
                 parts) or `state.error.message` -> TOOL_RESULT). "reasoning"
                 parts are thinking and are skipped.
- anything else ("system", "synthetic", "compaction", "idle", ...) is
                 steering/step bookkeeping, not user-facing transcript, and
                 skipped.

Legacy schema (OpenCode 1.x), still read as a fallback:

    session   - one row per session (same identity/metadata columns)
    message   - one row per top-level message (`data.role` = "user"/"assistant")
    part      - one row per message part (`data.type` = "text"/"tool"/...);
                a "tool" part carries `data.tool` + `state.input` -> TOOL_CALL,
                and `state.output` -> TOOL_RESULT, on a single row.

`session_v2.directory` (or `session.directory`) holds the absolute workspace
path, whose basename becomes `Conversation.project`. `time_updated` (epoch
millis) drives recency. `parent_id` links subagent sessions (e.g. a `task`
tool child) to their parent; subagent sessions are listed like any other.

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

#: Newest storage schema: session metadata + one row per message.
_V2_SESSION_TABLE = "session_v2"
_V2_MESSAGE_TABLE = "session_message"
#: Legacy storage schema: session metadata + message/part rows.
_LEGACY_SESSION_TABLE = "session"
_LEGACY_MESSAGE_TABLE = "message"
_LEGACY_PART_TABLE = "part"


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


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?"
        " LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _detect_schema(conn: sqlite3.Connection) -> str:
    """Return "v2" when the 2.x tables are present, else "legacy".

    Both tables are required: the v2 session table without the v2 message
    table would be a partially-migrated/half-written database, so it is safer
    to fall back to the legacy tables (which a 2.x DB still mirrors) than to
    read an incomplete v2 view.
    """
    if _has_table(conn, _V2_SESSION_TABLE) and _has_table(conn, _V2_MESSAGE_TABLE):
        return "v2"
    return "legacy"


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
            table = (
                _V2_SESSION_TABLE
                if _detect_schema(conn) == "v2"
                else _LEGACY_SESSION_TABLE
            )
            sql = f"SELECT id, time_updated FROM {table}"
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
            schema = _detect_schema(conn)
            if schema == "v2":
                project, messages = self._load_v2(conn, session_id)
            else:
                project, messages = self._load_legacy(conn, session_id)
        finally:
            conn.close()

        messages = _trim_before_first_user(messages)
        return Conversation(ref=ref, messages=messages, project=project)

    def _load_v2(
        self, conn: sqlite3.Connection, session_id: str
    ) -> tuple[Optional[str], list[Message]]:
        row = conn.execute(
            f"SELECT directory FROM {_V2_SESSION_TABLE} WHERE id=?", (session_id,)
        ).fetchone()
        project = None
        if row and row["directory"]:
            project = basename_from_path(row["directory"])

        msg_rows = conn.execute(
            f"SELECT type, data FROM {_V2_MESSAGE_TABLE} WHERE session_id=?"
            " ORDER BY seq",
            (session_id,),
        ).fetchall()

        messages: list[Message] = []
        for mr in msg_rows:
            try:
                mdata = json.loads(mr["data"])
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(mdata, dict):
                messages.extend(self._to_v2_messages(mr["type"], mdata))
        return project, messages

    def _load_legacy(
        self, conn: sqlite3.Connection, session_id: str
    ) -> tuple[Optional[str], list[Message]]:
        row = conn.execute(
            f"SELECT directory FROM {_LEGACY_SESSION_TABLE} WHERE id=?", (session_id,)
        ).fetchone()
        project = None
        if row and row["directory"]:
            project = basename_from_path(row["directory"])

        msg_rows = conn.execute(
            f"SELECT id, data FROM {_LEGACY_MESSAGE_TABLE} WHERE session_id=?"
            " ORDER BY time_created, id",
            (session_id,),
        ).fetchall()
        part_rows = conn.execute(
            f"SELECT id, message_id, data FROM {_LEGACY_PART_TABLE} WHERE session_id=?"
            " ORDER BY time_created, id",
            (session_id,),
        ).fetchall()

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
        return project, messages

    def conversation_metadata(self, ref: ConversationRef) -> tuple[Optional[str], Optional[str]]:
        return str(self._db_path), ref.locator

    def conversation_size(self, ref: ConversationRef) -> int:
        """Best-effort byte size of the session's stored content.

        There's no single file to stat -- OpenCode keeps everything in one
        shared SQLite DB -- so this sums the byte length of the `data` JSON
        across the session's message rows (`session_message` on 2.x, or
        `message` + `part` on legacy). It won't equal a file-backed
        provider's on-disk size, but it's cheap (SQL aggregates, no row
        materialization or JSON parsing) and monotonic with conversation
        length, which is all a size threshold needs.
        """
        conn = _connect_readonly(self._db_path)
        if conn is None:
            return 0
        try:
            if _detect_schema(conn) == "v2":
                tables = (_V2_MESSAGE_TABLE,)
            else:
                tables = (_LEGACY_MESSAGE_TABLE, _LEGACY_PART_TABLE)
            total = 0
            for table in tables:
                # CAST to BLOB so LENGTH counts bytes, not characters -- a
                # multi-byte UTF-8 transcript would otherwise under-report.
                row = conn.execute(
                    f"SELECT COALESCE(SUM(LENGTH(CAST(data AS BLOB))), 0)"
                    f" FROM {table} WHERE session_id=?",
                    (ref.locator,),
                ).fetchone()
                if row and row[0]:
                    total += int(row[0])
            return total
        except sqlite3.Error:
            return 0
        finally:
            conn.close()

    @staticmethod
    def _to_v2_messages(msg_type: str, data: dict) -> list[Message]:
        """Convert one OpenCode 2.x `session_message` row into Messages.

        Only "user" and "assistant" rows are transcript content; every other
        `type` ("system", "synthetic", "compaction", "idle", ...) is
        steering/step bookkeeping and produces nothing.
        """
        if msg_type == "user":
            text = data.get("text", "")
            if isinstance(text, str) and text.strip():
                return [Message(role=Role.USER, text=text)]
            return []

        if msg_type == "assistant":
            results: list[Message] = []
            content = data.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        results.extend(OpenCodeProvider._to_v2_part(part))
            return results

        return []

    @staticmethod
    def _to_v2_part(part: dict) -> list[Message]:
        """Convert one element of a 2.x assistant message's `content` array."""
        part_type = part.get("type", "")

        if part_type == "text":
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                return [Message(role=Role.ASSISTANT, text=text)]
            return []

        if part_type == "tool":
            tool_name = part.get("name") or "tool"
            state = part.get("state") or {}
            results = [
                Message(
                    role=Role.TOOL_CALL,
                    text=_dumps(state.get("input", {})),
                    label=tool_name,
                )
            ]
            output = _v2_tool_output(state)
            if output is not None and output.strip():
                results.append(Message(role=Role.TOOL_RESULT, text=output))
            return results

        # "reasoning" (thinking) is intentionally dropped.
        return []

    @staticmethod
    def _to_messages(part: dict, role_str: Optional[str]) -> list[Message]:
        """Convert one legacy OpenCode part into zero or more Messages."""
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


def _v2_tool_output(state: dict) -> Optional[str]:
    """Extract the tool result text from a 2.x tool part's `state`.

    A completed tool stores its result as `state.content`, a list of parts
    (normally text). A failed tool stores `state.error.message` instead. A
    tool still running has neither -- there is nothing to show yet.
    """
    status = state.get("status")

    if status == "error":
        error = state.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
            return _dumps(error)
        if isinstance(error, str):
            return error
        return None

    content = state.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    if content is not None:
        return _dumps(content)
    return None
