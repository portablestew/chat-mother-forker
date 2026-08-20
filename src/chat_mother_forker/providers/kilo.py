"""Provider for Kilo Code (CLI/VS Code extension).

Kilo Code stores its conversation history in a single SQLite database (not
flat JSON/JSONL files). Default location:

    ~/.local/share/kilo/kilo.db          (Windows, macOS, Linux)
    override via KILO_DB env var, or XDG_DATA_HOME/kilo/kilo.db on Linux

Three tables hold the conversation content:

    session   - one row per session (id, title, directory, model, tokens,
                time_created/time_updated, ...)
    message   - one row per top-level message (id, session_id, data=JSON);
                `data.role` is "user" or "assistant"
    part      - one row per message part (id, message_id, session_id,
                data=JSON); `data.type` discriminates the part shape

`part.data` types and their mapping:
- "text"      -> USER/ASSISTANT text (role taken from the parent message's
                `data.role`); `data.text`
- "tool"      -> a combined tool call + result: `data.tool` is the tool name,
                `data.state.input` the arguments (-> TOOL_CALL), and
                `data.state.output` the result (-> TOOL_RESULT)
- "step-start", "step-finish", "patch", "reasoning" -> skipped (metadata /
  internal thinking / file-change info, not user-facing transcript)

`session.directory` holds the absolute workspace path, whose basename becomes
`Conversation.project`. `session.time_updated` (epoch millis) drives recency.

A read-only connection is opened per call (``mode=ro`` URI) so a running
Kilo Code instance can keep writing the database while history is read.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role, basename_from_path
from chat_mother_forker.providers.base import ChatProvider


def _kilo_db_path() -> Path:
    override = os.environ.get("KILO_DB")
    if override:
        return Path(override)
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home) / "kilo" / "kilo.db"
    return Path.home() / ".local" / "share" / "kilo" / "kilo.db"


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


class KiloProvider(ChatProvider):
    name = "kilo"

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _kilo_db_path()

    def list_candidates(self) -> Iterable[ConversationRef]:
        conn = _connect_readonly(self._db_path)
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT id, time_updated FROM session ORDER BY time_updated DESC"
            ).fetchall()
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

            rows = conn.execute(
                """
                SELECT m.data AS mdata, p.data AS pdata
                FROM message m
                LEFT JOIN part p ON p.message_id = m.id
                WHERE m.session_id = ?
                ORDER BY m.time_created, p.time_created, p.id
                """,
                (session_id,),
            ).fetchall()
        finally:
            conn.close()

        messages: list[Message] = []
        for r in rows:
            mdata = r["mdata"]
            role_str = None
            if mdata:
                try:
                    role_str = json.loads(mdata).get("role")
                except (json.JSONDecodeError, ValueError):
                    role_str = None

            pdata = r["pdata"]
            if not pdata:
                continue
            try:
                part = json.loads(pdata)
            except (json.JSONDecodeError, ValueError):
                continue
            messages.extend(self._to_messages(part, role_str))

        messages = _trim_before_first_user(messages)
        return Conversation(ref=ref, messages=messages, project=project)

    @staticmethod
    def _to_messages(part: dict, role_str: Optional[str]) -> list[Message]:
        """Convert one Kilo part into zero or more normalized Messages."""
        if not isinstance(part, dict):
            return []

        part_type = part.get("type", "")
        results: list[Message] = []

        if part_type == "text":
            text = part.get("text", "")
            if text.strip():
                role = Role.USER if role_str == "user" else Role.ASSISTANT
                results.append(Message(role=role, text=text))

        elif part_type == "tool":
            tool_name = part.get("tool", "tool")
            state = part.get("state") or {}
            tool_input = state.get("input", {})
            try:
                args_text = json.dumps(tool_input, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                args_text = str(tool_input)
            results.append(
                Message(role=Role.TOOL_CALL, text=args_text, label=tool_name)
            )

            output = state.get("output")
            if isinstance(output, str) and output.strip():
                results.append(Message(role=Role.TOOL_RESULT, text=output))

        return results