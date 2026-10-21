"""Provider for Command Code (`cmdc` CLI).

Layout on disk (rooted at ~/.commandcode/projects by default):

    <projects_root>/<encoded-cwd>/<session-uuid>.jsonl
    <projects_root>/<encoded-cwd>/<session-uuid>.meta.json          (sidecar, not read)
    <projects_root>/<encoded-cwd>/<session-uuid>.checkpoints.jsonl  (sidecar, not read)

The `.jsonl` file is an append-only event log. Line 1 is a session header:

    {"type":"session","version":3,"id":"9daf97ff-...","timestamp":"2026-10-21T05:54:57.989Z","cwd":"C:\\\\Dev\\\\scratch\\\\cmdc-probe"}

`cwd` gives the workspace directory; its basename becomes `Conversation.project`.
A forked session copies the parent's `id` into the header and carries its own id
in `sessionId`, so conversations are identified by the filename stem instead.

Message records are linked by `id`/`parentId`:

    {"type":"message","id":"80099184","parentId":"ea78c826","timestamp":"2026-10-21T05:55:05.668Z","message":{"role":"user","content":[{"type":"text","text":"Reply with exactly the word PONG and nothing else."}],"meta":{"source":"user","createdAt":1792562098004,"messageId":"64802f05-..."}}}

Assistant messages may carry `text`, `thinking` and `tool_use` blocks:

    {"type":"message","id":"8b820e70","parentId":"41d5f53a","timestamp":"2026-10-21T05:55:13.293Z","message":{"role":"assistant","content":[{"type":"text","text":"I'll read the file now."},{"type":"tool_use","id":"call_01a15d0729e974599680b0cd","name":"read_file","input":{"file_path":"C:\\\\Dev\\\\scratch\\\\cmdc-probe\\\\notes.txt"}}],"meta":{"source":"model","createdAt":1792562113279,"messageId":"622b8b28-..."}},"usage":{"inputTokens":20185,"outputTokens":47,"cacheReadTokens":128,"cacheWriteTokens":0,"costUsd":0},"model":"minimax/minimax-m3.1-flash:free"}

A tool result is a `tool_result` block inside a record whose role is `user`
(its `meta.source` is `"tool"`):

    {"type":"message","id":"c3b32af1","parentId":"8b820e70","timestamp":"2026-10-21T05:55:13.293Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_01a15d0729e974599680b0cd","content":[{"type":"text","text":"Read 1/1 file, 1 line\\n1: probe-marker-42"}]}],"meta":{"source":"tool","messageId":"59e219ca-..."}}}

So a `role == "user"` record is not necessarily a user turn: a `tool_result`
block there maps to `Role.TOOL_RESULT`.

`session_info` records (name), `thinking` blocks, and the `.meta.json` and
`.checkpoints.jsonl` sidecars are not part of the conversation text and are ignored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role, basename_from_path
from chat_mother_forker.providers.base import ChatProvider

_JSONL_SUFFIX = ".jsonl"
_CHECKPOINTS_SUFFIX = ".checkpoints.jsonl"


def _default_projects_root() -> Path:
    return Path.home() / ".commandcode" / "projects"


def _trim_before_first_user(messages: list[Message]) -> list[Message]:
    """Drop any messages before the first user message."""
    for i, m in enumerate(messages):
        if m.role is Role.USER:
            return messages[i:]
    return messages


class CommandCodeProvider(ChatProvider):
    """Reads Command Code transcripts from `<projects_root>/<cwd>/<uuid>.jsonl`."""

    name = "command_code"

    def __init__(self, projects_root: Optional[Path] = None):
        self._projects_root = projects_root if projects_root is not None else _default_projects_root()

    def list_candidates(self) -> Iterable[ConversationRef]:
        if not self._projects_root.is_dir():
            return []

        refs = []
        for project_dir in self._projects_root.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl_path in project_dir.glob("*" + _JSONL_SUFFIX):
                # `*.jsonl` also matches the `.checkpoints.jsonl` sidecar.
                if jsonl_path.name.endswith(_CHECKPOINTS_SUFFIX):
                    continue
                try:
                    stat = jsonl_path.stat()
                except OSError:
                    continue
                if stat.st_size == 0:
                    continue

                refs.append(
                    ConversationRef(
                        provider=self.name,
                        conversation_id=jsonl_path.stem,
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
                if not isinstance(event, dict):
                    continue

                if project is None and event.get("type") == "session":
                    cwd = event.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        project = basename_from_path(cwd)

                messages.extend(self._to_messages(event))

        messages = _trim_before_first_user(messages)
        return Conversation(ref=ref, messages=messages, project=project)

    def conversation_size(self, ref: ConversationRef) -> int:
        """Size of the `.jsonl` transcript file (`ref.locator`)."""
        try:
            return os.path.getsize(ref.locator)
        except OSError:
            return 0

    @staticmethod
    def _to_messages(event: dict) -> list[Message]:
        """Convert one JSONL record into zero or more Message objects."""
        if event.get("type") != "message":
            return []

        msg = event.get("message")
        if not isinstance(msg, dict):
            return []

        timestamp = event.get("timestamp")
        role_str = msg.get("role")
        content = msg.get("content", "")
        is_user = role_str == "user"

        if isinstance(content, str):
            if content.strip():
                role = Role.USER if is_user else Role.ASSISTANT
                return [Message(role=role, text=content, timestamp=timestamp)]
            return []

        if not isinstance(content, list):
            return []

        results: list[Message] = []
        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            if block_type == "text":
                text = block.get("text", "")
                if text.strip():
                    role = Role.USER if is_user else Role.ASSISTANT
                    results.append(Message(role=role, text=text, timestamp=timestamp))

            elif block_type == "tool_use":
                tool_name = block.get("name") or "tool"
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
                    text = "\n".join(
                        part.get("text", "")
                        for part in result_content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                else:
                    text = json.dumps(result_content, default=str)
                if text.strip():
                    results.append(Message(role=Role.TOOL_RESULT, text=text, timestamp=timestamp))

        return results
