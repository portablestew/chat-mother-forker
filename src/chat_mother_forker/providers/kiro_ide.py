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
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from chat_mother_forker.models import Conversation, ConversationRef, Message, Role
from chat_mother_forker.providers.base import ChatProvider

# Well-known subdirectory name where execution logs live within each workspace hash dir.
_EXEC_LOGS_SUBDIR = "414d1636299d2b9e4ce7e17fb11f63e9"

# Matches the first fileTree entry Kiro IDE embeds in a "document" entry's
# staticDirectoryView text, e.g. "<folder name='c:\Dev\github\my-project\.git'
# closed />". The path's second-to-last segment is the workspace root name
# (the last segment is always some child of it, .git/.gitignore/etc.).
_FILETREE_ENTRY_RE = re.compile(r"<(?:folder|file) name='([^']+)'")

# Maximum age (in days) of execution log files to consider during indexing.
# Files with a filesystem mtime older than this are skipped entirely,
# avoiding expensive JSON parsing of stale data.
MAX_INDEX_AGE_DAYS = 30

# Execution logs can be multiple MB (the full conversation history/tool
# output lives in `context`/`actions`), but `chatSessionId`, `startTime`,
# and `status` all sit near the top of the JSON object, well before those
# large arrays. Rather than json.loads()-ing the whole file just to read
# three scalar fields, `_read_header_fields` reads in exponentially
# growing chunks and regex-matches for them directly, stopping as soon as
# everything needed has been found (or `_HEADER_READ_MAX_BYTES` is hit).
_HEADER_READ_INITIAL_CHUNK = 4096
_HEADER_READ_MAX_BYTES = 256 * 1024

_START_TIME_RE = re.compile(rb'"startTime"\s*:\s*(\d+)')
_SESSION_ID_RE = re.compile(rb'"chatSessionId"\s*:\s*"([^"]*)"')
_STATUS_RE = re.compile(rb'"status"\s*:\s*"([^"]*)"')


def _read_header_fields(
    path: Path, *, need_status: bool = False
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Cheaply extract `chatSessionId`, `startTime`, and (optionally)
    `status` from an execution log without fully parsing its JSON body.

    Reads the file in growing binary chunks and regex-searches each field
    directly out of the raw bytes, stopping as soon as every requested
    field has been found or the read cap is reached. Returns None for any
    field not found -- the same as what `dict.get()` would yield for a
    missing key after a full `json.loads()`, except this never pays the
    cost of parsing the (often huge) `context`/`actions` arrays that
    trail these fields in the file.

    Note this does not validate that the file is well-formed JSON; a
    caller that needs the full parsed object must still `json.loads()` it
    separately once it knows (cheaply) that this is the file it wants.
    """
    session_id: Optional[str] = None
    start_time: Optional[int] = None
    status: Optional[str] = None

    try:
        with path.open("rb") as f:
            buf = b""
            chunk_size = _HEADER_READ_INITIAL_CHUNK
            while len(buf) < _HEADER_READ_MAX_BYTES:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buf += chunk

                if start_time is None:
                    m = _START_TIME_RE.search(buf)
                    if m:
                        start_time = int(m.group(1))
                if session_id is None:
                    m = _SESSION_ID_RE.search(buf)
                    if m:
                        session_id = m.group(1).decode("utf-8", errors="replace")
                if need_status and status is None:
                    m = _STATUS_RE.search(buf)
                    if m:
                        status = m.group(1).decode("utf-8", errors="replace")

                if start_time is not None and session_id is not None and (
                    not need_status or status is not None
                ):
                    break
                chunk_size *= 2
    except OSError:
        return None, None, None

    return session_id, start_time, status


def _extract_project(exec_data: dict) -> Optional[str]:
    """Best-effort project/workspace name for an execution log.

    Kiro IDE doesn't store the workspace path as a discrete field anywhere
    in the execution log -- it only appears embedded in the
    ``staticDirectoryView`` text of a "document" tool-result entry (the
    rendered `<fileTree>` block shown to the model). Scan context.messages
    for the first such entry and pull the workspace root name out of its
    first listed path, e.g. "c:\\Dev\\github\\my-project\\.git" ->
    "my-project".
    """
    for ctx_msg in exec_data.get("context", {}).get("messages", []):
        for entry in ctx_msg.get("entries", []):
            if not isinstance(entry, dict) or entry.get("type") != "document":
                continue
            document = entry.get("document")
            if not isinstance(document, dict):
                continue
            sdv = document.get("staticDirectoryView", "")
            if not isinstance(sdv, str):
                continue
            match = _FILETREE_ENTRY_RE.search(sdv)
            if not match:
                continue
            parts = re.split(r"[\\/]+", match.group(1))
            if len(parts) >= 2 and parts[-2]:
                return parts[-2]
    return None


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

    def __init__(
        self,
        storage_root: Optional[Path] = None,
        *,
        max_sessions: Optional[int] = None,
        scan_multiplier: float = 2.0,
    ):
        """
        `max_sessions`, if given, lets `_ensure_index` stop scanning once it
        has collected `max_sessions * scan_multiplier` unique sessions,
        instead of always walking every execution log within
        `MAX_INDEX_AGE_DAYS`. Files are scanned in filesystem-mtime order
        (most recent first), so this is a safe cutoff for "give me the N
        most recent sessions" callers (e.g. `chat_search`/`chat_fork`,
        which cap to `CANDIDATES_PER_PROVIDER`) -- the multiplier exists
        because `_ensure_index` dedups by `chatSessionId` using each
        execution's own `startTime`, not the file's mtime, and those two
        orderings can occasionally disagree by a file or two for a given
        session; the extra margin makes it very unlikely a session that
        truly belongs in the top `max_sessions` gets missed just because
        one of its execution files happened to sort slightly differently
        by mtime than by startTime.

        Leave `max_sessions` as None (the default) to scan every session
        within the age window, matching `list_candidates()`'s documented
        contract of returning every conversation the provider can see.
        """
        self._storage_root = storage_root or _global_storage_root()
        self._max_sessions = max_sessions
        self._scan_multiplier = scan_multiplier

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
        project = _extract_project(data)

        # When the newest execution is still running, its context.messages
        # is typically empty (Kiro IDE backfills it only after the turn
        # completes). In that case, stitch together the full prior history
        # from the previous terminal-status execution with the running
        # file's own new-turn content (input.data.messages + actions).
        if (
            data.get("status") == "running"
            and not data.get("context", {}).get("messages")
        ):
            messages, stitched_project = self._stitch_running_execution(data, ref.conversation_id)
            if project is None:
                project = stitched_project

        messages = _trim_before_first_user(messages)
        return Conversation(ref=ref, messages=messages, project=project)

    def _stitch_running_execution(
        self, running_data: dict, session_id: str
    ) -> tuple[list[Message], Optional[str]]:
        """Combine the previous terminal execution's full history with the
        running execution's own new-turn content.

        The previous terminal execution's ``_parse_execution_log`` output
        contains all prior turns (its context.messages + its own
        input/actions). The running execution contributes only its
        ``input.data.messages`` (the new user prompt) and ``actions[]``
        (the agent's in-progress work), since its ``context.messages`` is
        empty at this point.

        Returns the stitched messages plus the project name recovered from
        the previous execution's context (the running file has none of its
        own to offer, since its context.messages is empty).
        """
        start_time = running_data.get("startTime", float("inf"))
        previous = self._find_previous_terminal_execution(session_id, start_time)

        if previous is None:
            # No prior terminal execution to fall back on -- just parse
            # whatever the running file itself has (likely sparse).
            return self._parse_execution_log(running_data), None

        # Full prior history from the completed execution
        messages = self._parse_execution_log(previous)
        project = _extract_project(previous)

        # Append the running execution's new-turn content only
        input_messages = running_data.get("input", {}).get("data", {}).get("messages", [])
        if input_messages:
            last_user = input_messages[-1]
            text = self._extract_text_from_content_blocks(last_user.get("content", []))
            if text.strip():
                messages.append(Message(role=Role.USER, text=text))

        for action in running_data.get("actions", []):
            messages.extend(self._parse_action(action))

        return messages, project

    _TERMINAL_STATUSES = ("succeed", "aborted", "yielded")

    def _find_previous_terminal_execution(
        self, session_id: str, before_start_time: float
    ) -> Optional[dict]:
        """Scan recent execution logs for `session_id`, and return the raw
        dict of the one with the highest `startTime` that is still <
        `before_start_time` and has a terminal `status`. This is the
        candidate a "fall back to the last completed execution" fix would
        use in place of a `running` execution with an empty/incomplete
        context.

        Uses the same `_gather_recent_files` pool (age-filtered, mtime-sorted)
        and `_scan_cap()` bound as `_ensure_index`, rather than walking the
        entire on-disk corpus -- a session actively producing a `running`
        execution is, by definition, one of the most recently touched
        sessions, so its previous terminal execution should also be recent
        and near the front of that same mtime-sorted pool. This does mean a
        session whose *only* prior terminal execution is older than
        `MAX_INDEX_AGE_DAYS`, or ranks beyond `_scan_cap()` files back, won't
        be found here -- `_stitch_running_execution` falls back to the
        running file's own (sparse) content in that case, same as when no
        prior execution exists at all.

        Each file is screened cheaply via `_read_header_fields` (header-only
        read) for session id, status, and startTime; only the one file that
        actually ends up winning gets a full `json.loads()`.
        """
        candidates = self._gather_recent_files()
        scan_cap = self._scan_cap()
        if scan_cap is not None:
            candidates = candidates[:scan_cap]

        best_path: Optional[Path] = None
        best_start_time = -1.0
        for _mtime, f in candidates:
            sid, st, status = _read_header_fields(f, need_status=True)
            if sid != session_id:
                continue
            if status not in self._TERMINAL_STATUSES:
                continue
            if not st or st >= before_start_time:
                continue
            if st > best_start_time:
                best_path, best_start_time = f, st

        if best_path is None:
            return None

        try:
            return json.loads(best_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return None

    def _scan_cap(self) -> Optional[int]:
        """The shared "how many candidates is enough" bound, derived from
        `self._max_sessions` (see `__init__`). None means unbounded --
        both `_ensure_index` and `_find_previous_terminal_execution` fall
        back to scanning every age-filtered candidate in that case.
        """
        if self._max_sessions is None:
            return None
        return int(self._max_sessions * self._scan_multiplier)

    def _gather_recent_files(self) -> list[tuple[float, Path]]:
        """Collect every execution log across all workspace-hash dirs whose
        filesystem mtime is within `MAX_INDEX_AGE_DAYS`, sorted by that
        mtime descending (most recent first).

        This is the one age/recency pre-filter shared by both
        `_ensure_index` (building the session index) and
        `_find_previous_terminal_execution` (the running-execution
        stitching fallback) -- both need "the same pool of recent
        candidate files", just processed differently, so the filtering
        lives here once rather than being duplicated (and potentially
        drifting) between them.
        """
        if self._storage_root is None or not self._storage_root.is_dir():
            return []

        cutoff = time.time() - (MAX_INDEX_AGE_DAYS * 86400)

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

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _ensure_index(self) -> dict[str, tuple[float, Path]]:
        """Scan recent execution logs and build a session -> latest execution index.

        Re-scans on every call rather than caching, because execution logs
        are written during/after each turn -- a cached index goes stale
        within the lifetime of a long-running MCP server process, causing
        chat_fork to return incomplete transcripts.

        Performance optimization: uses `_gather_recent_files` to pre-filter
        by age and sort by mtime before reading anything, and reads each
        file via `_read_header_fields` rather than a full JSON parse (see
        that function's docstring).

        If `self._max_sessions` is set, scanning stops early once
        `self._scan_cap()` unique sessions have been indexed -- see
        `__init__` for why the multiplier exists.
        """
        session_index: dict[str, tuple[float, Path]] = {}
        scan_cap = self._scan_cap()

        for _mtime, exec_file in self._gather_recent_files():
            self._index_execution_file(session_index, exec_file)
            if scan_cap is not None and len(session_index) >= scan_cap:
                break

        return session_index

    def _index_execution_file(self, session_index: dict[str, tuple[float, Path]], exec_file: Path) -> None:
        """Read just enough of an execution log to index it by session.

        Uses `_read_header_fields` instead of a full `json.loads()` --
        indexing only ever needs `chatSessionId`/`startTime`, both of
        which sit near the top of the file, so there's no reason to pay
        for parsing the full (often multi-MB) `context`/`actions` body.
        """
        session_id, start_time, _status = _read_header_fields(exec_file)
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
