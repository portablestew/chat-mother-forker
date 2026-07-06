"""Tests for the Kiro IDE provider.

Exercises the meaty logic: execution-log indexing, session deduplication,
parsing of context.messages / input / actions, stitching of running
executions, and the _trim_before_first_user helper.
"""

import json
import os
from pathlib import Path

import pytest

from chat_mother_forker.models import Role
from chat_mother_forker.providers.kiro_ide import (
    KiroIdeProvider,
    _global_storage_root,
    _trim_before_first_user,
    _EXEC_LOGS_SUBDIR,
)
from chat_mother_forker.models import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workspace_hash(storage_root: Path, workspace_hash: str = "a" * 32) -> Path:
    """Create the workspace-hash/exec-logs directory structure."""
    exec_dir = storage_root / workspace_hash / _EXEC_LOGS_SUBDIR
    exec_dir.mkdir(parents=True, exist_ok=True)
    return exec_dir


def _write_execution(exec_dir: Path, filename: str, data: dict) -> Path:
    """Write an execution log JSON file."""
    path = exec_dir / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_execution(
    session_id: str = "session-1",
    start_time: int = 1700000000000,
    status: str = "succeed",
    context_messages=None,
    input_messages=None,
    actions=None,
) -> dict:
    """Build a minimal execution log dict with sensible defaults."""
    return {
        "chatSessionId": session_id,
        "startTime": start_time,
        "status": status,
        "context": {"messages": context_messages or []},
        "input": {"data": {"messages": input_messages or []}},
        "actions": actions or [],
    }


# ---------------------------------------------------------------------------
# _trim_before_first_user
# ---------------------------------------------------------------------------


class TestTrimBeforeFirstUser:
    def test_drops_leading_non_user_messages(self):
        msgs = [
            Message(role=Role.ASSISTANT, text="system ack"),
            Message(role=Role.TOOL_CALL, text="file tree", label="readFiles"),
            Message(role=Role.USER, text="hello"),
            Message(role=Role.ASSISTANT, text="hi back"),
        ]
        result = _trim_before_first_user(msgs)
        assert len(result) == 2
        assert result[0].role is Role.USER
        assert result[0].text == "hello"

    def test_returns_all_when_no_user_message(self):
        """When no user message exists, all messages are preserved as-is."""
        msgs = [
            Message(role=Role.ASSISTANT, text="only bots here"),
            Message(role=Role.TOOL_RESULT, text="some tool output"),
        ]
        assert _trim_before_first_user(msgs) == msgs

    def test_preserves_all_when_first_is_user(self):
        msgs = [
            Message(role=Role.USER, text="first"),
            Message(role=Role.ASSISTANT, text="second"),
        ]
        assert _trim_before_first_user(msgs) == msgs


# ---------------------------------------------------------------------------
# _global_storage_root
# ---------------------------------------------------------------------------


class TestGlobalStorageRoot:
    def test_respects_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KIRO_IDE_STORAGE", str(tmp_path / "custom"))
        assert _global_storage_root() == tmp_path / "custom"

    def test_returns_platform_specific_path_when_no_override(self, monkeypatch):
        monkeypatch.delenv("KIRO_IDE_STORAGE", raising=False)
        result = _global_storage_root()
        # Should return a Path ending with kiro.kiroagent on all platforms
        assert result is not None
        assert result.name == "kiro.kiroagent"


# ---------------------------------------------------------------------------
# Indexing and list_candidates
# ---------------------------------------------------------------------------


class TestListCandidates:
    def test_returns_empty_when_storage_root_missing(self, tmp_path):
        provider = KiroIdeProvider(storage_root=tmp_path / "nonexistent")
        assert list(provider.list_candidates()) == []

    def test_returns_empty_when_no_workspace_hash_dirs(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        provider = KiroIdeProvider(storage_root=tmp_path)
        assert list(provider.list_candidates()) == []

    def test_skips_dirs_with_wrong_hash_length(self, tmp_path):
        # Create a dir with wrong name length (not 32 chars)
        bad_dir = tmp_path / "short" / _EXEC_LOGS_SUBDIR
        bad_dir.mkdir(parents=True)
        _write_execution(bad_dir, "exec1.json", _minimal_execution())

        provider = KiroIdeProvider(storage_root=tmp_path)
        assert list(provider.list_candidates()) == []

    def test_discovers_single_session(self, tmp_path):
        exec_dir = _make_workspace_hash(tmp_path)
        _write_execution(exec_dir, "exec1.json", _minimal_execution(
            session_id="sess-abc", start_time=1700000000000
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        refs = list(provider.list_candidates())

        assert len(refs) == 1
        assert refs[0].conversation_id == "sess-abc"
        assert refs[0].provider == "kiro_ide"
        assert refs[0].mtime == 1700000000.0

    def test_deduplicates_sessions_keeping_latest_execution(self, tmp_path):
        """Multiple executions for the same session -> only the latest is returned."""
        exec_dir = _make_workspace_hash(tmp_path)
        _write_execution(exec_dir, "exec_old.json", _minimal_execution(
            session_id="sess-1", start_time=1000
        ))
        latest_path = _write_execution(exec_dir, "exec_new.json", _minimal_execution(
            session_id="sess-1", start_time=5000
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        refs = list(provider.list_candidates())

        assert len(refs) == 1
        assert refs[0].locator == str(latest_path)
        assert refs[0].mtime == 5.0  # 5000ms -> 5s

    def test_multiple_sessions_across_workspaces(self, tmp_path):
        """Sessions from different workspace-hash dirs are all discovered."""
        exec_dir_1 = _make_workspace_hash(tmp_path, "a" * 32)
        exec_dir_2 = _make_workspace_hash(tmp_path, "b" * 32)
        _write_execution(exec_dir_1, "e1.json", _minimal_execution(
            session_id="sess-from-ws1", start_time=2000
        ))
        _write_execution(exec_dir_2, "e2.json", _minimal_execution(
            session_id="sess-from-ws2", start_time=3000
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        refs = list(provider.list_candidates())
        ids = {r.conversation_id for r in refs}

        assert ids == {"sess-from-ws1", "sess-from-ws2"}

    def test_skips_malformed_json_files(self, tmp_path):
        exec_dir = _make_workspace_hash(tmp_path)
        # Write valid execution
        _write_execution(exec_dir, "good.json", _minimal_execution(session_id="good"))
        # Write garbage
        (exec_dir / "bad.json").write_text("{not valid json", encoding="utf-8")

        provider = KiroIdeProvider(storage_root=tmp_path)
        refs = list(provider.list_candidates())

        assert len(refs) == 1
        assert refs[0].conversation_id == "good"

    def test_skips_executions_missing_session_id_or_start_time(self, tmp_path):
        exec_dir = _make_workspace_hash(tmp_path)
        _write_execution(exec_dir, "no_session.json", {
            "startTime": 1000, "status": "succeed",
            "context": {"messages": []}, "input": {"data": {"messages": []}}, "actions": [],
        })
        _write_execution(exec_dir, "no_time.json", {
            "chatSessionId": "sess-x", "status": "succeed",
            "context": {"messages": []}, "input": {"data": {"messages": []}}, "actions": [],
        })

        provider = KiroIdeProvider(storage_root=tmp_path)
        assert list(provider.list_candidates()) == []


# ---------------------------------------------------------------------------
# Parsing: _parse_execution_log via load()
# ---------------------------------------------------------------------------


class TestLoadParsing:
    def _load_single(self, tmp_path, exec_data):
        """Write one execution log and load it."""
        exec_dir = _make_workspace_hash(tmp_path)
        path = _write_execution(exec_dir, "exec.json", exec_data)
        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = next(iter(provider.list_candidates()))
        return provider.load(ref)

    def test_parses_context_text_messages(self, tmp_path):
        """context.messages with role=human/bot text entries become USER/ASSISTANT."""
        data = _minimal_execution(context_messages=[
            {"role": "human", "entries": [{"type": "text", "text": "What is this?"}]},
            {"role": "bot", "entries": [{"type": "text", "text": "It's a project."}]},
        ])
        conv = self._load_single(tmp_path, data)

        assert len(conv.messages) == 2
        assert conv.messages[0].role is Role.USER
        assert conv.messages[0].text == "What is this?"
        assert conv.messages[1].role is Role.ASSISTANT
        assert conv.messages[1].text == "It's a project."

    def test_parses_context_tool_use_entries(self, tmp_path):
        """toolUse entries in context.messages become TOOL_CALL messages."""
        data = _minimal_execution(context_messages=[
            {"role": "bot", "entries": [
                {"type": "toolUse", "name": "grep_search", "args": {"query": "hello", "limit": 5}},
            ]},
        ])
        conv = self._load_single(tmp_path, data)

        assert len(conv.messages) == 1
        msg = conv.messages[0]
        assert msg.role is Role.TOOL_CALL
        assert msg.label == "grep_search"
        assert json.loads(msg.text) == {"query": "hello", "limit": 5}

    def test_parses_context_tool_use_response_entries(self, tmp_path):
        """toolUseResponse entries become TOOL_RESULT messages."""
        data = _minimal_execution(context_messages=[
            {"role": "bot", "entries": [
                {"type": "toolUseResponse", "message": "Found 3 matches"},
            ]},
        ])
        conv = self._load_single(tmp_path, data)

        assert len(conv.messages) == 1
        assert conv.messages[0].role is Role.TOOL_RESULT
        assert conv.messages[0].text == "Found 3 matches"

    def test_parses_user_prompt_from_input_data_messages(self, tmp_path):
        """The last entry in input.data.messages is the new user prompt."""
        data = _minimal_execution(input_messages=[
            {"content": [{"type": "text", "text": "previous context"}]},
            {"content": [{"type": "text", "text": "my actual question"}]},
        ])
        conv = self._load_single(tmp_path, data)

        # Only the last message from input is used as the new prompt
        assert any(m.text == "my actual question" and m.role is Role.USER for m in conv.messages)

    def test_parses_say_action(self, tmp_path):
        """A say action becomes an ASSISTANT message."""
        data = _minimal_execution(actions=[
            {"actionType": "say", "output": {"message": "Here's my answer."}},
        ])
        conv = self._load_single(tmp_path, data)

        assert len(conv.messages) == 1
        assert conv.messages[0].role is Role.ASSISTANT
        assert conv.messages[0].text == "Here's my answer."

    def test_parses_mcp_action_with_tool_call_and_response(self, tmp_path):
        """An mcp action produces both TOOL_CALL and TOOL_RESULT messages."""
        data = _minimal_execution(actions=[
            {
                "actionType": "mcp",
                "input": {"toolName": "chat_search", "toolArgs": {"query": "auth"}},
                "output": {"response": "Found 2 conversations"},
            },
        ])
        conv = self._load_single(tmp_path, data)

        assert len(conv.messages) == 2
        assert conv.messages[0].role is Role.TOOL_CALL
        assert conv.messages[0].label == "chat_search"
        assert json.loads(conv.messages[0].text) == {"query": "auth"}
        assert conv.messages[1].role is Role.TOOL_RESULT
        assert conv.messages[1].text == "Found 2 conversations"

    def test_parses_run_command_action(self, tmp_path):
        """A runCommand action becomes a TOOL_CALL with label 'shell'."""
        data = _minimal_execution(actions=[
            {"actionType": "runCommand", "input": {"command": "npm run test"}, "output": {}},
        ])
        conv = self._load_single(tmp_path, data)

        assert len(conv.messages) == 1
        assert conv.messages[0].role is Role.TOOL_CALL
        assert conv.messages[0].label == "shell"
        assert conv.messages[0].text == "npm run test"

    def test_skips_system_prompt_context_messages(self, tmp_path):
        """Text entries starting with <key_kiro_features> or <EnvironmentContext> are skipped."""
        data = _minimal_execution(context_messages=[
            {"role": "human", "entries": [
                {"type": "text", "text": "<key_kiro_features>\nstuff\n</key_kiro_features>"},
            ]},
            {"role": "human", "entries": [
                {"type": "text", "text": "<EnvironmentContext>\nmore stuff\n</EnvironmentContext>"},
            ]},
            {"role": "human", "entries": [{"type": "text", "text": "real question"}]},
        ])
        conv = self._load_single(tmp_path, data)

        assert len(conv.messages) == 1
        assert conv.messages[0].text == "real question"

    def test_skips_blank_text_entries(self, tmp_path):
        """Whitespace-only text entries are dropped."""
        data = _minimal_execution(context_messages=[
            {"role": "bot", "entries": [{"type": "text", "text": "   "}]},
            {"role": "human", "entries": [{"type": "text", "text": "real"}]},
        ])
        conv = self._load_single(tmp_path, data)

        assert len(conv.messages) == 1
        assert conv.messages[0].text == "real"

    def test_skips_blank_tool_use_responses(self, tmp_path):
        """toolUseResponse entries with only whitespace are dropped."""
        data = _minimal_execution(context_messages=[
            {"role": "bot", "entries": [{"type": "toolUseResponse", "message": "  \n "}]},
        ])
        conv = self._load_single(tmp_path, data)
        assert conv.messages == []

    def test_handles_empty_mcp_response(self, tmp_path):
        """An mcp action with no response text only produces a TOOL_CALL."""
        data = _minimal_execution(actions=[
            {
                "actionType": "mcp",
                "input": {"toolName": "read_file", "toolArgs": {"path": "foo.py"}},
                "output": {"response": ""},
            },
        ])
        conv = self._load_single(tmp_path, data)

        assert len(conv.messages) == 1
        assert conv.messages[0].role is Role.TOOL_CALL

    def test_full_multi_turn_conversation(self, tmp_path):
        """A realistic multi-turn execution log parses into the expected sequence."""
        data = _minimal_execution(
            context_messages=[
                {"role": "human", "entries": [{"type": "text", "text": "first question"}]},
                {"role": "bot", "entries": [
                    {"type": "text", "text": "let me check"},
                    {"type": "toolUse", "name": "read_file", "args": {"path": "README.md"}},
                ]},
                {"role": "bot", "entries": [
                    {"type": "toolUseResponse", "message": "# Title\nContent"},
                ]},
                {"role": "bot", "entries": [{"type": "text", "text": "here's what I found"}]},
            ],
            input_messages=[
                {"content": [{"type": "text", "text": "follow-up question"}]},
            ],
            actions=[
                {"actionType": "say", "output": {"message": "Sure, let me help."}},
                {
                    "actionType": "mcp",
                    "input": {"toolName": "grep_search", "toolArgs": {"query": "test"}},
                    "output": {"response": "2 matches"},
                },
                {"actionType": "say", "output": {"message": "Done!"}},
            ],
        )
        conv = self._load_single(tmp_path, data)

        roles = [m.role for m in conv.messages]
        expected = [
            Role.USER,        # first question (context)
            Role.ASSISTANT,   # let me check (context)
            Role.TOOL_CALL,   # read_file (context)
            Role.TOOL_RESULT, # file content (context)
            Role.ASSISTANT,   # here's what I found (context)
            Role.USER,        # follow-up question (input)
            Role.ASSISTANT,   # Sure, let me help (action: say)
            Role.TOOL_CALL,   # grep_search (action: mcp)
            Role.TOOL_RESULT, # 2 matches (action: mcp response)
            Role.ASSISTANT,   # Done! (action: say)
        ]
        assert roles == expected

    def test_extract_text_skips_environment_context_blocks(self, tmp_path):
        """Content blocks with <EnvironmentContext> are filtered out of input messages."""
        data = _minimal_execution(input_messages=[
            {"content": [
                {"type": "text", "text": "<EnvironmentContext>\nstuff\n</EnvironmentContext>"},
                {"type": "text", "text": "actual prompt"},
            ]},
        ])
        conv = self._load_single(tmp_path, data)

        user_msgs = [m for m in conv.messages if m.role is Role.USER]
        assert len(user_msgs) == 1
        assert "EnvironmentContext" not in user_msgs[0].text
        assert "actual prompt" in user_msgs[0].text

    def test_extract_text_handles_string_content(self, tmp_path):
        """input.data.messages content can be a plain string instead of list."""
        data = _minimal_execution(input_messages=[
            {"content": "plain string prompt"},
        ])
        conv = self._load_single(tmp_path, data)

        user_msgs = [m for m in conv.messages if m.role is Role.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].text == "plain string prompt"

    def test_load_returns_empty_on_unreadable_file(self, tmp_path):
        """If the locator points to a file that can't be parsed, return empty."""
        exec_dir = _make_workspace_hash(tmp_path)
        bad_path = exec_dir / "corrupt.json"
        bad_path.write_text("not json at all", encoding="utf-8")

        from chat_mother_forker.models import ConversationRef
        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = ConversationRef(
            provider="kiro_ide",
            conversation_id="fake",
            locator=str(bad_path),
            mtime=0.0,
        )
        conv = provider.load(ref)
        assert conv.messages == []


# ---------------------------------------------------------------------------
# Stitching: running execution with empty context
# ---------------------------------------------------------------------------


class TestStitchRunningExecution:
    def test_stitches_prior_terminal_execution_with_running(self, tmp_path):
        """When latest execution is 'running' with empty context, prior terminal
        execution's full history is combined with the running one's new content."""
        exec_dir = _make_workspace_hash(tmp_path)

        # Prior completed execution with full history
        _write_execution(exec_dir, "exec_done.json", _minimal_execution(
            session_id="sess-1",
            start_time=1000,
            status="succeed",
            context_messages=[
                {"role": "human", "entries": [{"type": "text", "text": "earlier question"}]},
                {"role": "bot", "entries": [{"type": "text", "text": "earlier answer"}]},
            ],
            input_messages=[
                {"content": [{"type": "text", "text": "second question"}]},
            ],
            actions=[
                {"actionType": "say", "output": {"message": "second answer"}},
            ],
        ))

        # Running execution with empty context (Kiro hasn't backfilled yet)
        running_path = _write_execution(exec_dir, "exec_running.json", _minimal_execution(
            session_id="sess-1",
            start_time=2000,
            status="running",
            context_messages=[],  # empty - not yet backfilled
            input_messages=[
                {"content": [{"type": "text", "text": "third question"}]},
            ],
            actions=[
                {"actionType": "say", "output": {"message": "working on it..."}},
            ],
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        refs = list(provider.list_candidates())
        # Should pick the running one (latest startTime)
        assert len(refs) == 1
        assert refs[0].locator == str(running_path)

        conv = provider.load(refs[0])
        texts = [m.text for m in conv.messages]

        # Should have prior history stitched in
        assert "earlier question" in texts
        assert "earlier answer" in texts
        assert "second question" in texts
        assert "second answer" in texts
        # Plus the running execution's own new content
        assert "third question" in texts
        assert "working on it..." in texts

    def test_stitch_falls_back_to_running_data_when_no_prior(self, tmp_path):
        """When no prior terminal execution exists, use the running file's own data."""
        exec_dir = _make_workspace_hash(tmp_path)

        _write_execution(exec_dir, "exec_running.json", _minimal_execution(
            session_id="sess-new",
            start_time=1000,
            status="running",
            context_messages=[],
            input_messages=[
                {"content": [{"type": "text", "text": "very first prompt"}]},
            ],
            actions=[
                {"actionType": "say", "output": {"message": "starting work"}},
            ],
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)

        # Should still get the input and actions even without prior execution
        texts = [m.text for m in conv.messages]
        assert "very first prompt" in texts
        assert "starting work" in texts

    def test_stitch_ignores_non_terminal_prior_executions(self, tmp_path):
        """Only terminal statuses (succeed, aborted, yielded) are used for stitching."""
        exec_dir = _make_workspace_hash(tmp_path)

        # Another running execution (not terminal, should be skipped)
        _write_execution(exec_dir, "also_running.json", _minimal_execution(
            session_id="sess-1",
            start_time=500,
            status="running",
            context_messages=[
                {"role": "human", "entries": [{"type": "text", "text": "should be ignored"}]},
            ],
        ))

        # The actual running execution we're loading
        _write_execution(exec_dir, "current_running.json", _minimal_execution(
            session_id="sess-1",
            start_time=1000,
            status="running",
            context_messages=[],
            input_messages=[{"content": [{"type": "text", "text": "new prompt"}]}],
            actions=[],
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        refs = list(provider.list_candidates())
        assert len(refs) == 1

        conv = provider.load(refs[0])
        texts = [m.text for m in conv.messages]
        # The non-terminal prior execution should NOT have been used for stitching
        assert "should be ignored" not in texts

    def test_stitch_picks_most_recent_terminal_before_running(self, tmp_path):
        """If multiple terminal executions exist, use the one closest in time."""
        exec_dir = _make_workspace_hash(tmp_path)

        _write_execution(exec_dir, "old_done.json", _minimal_execution(
            session_id="sess-1",
            start_time=100,
            status="succeed",
            context_messages=[
                {"role": "human", "entries": [{"type": "text", "text": "old history"}]},
            ],
            input_messages=[{"content": [{"type": "text", "text": "old prompt"}]}],
            actions=[{"actionType": "say", "output": {"message": "old response"}}],
        ))

        _write_execution(exec_dir, "recent_done.json", _minimal_execution(
            session_id="sess-1",
            start_time=500,
            status="aborted",
            context_messages=[
                {"role": "human", "entries": [{"type": "text", "text": "old history"}]},
                {"role": "bot", "entries": [{"type": "text", "text": "old response"}]},
            ],
            input_messages=[{"content": [{"type": "text", "text": "recent prompt"}]}],
            actions=[{"actionType": "say", "output": {"message": "recent response"}}],
        ))

        _write_execution(exec_dir, "now_running.json", _minimal_execution(
            session_id="sess-1",
            start_time=1000,
            status="running",
            context_messages=[],
            input_messages=[{"content": [{"type": "text", "text": "current prompt"}]}],
            actions=[],
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)
        texts = [m.text for m in conv.messages]

        # Should include content from the most recent terminal (start_time=500)
        assert "recent prompt" in texts
        assert "recent response" in texts
        # And the running execution's own input
        assert "current prompt" in texts

    def test_no_stitch_when_running_has_populated_context(self, tmp_path):
        """If a running execution already has context.messages, no stitching occurs."""
        exec_dir = _make_workspace_hash(tmp_path)

        _write_execution(exec_dir, "prior.json", _minimal_execution(
            session_id="sess-1",
            start_time=500,
            status="succeed",
            context_messages=[
                {"role": "human", "entries": [{"type": "text", "text": "should not appear"}]},
            ],
            input_messages=[{"content": [{"type": "text", "text": "old prompt"}]}],
            actions=[],
        ))

        # Running execution with populated context (already backfilled)
        _write_execution(exec_dir, "running.json", _minimal_execution(
            session_id="sess-1",
            start_time=1000,
            status="running",
            context_messages=[
                {"role": "human", "entries": [{"type": "text", "text": "backfilled history"}]},
            ],
            input_messages=[{"content": [{"type": "text", "text": "new prompt"}]}],
            actions=[],
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)
        texts = [m.text for m in conv.messages]

        # Stitching should NOT have happened since context was populated
        assert "should not appear" not in texts
        assert "backfilled history" in texts
        assert "new prompt" in texts


# ---------------------------------------------------------------------------
# Checkpoint integration
# ---------------------------------------------------------------------------


class TestCheckpointIntegration:
    def test_checkpoint_discoverable_via_mcp_action_response(self, tmp_path):
        """A checkpoint emitted as an mcp action's response is found by find_checkpoints."""
        from chat_mother_forker.checkpoint import find_checkpoints

        checkpoint_line = "CHAT CHECKPOINT UUID=11111111-2222-3333-4444-555555555555 SLUG=my-marker"
        data = _minimal_execution(
            input_messages=[{"content": [{"type": "text", "text": "set a checkpoint"}]}],
            actions=[
                {
                    "actionType": "mcp",
                    "input": {"toolName": "chat_checkpoint", "toolArgs": {"slug": "my-marker"}},
                    "output": {"response": checkpoint_line},
                },
            ],
        )

        exec_dir = _make_workspace_hash(tmp_path)
        _write_execution(exec_dir, "exec.json", data)

        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)

        checkpoints = find_checkpoints(conv)
        assert len(checkpoints) == 1
        assert checkpoints[0].uuid == "11111111-2222-3333-4444-555555555555"
        assert checkpoints[0].slug == "my-marker"

    def test_checkpoint_discoverable_via_context_tool_use_response(self, tmp_path):
        """A checkpoint stored in context.messages toolUseResponse is also found."""
        from chat_mother_forker.checkpoint import find_checkpoints

        checkpoint_line = "CHAT CHECKPOINT UUID=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee SLUG=ctx-cp"
        data = _minimal_execution(context_messages=[
            {"role": "human", "entries": [{"type": "text", "text": "checkpoint please"}]},
            {"role": "bot", "entries": [
                {"type": "toolUse", "name": "chat_checkpoint", "args": {"slug": "ctx-cp"}},
            ]},
            {"role": "bot", "entries": [
                {"type": "toolUseResponse", "message": checkpoint_line},
            ]},
        ])

        exec_dir = _make_workspace_hash(tmp_path)
        _write_execution(exec_dir, "exec.json", data)

        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)

        checkpoints = find_checkpoints(conv)
        assert len(checkpoints) == 1
        assert checkpoints[0].slug == "ctx-cp"

    def test_checkpoint_found_during_stitched_running_execution(self, tmp_path):
        """Checkpoints from the prior terminal execution survive stitching."""
        from chat_mother_forker.checkpoint import find_checkpoints

        checkpoint_line = "CHAT CHECKPOINT UUID=12345678-1234-1234-1234-123456789012 SLUG=stitched-cp"
        exec_dir = _make_workspace_hash(tmp_path)

        # Prior execution had a checkpoint in its mcp action
        _write_execution(exec_dir, "prior.json", _minimal_execution(
            session_id="sess-1",
            start_time=1000,
            status="succeed",
            input_messages=[{"content": [{"type": "text", "text": "set checkpoint"}]}],
            actions=[{
                "actionType": "mcp",
                "input": {"toolName": "chat_checkpoint", "toolArgs": {"slug": "stitched-cp"}},
                "output": {"response": checkpoint_line},
            }],
        ))

        # Running execution triggers stitching
        _write_execution(exec_dir, "running.json", _minimal_execution(
            session_id="sess-1",
            start_time=2000,
            status="running",
            context_messages=[],
            input_messages=[{"content": [{"type": "text", "text": "continue"}]}],
            actions=[],
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)

        checkpoints = find_checkpoints(conv)
        assert len(checkpoints) == 1
        assert checkpoints[0].slug == "stitched-cp"


# ---------------------------------------------------------------------------
# Project extraction from staticDirectoryView
# ---------------------------------------------------------------------------


def _static_directory_view_entry(root_path: str) -> dict:
    """Build a context.messages 'document' entry containing a fileTree whose
    first entry is `root_path` (mirrors what Kiro IDE actually embeds).
    """
    sdv = (
        "You are operating in a workspace with files and folders.\n\n"
        f"<fileTree>\n<folder name='{root_path}\\.git' closed />\n"
        f"<file name='{root_path}\\.gitignore' />\n</fileTree>"
    )
    return {
        "role": "tool",
        "entries": [
            {
                "type": "document",
                "document": {
                    "type": "directory",
                    "target": 500,
                    "expandedPaths": [],
                    "openedFiles": [],
                    "staticDirectoryView": sdv,
                },
            },
            {"type": "toolUseResponse", "message": "ok"},
        ],
    }


class TestProjectExtraction:
    def test_extracts_project_from_static_directory_view(self, tmp_path):
        data = _minimal_execution(context_messages=[
            {"role": "human", "entries": [{"type": "text", "text": "hi"}]},
            _static_directory_view_entry("c:\\Dev\\github\\chat-mother-forker"),
        ])
        exec_dir = _make_workspace_hash(tmp_path)
        _write_execution(exec_dir, "exec.json", data)

        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)

        assert conv.project == "chat-mother-forker"

    def test_project_is_none_when_no_document_entries(self, tmp_path):
        data = _minimal_execution(context_messages=[
            {"role": "human", "entries": [{"type": "text", "text": "hi"}]},
        ])
        exec_dir = _make_workspace_hash(tmp_path)
        _write_execution(exec_dir, "exec.json", data)

        provider = KiroIdeProvider(storage_root=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)

        assert conv.project is None

    def test_project_recovered_during_stitching_from_prior_terminal_execution(self, tmp_path):
        """When the newest execution is 'running' with empty context, the
        project name should still be recovered from the prior terminal
        execution used for stitching.
        """
        exec_dir = _make_workspace_hash(tmp_path)

        _write_execution(exec_dir, "exec_done.json", _minimal_execution(
            session_id="sess-1",
            start_time=1000,
            status="succeed",
            context_messages=[
                {"role": "human", "entries": [{"type": "text", "text": "earlier question"}]},
                _static_directory_view_entry("c:\\Dev\\github\\chat-mother-forker"),
            ],
        ))

        running_path = _write_execution(exec_dir, "exec_running.json", _minimal_execution(
            session_id="sess-1",
            start_time=2000,
            status="running",
            context_messages=[],
            input_messages=[{"content": [{"type": "text", "text": "new question"}]}],
        ))

        provider = KiroIdeProvider(storage_root=tmp_path)
        refs = list(provider.list_candidates())
        assert refs[0].locator == str(running_path)

        conv = provider.load(refs[0])
        assert conv.project == "chat-mother-forker"
