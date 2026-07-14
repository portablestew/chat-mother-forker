"""Tests for the Kiro IDE v2 provider (new ~/.kiro/sessions/<hash>/sess_<uuid>/
layout: one messages.jsonl + session.json per session directory).
"""

import json
import os

from chat_mother_forker.models import Role
from chat_mother_forker.providers.kiro_ide_v2 import KiroIdeV2Provider, _kiro_home


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_session(kiro_home, workspace_hash, session_uuid, events, session_json=None):
    """Write a v2 session directory:
    <kiro_home>/sessions/<workspace_hash>/sess_<session_uuid>/messages.jsonl
    <kiro_home>/sessions/<workspace_hash>/sess_<session_uuid>/session.json
    """
    session_dir = kiro_home / "sessions" / workspace_hash / f"sess_{session_uuid}"
    session_dir.mkdir(parents=True, exist_ok=True)

    messages_path = session_dir / "messages.jsonl"
    with messages_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    if session_json is not None:
        (session_dir / "session.json").write_text(json.dumps(session_json), encoding="utf-8")

    return session_dir


def _event(payload, event_id="evt", timestamp="2026-07-14T00:00:00.000Z"):
    return {"id": event_id, "timestamp": timestamp, "payload": payload}


def _user(text, event_id="u1"):
    return _event({"type": "user", "content": text, "images": [], "documents": []}, event_id=event_id)


def _assistant(text, event_id="a1"):
    return _event({"type": "assistant", "content": text, "operationType": "Say"}, event_id=event_id)


def _tool_call(tool_call_id, tool_name, args, event_id=None):
    return _event(
        {"type": "tool_call", "toolCallId": tool_call_id, "toolName": tool_name, "args": args, "status": "completed"},
        event_id=event_id or f"{tool_call_id}-call",
    )


def _tool_result(tool_call_id, content, success=True, event_id=None):
    return _event(
        {"type": "tool_result", "toolCallId": tool_call_id, "content": content, "success": success},
        event_id=event_id or f"{tool_call_id}-result",
    )


def _mcp_tool_result(tool_call_id, response_text, event_id=None):
    """An MCP-style tool_result whose content is a JSON object with a
    "response" key, e.g. what chat_checkpoint/chat_search return.
    """
    content = json.dumps({"response": response_text, "imageBase64Urls": []})
    return _tool_result(tool_call_id, content, event_id=event_id)


def _bookkeeping(payload_type, event_id="bk1", **extra):
    return _event({"type": payload_type, **extra}, event_id=event_id)


# ---------------------------------------------------------------------------
# _kiro_home
# ---------------------------------------------------------------------------


def test_kiro_home_defaults_to_dot_kiro_when_env_unset(monkeypatch):
    monkeypatch.delenv("KIRO_HOME", raising=False)
    home = _kiro_home()
    assert str(home).endswith(".kiro")


def test_kiro_home_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KIRO_HOME", str(tmp_path / "custom"))
    assert _kiro_home() == tmp_path / "custom"


# ---------------------------------------------------------------------------
# list_candidates
# ---------------------------------------------------------------------------


class TestListCandidates:
    def test_returns_empty_when_sessions_dir_missing(self, tmp_path):
        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        assert list(provider.list_candidates()) == []

    def test_discovers_single_session(self, tmp_path):
        _write_session(tmp_path, "abc123", "11111111-1111-1111-1111-111111111111", [_user("hi")])

        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        refs = list(provider.list_candidates())

        assert len(refs) == 1
        assert refs[0].conversation_id == "11111111-1111-1111-1111-111111111111"
        assert refs[0].provider == "kiro_ide_v2"

    def test_discovers_sessions_across_multiple_workspace_hashes(self, tmp_path):
        _write_session(tmp_path, "hash-a", "aaaaaaaa-1111-1111-1111-111111111111", [_user("from a")])
        _write_session(tmp_path, "hash-b", "bbbbbbbb-2222-2222-2222-222222222222", [_user("from b")])

        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        refs = list(provider.list_candidates())
        ids = {r.conversation_id for r in refs}

        assert ids == {"aaaaaaaa-1111-1111-1111-111111111111", "bbbbbbbb-2222-2222-2222-222222222222"}

    def test_skips_cli_sessions_directory(self, tmp_path):
        """The 'cli' subdirectory under sessions/ belongs to KiroCliProvider,
        not this provider -- must not be treated as a workspace hash dir.
        """
        cli_dir = tmp_path / "sessions" / "cli"
        cli_dir.mkdir(parents=True)
        (cli_dir / "some-session.jsonl").write_text('{"kind":"Prompt"}\n', encoding="utf-8")

        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        assert list(provider.list_candidates()) == []

    def test_skips_non_session_directories(self, tmp_path):
        """Directories not starting with 'sess_' under a workspace hash are ignored."""
        workspace_dir = tmp_path / "sessions" / "hash-a"
        other_dir = workspace_dir / "not-a-session"
        other_dir.mkdir(parents=True)
        (other_dir / "messages.jsonl").write_text('{}\n', encoding="utf-8")

        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        assert list(provider.list_candidates()) == []

    def test_skips_session_dirs_missing_messages_file(self, tmp_path):
        session_dir = tmp_path / "sessions" / "hash-a" / "sess_11111111-1111-1111-1111-111111111111"
        session_dir.mkdir(parents=True)
        # No messages.jsonl written.

        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        assert list(provider.list_candidates()) == []

    def test_mtime_reflects_messages_file_modification_time(self, tmp_path):
        session_dir = _write_session(tmp_path, "hash-a", "11111111-1111-1111-1111-111111111111", [_user("hi")])
        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        refs = list(provider.list_candidates())

        expected_mtime = os.stat(session_dir / "messages.jsonl").st_mtime
        assert refs[0].mtime == expected_mtime

    def test_locator_points_to_session_directory(self, tmp_path):
        session_dir = _write_session(tmp_path, "hash-a", "11111111-1111-1111-1111-111111111111", [_user("hi")])
        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        refs = list(provider.list_candidates())

        assert refs[0].locator == str(session_dir)


# ---------------------------------------------------------------------------
# load: message parsing
# ---------------------------------------------------------------------------


class TestLoadParsing:
    def _load_single(self, tmp_path, events, session_json=None):
        _write_session(tmp_path, "hash-a", "11111111-1111-1111-1111-111111111111", events, session_json=session_json)
        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        ref = next(iter(provider.list_candidates()))
        return provider.load(ref)

    def test_parses_user_and_assistant_messages(self, tmp_path):
        conv = self._load_single(tmp_path, [_user("hello there"), _assistant("hi back")])

        assert [m.role for m in conv.messages] == [Role.USER, Role.ASSISTANT]
        assert conv.messages[0].text == "hello there"
        assert conv.messages[1].text == "hi back"

    def test_parses_tool_call_with_name_and_args(self, tmp_path):
        conv = self._load_single(tmp_path, [
            _tool_call("tc1", "grep_search", {"query": "foo", "limit": 10}),
        ])

        assert len(conv.messages) == 1
        msg = conv.messages[0]
        assert msg.role is Role.TOOL_CALL
        assert msg.label == "grep_search"
        assert json.loads(msg.text) == {"query": "foo", "limit": 10}

    def test_parses_builtin_tool_result_as_plain_text(self, tmp_path):
        """Built-in tools (e.g. read_file) return plain, non-JSON text."""
        conv = self._load_single(tmp_path, [
            _tool_call("tc1", "read_file", {"path": "README.md"}),
            _tool_result("tc1", "# Title\nSome file content"),
        ])

        tool_results = [m for m in conv.messages if m.role is Role.TOOL_RESULT]
        assert len(tool_results) == 1
        assert tool_results[0].text == "# Title\nSome file content"

    def test_parses_mcp_tool_result_response_field(self, tmp_path):
        """MCP tools wrap their text in {"response": "..."}; that value is
        what should end up as the TOOL_RESULT text, not the raw JSON.
        """
        conv = self._load_single(tmp_path, [
            _tool_call("tc1", "mcp_chat_mother_forker_chat_search", {}),
            _mcp_tool_result("tc1", "3 conversation(s) found"),
        ])

        tool_results = [m for m in conv.messages if m.role is Role.TOOL_RESULT]
        assert len(tool_results) == 1
        assert tool_results[0].text == "3 conversation(s) found"

    def test_skips_tool_result_with_empty_json_object(self, tmp_path):
        """Some built-in tools (e.g. update_session_information) return
        a bare '{}' with nothing useful -- should produce no message.
        """
        conv = self._load_single(tmp_path, [
            _tool_call("tc1", "update_session_information", {"title": "x"}),
            _tool_result("tc1", "{}"),
        ])

        tool_results = [m for m in conv.messages if m.role is Role.TOOL_RESULT]
        assert tool_results == []
        # The tool_call itself should still be present.
        assert any(m.role is Role.TOOL_CALL for m in conv.messages)

    def test_skips_blank_user_and_assistant_content(self, tmp_path):
        conv = self._load_single(tmp_path, [
            _user("   "),
            _assistant(""),
            _user("real message"),
        ])

        assert len(conv.messages) == 1
        assert conv.messages[0].text == "real message"

    def test_skips_bookkeeping_event_types(self, tmp_path):
        conv = self._load_single(tmp_path, [
            _bookkeeping("turn_start", executionId="exec-1"),
            _bookkeeping("steering_inclusion", documents=["file:///foo.md"]),
            _bookkeeping("session_metadata", key="contextUsage", value={"usagePercentage": 1.0}),
            _bookkeeping("usage_summary", promptTurnSummaries=[]),
            _bookkeeping("session_event", category="session_pause"),
            _bookkeeping("turn_end", stopReason="done"),
            _bookkeeping("pending_interaction", interactionType="tool_approval"),
            _bookkeeping("interaction_resolved", outcome="cancelled"),
            _user("the only real message"),
        ])

        assert len(conv.messages) == 1
        assert conv.messages[0].text == "the only real message"

    def test_skips_malformed_json_lines_without_crashing(self, tmp_path):
        session_dir = tmp_path / "sessions" / "hash-a" / "sess_11111111-1111-1111-1111-111111111111"
        session_dir.mkdir(parents=True)
        messages_path = session_dir / "messages.jsonl"
        with messages_path.open("w", encoding="utf-8") as f:
            f.write("{not valid json\n")
            f.write(json.dumps(_user("still works")) + "\n")
            f.write("\n")  # blank line

        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)

        assert len(conv.messages) == 1
        assert conv.messages[0].text == "still works"

    def test_load_returns_empty_when_messages_file_unreadable(self, tmp_path):
        """If the locator's messages.jsonl vanished/can't be read, return
        an empty conversation rather than raising.
        """
        from chat_mother_forker.models import ConversationRef

        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        ref = ConversationRef(
            provider="kiro_ide_v2",
            conversation_id="fake",
            locator=str(tmp_path / "sessions" / "hash-a" / "sess_nonexistent"),
            mtime=0.0,
        )
        conv = provider.load(ref)
        assert conv.messages == []

    def test_preserves_timestamp_on_messages(self, tmp_path):
        conv = self._load_single(tmp_path, [
            _user("hello", event_id="u1"),
        ])
        # timestamp defaults to the fixed value used by _event/_user helpers
        assert conv.messages[0].timestamp == "2026-07-14T00:00:00.000Z"

    def test_full_multi_turn_conversation(self, tmp_path):
        """A realistic turn: user prompt, assistant text, tool_call/result
        pair (MCP-style), followed by another assistant message.
        """
        conv = self._load_single(tmp_path, [
            _user("What's in the project?"),
            _assistant("Let me check."),
            _tool_call("tc1", "mcp_chat_mother_forker_chat_search", {}),
            _mcp_tool_result("tc1", "1 conversation(s) found"),
            _assistant("Found one prior conversation."),
        ])

        roles = [m.role for m in conv.messages]
        assert roles == [
            Role.USER,
            Role.ASSISTANT,
            Role.TOOL_CALL,
            Role.TOOL_RESULT,
            Role.ASSISTANT,
        ]
        assert conv.messages[3].text == "1 conversation(s) found"


# ---------------------------------------------------------------------------
# load: project extraction from session.json
# ---------------------------------------------------------------------------


class TestProjectExtraction:
    def test_extracts_project_from_workspace_paths(self, tmp_path):
        conv = self._load_with_metadata(tmp_path, {"workspacePaths": ["j:\\Git\\chat-mother-forker"]})
        assert conv.project == "chat-mother-forker"

    def test_project_none_when_session_json_missing(self, tmp_path):
        conv = self._load_with_metadata(tmp_path, session_json=None)
        assert conv.project is None

    def test_project_none_when_workspace_paths_empty(self, tmp_path):
        conv = self._load_with_metadata(tmp_path, {"workspacePaths": []})
        assert conv.project is None

    def test_project_none_when_session_json_malformed(self, tmp_path):
        session_dir = _write_session(
            tmp_path, "hash-a", "11111111-1111-1111-1111-111111111111", [_user("hi")]
        )
        (session_dir / "session.json").write_text("{not valid json", encoding="utf-8")

        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)
        assert conv.project is None

    def test_project_handles_posix_style_workspace_path(self, tmp_path):
        conv = self._load_with_metadata(tmp_path, {"workspacePaths": ["/home/dev/my-project"]})
        assert conv.project == "my-project"

    def _load_with_metadata(self, tmp_path, session_json):
        _write_session(
            tmp_path, "hash-a", "11111111-1111-1111-1111-111111111111", [_user("hi")],
            session_json=session_json,
        )
        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        ref = next(iter(provider.list_candidates()))
        return provider.load(ref)


# ---------------------------------------------------------------------------
# Checkpoint integration
# ---------------------------------------------------------------------------


class TestCheckpointIntegration:
    def test_checkpoint_discoverable_via_mcp_tool_result(self, tmp_path):
        from chat_mother_forker.checkpoint import find_checkpoints

        checkpoint_line = "CHAT CHECKPOINT UUID=11111111-2222-3333-4444-555555555555 SLUG=my-marker"
        _write_session(tmp_path, "hash-a", "11111111-1111-1111-1111-111111111111", [
            _user("set a checkpoint"),
            _tool_call("tc1", "mcp_chat_mother_forker_chat_checkpoint", {"slug": "my-marker"}),
            _mcp_tool_result("tc1", checkpoint_line),
        ])

        provider = KiroIdeV2Provider(kiro_home=tmp_path)
        ref = next(iter(provider.list_candidates()))
        conv = provider.load(ref)

        checkpoints = find_checkpoints(conv)
        assert len(checkpoints) == 1
        assert checkpoints[0].uuid == "11111111-2222-3333-4444-555555555555"
        assert checkpoints[0].slug == "my-marker"


# ---------------------------------------------------------------------------
# Cross-provider merge sanity check
# ---------------------------------------------------------------------------


def test_two_provider_instances_can_represent_different_homes(tmp_path):
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    dir_a = _write_session(home_a, "hash-a", "aaaaaaaa-1111-1111-1111-111111111111", [_user("from A")])
    dir_b = _write_session(home_b, "hash-b", "bbbbbbbb-2222-2222-2222-222222222222", [_user("from B")])
    os.utime(dir_a / "messages.jsonl", (1000, 1000))
    os.utime(dir_b / "messages.jsonl", (2000, 2000))

    provider_a = KiroIdeV2Provider(kiro_home=home_a)
    provider_b = KiroIdeV2Provider(kiro_home=home_b)

    from chat_mother_forker.search import gather_sorted_candidates

    refs = gather_sorted_candidates([provider_a, provider_b], per_provider=100)
    assert [r.conversation_id for r in refs] == [
        "bbbbbbbb-2222-2222-2222-222222222222",
        "aaaaaaaa-1111-1111-1111-111111111111",
    ]
