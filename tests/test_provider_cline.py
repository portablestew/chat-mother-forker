import json
import os

from chat_mother_forker.checkpoint import find_checkpoints
from chat_mother_forker.models import Role
from chat_mother_forker.providers.cline import ClineProvider, _cline_home


def _write_session(cline_home, session_id, meta, messages):
    """Write a session at the expected location:
    <cline_home>/data/sessions/<session_id>/<session_id>.json
    <cline_home>/data/sessions/<session_id>/<session_id>.messages.json
    """
    session_dir = cline_home / "data" / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    meta_path = session_dir / f"{session_id}.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f)

    messages_path = session_dir / f"{session_id}.messages.json"
    messages_data = {
        "version": 1,
        "updated_at": "2026-08-16T04:55:19.373Z",
        "agent": "lead",
        "sessionId": session_id,
        "origin": {"source": "vscode", "mode": "user", "sessionId": session_id, "version": "4.1.10"},
        "messages": messages,
        "system_prompt": "You are Cline, an AI coding agent...",
    }
    with messages_path.open("w", encoding="utf-8") as f:
        json.dump(messages_data, f)

    return session_dir


def _user(text, timestamp=1786853510060):
    return {
        "id": "msg_user_1",
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "ts": timestamp,
    }


def _assistant_text(text, timestamp=1786853513589):
    return {
        "id": "msg_assistant_1",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "ts": timestamp,
    }


def _assistant_tool_use(name, tool_input, timestamp=1786853513589):
    return {
        "id": "msg_tool_1",
        "role": "assistant",
        "content": [{"type": "tool_use", "name": name, "input": tool_input}],
        "ts": timestamp,
    }


def _user_tool_result(content, timestamp=1786853515858):
    """A tool_result block lives inside a *user* message in Cline."""
    return {
        "id": "msg_tool_result_1",
        "role": "user",
        "content": [{"type": "tool_result", "content": content}],
        "ts": timestamp,
    }


# --- home resolution ---


def test_cline_home_defaults_to_dot_cline_when_env_unset(monkeypatch):
    monkeypatch.delenv("CLINE_HOME", raising=False)
    assert str(_cline_home()).endswith(".cline")


def test_cline_home_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CLINE_HOME", str(tmp_path / "custom"))
    assert _cline_home() == tmp_path / "custom"
# --- candidate discovery ---


def test_list_candidates_returns_empty_when_sessions_dir_missing(tmp_path):
    provider = ClineProvider(cline_home=tmp_path)
    assert list(provider.list_candidates()) == []


def test_list_candidates_discovers_sessions(tmp_path):
    meta1 = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj-a", "title": "Session A"}
    meta2 = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj-b", "title": "Session B"}
    messages = [_user("hello")]

    _write_session(tmp_path, "session-1", meta1, messages)
    _write_session(tmp_path, "session-2", meta2, messages)

    provider = ClineProvider(cline_home=tmp_path)
    refs = list(provider.list_candidates())

    ids = {r.conversation_id for r in refs}
    assert ids == {"session-1", "session-2"}
    assert all(r.provider == "cline" for r in refs)


def test_list_candidates_skips_empty_metadata_files(tmp_path):
    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj", "title": "Session"}
    messages = [_user("hi")]
    _write_session(tmp_path, "session-1", meta, messages)

    session_dir = tmp_path / "data" / "sessions" / "empty-session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "empty-session.json").write_text("")

    provider = ClineProvider(cline_home=tmp_path)
    refs = list(provider.list_candidates())

    ids = {r.conversation_id for r in refs}
    assert ids == {"session-1"}


def test_list_candidates_reads_project_from_workspace_root(tmp_path):
    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\my-project", "title": "Test"}
    _write_session(tmp_path, "session-1", meta, [_user("hello")])

    provider = ClineProvider(cline_home=tmp_path)
    refs = list(provider.list_candidates())

    assert len(refs) == 1
    conv = provider.load(refs[0])
    assert conv.project == "my-project"


def test_list_candidates_falls_back_to_cwd_when_no_workspace_root(tmp_path):
    meta = {"cwd": "C:\\\\Dev\\\\github\\\\fallback-project", "title": "Test"}
    _write_session(tmp_path, "session-1", meta, [_user("hello")])

    provider = ClineProvider(cline_home=tmp_path)
    refs = list(provider.list_candidates())

    conv = provider.load(refs[0])
    assert conv.project == "fallback-project"


# --- message parsing ---


def test_load_parses_user_text_message(tmp_path):
    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj", "title": "Test"}
    messages = [_user("Hello world")]

    _write_session(tmp_path, "session-1", meta, messages)
    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert len(conv.messages) == 1
    assert conv.messages[0].role is Role.USER
    assert conv.messages[0].text == "Hello world"


def test_load_parses_assistant_text_message(tmp_path):
    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj", "title": "Test"}
    messages = [_user("hi"), _assistant_text("Hello!")]

    _write_session(tmp_path, "session-1", meta, messages)
    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert len(conv.messages) == 2
    assert conv.messages[1].role is Role.ASSISTANT
    assert conv.messages[1].text == "Hello!"
def test_load_parses_tool_call(tmp_path):
    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj", "title": "Test"}
    messages = [
        _user("run ls"),
        _assistant_tool_use("bash", {"command": "ls -la"}),
        _user_tool_result("file1.txt\nfile2.txt"),
    ]

    _write_session(tmp_path, "session-1", meta, messages)
    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    tool_calls = [m for m in conv.messages if m.role is Role.TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0].label == "bash"
    assert "ls -la" in tool_calls[0].text


def test_load_parses_tool_result_string(tmp_path):
    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj", "title": "Test"}
    messages = [
        _user("run ls"),
        _assistant_tool_use("bash", {"command": "ls"}),
        _user_tool_result("file1.txt\nfile2.txt"),
    ]

    _write_session(tmp_path, "session-1", meta, messages)
    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    tool_results = [m for m in conv.messages if m.role is Role.TOOL_RESULT]
    assert len(tool_results) == 1
    assert "file1.txt" in tool_results[0].text
    assert "file2.txt" in tool_results[0].text


def test_load_parses_tool_result_list_of_text_blocks(tmp_path):
    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj", "title": "Test"}
    messages = [
        _user("run ls"),
        _assistant_tool_use("bash", {"command": "ls"}),
        {
            "id": "msg_1",
            "role": "user",
            "content": [{"type": "tool_result", "content": [{"type": "text", "text": "output1"}, {"type": "text", "text": "output2"}]}],
            "ts": 1786853515858,
        },
    ]

    _write_session(tmp_path, "session-1", meta, messages)
    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    tool_results = [m for m in conv.messages if m.role is Role.TOOL_RESULT]
    assert len(tool_results) == 1
    assert "output1" in tool_results[0].text
    assert "output2" in tool_results[0].text


def test_load_trims_before_first_user(tmp_path):
    """Messages before the first user message should be dropped."""
    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj", "title": "Test"}
    # System/assistant message before any user message
    messages = [
        {
            "id": "msg_sys",
            "role": "assistant",
            "content": [{"type": "text", "text": "System prompt"}],
            "ts": 1786853510000,
        },
        _user("first user message"),
        _assistant_text("response"),
    ]

    _write_session(tmp_path, "session-1", meta, messages)
    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    # First message should be the user message
    assert conv.messages[0].role is Role.USER
    assert conv.messages[0].text == "first user message"


# --- checkpoint discovery ---


def test_checkpoint_discovery_through_cline_provider(tmp_path):
    """Cline stores tool results with content that may include checkpoint output."""
    uuid = "27ebccde-2451-45c6-91b2-acc9156ef44e"
    checkpoint_text = f"CHAT CHECKPOINT UUID={uuid} SLUG=my-slug"

    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj", "title": "Test"}
    messages = [
        _user("place a checkpoint"),
        _assistant_tool_use("chat_checkpoint", {"slug": "my-slug"}),
        _user_tool_result(checkpoint_text),
        _assistant_text(f"Checkpoint set: {uuid}"),
    ]

    _write_session(tmp_path, "session-1", meta, messages)
    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    checkpoints = find_checkpoints(conv)
    assert len(checkpoints) == 1
    assert checkpoints[0].uuid == uuid
    assert checkpoints[0].slug == "my-slug"


def test_nested_fork_transcript_does_not_produce_phantom_checkpoint(tmp_path):
    """A chat_fork result should not register as a checkpoint."""
    nested = "## USER\n> hello\n\nTOOL_RESULT\n> CHAT CHECKPOINT UUID=aaaa SLUG=other"

    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\proj", "title": "Test"}
    messages = [
        _user("fork it"),
        _user_tool_result(nested),
    ]

    _write_session(tmp_path, "session-1", meta, messages)
    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert find_checkpoints(conv) == []


# --- project extraction from cwd ---


def test_load_sets_project_from_workspace_root_field(tmp_path):
    meta = {"workspace_root": "C:\\\\Dev\\\\github\\\\chat-mother-forker", "title": "Test"}
    _write_session(tmp_path, "session-1", meta, [_user("hello")])

    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert conv.project == "chat-mother-forker"


def test_load_project_is_none_when_no_cwd_or_workspace_root(tmp_path):
    meta = {"title": "Test"}
    _write_session(tmp_path, "session-1", meta, [_user("hello")])

    provider = ClineProvider(cline_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert conv.project is None