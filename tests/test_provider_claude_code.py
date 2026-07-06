import json
import os

from chat_mother_forker.checkpoint import find_checkpoints
from chat_mother_forker.models import Role
from chat_mother_forker.providers.claude_code import (
    ClaudeCodeProvider,
    _claude_home,
    _unwrap_structured_result,
)


def _write_session(claude_home, project, session_id, events, cwd=None):
    """Write a session JSONL file at the expected location:
    <claude_home>/projects/<project>/<session_id>.jsonl

    If `cwd` is given, it's stamped onto every event (mirroring the real
    format, where every JSONL line carries a `cwd` field).
    """
    project_dir = claude_home / "projects" / project
    project_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = project_dir / f"{session_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for event in events:
            if cwd is not None:
                event = {**event, "cwd": cwd}
            f.write(json.dumps(event) + "\n")
    return jsonl_path


def _user(text, timestamp="2026-07-04T00:00:00.000Z"):
    return {"type": "user", "message": {"role": "user", "content": text}, "timestamp": timestamp}


def _assistant_text(text, timestamp="2026-07-04T00:00:01.000Z"):
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "timestamp": timestamp,
    }


def _assistant_tool_use(name, tool_input, timestamp="2026-07-04T00:00:02.000Z"):
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": name, "input": tool_input}],
        },
        "timestamp": timestamp,
    }


def _user_tool_result(content, timestamp="2026-07-04T00:00:03.000Z"):
    """A tool_result block lives inside a *user* event in Claude Code."""
    return {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "tool_result", "content": content}]},
        "timestamp": timestamp,
    }


# --- home resolution ---


def test_claude_home_defaults_to_dot_claude_when_env_unset(monkeypatch):
    monkeypatch.delenv("CLAUDE_HOME", raising=False)
    assert str(_claude_home()).endswith(".claude")


def test_claude_home_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_HOME", str(tmp_path / "custom"))
    assert _claude_home() == tmp_path / "custom"


# --- candidate discovery ---


def test_list_candidates_returns_empty_when_projects_dir_missing(tmp_path):
    provider = ClaudeCodeProvider(claude_home=tmp_path)
    assert list(provider.list_candidates()) == []


def test_list_candidates_discovers_sessions_across_projects(tmp_path):
    _write_session(tmp_path, "proj-a", "session-1", [_user("hi")])
    _write_session(tmp_path, "proj-b", "session-2", [_user("hello")])

    provider = ClaudeCodeProvider(claude_home=tmp_path)
    refs = list(provider.list_candidates())

    ids = {r.conversation_id for r in refs}
    assert ids == {"session-1", "session-2"}
    assert all(r.provider == "claude_code" for r in refs)


def test_list_candidates_skips_empty_jsonl_files(tmp_path):
    _write_session(tmp_path, "proj", "session-1", [_user("hi")])
    project_dir = tmp_path / "projects" / "proj"
    (project_dir / "empty.jsonl").write_text("")

    provider = ClaudeCodeProvider(claude_home=tmp_path)
    refs = list(provider.list_candidates())

    assert [r.conversation_id for r in refs] == ["session-1"]


# --- message parsing ---


def test_load_parses_string_and_block_messages(tmp_path):
    _write_session(
        tmp_path,
        "proj",
        "session-1",
        [_user("hello there"), _assistant_text("hi back")],
    )
    provider = ClaudeCodeProvider(claude_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert [m.role for m in conversation.messages] == [Role.USER, Role.ASSISTANT]
    assert conversation.messages[0].text == "hello there"
    assert conversation.messages[1].text == "hi back"


def test_load_parses_tool_use(tmp_path):
    _write_session(
        tmp_path,
        "proj",
        "session-1",
        [_user("go"), _assistant_tool_use("chat_search", {"search": "foo"})],
    )
    provider = ClaudeCodeProvider(claude_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    tool_call = [m for m in conversation.messages if m.role == Role.TOOL_CALL][0]
    assert tool_call.label == "chat_search"
    assert json.loads(tool_call.text) == {"search": "foo"}


def test_load_parses_tool_result_inside_user_event(tmp_path):
    _write_session(
        tmp_path,
        "proj",
        "session-1",
        [_user("go"), _user_tool_result("plain output")],
    )
    provider = ClaudeCodeProvider(claude_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    tool_results = [m for m in conversation.messages if m.role == Role.TOOL_RESULT]
    assert len(tool_results) == 1
    assert tool_results[0].text == "plain output"


def test_load_skips_malformed_json_lines_without_crashing(tmp_path):
    project_dir = tmp_path / "projects" / "proj"
    project_dir.mkdir(parents=True)
    jsonl_path = project_dir / "session-1.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write(json.dumps(_user("still works")) + "\n")
        f.write("\n")

    provider = ClaudeCodeProvider(claude_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert [m.text for m in conversation.messages] == ["still works"]


# --- structured-content unwrapping (the FastMCP {"result": ...} wrapper) ---


def test_unwrap_structured_result_unwraps_single_result_key():
    wrapped = json.dumps({"result": "CHAT CHECKPOINT UUID=x SLUG=y"})
    assert _unwrap_structured_result(wrapped) == "CHAT CHECKPOINT UUID=x SLUG=y"


def test_unwrap_structured_result_stringifies_non_string_inner():
    wrapped = json.dumps({"result": {"nested": 1}})
    assert _unwrap_structured_result(wrapped) == '{"nested": 1}'


def test_unwrap_structured_result_leaves_plain_text_untouched():
    assert _unwrap_structured_result("just some output") == "just some output"


def test_unwrap_structured_result_leaves_multi_key_json_untouched():
    payload = json.dumps({"result": "x", "isError": False})
    assert _unwrap_structured_result(payload) == payload


def test_unwrap_structured_result_leaves_malformed_json_untouched():
    text = '{"result": broken'
    assert _unwrap_structured_result(text) == text


# --- the regression: checkpoint discovery through the real Claude Code shape ---


def test_checkpoint_discovery_through_claude_code_provider(tmp_path):
    """Claude Code stores the checkpoint tool_result as the FastMCP structured
    wrapper: {"result":"CHAT CHECKPOINT ..."}. find_checkpoints() must still
    discover it after the provider normalizes that wrapper.
    """
    uuid = "27ebccde-2451-45c6-91b2-acc9156ef44e"
    wrapper = json.dumps({"result": f"CHAT CHECKPOINT UUID={uuid} SLUG=my-slug"})

    _write_session(
        tmp_path,
        "proj",
        "session-1",
        [
            _user("place a checkpoint"),
            _assistant_tool_use("chat_checkpoint", {"slug": "my-slug"}),
            _user_tool_result(wrapper),
            _assistant_text(f"Checkpoint set: {uuid}"),
        ],
    )
    provider = ClaudeCodeProvider(claude_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    checkpoints = find_checkpoints(conversation)
    assert len(checkpoints) == 1
    assert checkpoints[0].uuid == uuid
    assert checkpoints[0].slug == "my-slug"


def test_nested_fork_transcript_does_not_produce_phantom_checkpoint(tmp_path):
    """A chat_fork result -- itself wrapped as {"result": "## USER ..."} -- must
    not register as a checkpoint even though the quoted transcript inside it may
    mention 'CHAT CHECKPOINT'. After unwrapping, the text starts with '## USER',
    so position-0 anchoring holds.
    """
    nested = "## USER\n> hello\n\nTOOL_RESULT\n> CHAT CHECKPOINT UUID=aaaa SLUG=other"
    wrapper = json.dumps({"result": nested})

    _write_session(
        tmp_path,
        "proj",
        "session-1",
        [_user("fork it"), _user_tool_result(wrapper)],
    )
    provider = ClaudeCodeProvider(claude_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert find_checkpoints(conversation) == []


def test_two_provider_instances_can_represent_different_homes(tmp_path):
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    path_a = _write_session(home_a, "proj", "session-a", [_user("from A")])
    path_b = _write_session(home_b, "proj", "session-b", [_user("from B")])
    os.utime(path_a, (1000, 1000))
    os.utime(path_b, (2000, 2000))

    from chat_mother_forker.search import gather_sorted_candidates

    provider_a = ClaudeCodeProvider(claude_home=home_a)
    provider_b = ClaudeCodeProvider(claude_home=home_b)
    refs = gather_sorted_candidates([provider_a, provider_b], per_provider=100)
    assert [r.conversation_id for r in refs] == ["session-b", "session-a"]


# --- project extraction from cwd ---


def test_load_sets_project_from_cwd_field(tmp_path):
    _write_session(
        tmp_path,
        "proj",
        "session-1",
        [_user("hello")],
        cwd="C:\\Dev\\github\\chat-mother-forker",
    )
    provider = ClaudeCodeProvider(claude_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert conversation.project == "chat-mother-forker"


def test_load_project_is_none_when_cwd_absent(tmp_path):
    _write_session(tmp_path, "proj", "session-1", [_user("hello")])
    provider = ClaudeCodeProvider(claude_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert conversation.project is None
