import json
import os

from chat_mother_forker.models import Role
from chat_mother_forker.providers.kiro_cli import KiroCliProvider, _kiro_home


def _write_session(kiro_home, session_id, lines):
    """Write a session JSONL file at the expected location:
    <kiro_home>/sessions/cli/<session_id>.jsonl
    """
    cli_dir = kiro_home / "sessions" / "cli"
    cli_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = cli_dir / f"{session_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return jsonl_path


def _prompt(text, timestamp=None):
    """Create a Prompt event in the real Kiro CLI v1 format."""
    data = {"content": [{"kind": "text", "data": text}]}
    if timestamp is not None:
        data["meta"] = {"timestamp": timestamp}
    return {"version": "v1", "kind": "Prompt", "data": data}


def _assistant(text):
    """Create an AssistantMessage event in the real Kiro CLI v1 format."""
    return {"version": "v1", "kind": "AssistantMessage", "data": {"content": [{"kind": "text", "data": text}]}}


def _tool_use(name, tool_input):
    """Create a ToolUse event."""
    return {"version": "v1", "kind": "ToolUse", "data": {"name": name, "input": tool_input}}


def _tool_result(content):
    """Create a ToolResult event."""
    if isinstance(content, str):
        return {"version": "v1", "kind": "ToolResult", "data": {"content": content}}
    # Structured content as a list
    return {"version": "v1", "kind": "ToolResult", "data": {"content": content}}


def test_kiro_home_defaults_to_dot_kiro_when_env_unset(monkeypatch):
    monkeypatch.delenv("KIRO_HOME", raising=False)
    home = _kiro_home()
    assert str(home).endswith(".kiro")


def test_kiro_home_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KIRO_HOME", str(tmp_path / "custom"))
    assert _kiro_home() == tmp_path / "custom"


def test_list_candidates_returns_empty_when_sessions_dir_missing(tmp_path):
    provider = KiroCliProvider(kiro_home=tmp_path)
    assert list(provider.list_candidates()) == []


def test_list_candidates_discovers_sessions(tmp_path):
    _write_session(tmp_path, "session-1", [_prompt("hi")])
    _write_session(tmp_path, "session-2", [_prompt("hello")])

    provider = KiroCliProvider(kiro_home=tmp_path)
    refs = list(provider.list_candidates())

    ids = {r.conversation_id for r in refs}
    assert ids == {"session-1", "session-2"}
    assert all(r.provider == "kiro_cli" for r in refs)


def test_list_candidates_skips_empty_jsonl_files(tmp_path):
    _write_session(tmp_path, "session-1", [_prompt("hi")])
    # Create an empty file
    cli_dir = tmp_path / "sessions" / "cli"
    (cli_dir / "session-empty.jsonl").write_text("")

    provider = KiroCliProvider(kiro_home=tmp_path)
    refs = list(provider.list_candidates())

    assert len(refs) == 1
    assert refs[0].conversation_id == "session-1"


def test_list_candidates_mtime_reflects_file_modification_time(tmp_path):
    jsonl_path = _write_session(tmp_path, "session-1", [_prompt("hi")])
    provider = KiroCliProvider(kiro_home=tmp_path)
    refs = list(provider.list_candidates())

    assert len(refs) == 1
    expected_mtime = os.stat(jsonl_path).st_mtime
    assert refs[0].mtime == expected_mtime


def test_load_parses_user_and_assistant_messages(tmp_path):
    _write_session(tmp_path, "session-1", [_prompt("hello there"), _assistant("hi back")])

    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert [m.role for m in conversation.messages] == [Role.USER, Role.ASSISTANT]
    assert conversation.messages[0].text == "hello there"
    assert conversation.messages[1].text == "hi back"


def test_load_parses_tool_use_with_name_and_input(tmp_path):
    _write_session(
        tmp_path,
        "session-1",
        [_tool_use("grep_search", {"query": "foo", "limit": 10})],
    )
    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert len(conversation.messages) == 1
    message = conversation.messages[0]
    assert message.role == Role.TOOL_CALL
    assert message.label == "grep_search"
    parsed_args = json.loads(message.text)
    assert parsed_args == {"query": "foo", "limit": 10}


def test_load_parses_tool_result_with_string_content(tmp_path):
    _write_session(tmp_path, "session-1", [_tool_result("some output text")])

    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert conversation.messages[0].role == Role.TOOL_RESULT
    assert conversation.messages[0].text == "some output text"


def test_load_parses_tool_result_with_structured_content(tmp_path):
    _write_session(
        tmp_path,
        "session-1",
        [{"version": "v1", "kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "item 1"}, {"kind": "text", "data": "item 2"}]}}],
    )
    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert conversation.messages[0].role == Role.TOOL_RESULT
    assert "item 1" in conversation.messages[0].text
    assert "item 2" in conversation.messages[0].text


def test_load_skips_bookkeeping_event_types(tmp_path):
    _write_session(
        tmp_path,
        "session-1",
        [
            {"version": "v1", "kind": "TurnStart", "data": {}},
            {"version": "v1", "kind": "SessionMetadata", "data": {"key": "contextUsage"}},
            _prompt("real message"),
        ],
    )
    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].text == "real message"


def test_load_skips_blank_and_missing_content(tmp_path):
    _write_session(
        tmp_path,
        "session-1",
        [
            {"version": "v1", "kind": "Prompt", "data": {"content": [{"kind": "text", "data": "   "}]}},
            {"version": "v1", "kind": "AssistantMessage", "data": {}},  # no content
            _prompt("real one"),
        ],
    )
    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].text == "real one"


def test_load_skips_malformed_json_lines_without_crashing(tmp_path):
    cli_dir = tmp_path / "sessions" / "cli"
    cli_dir.mkdir(parents=True)
    jsonl_path = cli_dir / "session-1.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write(json.dumps(_prompt("still works")) + "\n")
        f.write("\n")  # blank line

    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].text == "still works"


def test_two_provider_instances_can_represent_different_homes(tmp_path):
    # Simulates the "same tool, different machine" case: two KiroCliProvider
    # instances pointed at different KIRO_HOME roots can be merged by the
    # same recency logic used for cross-tool merging.
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    path_a = _write_session(home_a, "session-a", [_prompt("from A")])
    path_b = _write_session(home_b, "session-b", [_prompt("from B")])
    # Make sure they have distinguishable mtimes.
    os.utime(path_a, (1000, 1000))
    os.utime(path_b, (2000, 2000))

    provider_a = KiroCliProvider(kiro_home=home_a)
    provider_b = KiroCliProvider(kiro_home=home_b)

    from chat_mother_forker.search import gather_sorted_candidates

    refs = gather_sorted_candidates([provider_a, provider_b], per_provider=100)
    assert [r.conversation_id for r in refs] == ["session-b", "session-a"]


def test_load_preserves_timestamp_from_meta(tmp_path):
    _write_session(tmp_path, "session-1", [_prompt("hello", timestamp=1780083415)])

    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert conversation.messages[0].timestamp == "1780083415"
