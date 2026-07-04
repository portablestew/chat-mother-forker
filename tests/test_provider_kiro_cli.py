import json
import os

from chat_mother_forker.models import Role
from chat_mother_forker.providers.kiro_cli import KiroCliProvider, _kiro_home


def _write_session(kiro_home, workspace_hash, session_id, lines):
    session_dir = kiro_home / "sessions" / workspace_hash / session_id
    session_dir.mkdir(parents=True)
    messages_path = session_dir / "messages.jsonl"
    with messages_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return messages_path


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


def test_list_candidates_discovers_sessions_across_workspaces(tmp_path):
    _write_session(tmp_path, "workspace-a", "session-1", [{"payload": {"type": "user", "content": "hi"}}])
    _write_session(tmp_path, "workspace-b", "session-2", [{"payload": {"type": "user", "content": "hi"}}])

    provider = KiroCliProvider(kiro_home=tmp_path)
    refs = list(provider.list_candidates())

    ids = {r.conversation_id for r in refs}
    assert ids == {"session-1", "session-2"}
    assert all(r.provider == "kiro_cli" for r in refs)


def test_list_candidates_mtime_reflects_file_modification_time(tmp_path):
    messages_path = _write_session(
        tmp_path, "ws", "session-1", [{"payload": {"type": "user", "content": "hi"}}]
    )
    provider = KiroCliProvider(kiro_home=tmp_path)
    refs = list(provider.list_candidates())

    assert len(refs) == 1
    expected_mtime = os.stat(messages_path).st_mtime
    assert refs[0].mtime == expected_mtime


def test_load_parses_user_and_assistant_messages(tmp_path):
    _write_session(
        tmp_path,
        "ws",
        "session-1",
        [
            {"payload": {"type": "user", "content": "hello there"}},
            {"payload": {"type": "assistant", "content": "hi back"}},
        ],
    )
    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert [m.role for m in conversation.messages] == [Role.USER, Role.ASSISTANT]
    assert conversation.messages[0].text == "hello there"
    assert conversation.messages[1].text == "hi back"


def test_load_parses_tool_call_with_name_and_args(tmp_path):
    _write_session(
        tmp_path,
        "ws",
        "session-1",
        [
            {
                "payload": {
                    "type": "tool_call",
                    "toolName": "grep_search",
                    "args": {"query": "foo", "limit": 10},
                }
            }
        ],
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
    _write_session(
        tmp_path,
        "ws",
        "session-1",
        [{"payload": {"type": "tool_result", "content": "some output text"}}],
    )
    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert conversation.messages[0].role == Role.TOOL_RESULT
    assert conversation.messages[0].text == "some output text"


def test_load_parses_tool_result_with_structured_content_as_json(tmp_path):
    _write_session(
        tmp_path,
        "ws",
        "session-1",
        [{"payload": {"type": "tool_result", "content": {"items": [1, 2, 3]}}}],
    )
    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert conversation.messages[0].role == Role.TOOL_RESULT
    parsed = json.loads(conversation.messages[0].text)
    assert parsed == {"items": [1, 2, 3]}


def test_load_skips_bookkeeping_event_types(tmp_path):
    _write_session(
        tmp_path,
        "ws",
        "session-1",
        [
            {"payload": {"type": "turn_start"}},
            {"payload": {"type": "session_metadata", "key": "contextUsage", "value": {}}},
            {"payload": {"type": "user", "content": "real message"}},
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
        "ws",
        "session-1",
        [
            {"payload": {"type": "user", "content": "   "}},
            {"payload": {"type": "assistant"}},  # no content key at all
            {"payload": {"type": "user", "content": "real one"}},
        ],
    )
    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].text == "real one"


def test_load_skips_malformed_json_lines_without_crashing(tmp_path):
    session_dir = tmp_path / "sessions" / "ws" / "session-1"
    session_dir.mkdir(parents=True)
    messages_path = session_dir / "messages.jsonl"
    with messages_path.open("w", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write(json.dumps({"payload": {"type": "user", "content": "still works"}}) + "\n")
        f.write("\n")  # blank line

    provider = KiroCliProvider(kiro_home=tmp_path)
    ref = next(iter(provider.list_candidates()))
    conversation = provider.load(ref)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].text == "still works"


def test_two_provider_instances_can_represent_two_different_workspaces_or_hosts(tmp_path):
    # Simulates the "same tool, different workspace" case: two KiroCliProvider
    # instances pointed at different KIRO_HOME roots (e.g. copied from two
    # machines, or two profiles) can be merged by the same recency logic
    # used for cross-tool merging.
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    _write_session(home_a, "ws", "session-a", [{"payload": {"type": "user", "content": "from A"}}])
    _write_session(home_b, "ws", "session-b", [{"payload": {"type": "user", "content": "from B"}}])
    # Make sure they have distinguishable mtimes.
    os.utime(home_a / "sessions" / "ws" / "session-a" / "messages.jsonl", (1000, 1000))
    os.utime(home_b / "sessions" / "ws" / "session-b" / "messages.jsonl", (2000, 2000))

    provider_a = KiroCliProvider(kiro_home=home_a)
    provider_b = KiroCliProvider(kiro_home=home_b)

    from chat_mother_forker.search import gather_sorted_candidates

    refs = gather_sorted_candidates([provider_a, provider_b], per_provider=100)
    assert [r.conversation_id for r in refs] == ["session-b", "session-a"]
