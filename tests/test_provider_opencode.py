import json
import sqlite3

from chat_mother_forker.checkpoint import find_checkpoints
from chat_mother_forker.models import Role
from chat_mother_forker.providers.opencode import OpenCodeProvider, _opencode_db_path


def _make_db(tmp_path, rows):
    """Create a minimal opencode.db with the given session/message/part rows.

    `rows` is a list of (table, values) tuples where values is a list of
    row dicts. Only the columns the provider actually reads are declared.
    """
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db))
    c = conn.cursor()
    c.execute(
        "CREATE TABLE session ("
        "id TEXT PRIMARY KEY, project_id TEXT NOT NULL, slug TEXT NOT NULL, "
        "directory TEXT NOT NULL, title TEXT NOT NULL, version TEXT NOT NULL, "
        "time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL)"
    )
    c.execute(
        "CREATE TABLE message ("
        "id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
        "time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, data TEXT NOT NULL)"
    )
    c.execute(
        "CREATE TABLE part ("
        "id TEXT PRIMARY KEY, message_id TEXT NOT NULL, session_id TEXT NOT NULL, "
        "time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, data TEXT NOT NULL)"
    )
    for table, values in rows:
        for v in values:
            cols = ", ".join(v.keys())
            placeholders = ", ".join("?" for _ in v)
            c.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                list(v.values()),
            )
    conn.commit()
    conn.close()
    return db


def _session(sid, directory="C:/Dev/github/proj", time_updated=1790000000000):
    return {
        "id": sid,
        "project_id": "p1",
        "slug": "slug-" + sid,
        "directory": directory,
        "title": "T",
        "version": "1.18.32",
        "time_created": time_updated - 1000,
        "time_updated": time_updated,
    }


def _message(mid, session_id, role, time_created):
    return {
        "id": mid,
        "session_id": session_id,
        "time_created": time_created,
        "time_updated": time_created,
        "data": json.dumps({"role": role}),
    }


def _part(pid, message_id, session_id, time_created, data):
    return {
        "id": pid,
        "message_id": message_id,
        "session_id": session_id,
        "time_created": time_created,
        "time_updated": time_created,
        "data": json.dumps(data),
    }


# --- path resolution ---


def test_opencode_db_path_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCODE_DB", str(tmp_path / "custom.db"))
    assert _opencode_db_path() == tmp_path / "custom.db"


def test_opencode_db_path_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert _opencode_db_path() == tmp_path / "opencode" / "opencode.db"


# --- candidate discovery ---


def test_list_candidates_returns_empty_when_db_missing(tmp_path):
    provider = OpenCodeProvider(db_path=tmp_path / "missing.db")
    assert list(provider.list_candidates()) == []


def test_list_candidates_discovers_sessions_sorted_by_mtime(tmp_path):
    db = _make_db(
        tmp_path,
        [
            (
                "session",
                [
                    _session("ses_old", time_updated=1000000),
                    _session("ses_new", time_updated=2000000),
                ],
            )
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    refs = list(provider.list_candidates())

    assert [r.conversation_id for r in refs] == ["ses_new", "ses_old"]
    assert all(r.provider == "opencode" for r in refs)
    # epoch millis -> epoch seconds
    assert refs[0].mtime == 2000.0
    assert refs[1].mtime == 1000.0
    # locator must round-trip through load (same session id)
    assert refs[0].locator == "ses_new"


def test_list_candidates_max_sessions_pushes_limit_into_sql(tmp_path):
    sessions = [
        ("session", [_session(f"ses_{i}", time_updated=1000 * (i + 1)) for i in range(5)])
    ]
    db = _make_db(tmp_path, sessions)
    provider = OpenCodeProvider(db_path=db, max_sessions=2)
    refs = list(provider.list_candidates())

    # newest two, in recency order -- exact cut, no scan_multiplier needed
    assert [r.conversation_id for r in refs] == ["ses_4", "ses_3"]


# --- message parsing ---


def test_load_parses_user_and_assistant_text(tmp_path):
    sid = "ses_1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            (
                "message",
                [
                    _message("msg_1", sid, "user", 1000),
                    _message("msg_2", sid, "assistant", 2000),
                ],
            ),
            (
                "part",
                [
                    _part("prt_1", "msg_1", sid, 1000, {"type": "text", "text": "Hello!"}),
                    _part("prt_2", "msg_2", sid, 2000, {"type": "text", "text": "Hi there"}),
                ],
            ),
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert len(conv.messages) == 2
    assert conv.messages[0].role is Role.USER
    assert conv.messages[0].text == "Hello!"
    assert conv.messages[1].role is Role.ASSISTANT
    assert conv.messages[1].text == "Hi there"
    assert conv.project == "proj"


def test_load_skips_reasoning_and_step_parts(tmp_path):
    sid = "ses_1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            (
                "message",
                [
                    _message("msg_1", sid, "user", 1000),
                    _message("msg_2", sid, "assistant", 2000),
                ],
            ),
            (
                "part",
                [
                    _part("prt_1", "msg_1", sid, 1000, {"type": "text", "text": "go"}),
                    _part("prt_2", "msg_2", sid, 2000, {"type": "step-start"}),
                    _part("prt_3", "msg_2", sid, 2100, {"type": "reasoning", "text": "thinking"}),
                    _part("prt_4", "msg_2", sid, 2200, {"type": "text", "text": "done"}),
                    _part("prt_5", "msg_2", sid, 2300, {"type": "step-finish"}),
                ],
            ),
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert [m.text for m in conv.messages] == ["go", "done"]


def test_load_parses_tool_call_and_result(tmp_path):
    sid = "ses_1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            (
                "message",
                [
                    _message("msg_1", sid, "user", 1000),
                    _message("msg_2", sid, "assistant", 2000),
                ],
            ),
            (
                "part",
                [
                    _part("prt_1", "msg_1", sid, 1000, {"type": "text", "text": "run it"}),
                    _part(
                        "prt_2",
                        "msg_2",
                        sid,
                        2000,
                        {
                            "type": "tool",
                            "tool": "bash",
                            "callID": "call_abc",
                            "state": {
                                "status": "completed",
                                "input": {"command": "ls -la"},
                                "output": "file1.txt\nfile2.txt",
                            },
                        },
                    ),
                ],
            ),
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    tool_calls = [m for m in conv.messages if m.role is Role.TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0].label == "bash"
    assert "ls -la" in tool_calls[0].text

    tool_results = [m for m in conv.messages if m.role is Role.TOOL_RESULT]
    assert len(tool_results) == 1
    assert "file1.txt" in tool_results[0].text


def test_load_tool_with_non_string_output(tmp_path):
    sid = "ses_1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            (
                "message",
                [
                    _message("msg_1", sid, "user", 1000),
                    _message("msg_2", sid, "assistant", 2000),
                ],
            ),
            (
                "part",
                [
                    _part("prt_1", "msg_1", sid, 1000, {"type": "text", "text": "go"}),
                    _part(
                        "prt_2",
                        "msg_2",
                        sid,
                        2000,
                        {
                            "type": "tool",
                            "tool": "read",
                            "state": {
                                "status": "completed",
                                "input": {"filePath": "a.txt"},
                                "output": {"content": "hi", "truncated": False},
                            },
                        },
                    ),
                ],
            ),
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    tool_results = [m for m in conv.messages if m.role is Role.TOOL_RESULT]
    assert len(tool_results) == 1
    assert "hi" in tool_results[0].text


def test_load_skips_malformed_part_json(tmp_path):
    sid = "ses_1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            ("message", [_message("msg_1", sid, "user", 1000)]),
            (
                "part",
                [
                    {
                        "id": "prt_bad",
                        "message_id": "msg_1",
                        "session_id": sid,
                        "time_created": 999,
                        "time_updated": 999,
                        "data": "{not json",
                    },
                    _part("prt_1", "msg_1", sid, 1000, {"type": "text", "text": "ok"}),
                ],
            ),
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert [m.text for m in conv.messages] == ["ok"]


def test_load_trims_before_first_user(tmp_path):
    sid = "ses_1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            (
                "message",
                [
                    _message("msg_1", sid, "assistant", 1000),
                    _message("msg_2", sid, "user", 2000),
                ],
            ),
            (
                "part",
                [
                    _part("prt_1", "msg_1", sid, 1000, {"type": "text", "text": "system ack"}),
                    _part("prt_2", "msg_2", sid, 2000, {"type": "text", "text": "first user"}),
                ],
            ),
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert conv.messages[0].role is Role.USER
    assert conv.messages[0].text == "first user"


# --- checkpoint discovery ---


def test_checkpoint_discovery_through_opencode_provider(tmp_path):
    uuid = "27ebccde-2451-45c6-91b2-acc9156ef44e"
    checkpoint_text = f"CHAT CHECKPOINT UUID={uuid} SLUG=my-slug"
    sid = "ses_1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            (
                "message",
                [
                    _message("msg_1", sid, "user", 1000),
                    _message("msg_2", sid, "assistant", 2000),
                ],
            ),
            (
                "part",
                [
                    _part("prt_1", "msg_1", sid, 1000, {"type": "text", "text": "checkpoint please"}),
                    _part(
                        "prt_2",
                        "msg_2",
                        sid,
                        2000,
                        {
                            "type": "tool",
                            "tool": "chat-mother-forker_chat_checkpoint",
                            "state": {
                                "status": "completed",
                                "input": {"slug": "my-slug"},
                                "output": checkpoint_text,
                            },
                        },
                    ),
                ],
            ),
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    checkpoints = find_checkpoints(conv)
    assert len(checkpoints) == 1
    assert checkpoints[0].uuid == uuid
    assert checkpoints[0].slug == "my-slug"


def test_conversation_metadata_returns_db_path_and_session(tmp_path):
    sid = "ses_1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            ("message", [_message("msg_1", sid, "user", 1000)]),
            ("part", [_part("prt_1", "msg_1", sid, 1000, {"type": "text", "text": "hi"})]),
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    database, session_id = provider.conversation_metadata(ref)

    assert database == str(db)
    assert session_id == sid


def test_project_is_none_when_directory_empty(tmp_path):
    sid = "ses_1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid, directory="")]),
            ("message", [_message("msg_1", sid, "user", 1000)]),
            ("part", [_part("prt_1", "msg_1", sid, 1000, {"type": "text", "text": "hi"})]),
        ],
    )
    provider = OpenCodeProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert conv.project is None
