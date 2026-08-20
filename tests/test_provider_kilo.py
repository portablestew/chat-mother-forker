import json
import sqlite3

from chat_mother_forker.checkpoint import find_checkpoints
from chat_mother_forker.models import Role
from chat_mother_forker.providers.kilo import KiloProvider, _kilo_db_path


def _make_db(tmp_path, rows):
    """Create a minimal kilo.db with the given session/message/part rows.

    `rows` is a list of (table, values) tuples where values is a list of
    row dicts. Simple helper: build tables directly.
    """
    db = tmp_path / "kilo.db"
    conn = sqlite3.connect(str(db))
    c = conn.cursor()
    c.execute(
        "CREATE TABLE session ("
        "id TEXT PRIMARY KEY, project_id TEXT NOT NULL, slug TEXT NOT NULL, "
        "directory TEXT NOT NULL, title TEXT NOT NULL, version TEXT NOT NULL, "
        "cost REAL NOT NULL, tokens_input INTEGER NOT NULL, tokens_output INTEGER NOT NULL, "
        "tokens_reasoning INTEGER NOT NULL, tokens_cache_read INTEGER NOT NULL, "
        "tokens_cache_write INTEGER NOT NULL, time_created INTEGER NOT NULL, "
        "time_updated INTEGER NOT NULL)"
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


def _session(sid, directory="C:/Dev/github/proj", time_updated=1787259677872):
    return {
        "id": sid,
        "project_id": "p1",
        "slug": "slug-" + sid,
        "directory": directory,
        "title": "T",
        "version": "1",
        "cost": 0.0,
        "tokens_input": 0,
        "tokens_output": 0,
        "tokens_reasoning": 0,
        "tokens_cache_read": 0,
        "tokens_cache_write": 0,
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


# --- home resolution ---


def test_kilo_db_path_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KILO_DB", str(tmp_path / "custom.db"))
    assert _kilo_db_path() == tmp_path / "custom.db"


def test_kilo_db_path_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("KILO_DB", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert _kilo_db_path() == tmp_path / "kilo" / "kilo.db"


# --- candidate discovery ---


def test_list_candidates_returns_empty_when_db_missing(tmp_path):
    provider = KiloProvider(db_path=tmp_path / "missing.db")
    assert list(provider.list_candidates()) == []


def test_list_candidates_discovers_sessions_sorted_by_mtime(tmp_path):
    db = _make_db(
        tmp_path,
        [
            (
                "session",
                [
                    _session("old", time_updated=1000000),
                    _session("new", time_updated=2000000),
                ],
            )
        ],
    )
    provider = KiloProvider(db_path=db)
    refs = list(provider.list_candidates())

    assert [r.conversation_id for r in refs] == ["new", "old"]
    assert all(r.provider == "kilo" for r in refs)
    assert refs[0].mtime == 2000.0
    assert refs[1].mtime == 1000.0
    # locator must round-trip through load (same session id)
    assert refs[0].locator == "new"


# --- message parsing ---


def test_load_parses_user_and_assistant_text(tmp_path):
    sid = "s1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            (
                "message",
                [
                    _message("m1", sid, "user", 1000),
                    _message("m2", sid, "assistant", 2000),
                ],
            ),
            (
                "part",
                [
                    _part("p1", "m1", sid, 1000, {"type": "text", "text": "Hello!"}),
                    _part("p2", "m2", sid, 2000, {"type": "text", "text": "Hi there"}),
                ],
            ),
        ],
    )
    provider = KiloProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert len(conv.messages) == 2
    assert conv.messages[0].role is Role.USER
    assert conv.messages[0].text == "Hello!"
    assert conv.messages[1].role is Role.ASSISTANT
    assert conv.messages[1].text == "Hi there"
    assert conv.project == "proj"


def test_load_skips_reasoning_and_step_parts(tmp_path):
    sid = "s1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            (
                "message",
                [
                    _message("m1", sid, "user", 1000),
                    _message("m2", sid, "assistant", 2000),
                ],
            ),
            (
                "part",
                [
                    _part("p1", "m1", sid, 1000, {"type": "text", "text": "go"}),
                    _part("p2", "m2", sid, 2000, {"type": "step-start"}),
                    _part("p3", "m2", sid, 2100, {"type": "reasoning", "text": "thinking"}),
                    _part("p4", "m2", sid, 2200, {"type": "text", "text": "done"}),
                    _part("p5", "m2", sid, 2300, {"type": "step-finish"}),
                ],
            ),
        ],
    )
    provider = KiloProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert [m.text for m in conv.messages] == ["go", "done"]


def test_load_parses_tool_call_and_result(tmp_path):
    sid = "s1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            ("message", [_message("m1", sid, "user", 1000), _message("m2", sid, "assistant", 2000)]),
            (
                "part",
                [
                    _part("p1", "m1", sid, 1000, {"type": "text", "text": "run it"}),
                    _part(
                        "p2",
                        "m2",
                        sid,
                        2000,
                        {
                            "type": "tool",
                            "tool": "bash",
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
    provider = KiloProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    tool_calls = [m for m in conv.messages if m.role is Role.TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0].label == "bash"
    assert "ls -la" in tool_calls[0].text

    tool_results = [m for m in conv.messages if m.role is Role.TOOL_RESULT]
    assert len(tool_results) == 1
    assert "file1.txt" in tool_results[0].text


def test_load_trims_before_first_user(tmp_path):
    sid = "s1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            (
                "message",
                [
                    _message("m1", sid, "assistant", 1000),
                    _message("m2", sid, "user", 2000),
                ],
            ),
            (
                "part",
                [
                    _part("p1", "m1", sid, 1000, {"type": "text", "text": "system ack"}),
                    _part("p2", "m2", sid, 2000, {"type": "text", "text": "first user"}),
                ],
            ),
        ],
    )
    provider = KiloProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert conv.messages[0].role is Role.USER
    assert conv.messages[0].text == "first user"


# --- checkpoint discovery ---


def test_checkpoint_discovery_through_kilo_provider(tmp_path):
    uuid = "27ebccde-2451-45c6-91b2-acc9156ef44e"
    checkpoint_text = f"CHAT CHECKPOINT UUID={uuid} SLUG=my-slug"
    sid = "s1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid)]),
            ("message", [_message("m1", sid, "user", 1000), _message("m2", sid, "assistant", 2000)]),
            (
                "part",
                [
                    _part("p1", "m1", sid, 1000, {"type": "text", "text": "checkpoint please"}),
                    _part(
                        "p2",
                        "m2",
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
    provider = KiloProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    checkpoints = find_checkpoints(conv)
    assert len(checkpoints) == 1
    assert checkpoints[0].uuid == uuid
    assert checkpoints[0].slug == "my-slug"


def test_project_is_none_when_directory_empty(tmp_path):
    sid = "s1"
    db = _make_db(
        tmp_path,
        [
            ("session", [_session(sid, directory="")]),
            ("message", [_message("m1", sid, "user", 1000)]),
            ("part", [_part("p1", "m1", sid, 1000, {"type": "text", "text": "hi"})]),
        ],
    )
    provider = KiloProvider(db_path=db)
    ref = next(iter(provider.list_candidates()))
    conv = provider.load(ref)

    assert conv.project is None