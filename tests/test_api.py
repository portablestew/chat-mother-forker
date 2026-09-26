"""Tests for the public library API (`fork_chat`, `find_chats`, `ChatRef`).

These exercise the public library surface, using the
in-memory `FakeProvider` from conftest so no filesystem/DB is touched.
"""

from __future__ import annotations

import time

import pytest

from chat_mother_forker import ChatRef, find_chats, fork_chat, load_chat
from chat_mother_forker.api import default_providers
from conftest import FakeProvider, assistant, tool_call, tool_result, user


# --- fork_chat ---


def test_fork_chat_returns_transcript_string_with_footer(fake_provider):
    fake_provider.add(
        "conv-1",
        mtime=time.time() - 3600,
        messages=[user("build the thing"), assistant("done building")],
    )
    out = fork_chat("build the thing", providers=[fake_provider])

    assert "## USER" in out
    assert "build the thing" in out
    # The fork footer marks this as reference material, not instructions.
    assert "historical reference material" in out
    assert 'END CHAT SUMMARY ID="fake:conv-1"' in out


def test_fork_chat_no_match_returns_message(fake_provider):
    fake_provider.add("conv-1", mtime=time.time(), messages=[user("hello")])
    out = fork_chat("nonexistent-needle", providers=[fake_provider])
    assert "No conversation found" in out


def test_fork_chat_matches_by_composite_key(fake_provider):
    fake_provider.add("conv-1", mtime=time.time() - 100, messages=[user("hi")])
    # The `key` find_chats emits ("fake:conv-1") must round-trip into fork_chat.
    out = fork_chat("fake:conv-1", providers=[fake_provider])
    assert 'END CHAT SUMMARY ID="fake:conv-1"' in out


def test_fork_chat_defaults_to_default_providers(monkeypatch):
    """When `providers` is omitted, fork_chat uses default_providers()."""
    sentinel = FakeProvider(name="sentinel")
    sentinel.add("c1", mtime=time.time() - 10, messages=[user("marker-xyz")])
    monkeypatch.setattr("chat_mother_forker.api.default_providers", lambda: [sentinel])

    out = fork_chat("marker-xyz")
    assert 'END CHAT SUMMARY ID="sentinel:c1"' in out


# --- find_chats ---


def test_find_chats_returns_chatrefs_newest_first(fake_provider):
    now = time.time()
    fake_provider.add("old", mtime=now - 5000, messages=[user("old chat")])
    fake_provider.add("new", mtime=now - 100, messages=[user("new chat")])

    refs = find_chats(providers=[fake_provider])

    assert all(isinstance(r, ChatRef) for r in refs)
    assert [r.key for r in refs] == ["fake:new", "fake:old"]


def test_find_chats_key_feeds_fork_chat(fake_provider):
    fake_provider.add("conv-1", mtime=time.time() - 100, messages=[user("hello world")])
    refs = find_chats("hello world", providers=[fake_provider])

    assert len(refs) == 1
    out = fork_chat(refs[0].key, providers=[fake_provider])
    assert 'END CHAT SUMMARY ID="fake:conv-1"' in out


def test_find_chats_filters_by_search(fake_provider):
    now = time.time()
    fake_provider.add("a", mtime=now - 10, messages=[user("apples")])
    fake_provider.add("b", mtime=now - 20, messages=[user("oranges")])

    refs = find_chats("oranges", providers=[fake_provider])
    assert [r.key for r in refs] == ["fake:b"]


def test_find_chats_search_matches_checkpoint_slug(fake_provider):
    uuid = "27ebccde-2451-45c6-91b2-acc9156ef44e"
    fake_provider.add(
        "conv-1",
        mtime=time.time() - 10,
        messages=[
            user("do work"),
            tool_call("chat_checkpoint", '{"slug":"vision-done"}'),
            tool_result(f"CHAT CHECKPOINT UUID={uuid} SLUG=vision-done"),
        ],
    )
    refs = find_chats("vision-done", providers=[fake_provider])
    assert len(refs) == 1
    assert refs[0].checkpoints[0].slug == "vision-done"
    assert refs[0].checkpoints[0].uuid == uuid


def test_find_chats_search_excludes_tool_text(fake_provider):
    """Tool call/result text is not searchable (mirrors chat_search)."""
    fake_provider.add(
        "conv-1",
        mtime=time.time() - 10,
        messages=[user("hello"), tool_call("grep", '{"q":"secret-token"}')],
    )
    assert find_chats("secret-token", providers=[fake_provider]) == []


def test_find_chats_carries_size_and_project(fake_provider):
    fake_provider.add(
        "conv-1",
        mtime=time.time() - 10,
        messages=[user("hello there")],
        project="my-workspace",
        size_bytes=123456,
    )
    refs = find_chats(providers=[fake_provider])
    assert refs[0].size_bytes == 123456
    assert refs[0].project == "my-workspace"
    # File-backed FakeProvider: locator set, no database/session.
    assert refs[0].locator == "conv-1"
    assert refs[0].database is None


def test_find_chats_respects_limit(fake_provider):
    now = time.time()
    for i in range(5):
        fake_provider.add(f"c{i}", mtime=now - i, messages=[user(f"chat {i}")])

    refs = find_chats(providers=[fake_provider], limit=2)
    assert len(refs) == 2


def test_find_chats_filter_by_project_workflow(fake_provider):
    """The intended harness pattern: list, filter by workspace, pick newest."""
    now = time.time()
    fake_provider.add("mine-old", mtime=now - 900, messages=[user("x")], project="ws")
    fake_provider.add("mine-new", mtime=now - 100, messages=[user("y")], project="ws")
    fake_provider.add("other", mtime=now - 50, messages=[user("z")], project="elsewhere")

    mine = [r for r in find_chats(providers=[fake_provider]) if r.project == "ws"]
    mine.sort(key=lambda r: r.mtime, reverse=True)
    assert [r.key for r in mine] == ["fake:mine-new", "fake:mine-old"]


def test_default_providers_returns_fresh_list_each_call():
    a = default_providers()
    b = default_providers()
    assert a is not b
    assert [p.name for p in a] == [p.name for p in b]


# --- load_chat ---


def test_load_chat_returns_full_untruncated_transcript(fake_provider):
    fake_provider.add(
        "conv-1",
        mtime=time.time() - 3600,
        messages=[user("build the thing"), assistant("done building")],
    )
    out = load_chat("build the thing", providers=[fake_provider])

    assert "## USER" in out
    assert "build the thing" in out
    assert "done building" in out
    # No fork footer: load_chat is a raw data API.
    assert "historical reference material" not in out
    assert "END CHAT SUMMARY" not in out


def test_load_chat_no_match_raises_lookup_error(fake_provider):
    fake_provider.add("conv-1", mtime=time.time(), messages=[user("hello")])
    with pytest.raises(LookupError) as exc_info:
        load_chat("nonexistent-needle", providers=[fake_provider])
    assert "nonexistent-needle" in str(exc_info.value)


def test_load_chat_matches_by_composite_key(fake_provider):
    fake_provider.add("conv-1", mtime=time.time() - 100, messages=[user("hi")])
    out = load_chat("fake:conv-1", providers=[fake_provider])
    assert "hi" in out


def test_load_chat_includes_tool_calls_and_results(fake_provider):
    """load_chat is untruncated, so tool call/result parts survive --
    the whole point vs. fork_chat's size-bounded summary."""
    fake_provider.add(
        "conv-1",
        mtime=time.time() - 10,
        messages=[
            user("check the tree"),
            assistant("running git status"),
            tool_call("bash", "git status"),
            tool_result("On branch main\nnothing to commit, clean tree"),
        ],
    )
    out = load_chat("check the tree", providers=[fake_provider])

    assert "TOOL_CALL: bash" in out
    assert "On branch main" in out
    assert "nothing to commit, clean tree" in out


def test_load_chat_defaults_to_default_providers(monkeypatch):
    """When `providers` is omitted, load_chat uses default_providers()."""
    sentinel = FakeProvider(name="sentinel")
    sentinel.add("c1", mtime=time.time() - 10, messages=[user("marker-xyz")])
    monkeypatch.setattr("chat_mother_forker.api.default_providers", lambda: [sentinel])

    out = load_chat("marker-xyz")
    assert "marker-xyz" in out


def test_load_chat_does_not_truncate_very_long_turns(fake_provider):
    """A turn longer than fork_chat's per-turn budget must come through whole."""
    long_body = "x" * 100_000
    fake_provider.add("conv-1", mtime=time.time() - 10, messages=[user("needle"), assistant(long_body)])
    out = load_chat("needle", providers=[fake_provider])

    assert long_body in out
    assert "characters truncated" not in out


def test_load_chat_uncaps_turn_count_where_fork_truncates(fake_provider):
    """Contrast on the same data: fork_chat drops the middle turns per
    MAX_TURNS; load_chat returns every one of them."""
    messages = []
    for i in range(60):
        if i % 2 == 0:
            messages.append(user(f"turn {i} of the whole story"))
        else:
            messages.append(assistant(f"turn {i} answer"))
    fake_provider.add("conv-1", mtime=time.time() - 10, messages=messages)

    forked = fork_chat("conv-1", providers=[fake_provider])
    loaded = load_chat("conv-1", providers=[fake_provider])

    assert "turns truncated" in forked
    assert "turns truncated" not in loaded
    assert "turn 30 of the whole story" in loaded
    assert "turn 30 of the whole story" not in forked
