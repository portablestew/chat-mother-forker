import time

from chat_mother_forker.checkpoint import format_checkpoint_line
from chat_mother_forker.search import (
    SearchResults,
    gather_sorted_candidates,
    render_search_results,
    search_conversations,
)
from conftest import FakeProvider, assistant, tool_call, tool_result, user


def test_gather_sorted_candidates_merges_multiple_providers_by_recency():
    p1 = FakeProvider("p1")
    p1.add("a", mtime=10, messages=[user("a")])
    p1.add("b", mtime=30, messages=[user("b")])

    p2 = FakeProvider("p2")
    p2.add("c", mtime=20, messages=[user("c")])

    refs = gather_sorted_candidates([p1, p2], per_provider=100)
    assert [r.conversation_id for r in refs] == ["b", "c", "a"]


def test_gather_sorted_candidates_caps_per_provider_before_merging():
    p1 = FakeProvider("p1")
    for i in range(10):
        p1.add(f"p1-{i}", mtime=float(i), messages=[user("x")])

    p2 = FakeProvider("p2")
    p2.add("p2-only", mtime=1000.0, messages=[user("x")])

    # Cap p1 to its 2 most recent; p2 (only 1 conversation) is unaffected.
    refs = gather_sorted_candidates([p1, p2], per_provider=2)
    ids = [r.conversation_id for r in refs]
    assert ids == ["p2-only", "p1-9", "p1-8"]


def test_search_conversations_no_filter_returns_all_up_to_max_results(fake_provider):
    for i in range(5):
        fake_provider.add(f"c{i}", mtime=float(i), messages=[user(f"prompt {i}")])

    results = search_conversations([fake_provider], search=None, max_results=3)
    assert len(results) == 3
    # Newest first.
    assert [r.conversation_id for r in results] == ["c4", "c3", "c2"]


def test_search_conversations_current_chat_does_not_count_against_max_results(fake_provider):
    """The current chat is reported separately from `others` and shouldn't
    eat into `max_results` -- requesting 3 results should yield 3 others
    plus the current chat, not 2 others."""
    fake_provider.add(
        "current-convo",
        mtime=time.time(),
        messages=[user("hi"), tool_call("chat_search", text="{}")],
    )
    for i in range(5):
        fake_provider.add(f"c{i}", mtime=float(i), messages=[user(f"prompt {i}")])

    results = search_conversations([fake_provider], search=None, max_results=3)
    assert results.current is not None
    assert results.current.conversation_id == "current-convo"
    assert len(results.others) == 3
    assert [r.conversation_id for r in results.others] == ["c4", "c3", "c2"]


def test_search_conversations_filters_by_conversation_id_substring(fake_provider):
    fake_provider.add("alpha-123", mtime=1, messages=[user("hi")])
    fake_provider.add("beta-456", mtime=2, messages=[user("hi")])

    results = search_conversations([fake_provider], search="alpha")
    assert len(results) == 1
    assert results[0].conversation_id == "alpha-123"


def test_search_conversations_filters_by_transcript_substring(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("talking about widgets")])
    fake_provider.add("c2", mtime=2, messages=[user("talking about gadgets")])

    results = search_conversations([fake_provider], search="widgets")
    assert len(results) == 1
    assert results[0].conversation_id == "c1"


def test_search_conversations_filter_is_case_insensitive(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("Discussing WIDGETS today")])

    results = search_conversations([fake_provider], search="widgets")
    assert len(results) == 1


def test_search_conversations_filters_by_checkpoint_slug(fake_provider):
    line = format_checkpoint_line("release-plan")
    fake_provider.add(
        "c1", mtime=1, messages=[user("plan release"), tool_result(line)]
    )
    fake_provider.add("c2", mtime=2, messages=[user("unrelated")])

    results = search_conversations([fake_provider], search="release-plan")
    assert len(results) == 1
    assert results[0].conversation_id == "c1"


def test_search_conversations_filters_by_checkpoint_uuid(fake_provider):
    line = format_checkpoint_line("some-slug")
    checkpoint_uuid = line.split("UUID=")[1].split(" ")[0]
    fake_provider.add("c1", mtime=1, messages=[tool_result(line)])
    fake_provider.add("c2", mtime=2, messages=[user("unrelated")])

    results = search_conversations([fake_provider], search=checkpoint_uuid)
    assert len(results) == 1
    assert results[0].conversation_id == "c1"


def test_search_conversations_reports_all_checkpoints_regardless_of_search_term(fake_provider):
    line1 = format_checkpoint_line("first-slug")
    line2 = format_checkpoint_line("second-slug")
    fake_provider.add(
        "c1",
        mtime=1,
        messages=[user("looking for first-slug"), tool_result(line1), tool_result(line2)],
    )

    results = search_conversations([fake_provider], search="first-slug")
    assert len(results) == 1
    slugs = {cp.slug for cp in results[0].checkpoints}
    # Both checkpoints are reported even though the search term only matches one.
    assert slugs == {"first-slug", "second-slug"}


def test_search_conversations_no_match_returns_empty_list(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hello")])
    results = search_conversations([fake_provider], search="nonexistent-xyz")
    assert len(results) == 0


def test_search_conversations_ignores_tool_call_and_result_text(fake_provider):
    fake_provider.add(
        "c1",
        mtime=1,
        messages=[
            user("hello"),
            tool_call("grep", text='{"query":"unique-tool-phrase"}'),
            tool_result("output containing unique-tool-phrase"),
        ],
    )
    results = search_conversations([fake_provider], search="unique-tool-phrase")
    assert len(results) == 0


def test_search_conversations_preview_uses_first_user_message_only(fake_provider):
    fake_provider.add(
        "c1",
        mtime=1,
        messages=[
            assistant("some earlier assistant text should not appear"),
            user("the real first user prompt"),
            user("a later user message"),
        ],
    )
    results = search_conversations([fake_provider], search=None)
    assert results[0].preview == "the real first user prompt"


def test_render_search_results_no_results_no_search_term():
    assert render_search_results(SearchResults(), search=None) == "No conversations found."


def test_render_search_results_no_results_with_search_term():
    out = render_search_results(SearchResults(), search="xyz")
    assert "xyz" in out
    assert "No conversations" in out


def test_render_search_results_includes_slugs_when_present(fake_provider):
    line = format_checkpoint_line("my-slug")
    fake_provider.add("c1", mtime=1, messages=[user("hi"), tool_result(line)])
    results = search_conversations([fake_provider], search=None)
    out = render_search_results(results, search=None)
    assert "my-slug" in out
    assert "UUID=" in out


def test_render_search_results_omits_slugs_line_when_none_present(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hi")])
    results = search_conversations([fake_provider], search=None)
    out = render_search_results(results, search=None)
    assert "slugs:" not in out


def test_search_conversations_matched_in_reports_transcript_hit(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("talking about widgets today")])

    results = search_conversations([fake_provider], search="widgets")
    assert results[0].matched_in == ["transcript"]
    assert results[0].transcript_hit_count == 1


def test_search_conversations_matched_in_reports_conversation_id(fake_provider):
    fake_provider.add("alpha-123", mtime=1, messages=[user("hi")])

    results = search_conversations([fake_provider], search="alpha")
    assert results[0].matched_in == ["conversation_id"]
    assert results[0].transcript_hit_count == 0


def test_search_conversations_matched_in_reports_checkpoint_slug(fake_provider):
    line = format_checkpoint_line("release-plan")
    fake_provider.add("c1", mtime=1, messages=[user("hi"), tool_result(line)])

    results = search_conversations([fake_provider], search="release-plan")
    assert results[0].matched_in == ["checkpoint_slug"]


def test_search_conversations_matched_in_reports_checkpoint_uuid(fake_provider):
    line = format_checkpoint_line("some-slug")
    checkpoint_uuid = line.split("UUID=")[1].split(" ")[0]
    fake_provider.add("c1", mtime=1, messages=[tool_result(line)])

    results = search_conversations([fake_provider], search=checkpoint_uuid)
    assert results[0].matched_in == ["checkpoint_uuid"]


def test_search_conversations_matched_in_can_report_multiple_reasons(fake_provider):
    line = format_checkpoint_line("widgets-plan")
    fake_provider.add(
        "widgets-convo", mtime=1, messages=[user("talking about widgets"), tool_result(line)]
    )

    results = search_conversations([fake_provider], search="widgets")
    assert set(results[0].matched_in) == {"conversation_id", "checkpoint_slug", "transcript"}


def test_search_conversations_first_context_empty_when_id_only_match(fake_provider):
    fake_provider.add("alpha-123", mtime=1, messages=[user("hello there")])

    results = search_conversations([fake_provider], search="alpha")
    assert results[0].first_context == ""
    assert results[0].last_context == ""


def test_search_conversations_first_context_bolds_match(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("talking about widgets today")])

    results = search_conversations([fake_provider], search="widgets")
    assert "**widgets**" in results[0].first_context


def test_search_conversations_last_context_empty_when_single_hit(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("talking about widgets today")])

    results = search_conversations([fake_provider], search="widgets")
    assert results[0].transcript_hit_count == 1
    assert results[0].last_context == ""
    assert results[0].first_context != ""


def test_search_conversations_last_context_set_when_multiple_hits(fake_provider):
    fake_provider.add(
        "c1",
        mtime=1,
        messages=[user("widgets are great"), assistant("more widgets over here")],
    )

    results = search_conversations([fake_provider], search="widgets")
    assert results[0].transcript_hit_count == 2
    assert "**widgets**" in results[0].first_context
    assert "**widgets**" in results[0].last_context
    assert results[0].first_context != results[0].last_context


def test_search_conversations_transcript_hit_count_sums_across_messages(fake_provider):
    fake_provider.add(
        "c1",
        mtime=1,
        messages=[
            user("widgets widgets widgets"),
            assistant("one more widgets mention"),
        ],
    )

    results = search_conversations([fake_provider], search="widgets")
    assert results[0].transcript_hit_count == 4


def test_search_conversations_matched_in_empty_when_no_search(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hello")])

    results = search_conversations([fake_provider], search=None)
    assert results[0].matched_in == []
    assert results[0].first_context == ""


def test_render_search_results_shows_matched_in_and_context(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("talking about widgets today")])

    results = search_conversations([fake_provider], search="widgets")
    out = render_search_results(results, search="widgets")

    assert "matched: transcript (1 hit)" in out
    assert "first:" in out
    assert "**widgets**" in out
    assert "last:" not in out


def test_render_search_results_shows_plural_hits(fake_provider):
    fake_provider.add(
        "c1",
        mtime=1,
        messages=[user("widgets here"), assistant("widgets there")],
    )

    results = search_conversations([fake_provider], search="widgets")
    out = render_search_results(results, search="widgets")

    assert "matched: transcript (2 hits)" in out
    assert "first:" in out
    assert "last:" in out


def test_render_search_results_no_matched_line_when_no_search(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hi")])

    results = search_conversations([fake_provider], search=None)
    out = render_search_results(results, search=None)
    assert "matched:" not in out


def test_search_conversations_includes_project_when_available(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hi")], project="chat-mother-forker")

    results = search_conversations([fake_provider], search=None)
    assert results[0].project == "chat-mother-forker"


def test_search_conversations_project_is_none_when_unavailable(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hi")])

    results = search_conversations([fake_provider], search=None)
    assert results[0].project is None


def test_render_search_results_includes_project_in_header_line(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hi")], project="chat-mother-forker")

    results = search_conversations([fake_provider], search=None)
    out = render_search_results(results, search=None)

    lines = out.splitlines()
    header_line = next(line for line in lines if line.startswith("- "))
    assert "fake:c1" in header_line
    assert "chat-mother-forker" in header_line


def test_render_search_results_omits_project_when_none(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hi")])

    results = search_conversations([fake_provider], search=None)
    out = render_search_results(results, search=None)

    lines = out.splitlines()
    header_line = next(line for line in lines if line.startswith("- "))
    # No trailing " | " with nothing after it, and no stray "None".
    assert not header_line.endswith("| ")
    assert "None" not in header_line


def test_search_conversations_flags_current_chat(fake_provider):
    fake_provider.add(
        "current-convo",
        mtime=time.time(),
        messages=[user("test the search"), tool_call("chat_search", text='{"search":"x"}')],
    )
    fake_provider.add("other-convo", mtime=time.time(), messages=[user("hi")])

    results = search_conversations([fake_provider], search=None)
    assert results.current is not None
    assert results.current.conversation_id == "current-convo"
    assert [r.conversation_id for r in results.others] == ["other-convo"]


def test_search_conversations_not_current_when_last_message_is_not_chat_search_call(fake_provider):
    fake_provider.add(
        "c1",
        mtime=time.time(),
        messages=[user("hi"), tool_call("grep_search", text="{}")],
    )

    results = search_conversations([fake_provider], search=None)
    assert results.current is None
    assert results[0].is_current is False


def test_search_conversations_current_when_tool_result_already_present_for_own_tool(fake_provider):
    """A TOOL_RESULT immediately following a TOOL_CALL to one of our own
    tools is treated as current (not dismissed) -- this is what an
    atomic-write provider's just-returned call looks like on disk. See
    current_chat.py's module docstring for the full rationale.
    """
    fake_provider.add(
        "c1",
        mtime=time.time(),
        messages=[
            user("hi"),
            tool_call("chat_search", text="{}"),
            tool_result("1 conversation(s)"),
        ],
    )

    results = search_conversations([fake_provider], search=None)
    assert results.current is not None
    assert results.current.conversation_id == "c1"


def test_search_conversations_not_current_when_too_old(fake_provider):
    fake_provider.add(
        "c1",
        mtime=time.time() - 3600,
        messages=[user("hi"), tool_call("chat_search", text="{}")],
    )

    results = search_conversations([fake_provider], search=None)
    assert results.current is None
    assert results[0].is_current is False


def test_search_conversations_only_flags_one_current_chat(fake_provider):
    """If two candidates both heuristically look like the current chat
    (e.g. two IDE windows each mid-chat_search), only the newest is kept
    as `current` -- the other is demoted to a normal result."""
    fake_provider.add(
        "older",
        mtime=time.time() - 1,
        messages=[user("hi"), tool_call("chat_search", text="{}")],
    )
    fake_provider.add(
        "newer",
        mtime=time.time(),
        messages=[user("hi"), tool_call("chat_search", text="{}")],
    )

    results = search_conversations([fake_provider], search=None)
    assert results.current is not None
    assert results.current.conversation_id == "newer"
    assert [r.conversation_id for r in results.others] == ["older"]
    # The demoted candidate's is_current flag is cleared too.
    assert results.others[0].is_current is False


def test_search_conversations_current_chat_is_first_regardless_of_recency(fake_provider):
    """The current chat is always first in iteration order, even when a
    more recent (non-current) conversation exists."""
    fake_provider.add(
        "current-but-older",
        mtime=time.time() - 10,
        messages=[user("hi"), tool_call("chat_search", text="{}")],
    )
    fake_provider.add("newer-not-current", mtime=time.time(), messages=[user("hi")])

    results = search_conversations([fake_provider], search=None)
    assert [r.conversation_id for r in results] == ["current-but-older", "newer-not-current"]


def test_render_search_results_shows_current_chat_id_line(fake_provider):
    fake_provider.add(
        "current-convo",
        mtime=time.time(),
        messages=[user("test the search"), tool_call("chat_search", text="{}")],
    )

    results = search_conversations([fake_provider], search=None)
    out = render_search_results(results, search=None)

    assert "current chat id = fake:current-convo" in out
    # The usual metadata block is suppressed for the current-chat entry.
    assert "prompt:" not in out
    assert "messages:" not in out


def test_render_search_results_current_chat_not_counted_in_total(fake_provider):
    fake_provider.add(
        "current-convo",
        mtime=time.time(),
        messages=[user("test the search"), tool_call("chat_search", text="{}")],
    )
    for i in range(5):
        fake_provider.add(f"other-{i}", mtime=float(i), messages=[user(f"prompt {i}")])

    results = search_conversations([fake_provider], search=None)
    out = render_search_results(results, search=None)
    lines = out.splitlines()

    assert lines[0] == "current chat id = fake:current-convo"
    assert lines[1] == "5 conversation(s):"


def test_search_conversations_flags_current_chat_for_atomically_flushed_result(fake_provider):
    """Providers that flush a tool_call and its tool_result together
    (e.g. kiro_ide_v2, claude_code) never have an unanswered call on
    disk -- current-chat detection must still catch this via the
    TOOL_RESULT-preceded-by-our-own-TOOL_CALL case.
    """
    fake_provider.add(
        "current-convo",
        mtime=time.time(),
        messages=[
            user("test the search"),
            tool_call("mcp_chat_mother_forker_chat_search", text="{}"),
            tool_result("50 conversation(s):\n..."),
        ],
    )
    fake_provider.add("other-convo", mtime=time.time(), messages=[user("hi")])

    results = search_conversations([fake_provider], search=None)
    assert results.current is not None
    assert results.current.conversation_id == "current-convo"
    assert [r.conversation_id for r in results.others] == ["other-convo"]


def test_search_conversations_not_current_for_atomically_flushed_unrelated_tool(fake_provider):
    """A newest conversation whose flushed call+result pair is for an
    unrelated tool (not one of our own) must not be flagged current.
    """
    fake_provider.add(
        "c1",
        mtime=time.time(),
        messages=[user("hi"), tool_call("grep_search", text="{}"), tool_result("results")],
    )

    results = search_conversations([fake_provider], search=None)
    assert results.current is None
    assert results[0].is_current is False


def test_render_search_results_current_chat_with_zero_other_matches(fake_provider):
    fake_provider.add(
        "current-convo",
        mtime=time.time(),
        messages=[user("test the search"), tool_call("chat_search", text="{}")],
    )

    results = search_conversations([fake_provider], search="pippin docs")
    out = render_search_results(results, search="pippin docs")

    lines = out.splitlines()
    assert lines[0] == "current chat id = fake:current-convo"
    assert lines[1] == '0 conversation(s) matching "pippin docs".'
