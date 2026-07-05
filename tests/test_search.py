from chat_mother_forker.checkpoint import format_checkpoint_line
from chat_mother_forker.search import (
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
    assert results == []


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
    assert results == []


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
    assert render_search_results([], search=None) == "No conversations found."


def test_render_search_results_no_results_with_search_term():
    out = render_search_results([], search="xyz")
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
