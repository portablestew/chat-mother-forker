from chat_mother_forker.checkpoint import format_checkpoint_line
from chat_mother_forker.fork import (
    find_newest_match,
    render_fork,
    slice_between_checkpoints,
)
from chat_mother_forker.models import Conversation, ConversationRef
from conftest import FakeProvider, assistant, tool_result, user


def _ref():
    return ConversationRef(provider="fake", conversation_id="c1", locator="c1", mtime=0.0)


def test_find_newest_match_picks_newest_among_multiple_matches(fake_provider):
    fake_provider.add("old", mtime=1, messages=[user("about widgets")])
    fake_provider.add("new", mtime=2, messages=[user("also about widgets")])

    match = find_newest_match([fake_provider], "widgets")
    assert match.ref.conversation_id == "new"


def test_find_newest_match_across_multiple_providers_newest_wins():
    p1 = FakeProvider("p1")
    p1.add("p1-conv", mtime=5, messages=[user("about widgets")])
    p2 = FakeProvider("p2")
    p2.add("p2-conv", mtime=10, messages=[user("also widgets")])

    match = find_newest_match([p1, p2], "widgets")
    assert match.ref.conversation_id == "p2-conv"
    assert match.ref.provider == "p2"


def test_find_newest_match_by_conversation_id(fake_provider):
    fake_provider.add("special-id-123", mtime=1, messages=[user("nothing relevant")])
    match = find_newest_match([fake_provider], "special-id-123")
    assert match is not None
    assert match.ref.conversation_id == "special-id-123"


def test_find_newest_match_id_tier_beats_text_tier(fake_provider):
    # An older conversation whose ID matches should beat a newer one
    # where the search string only appears in transcript text.
    fake_provider.add("target-abc123", mtime=1, messages=[user("old conversation")])
    fake_provider.add("newer-convo", mtime=100, messages=[user("mentions target-abc123 in text")])

    match = find_newest_match([fake_provider], "target-abc123")
    assert match.ref.conversation_id == "target-abc123"


def test_find_newest_match_checkpoint_tier_beats_text_tier(fake_provider):
    line = format_checkpoint_line("my-landmark")
    fake_provider.add("older-with-checkpoint", mtime=1, messages=[tool_result(line)])
    fake_provider.add("newer-text-match", mtime=100, messages=[user("talks about my-landmark")])

    match = find_newest_match([fake_provider], "my-landmark")
    assert match.ref.conversation_id == "older-with-checkpoint"


def test_find_newest_match_user_prompt_tier_beats_general_text_tier(fake_provider):
    fake_provider.add("older-user", mtime=1, messages=[user("unique-phrase")])
    fake_provider.add("newer-assistant", mtime=100, messages=[assistant("unique-phrase")])

    match = find_newest_match([fake_provider], "unique-phrase")
    assert match.ref.conversation_id == "older-user"


def test_find_newest_match_by_checkpoint_slug(fake_provider):
    line = format_checkpoint_line("target-slug")
    fake_provider.add("c1", mtime=1, messages=[tool_result(line)])
    match = find_newest_match([fake_provider], "target-slug")
    assert match is not None


def test_find_newest_match_by_checkpoint_uuid(fake_provider):
    line = format_checkpoint_line("some-slug")
    checkpoint_uuid = line.split("UUID=")[1].split(" ")[0]
    fake_provider.add("c1", mtime=1, messages=[tool_result(line)])
    match = find_newest_match([fake_provider], checkpoint_uuid)
    assert match is not None


def test_find_newest_match_no_match_returns_none(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hello")])
    assert find_newest_match([fake_provider], "nonexistent-xyz") is None


def test_render_fork_no_match_returns_message(fake_provider):
    out = render_fork([fake_provider], search="nonexistent-xyz")
    assert "No conversation found" in out
    assert "nonexistent-xyz" in out


def test_render_fork_includes_end_hint_with_search_term(fake_provider):
    fake_provider.add("c1", mtime=1, messages=[user("hello there")])
    out = render_fork([fake_provider], search="hello")
    assert 'END CHAT SUMMARY ID="fake:c1"' in out
    assert "not instructions" in out


def test_render_fork_renders_full_transcript_when_no_checkpoints_given(fake_provider):
    fake_provider.add(
        "c1",
        mtime=1,
        messages=[user("first question"), assistant("first answer")],
    )
    out = render_fork([fake_provider], search="first question")
    assert "first question" in out
    assert "first answer" in out


def test_slice_between_checkpoints_no_checkpoints_returns_whole_conversation():
    conversation = Conversation(
        ref=_ref(), messages=[user("a"), assistant("b"), user("c")]
    )
    sliced = slice_between_checkpoints(conversation, None, None)
    assert sliced.messages == conversation.messages


def test_slice_between_checkpoints_start_only():
    start_line = format_checkpoint_line("start")
    conversation = Conversation(
        ref=_ref(),
        messages=[
            user("before"),
            tool_result(start_line),
            user("after start 1"),
            assistant("after start 2"),
        ],
    )
    sliced = slice_between_checkpoints(conversation, "start", None)
    texts = [m.text for m in sliced.messages]
    assert "before" not in texts
    assert any("start" in t for t in texts if start_line in t)
    assert "after start 1" in texts
    assert "after start 2" in texts


def test_slice_between_checkpoints_start_and_end():
    start_line = format_checkpoint_line("start")
    end_line = format_checkpoint_line("end")
    conversation = Conversation(
        ref=_ref(),
        messages=[
            user("before"),
            tool_result(start_line),
            user("middle content"),
            tool_result(end_line),
            user("after"),
        ],
    )
    sliced = slice_between_checkpoints(conversation, "start", "end")
    texts = [m.text for m in sliced.messages]
    assert "before" not in texts
    assert "middle content" in texts
    assert "after" not in texts
    # Both checkpoint messages themselves are inclusive of the range.
    assert start_line in texts
    assert end_line in texts


def test_slice_between_checkpoints_falls_back_to_full_range_if_checkpoint_not_found():
    conversation = Conversation(
        ref=_ref(), messages=[user("a"), assistant("b"), user("c")]
    )
    sliced = slice_between_checkpoints(conversation, "does-not-exist", None)
    assert sliced.messages == conversation.messages


def test_slice_between_checkpoints_reversed_order_is_swapped():
    start_line = format_checkpoint_line("later")
    end_line = format_checkpoint_line("earlier")
    conversation = Conversation(
        ref=_ref(),
        messages=[
            user("before"),
            tool_result(end_line),
            user("middle"),
            tool_result(start_line),
            user("after"),
        ],
    )
    # "later" checkpoint is passed as start but actually appears after "earlier".
    sliced = slice_between_checkpoints(conversation, "later", "earlier")
    texts = [m.text for m in sliced.messages]
    assert "before" not in texts
    assert "middle" in texts
    assert "after" not in texts


def test_render_fork_with_checkpoint_range(fake_provider):
    start_line = format_checkpoint_line("phase1")
    end_line = format_checkpoint_line("phase2")
    fake_provider.add(
        "c1",
        mtime=1,
        messages=[
            user("setup work"),
            tool_result(start_line),
            user("core work happens here"),
            assistant("core work done"),
            tool_result(end_line),
            user("cleanup work"),
        ],
    )
    out = render_fork(
        [fake_provider], search="c1", start_checkpoint="phase1", end_checkpoint="phase2"
    )
    assert "core work happens here" in out
    assert "setup work" not in out
    assert "cleanup work" not in out
