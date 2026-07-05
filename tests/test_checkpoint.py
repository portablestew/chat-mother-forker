import re

from chat_mother_forker.checkpoint import (
    MAX_SLUG_CHARS,
    format_checkpoint_line,
    find_checkpoints,
)
from chat_mother_forker.models import Conversation, ConversationRef
from conftest import assistant, tool_result, user


def _ref():
    return ConversationRef(provider="fake", conversation_id="c1", locator="c1", mtime=0.0)


def test_format_checkpoint_line_matches_expected_shape():
    line = format_checkpoint_line("my-slug")
    match = re.match(r"^CHAT CHECKPOINT UUID=([0-9a-fA-F-]{36}) SLUG=my-slug$", line)
    assert match is not None


def test_format_checkpoint_line_generates_unique_uuids():
    line1 = format_checkpoint_line("same-slug")
    line2 = format_checkpoint_line("same-slug")
    assert line1 != line2


def test_format_checkpoint_line_truncates_long_slugs():
    long_slug = "s" * 500
    line = format_checkpoint_line(long_slug)
    slug_part = line.split("SLUG=", 1)[1]
    assert len(slug_part) == MAX_SLUG_CHARS


def test_format_checkpoint_line_strips_whitespace():
    line = format_checkpoint_line("   padded   ")
    assert line.endswith("SLUG=padded")


def test_find_checkpoints_extracts_from_tool_result():
    line = format_checkpoint_line("my-checkpoint")
    conversation = Conversation(
        ref=_ref(),
        messages=[user("checkpoint please"), tool_result(line)],
    )
    checkpoints = find_checkpoints(conversation)
    assert len(checkpoints) == 1
    assert checkpoints[0].slug == "my-checkpoint"
    assert len(checkpoints[0].uuid) == 36


def test_find_checkpoints_ignores_assistant_prose_mentioning_checkpoints():
    # An assistant *talking about* a checkpoint line (e.g. paraphrasing it in
    # a user-facing message) must not be picked up -- only TOOL_RESULT text
    # counts, since that's the only place the literal line reliably survives.
    prose = "I created a checkpoint with UUID=11111111-1111-1111-1111-111111111111 SLUG=fake"
    conversation = Conversation(ref=_ref(), messages=[assistant(prose)])
    assert find_checkpoints(conversation) == []


def test_find_checkpoints_ignores_user_text_even_if_it_matches_pattern():
    line = format_checkpoint_line("user-typed")
    conversation = Conversation(ref=_ref(), messages=[user(line)])
    assert find_checkpoints(conversation) == []


def test_find_checkpoints_multiple_checkpoints_across_conversation():
    line1 = format_checkpoint_line("first")
    line2 = format_checkpoint_line("second")
    conversation = Conversation(
        ref=_ref(),
        messages=[
            tool_result(line1),
            user("more work"),
            assistant("done"),
            tool_result(line2),
        ],
    )
    checkpoints = find_checkpoints(conversation)
    assert [cp.slug for cp in checkpoints] == ["first", "second"]


def test_find_checkpoints_one_checkpoint_per_tool_result_message():
    # A chat_checkpoint TOOL_RESULT is always exactly one message whose text
    # is exactly one checkpoint line -- one match per message, at most.
    line1 = format_checkpoint_line("a")
    line2 = format_checkpoint_line("b")
    conversation = Conversation(
        ref=_ref(),
        messages=[tool_result(line1), user("more work"), tool_result(line2)],
    )
    checkpoints = find_checkpoints(conversation)
    assert [cp.slug for cp in checkpoints] == ["a", "b"]


def test_find_checkpoints_no_checkpoints_returns_empty_list():
    conversation = Conversation(
        ref=_ref(),
        messages=[user("hello"), assistant("hi"), tool_result("plain output")],
    )
    assert find_checkpoints(conversation) == []


def test_find_checkpoints_ignores_tool_result_where_checkpoint_is_not_at_the_start():
    # If a checkpoint line appears anywhere other than the very start of a
    # TOOL_RESULT's text, it isn't a real chat_checkpoint output (that tool
    # never bundles anything else in) -- don't match it.
    line = format_checkpoint_line("embedded")
    noisy = f"some preamble\n{{\"result\": \"ok\"}}\n{line}\ntrailing junk"
    conversation = Conversation(ref=_ref(), messages=[tool_result(noisy)])
    assert find_checkpoints(conversation) == []


def test_find_checkpoints_ignores_checkpoint_quoted_inside_chat_fork_output():
    # A chat_fork TOOL_RESULT renders a nested transcript of another
    # conversation, always starting with a turn header. A checkpoint that
    # was genuinely created in that *other* conversation, however it's
    # quoted somewhere inside, must not be picked up here -- only a
    # checkpoint at this message's own start counts.
    line = format_checkpoint_line("from-another-conversation")
    nested_fork_output = f"## ASSISTANT\n> {line}\n\n---\nEND CHAT SUMMARY"
    conversation = Conversation(ref=_ref(), messages=[tool_result(nested_fork_output)])
    assert find_checkpoints(conversation) == []
