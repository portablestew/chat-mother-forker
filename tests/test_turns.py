from chat_mother_forker.models import Conversation, ConversationRef, Role
from chat_mother_forker.turns import (
    ASSISTANT_TURN,
    USER_TURN,
    group_into_turns,
    render_conversation,
    render_turn,
)
from conftest import assistant, tool_call, tool_result, user


def _ref():
    return ConversationRef(provider="fake", conversation_id="c1", locator="c1", mtime=0.0)


def test_group_into_turns_single_message_each_side():
    turns = group_into_turns([user("hi"), assistant("hello")])
    assert [t.kind for t in turns] == [USER_TURN, ASSISTANT_TURN]
    assert len(turns[0].messages) == 1
    assert len(turns[1].messages) == 1


def test_group_into_turns_merges_consecutive_same_side_messages():
    messages = [
        user("part one"),
        user("part two"),
        assistant("thinking"),
        tool_call("bash", "ls"),
        tool_result("file.txt"),
        user("thanks"),
    ]
    turns = group_into_turns(messages)

    assert [t.kind for t in turns] == [USER_TURN, ASSISTANT_TURN, USER_TURN]
    assert len(turns[0].messages) == 2
    assert len(turns[1].messages) == 3  # assistant text + tool_call + tool_result folded together
    assert len(turns[2].messages) == 1


def test_group_into_turns_empty_input():
    assert group_into_turns([]) == []


def test_group_into_turns_all_one_side():
    turns = group_into_turns([user("a"), user("b"), user("c")])
    assert len(turns) == 1
    assert turns[0].kind == USER_TURN
    assert len(turns[0].messages) == 3


def test_group_into_turns_alternating_every_message():
    messages = [user("1"), assistant("2"), user("3"), assistant("4")]
    turns = group_into_turns(messages)
    assert [t.kind for t in turns] == [USER_TURN, ASSISTANT_TURN, USER_TURN, ASSISTANT_TURN]
    assert all(len(t.messages) == 1 for t in turns)


def test_render_turn_user_has_header_and_quoted_text():
    turns = group_into_turns([user("hello world")])
    out = render_turn(turns[0])
    assert out.startswith(f"## {USER_TURN}")
    assert "> hello world" in out


def test_render_turn_multiline_text_quotes_every_line():
    turns = group_into_turns([user("line one\nline two\nline three")])
    out = render_turn(turns[0])
    assert "> line one" in out
    assert "> line two" in out
    assert "> line three" in out


def test_render_turn_tool_call_shows_tool_name_label():
    turns = group_into_turns([assistant("checking"), tool_call("grep", '{"query":"foo"}')])
    out = render_turn(turns[0])
    assert "TOOL_CALL: grep" in out
    assert '{"query":"foo"}' in out


def test_render_turn_tool_result_labeled_and_quoted():
    turns = group_into_turns([assistant("done"), tool_result("some output")])
    out = render_turn(turns[0])
    assert "TOOL_RESULT" in out
    assert "> some output" in out


def test_render_turn_applies_char_limit_to_whole_body():
    turns = group_into_turns([user("x" * 500)])
    out = render_turn(turns[0], max_chars=50)
    assert "characters truncated" in out
    # The header itself shouldn't count against the body limit in a way that
    # breaks rendering -- the body under the header should still be capped.
    body = out.split("\n", 1)[1]
    assert len(body) < 500


def test_render_turn_skips_messages_with_blank_text():
    turns = group_into_turns([assistant(""), tool_result("real output")])
    out = render_turn(turns[0])
    assert "real output" in out
    # Only one sub-section should appear since the blank assistant message
    # is skipped.
    assert out.count("TOOL_RESULT") == 1


def test_render_conversation_end_to_end_ordering_and_annotation():
    conversation = Conversation(
        ref=_ref(),
        messages=[
            user("What's the bug?"),
            assistant("Let me check."),
            tool_call("bash", "pytest"),
            tool_result("1 failed"),
            user("Here's more info"),
            assistant("Fixed it."),
        ],
    )
    rendered = render_conversation(conversation)

    # Turn headers appear in the right order.
    assert rendered.index(f"## {USER_TURN}") < rendered.index(f"## {ASSISTANT_TURN}")
    idx_first_assistant = rendered.index(f"## {ASSISTANT_TURN}")
    idx_second_user = rendered.index("Here's more info")
    idx_second_assistant = rendered.rindex(f"## {ASSISTANT_TURN}")
    assert idx_first_assistant < idx_second_user < idx_second_assistant

    assert "TOOL_CALL: bash" in rendered
    assert "1 failed" in rendered
    assert "Fixed it." in rendered


def test_render_conversation_caps_number_of_turns_with_middle_marker():
    # 60 alternating turns (user/assistant/user/assistant/...), well above
    # the default MAX_TURNS=50, forcing the turn-list truncation to kick in.
    messages = []
    for i in range(60):
        if i % 2 == 0:
            messages.append(user(f"user turn {i}"))
        else:
            messages.append(assistant(f"assistant turn {i}"))

    conversation = Conversation(ref=_ref(), messages=messages)
    rendered = render_conversation(conversation, max_turns=10)

    assert "turns truncated" in rendered
    # First few and last few turns should survive; middle ones should not.
    assert "user turn 0" in rendered
    assert "assistant turn 59" in rendered
    assert "user turn 30" not in rendered


def test_render_conversation_empty_messages_returns_empty_string():
    conversation = Conversation(ref=_ref(), messages=[])
    assert render_conversation(conversation) == ""
