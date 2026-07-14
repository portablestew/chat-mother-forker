import time

from chat_mother_forker.current_chat import is_current_conversation
from chat_mother_forker.models import Conversation, ConversationRef
from conftest import tool_call, tool_result, user


def _ref(mtime: float) -> ConversationRef:
    return ConversationRef(provider="fake", conversation_id="c1", locator="c1", mtime=mtime)


def test_is_current_conversation_true_for_recent_unanswered_chat_search_call():
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref, messages=[user("hi"), tool_call("chat_search", text="{}")]
    )
    assert is_current_conversation(ref, conversation) is True


def test_is_current_conversation_true_for_recent_unanswered_chat_fork_call():
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref, messages=[user("hi"), tool_call("chat_fork", text="{}")]
    )
    assert is_current_conversation(ref, conversation) is True


def test_is_current_conversation_true_for_recent_unanswered_chat_checkpoint_call():
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref, messages=[user("hi"), tool_call("chat_checkpoint", text="{}")]
    )
    assert is_current_conversation(ref, conversation) is True


def test_is_current_conversation_false_for_unrelated_tool_call():
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref, messages=[user("hi"), tool_call("grep_search", text="{}")]
    )
    assert is_current_conversation(ref, conversation) is False


def test_is_current_conversation_true_for_kiro_ide_v2_wrapped_label():
    """kiro_ide_v2 records tool_call labels as toolName
    "mcp_chat_mother_forker_chat_search" (single-underscore-joined), not
    the bare "chat_search" every other provider uses.
    """
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[user("hi"), tool_call("mcp_chat_mother_forker_chat_search", text="{}")],
    )
    assert is_current_conversation(ref, conversation) is True


def test_is_current_conversation_true_for_claude_code_wrapped_label():
    """claude_code records tool_call labels as
    "mcp__chat-mother-forker__chat_fork" (double-underscore-joined)."""
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[user("hi"), tool_call("mcp__chat-mother-forker__chat_fork", text="{}")],
    )
    assert is_current_conversation(ref, conversation) is True


def test_is_current_conversation_false_for_wrapped_label_of_unrelated_tool():
    """A provider-wrapped label for a *different* tool (not one of our own)
    must not accidentally match via the suffix check.
    """
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[user("hi"), tool_call("mcp_pyddock_run_python", text="{}")],
    )
    assert is_current_conversation(ref, conversation) is False


def test_is_current_conversation_false_for_wrapped_label_with_similar_suffix():
    """A tool name that merely *contains* one of our tool names as a
    substring, but not as a "_<name>" suffix, must not match -- e.g. some
    hypothetical "other_chat_searcher" tool.
    """
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[user("hi"), tool_call("other_chat_searcher", text="{}")],
    )
    assert is_current_conversation(ref, conversation) is False


def test_is_current_conversation_true_for_atomically_flushed_tool_result():
    """Some providers (kiro_ide_v2, claude_code) write a tool_call and its
    tool_result together -- same-millisecond timestamps, no moment where
    the call is genuinely unanswered. The transcript ending on a
    TOOL_RESULT whose immediately preceding message is a TOOL_CALL to one
    of our own tools must still be treated as current.
    """
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[
            user("hi"),
            tool_call("mcp_chat_mother_forker_chat_search", text="{}"),
            tool_result('50 conversation(s):\n...'),
        ],
    )
    assert is_current_conversation(ref, conversation) is True


def test_is_current_conversation_false_when_tool_result_precedes_unrelated_call():
    """A TOOL_RESULT following a TOOL_CALL to an unrelated tool (e.g.
    grep_search) must not be flagged current just because it's the newest
    conversation -- this was the bug in an earlier, too-broad "newest
    conversation wins" fallback attempt.
    """
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[user("hi"), tool_call("grep_search", text="{}"), tool_result("results here")],
    )
    assert is_current_conversation(ref, conversation) is False


def test_is_current_conversation_false_when_result_precedes_non_tool_call():
    """A TOOL_RESULT whose immediately preceding message is not a
    TOOL_CALL at all (e.g. it's a USER message -- a malformed/unusual
    transcript) must not be flagged current.
    """
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[tool_result("orphaned result"), user("hi")],
    )
    # Last message is USER here, not TOOL_RESULT, so this should be False
    # via the ordinary "last message isn't TOOL_CALL/TOOL_RESULT-preceded"
    # path. Covered for completeness alongside the next case.
    assert is_current_conversation(ref, conversation) is False


def test_is_current_conversation_true_for_atomically_flushed_wrapped_claude_code_label():
    """The atomic-flush fallback also normalizes Claude Code's
    double-underscore wrapped label.
    """
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[
            user("hi"),
            tool_call("mcp__chat-mother-forker__chat_fork", text="{}"),
            tool_result("## USER\n..."),
        ],
    )
    assert is_current_conversation(ref, conversation) is True


def test_is_current_conversation_false_for_atomically_flushed_result_when_too_old():
    ref = _ref(time.time() - 3600)
    conversation = Conversation(
        ref=ref,
        messages=[
            user("hi"),
            tool_call("mcp_chat_mother_forker_chat_search", text="{}"),
            tool_result("results"),
        ],
    )
    assert is_current_conversation(ref, conversation) is False


def test_is_current_conversation_false_for_single_message_tool_result():
    """A TOOL_RESULT as the very first (and only) message has no preceding
    message to check -- must not raise, must return False.
    """
    ref = _ref(time.time())
    conversation = Conversation(ref=ref, messages=[tool_result("orphaned")])
    assert is_current_conversation(ref, conversation) is False


def test_is_current_conversation_true_once_tool_result_present_for_own_tool():
    """Superseded expectation: a TOOL_RESULT immediately following a
    TOOL_CALL to one of our own tools is now treated as current, not
    dismissed. This was previously asserted False under the assumption
    that a written-back result always means the call resolved some time
    ago -- but atomic-write providers (kiro_ide_v2, claude_code) write the
    call and its result together, so this exact shape is what "the call
    that's returning right now" looks like on disk for them. See the
    atomically-flushed tests above for the full rationale.
    """
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[
            user("hi"),
            tool_call("chat_search", text="{}"),
            tool_result("1 conversation(s)"),
        ],
    )
    assert is_current_conversation(ref, conversation) is True


def test_is_current_conversation_false_when_too_old():
    ref = _ref(time.time() - 3600)
    conversation = Conversation(
        ref=ref, messages=[user("hi"), tool_call("chat_search", text="{}")]
    )
    assert is_current_conversation(ref, conversation) is False


def test_is_current_conversation_false_for_empty_conversation():
    ref = _ref(time.time())
    conversation = Conversation(ref=ref, messages=[])
    assert is_current_conversation(ref, conversation) is False
