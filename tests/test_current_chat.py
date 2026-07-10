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


def test_is_current_conversation_false_once_tool_result_present():
    ref = _ref(time.time())
    conversation = Conversation(
        ref=ref,
        messages=[
            user("hi"),
            tool_call("chat_search", text="{}"),
            tool_result("1 conversation(s)"),
        ],
    )
    assert is_current_conversation(ref, conversation) is False


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
