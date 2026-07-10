"""Shared heuristic for detecting "the conversation currently calling one of
our own tools" -- used by both `chat_search` (to report it as
`current chat id = ...` instead of a normal result, excluded from the
match count) and `chat_fork` (to avoid a search string that happens to
match the calling conversation's own freshly-written transcript text from
self-forking instead of finding the conversation the user actually meant).
"""

from __future__ import annotations

import time

from chat_mother_forker.models import Conversation, ConversationRef, Role

# A conversation is treated as "the one calling one of our tools right now"
# when its newest message is an unanswered TOOL_CALL to one of our own tools
# (the response hasn't been written back into the transcript yet -- that
# only happens once the call returns) and its ConversationRef.mtime is
# within this many seconds of "now". This is a heuristic, not a guarantee: a
# turn that does a lot of work before finally calling the tool can push
# mtime further from "now" than this window allows, in which case the
# conversation is simply not flagged as current (rather than risk
# misidentifying an older, unrelated conversation).
CURRENT_CHAT_MAX_AGE_SECONDS = 60

# Tool names whose unanswered call marks a conversation as "current". Kept
# as the full set of this package's own tools (not just chat_search) since
# chat_fork faces the exact same self-match risk.
OWN_TOOL_NAMES = frozenset({"chat_search", "chat_fork", "chat_checkpoint"})


def is_current_conversation(ref: ConversationRef, conversation: Conversation) -> bool:
    """Heuristic: is `conversation` the one whose in-flight turn is the very
    tool call being served right now?

    True when its last message is an unanswered TOOL_CALL to one of
    `OWN_TOOL_NAMES` and the conversation's mtime is recent enough to
    plausibly be "now".
    """
    if not conversation.messages:
        return False
    last = conversation.messages[-1]
    if last.role is not Role.TOOL_CALL:
        return False
    if (last.label or "").strip().lower() not in OWN_TOOL_NAMES:
        return False
    return abs(time.time() - ref.mtime) < CURRENT_CHAT_MAX_AGE_SECONDS
