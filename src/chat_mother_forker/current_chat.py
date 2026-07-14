"""Shared heuristic for detecting "the conversation currently calling one of
our own tools" -- used by both `chat_search` (to report it as
`current chat id = ...` instead of a normal result, excluded from the
match count) and `chat_fork` (to avoid a search string that happens to
match the calling conversation's own freshly-written transcript text from
self-forking instead of finding the conversation the user actually meant).

Different providers record a *different* raw label for the exact same MCP
tool call, since each host embeds its own naming convention for MCP tools
into the string it hands the model:

    kiro_cli / kiro_ide (v1) -> "chat_search"                          (bare)
    kiro_ide_v2            -> "mcp_chat_mother_forker_chat_search"    (single underscore)
    claude_code            -> "mcp__chat-mother-forker__chat_search"  (double underscore)

So matching against `OWN_TOOL_NAMES` can't be an exact-equality check --
it has to tolerate an arbitrary `<prefix>_<tool_name>` wrapper. See
`_matches_own_tool_name`.

Providers also differ in *when* a tool_call's response lands on disk.
Some (the original `kiro_ide` execution-log format, `kiro_cli`) write the
call first and only backfill the result once the tool returns, so mid-call
the transcript genuinely ends on an *unanswered* TOOL_CALL -- that's the
primary heuristic below. Others (`kiro_ide_v2`, `claude_code`) flush a
tool_call and its tool_result to disk together, same millisecond
timestamp -- an unanswered call never exists on disk for the primary
heuristic to catch. For those, `is_current_conversation` also accepts the
transcript ending on a TOOL_RESULT whose *immediately preceding* message
is a TOOL_CALL to one of our own tools (i.e. the call+response pair that
was just flushed as this very invocation returns).
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
# chat_fork faces the exact same self-match risk. These are the *bare*
# names -- see `_matches_own_tool_name` for how a provider-specific
# wrapped label (e.g. "mcp_chat_mother_forker_chat_search") is matched
# against this set.
OWN_TOOL_NAMES = frozenset({"chat_search", "chat_fork", "chat_checkpoint"})


def _matches_own_tool_name(label: str) -> bool:
    """True when `label` (a provider's raw tool_call label, already
    stripped/lowercased) refers to one of `OWN_TOOL_NAMES`.

    Accepts both the bare name and a "<host-prefix>_<tool_name>" wrapper --
    covers every naming convention seen in practice (see module docstring):
    bare ("chat_search"), single-underscore-joined
    ("mcp_chat_mother_forker_chat_search"), and double-underscore-joined
    ("mcp__chat-mother-forker__chat_search", which lowercasing plus this
    suffix check also handles since it still ends in "_chat_search").
    """
    if label in OWN_TOOL_NAMES:
        return True
    return any(label.endswith("_" + name) for name in OWN_TOOL_NAMES)


def is_current_conversation(ref: ConversationRef, conversation: Conversation) -> bool:
    """Heuristic: is `conversation` the one whose in-flight turn is the very
    tool call being served right now?

    True when the conversation's mtime is recent enough to plausibly be
    "now" (`CURRENT_CHAT_MAX_AGE_SECONDS`), and either:

    - its last message is an unanswered TOOL_CALL to one of `OWN_TOOL_NAMES`
      (bare or provider-wrapped, see `_matches_own_tool_name`) -- the
      response hasn't been backfilled yet; or
    - its last message is a TOOL_RESULT whose immediately preceding
      message is a TOOL_CALL to one of `OWN_TOOL_NAMES` -- covers
      providers that flush a call and its result together (see module
      docstring), where an unanswered call never exists on disk.
    """
    if not conversation.messages:
        return False
    if abs(time.time() - ref.mtime) >= CURRENT_CHAT_MAX_AGE_SECONDS:
        return False

    messages = conversation.messages
    last = messages[-1]

    if last.role is Role.TOOL_CALL:
        return _matches_own_tool_name((last.label or "").strip().lower())

    if last.role is Role.TOOL_RESULT and len(messages) >= 2:
        preceding = messages[-2]
        if preceding.role is Role.TOOL_CALL:
            return _matches_own_tool_name((preceding.label or "").strip().lower())

    return False
