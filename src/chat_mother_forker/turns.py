"""Turn grouping and rendering.

A "turn" is a maximal run of consecutive messages from the same side of the
conversation: all consecutive messages from the user form one user turn, and
all consecutive messages from anyone/anything else (assistant text, tool
calls, tool results, ...) form one assistant turn. A turn's rendered text is
the concatenation of its messages, each annotated with who/what produced it,
then middle-truncated as a whole if it's too long.
"""

from __future__ import annotations

from dataclasses import dataclass

from chat_mother_forker.models import Conversation, Message, Role
from chat_mother_forker.truncate import (
    MAX_TURN_CHARS,
    MAX_TURN_CHARS_SINGLE_MESSAGE,
    MAX_TURN_TAIL_CHARS_SINGLE_MESSAGE,
    MAX_TURNS,
    MAX_USER_TURN_CHARS,
    truncate_middle_list,
    truncate_middle_text,
)

USER_TURN = "USER"
ASSISTANT_TURN = "ASSISTANT"
_SYSTEM_MARKER_TURN = "__marker__"

_SUB_LABELS = {
    Role.USER: "USER",
    Role.ASSISTANT: "ASSISTANT",
    Role.TOOL_CALL: "TOOL_CALL",
    Role.TOOL_RESULT: "TOOL_RESULT",
    Role.OTHER: "OTHER",
}


@dataclass
class Turn:
    kind: str  # USER_TURN, ASSISTANT_TURN, or _SYSTEM_MARKER_TURN
    messages: list[Message]


def group_into_turns(messages: list[Message]) -> list[Turn]:
    """Group a flat message list into USER/ASSISTANT turns, preserving order."""
    turns: list[Turn] = []
    current_kind: str | None = None
    current_msgs: list[Message] = []

    for message in messages:
        kind = USER_TURN if message.role.is_user else ASSISTANT_TURN
        if current_kind is None:
            current_kind, current_msgs = kind, [message]
        elif kind == current_kind:
            current_msgs.append(message)
        else:
            turns.append(Turn(current_kind, current_msgs))
            current_kind, current_msgs = kind, [message]

    if current_msgs:
        turns.append(Turn(current_kind, current_msgs))

    return turns


def _quote(text: str) -> str:
    """Render `text` as a markdown quote block."""
    lines = text.splitlines() or [""]
    return "\n".join(f"> {line}" for line in lines)


def _sub_label(message: Message) -> str:
    label = _SUB_LABELS.get(message.role, "OTHER")
    if message.role is Role.TOOL_CALL and message.label:
        return f"{label}: {message.label}"
    return label


def render_turn(turn: Turn, max_chars: int | None = None) -> str:
    """Render a turn's body, middle-truncated to a character budget.

    If `max_chars` is given explicitly, it's applied as a plain symmetric
    head/tail split (backward-compatible override). Otherwise the budget is
    chosen per turn:

    - User turns get a smaller total budget (`MAX_USER_TURN_CHARS`).
    - A single-message assistant turn (e.g. one long assistant reply with no
      tool calls) gets a bigger tail than head, since the conclusion at the
      end tends to be the most useful context to preserve.
    - Any other assistant turn gets the general `MAX_TURN_CHARS` split evenly.
    """
    if turn.kind == _SYSTEM_MARKER_TURN:
        # A synthetic marker turn standing in for dropped turns -- rendered
        # plainly, no header/quoting.
        return turn.messages[0].text

    header = f"## {turn.kind}"
    sections = [f"{_sub_label(m)}\n{_quote(m.text)}" for m in turn.messages if m.text.strip()]
    body = "\n\n".join(sections)

    if max_chars is not None:
        body = truncate_middle_text(body, max_chars)
    elif turn.kind == USER_TURN:
        body = truncate_middle_text(body, MAX_USER_TURN_CHARS)
    elif len(sections) == 1:
        body = truncate_middle_text(
            body, MAX_TURN_CHARS_SINGLE_MESSAGE, tail_chars=MAX_TURN_TAIL_CHARS_SINGLE_MESSAGE
        )
    else:
        body = truncate_middle_text(body, MAX_TURN_CHARS)

    return f"{header}\n{body}"


def _turns_dropped_marker(count: int) -> Turn:
    return Turn(_SYSTEM_MARKER_TURN, [Message(role=Role.OTHER, text=f"[{count} turns truncated]")])


def render_conversation(
    conversation: Conversation,
    max_turns: int = MAX_TURNS,
    max_turn_chars: int | None = None,
) -> str:
    """Render a full conversation as annotated, truncated turns.

    Both the number of turns and the length of each individual turn are
    capped, dropping the middle in each case (per design: the important
    parts of a turn, and of a conversation, are the beginning and the end).

    `max_turn_chars`, if given, overrides the per-turn budget uniformly
    (see `render_turn`); otherwise each turn picks its own budget based on
    its kind and shape.
    """
    turns = group_into_turns(conversation.messages)
    turns = truncate_middle_list(turns, max_turns, _turns_dropped_marker)
    return "\n\n".join(render_turn(t, max_turn_chars) for t in turns)
