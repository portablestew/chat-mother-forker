"""Checkpoint format and extraction.

The `chat_checkpoint` tool's entire job is to emit one line of the form:

    CHAT CHECKPOINT UUID=<uuid> SLUG=<slug>

That line only reliably survives verbatim inside a TOOL_RESULT message (the
raw output of the chat_checkpoint tool call) -- an assistant relaying it to the
user in prose might paraphrase or reformat it. So extraction only looks at
TOOL_RESULT messages, which keeps the regex simple and avoids false
positives from an assistant merely *talking about* checkpoints.

Every provider maps one chat_checkpoint tool call to exactly one dedicated
TOOL_RESULT message whose entire text is that literal line -- nothing else
gets bundled into it. So the match is anchored to the *start of the
message's own text*, not scanned line-by-line within it. This matters
because chat_fork's own TOOL_RESULT can contain a full quoted transcript of
another conversation, and that transcript always starts with a turn header
("## USER"/"## ASSISTANT"), never with "CHAT CHECKPOINT" -- so a checkpoint
genuinely created in that other conversation, however it's quoted inside
the nested transcript, can never match at this message's start.

Extraction happens on the raw message text, before any turn truncation, so a
checkpoint is never lost to a `[N characters truncated]` marker.
"""

from __future__ import annotations

import re
import uuid as uuid_module
from dataclasses import dataclass

from chat_mother_forker.models import Conversation, Role

MAX_SLUG_CHARS = 256

_LINE_RE = re.compile(r"^CHAT CHECKPOINT UUID=([0-9a-fA-F-]{36}) SLUG=(.*)")


@dataclass
class Checkpoint:
    uuid: str
    slug: str


def format_checkpoint_line(slug: str) -> str:
    """Build the literal line the `chat_checkpoint` tool returns. `slug` is
    truncated to MAX_SLUG_CHARS if needed -- checkpoints are meant to be
    short labels, not content.
    """
    slug = slug.strip()[:MAX_SLUG_CHARS]
    return f"CHAT CHECKPOINT UUID={uuid_module.uuid4()} SLUG={slug}"


def find_checkpoints(conversation: Conversation) -> list[Checkpoint]:
    """Scan a conversation's TOOL_RESULT messages for checkpoint lines.

    Each message is checked for the checkpoint pattern only at the very
    start of its own text (see module docstring) -- a message contributes
    at most one checkpoint, since a chat_checkpoint TOOL_RESULT is never
    anything but that single line.
    """
    checkpoints: list[Checkpoint] = []
    for message in conversation.messages:
        if message.role is not Role.TOOL_RESULT:
            continue
        match = _LINE_RE.match(message.text)
        if match:
            checkpoints.append(Checkpoint(uuid=match.group(1), slug=match.group(2).strip()))
    return checkpoints
