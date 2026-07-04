"""Checkpoint format and extraction.

The `checkpoint` tool's entire job is to emit one line of the form:

    CHAT CHECKPOINT UUID=<uuid> SLUG=<slug>

That line only reliably survives verbatim inside a TOOL_RESULT message (the
raw output of the checkpoint tool call) -- an assistant relaying it to the
user in prose might paraphrase or reformat it. So extraction only looks at
TOOL_RESULT messages, which keeps the regex simple and avoids false
positives from an assistant merely *talking about* checkpoints.

Extraction happens on the raw message text, before any turn truncation, so a
checkpoint is never lost to a `[N characters truncated]` marker.
"""

from __future__ import annotations

import re
import uuid as uuid_module
from dataclasses import dataclass

from chat_mother_forker.models import Conversation, Role

MAX_SLUG_CHARS = 256

_LINE_RE = re.compile(r"CHAT CHECKPOINT UUID=([0-9a-fA-F-]{36}) SLUG=(.*)")


@dataclass
class Checkpoint:
    uuid: str
    slug: str


def format_checkpoint_line(slug: str) -> str:
    """Build the literal line the `checkpoint` tool returns. `slug` is
    truncated to MAX_SLUG_CHARS if needed -- checkpoints are meant to be
    short labels, not content.
    """
    slug = slug.strip()[:MAX_SLUG_CHARS]
    return f"CHAT CHECKPOINT UUID={uuid_module.uuid4()} SLUG={slug}"


def find_checkpoints(conversation: Conversation) -> list[Checkpoint]:
    """Scan a conversation's TOOL_RESULT messages for checkpoint lines."""
    checkpoints: list[Checkpoint] = []
    for message in conversation.messages:
        if message.role is not Role.TOOL_RESULT:
            continue
        for match in _LINE_RE.finditer(message.text):
            checkpoints.append(Checkpoint(uuid=match.group(1), slug=match.group(2).strip()))
    return checkpoints
