"""Stdio MCP server exposing chat_search, chat_checkpoint, and chat_fork."""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from chat_mother_forker.checkpoint import format_checkpoint_line
from chat_mother_forker.fork import render_fork
from chat_mother_forker.providers import ALL_PROVIDERS
from chat_mother_forker.search import render_search_results, search_conversations

mcp = FastMCP("chat-mother-forker")


@mcp.tool()
def chat_search(search: Optional[str] = None) -> str:
    """List the most recent conversations across every configured chat-history
    provider, merged and sorted by recency. If `search` is given, only
    conversations where it appears as a case-insensitive substring are
    returned -- matched against the conversation id, any checkpoint
    slug/uuid, or the user/assistant transcript text (tool call/result text
    is excluded to reduce noise).

    Returns, per matching conversation: last-modified date, a
    "provider:conversation_id" identifier, the project/workspace directory
    name if the provider could determine one, the first ~128 characters of
    the initial user prompt, and any checkpoint slugs found in it. When
    `search` is given, results also show which field(s) it matched
    (conversation id, checkpoint slug/uuid, and/or transcript with a hit
    count), plus up to ~128 characters of **bolded** context (newlines
    collapsed) around the first and last transcript occurrence.

    Call this proactively when the user references a prior conversation
    ambiguously ("like we discussed yesterday", "continue what I started
    earlier", "what did we decide about X") instead of asking them to
    repeat themselves. If results show checkpoints, surface the slugs to
    the user rather than picking one silently.
    """
    results = search_conversations(ALL_PROVIDERS, search=search)
    return render_search_results(results, search)


@mcp.tool()
def chat_checkpoint(slug: str) -> str:
    """Create a checkpoint in the current conversation so it can be found and
    sliced later via chat_search/chat_fork.

    `slug` is a short label (up to 256 characters) for human recall. The
    returned line includes a UUID -- pass that instead of the slug when
    precision matters.

    Call this right before delegating to a subagent or pausing work you'll
    want to resume, then pass the returned UUID in the subagent's prompt
    (e.g. "call chat_fork search='<uuid>'") instead of writing a summary --
    cheaper and lossless. Checkpoint both before and after a self-contained
    chunk of work to let a subagent fork just that slice later via
    start_checkpoint/end_checkpoint.
    """
    return format_checkpoint_line(slug)


@mcp.tool()
def chat_fork(
    search: str,
    start_checkpoint: Optional[str] = None,
    end_checkpoint: Optional[str] = None,
) -> str:
    """Find the newest conversation matching `search` (a case-insensitive
    substring match against the conversation id, a checkpoint slug/uuid, or
    any text in the transcript) and return it as an annotated, truncated
    transcript.

    If `start_checkpoint` and/or `end_checkpoint` are given, the returned
    transcript is sliced to the range between them (falling back to the
    whole conversation on either side if a checkpoint is omitted or not
    found). The response always ends with a hint that it is background
    context, not an instruction to follow.

    When delegating to a subagent, prefer passing a chat_fork search
    string (a checkpoint UUID, or this conversation's provider:id) in its
    prompt rather than a hand-written summary. This gives the subagent
    baseline context only -- still tell it specifically what to do,
    building on top of that background.
    """
    return render_fork(
        ALL_PROVIDERS,
        search=search,
        start_checkpoint=start_checkpoint,
        end_checkpoint=end_checkpoint,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
