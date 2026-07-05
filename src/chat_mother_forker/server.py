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
    """List the most recent conversations across all configured chat-history
    providers (Kiro CLI, and others as they're added), optionally filtered to
    those containing `search` as a substring (checked against the
    conversation id, any checkpoint slug/uuid, and the raw transcript text).

    Returns, per matching conversation: last-modified date, a
    "provider:conversation_id" identifier, the first ~256 characters of the
    initial user prompt, and any checkpoint slugs found in it.
    """
    results = search_conversations(ALL_PROVIDERS, search=search)
    return render_search_results(results, search)


@mcp.tool()
def chat_checkpoint(slug: str) -> str:
    """Create a checkpoint in the current conversation so it can be found and
    sliced later via chat_search/chat_fork.

    `slug` is a short label (up to 256 characters); it does not need to be
    unique.
    """
    return format_checkpoint_line(slug)


@mcp.tool()
def chat_fork(
    search: str,
    start_checkpoint: Optional[str] = None,
    end_checkpoint: Optional[str] = None,
) -> str:
    """Find the newest conversation matching `search` (a checkpoint slug, a
    checkpoint/conversation uuid, or any substring of the transcript) and
    return it as an annotated, truncated transcript.

    If `start_checkpoint` and/or `end_checkpoint` are given, the returned
    transcript is sliced to the range between them (falling back to the
    whole conversation on either side if a checkpoint is omitted or not
    found). The response always ends with a hint that it is background
    context, not an instruction to follow.
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
