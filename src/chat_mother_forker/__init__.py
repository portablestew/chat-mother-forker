"""chat-mother-forker: cross-tool chat search, checkpointing, and forking.

Usable two ways:

- As a stdio MCP server -- the `chat-mother-forker` console script
  (`chat_mother_forker.server`).
- As a library -- import the public API below. The core is dependency-free;
  `mcp` is only needed to run the server.

Library quickstart::

    from chat_mother_forker import fork_chat, find_chats

    # Newest conversation matching a marker/uuid, as a truncated transcript:
    preamble = fork_chat("some-root-uuid")

    # Lightweight listing to filter/rank/size before forking:
    for ref in find_chats(project_filter):
        if ref.size_bytes > 1_000_000:
            ...

`__version__` is the single source of truth for the package version.
pyproject.toml reads it from here (via tool.hatch.version), so this is the
only place to edit before a release.
"""

from chat_mother_forker.api import ChatRef, find_chats, fork_chat, load_chat
from chat_mother_forker.checkpoint import Checkpoint
from chat_mother_forker.providers import default_providers
from chat_mother_forker.providers.base import ChatProvider

__version__ = "0.2.1"

__all__ = [
    "__version__",
    # Primary library API
    "fork_chat",
    "load_chat",
    "find_chats",
    "ChatRef",
    # Providers (for typing / custom provider lists)
    "default_providers",
    "ChatProvider",
    # Value type surfaced on ChatRef
    "Checkpoint",
]
