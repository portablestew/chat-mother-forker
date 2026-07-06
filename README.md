# chat-mother-forker

A small stdio MCP server that lets a coding agent search, checkpoint, and
fork chat history — to subagents, different tools, different workspaces,
and across time. Chat context from any parent (a.k.a. "mother") can be
forked into as many child chats as needed, instead of relying on lossy
summarization or copy-pasting each time.

## Setup

Add it to your MCP client's config, e.g.:

```json
{
  "mcpServers": {
    "chat-mother-forker": {
      "command": "uvx",
      "args": ["chat-mother-forker"]
    }
  }
}
```

## Why

Coding agents constantly lose context that already exists on disk, just in
the wrong conversation:

- **Cross-tool continuation.** Started planning in one tool, want to keep
  going in another? `chat_fork` pulls the original conversation straight
  into the new one instead of you re-explaining everything.
- **Cross-workspace continuation, same tool.** A decision made in workspace
  A's chat is invisible to a session in workspace B, even though it's the
  same tool and the same person. Every provider here scans all of a tool's
  stored conversations, not just the current workspace's, so this falls
  out for free.
- **Cheap context handoff to subagents.** A hand-written summary for a
  subagent is expensive to write and inherently lossy. Handing it a
  `chat_fork` search string instead costs a couple of tokens and gets the
  real transcript.
- **Ad hoc recall.** "Apply what we learned in yesterday's chat about X"
  only works if the agent can actually go find yesterday's chat.

## Tools

### `chat_search(search=None)`

Lists the 50 most recent conversations across every configured provider
(see [Status](#status)), merged and sorted by recency. If `search` is
given, only conversations containing it as a substring are returned —
matched against the conversation id, any checkpoint slug/uuid found in the
conversation, or the raw transcript text.

For each matching conversation, returns the last-modified date, a
`provider:conversation_id` identifier, the project/workspace directory
name (when the provider could determine one — useful for telling apart
conversations from different projects), the first ~128 characters of the
initial user prompt, and every checkpoint slug/uuid found anywhere in it.
When `search` is given, results also show which field(s) it matched, plus
a hit count and ~128 characters of context around the first and last
transcript match.

### `chat_checkpoint(slug)`

Drops a named landmark in the *current* conversation so it can be found and
sliced out later. Returns a line of the form:

```
CHAT CHECKPOINT UUID=<random uuid> SLUG=<slug>
```

`slug` is a short label up to 256 characters; it doesn't need to be unique.
Pass the returned UUID to `chat_fork` when you need to target this exact
spot later, e.g. when handing off to a subagent.

### `chat_fork(search, start_checkpoint=None, end_checkpoint=None)`

Finds the **newest** conversation matching `search` (a checkpoint slug, a
checkpoint or conversation uuid, or any substring of the transcript) and
returns it as an annotated, truncated transcript — one you can hand
directly to yourself, another agent, or a subagent as background context.

Matching is tiered — a match on conversation id or checkpoint always beats
a match that's merely somewhere in the transcript text, regardless of
recency. Within the same tier, the newest conversation wins. If you want to
target one conversation unambiguously, search by its `provider:id` or a
checkpoint UUID.

If `start_checkpoint` and/or `end_checkpoint` are given, only the message
range between them (inclusive) is returned, falling back to the whole
conversation on either side if a checkpoint is omitted or not found.

The response always ends with a footer noting it's historical reference
material and not an instruction to act on, plus the exact
`provider:conversation_id` in case you need to fork or slice it again.

## How a conversation is rendered

Messages are grouped into **turns** — a run of consecutive user messages,
or a run of consecutive non-user messages (assistant text, tool calls,
tool results). Each turn gets a `## USER` / `## ASSISTANT` header, with
individual messages labeled (`USER`, `ASSISTANT`, `TOOL_CALL: <name>`,
`TOOL_RESULT`) and quoted as markdown.

To keep responses a manageable size, both an individual turn's text
(2000 characters) and the number of turns in a conversation (50) are
capped — when over the limit, the **middle** is dropped in favor of a
`[N truncated]` marker, on the idea that the beginning (intent) and end
(conclusion) matter more than the middle.

## Status

Three providers are implemented, one per tool:

- **`kiro_cli`** — Kiro CLI (`~/.kiro/sessions/cli/*.jsonl`)
- **`kiro_ide`** — Kiro IDE (execution logs under the extension's
  `globalStorage` directory)
- **`claude_code`** — Claude Code CLI
  (`~/.claude/projects/<encoded-workspace-path>/*.jsonl`)

## License

MIT, see [LICENSE](LICENSE).
