"""Tests for the Command Code provider.

Most fixtures are copies of the real probe records quoted in
docs/1-research.md (sections 3.1 to 3.5 and 3.8). Some are derived from those
by string replacement (new ids, parents and text, such as the resume and fork
turns). The helper builders (`_user_text_record` and the like), the tool
fixtures with `TOOL_STEM` and `MARKER`, and the inline edge-case lines are
synthetic. Everything is written under `tmp_path`. Nothing here reads
~/.commandcode.
"""

import json
import os

from chat_mother_forker.api import find_chats, fork_chat, load_chat
from chat_mother_forker.checkpoint import find_checkpoints
from chat_mother_forker.models import Role
from chat_mother_forker.providers import default_providers
from chat_mother_forker.providers.commandcode import CommandCodeProvider

# Probe A (plain chat), research 3.1, 3.2, 3.3, 3.4.
PROBE_A_HEADER = r'{"type":"session","version":3,"id":"9daf97ff-1d89-4407-8dbd-10425d9c2b04","timestamp":"2026-10-21T05:54:57.989Z","cwd":"C:\\Dev\\scratch\\cmdc-probe"}'
PROBE_A_NAME = r'{"type":"session_info","id":"ea78c826","parentId":null,"timestamp":"2026-10-21T05:54:57.997Z","name":"probe-plain"}'
PROBE_A_USER = r'{"type":"message","id":"80099184","parentId":"ea78c826","timestamp":"2026-10-21T05:55:05.668Z","message":{"role":"user","content":[{"type":"text","text":"Reply with exactly the word PONG and nothing else."}],"meta":{"source":"user","createdAt":1792562098004,"messageId":"64802f05-2cdd-4a49-9fc3-1f11a7c58108"}}}'
PROBE_A_ASSISTANT = r'{"type":"message","id":"410fc717","parentId":"80099184","timestamp":"2026-10-21T05:55:05.668Z","message":{"role":"assistant","content":[{"type":"text","text":"PONG"}],"meta":{"source":"model","createdAt":1792562105667,"messageId":"3623b66e-d771-4e15-a42a-a2ae3273a7e1"}},"usage":{"inputTokens":20178,"outputTokens":4,"cacheReadTokens":128,"cacheWriteTokens":0,"costUsd":0},"model":"minimax/minimax-m3.1-flash:free"}'

# Probe B (tool call), research 3.5.
PROBE_B_TOOL_USE = r'''{"type":"message","id":"8b820e70","parentId":"41d5f53a","timestamp":"2026-10-21T05:55:13.293Z","message":{"role":"assistant","content":[{"type":"text","text":"I'll read the file now."},{"type":"tool_use","id":"call_01a15d0729e974599680b0cd","name":"read_file","input":{"file_path":"C:\\Dev\\scratch\\cmdc-probe\\notes.txt"}}],"meta":{"source":"model","createdAt":1792562113279,"messageId":"622b8b28-8f94-4a46-b41b-48a78e70784c"}},"usage":{"inputTokens":20185,"outputTokens":47,"cacheReadTokens":128,"cacheWriteTokens":0,"costUsd":0},"model":"minimax/minimax-m3.1-flash:free"}'''
PROBE_B_TOOL_RESULT = r'{"type":"message","id":"c3b32af1","parentId":"8b820e70","timestamp":"2026-10-21T05:55:13.293Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_01a15d0729e974599680b0cd","content":[{"type":"text","text":"Read 1/1 file, 1 line\n1: probe-marker-42"}]}],"meta":{"source":"tool","messageId":"59e219ca-6971-4d73-9d3e-ed5ae48b25df"}}}'

# Probe B's leading user turn is not quoted in the research; the probe A user
# record stands in for it. The final assistant answer reuses the probe A
# assistant record with the marker as its text.
PROBE_B_FINAL_ASSISTANT = PROBE_A_ASSISTANT.replace('"text":"PONG"', '"text":"probe-marker-42"')


def _write_transcript(projects_root, project_dir, stem, lines):
    """Write `<projects_root>/<project_dir>/<stem>.jsonl` with the given raw lines."""
    directory = projects_root / project_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_list_returns_empty_when_projects_root_missing(tmp_path):
    provider = CommandCodeProvider(projects_root=tmp_path / "missing")

    assert list(provider.list_candidates()) == []


def test_list_finds_sessions_across_project_dirs(tmp_path):
    root = tmp_path / "projects"
    first = _write_transcript(root, "c-dev-scratch-cmdc-probe", "9daf97ff-1d89-4407-8dbd-10425d9c2b04", [PROBE_A_HEADER, PROBE_A_USER])
    second = _write_transcript(root, "c-dev-github-chat-mother-forker", "53aeabc4-c889-4267-aca4-79b604c72489", [PROBE_A_HEADER, PROBE_A_USER])

    refs = list(CommandCodeProvider(projects_root=root).list_candidates())

    by_id = {ref.conversation_id: ref for ref in refs}
    assert set(by_id) == {first.stem, second.stem}
    assert by_id[first.stem].provider == "command_code"
    assert by_id[first.stem].locator == str(first)
    assert isinstance(by_id[first.stem].mtime, float)


def test_list_excludes_checkpoint_meta_and_empty_files(tmp_path):
    root = tmp_path / "projects"
    project = "c-dev-scratch-cmdc-probe"
    real = _write_transcript(root, project, "9daf97ff-1d89-4407-8dbd-10425d9c2b04", [PROBE_A_HEADER, PROBE_A_USER])
    (root / project / "9daf97ff-1d89-4407-8dbd-10425d9c2b04.checkpoints.jsonl").write_text('{"turnNumber":1}\n', encoding="utf-8")
    (root / project / "9daf97ff-1d89-4407-8dbd-10425d9c2b04.meta.json").write_text('{"title":"probe-plain"}', encoding="utf-8")
    (root / project / "empty.jsonl").write_bytes(b"")

    refs = list(CommandCodeProvider(projects_root=root).list_candidates())

    assert [ref.locator for ref in refs] == [str(real)]


def test_load_parses_plain_user_and_assistant_text(tmp_path):
    root = tmp_path / "projects"
    _write_transcript(root, "c-dev-scratch-cmdc-probe", "9daf97ff-1d89-4407-8dbd-10425d9c2b04", [PROBE_A_HEADER, PROBE_A_NAME, PROBE_A_USER, PROBE_A_ASSISTANT])
    provider = CommandCodeProvider(projects_root=root)
    ref = next(iter(provider.list_candidates()))

    conv = provider.load(ref)

    assert [(m.role, m.text) for m in conv.messages] == [
        (Role.USER, "Reply with exactly the word PONG and nothing else."),
        (Role.ASSISTANT, "PONG"),
    ]


def test_load_maps_tool_use_to_tool_call_with_compact_input(tmp_path):
    root = tmp_path / "projects"
    _write_transcript(root, "c-dev-scratch-cmdc-probe", "3be24287-02b8-4cdd-9985-329bf7013f5e", [PROBE_A_HEADER, PROBE_A_USER, PROBE_B_TOOL_USE])
    provider = CommandCodeProvider(projects_root=root)
    ref = next(iter(provider.list_candidates()))

    conv = provider.load(ref)

    tool_calls = [m for m in conv.messages if m.role is Role.TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0].label == "read_file"
    assert json.loads(tool_calls[0].text) == {"file_path": "C:\\Dev\\scratch\\cmdc-probe\\notes.txt"}
    assert tool_calls[0].text == json.dumps({"file_path": "C:\\Dev\\scratch\\cmdc-probe\\notes.txt"}, separators=(",", ":"))


def test_load_maps_tool_result_in_user_record_to_tool_result(tmp_path):
    root = tmp_path / "projects"
    _write_transcript(
        root,
        "c-dev-scratch-cmdc-probe",
        "3be24287-02b8-4cdd-9985-329bf7013f5e",
        [PROBE_A_HEADER, PROBE_A_USER, PROBE_B_TOOL_USE, PROBE_B_TOOL_RESULT, PROBE_B_FINAL_ASSISTANT],
    )
    provider = CommandCodeProvider(projects_root=root)
    ref = next(iter(provider.list_candidates()))

    conv = provider.load(ref)

    assert [m.role for m in conv.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL_CALL,
        Role.TOOL_RESULT,
        Role.ASSISTANT,
    ]
    tool_result = conv.messages[3]
    assert tool_result.text == "Read 1/1 file, 1 line\n1: probe-marker-42"


def test_load_sets_project_from_session_cwd(tmp_path):
    root = tmp_path / "projects"
    _write_transcript(root, "c-dev-scratch-cmdc-probe", "9daf97ff-1d89-4407-8dbd-10425d9c2b04", [PROBE_A_HEADER, PROBE_A_USER])
    provider = CommandCodeProvider(projects_root=root)
    ref = next(iter(provider.list_candidates()))

    conv = provider.load(ref)

    assert conv.project == "cmdc-probe"


def test_conversation_size_is_file_size_and_zero_when_missing(tmp_path):
    root = tmp_path / "projects"
    path = _write_transcript(root, "c-dev-scratch-cmdc-probe", "9daf97ff-1d89-4407-8dbd-10425d9c2b04", [PROBE_A_HEADER, PROBE_A_USER, PROBE_A_ASSISTANT])
    provider = CommandCodeProvider(projects_root=root)
    ref = next(iter(provider.list_candidates()))

    assert provider.conversation_size(ref) == os.path.getsize(path)

    path.unlink()
    assert provider.conversation_size(ref) == 0


# --- A2: edge cases, fork, resume ---

PROBE_PARENT_HEADER = r'{"type":"session","version":3,"id":"26c6c53d-22f5-494f-b046-27ca3cdec792","timestamp":"2026-10-21T05:55:34.679Z","cwd":"C:\\Dev\\scratch\\cmdc-probe"}'
# Fork header, research 3.8 (verbatim).
PROBE_FORK_HEADER = r'{"type":"session","version":3,"id":"26c6c53d-22f5-494f-b046-27ca3cdec792","timestamp":"2026-10-21T05:55:34.679Z","cwd":"C:\\Dev\\scratch\\cmdc-probe","sessionId":"29e84c6a-89c8-4b99-8196-11f143a7d01a"}'
PARENT_STEM = "26c6c53d-22f5-494f-b046-27ca3cdec792"
FORK_STEM = "29e84c6a-89c8-4b99-8196-11f143a7d01a"
RESUME_STEM = "3be24287-02b8-4cdd-9985-329bf7013f5e"

# Probe C turn 2 (resume). Prompt text is from the 3.7 checkpoint line; the
# record layout is probe A's user record with a new id, parent and text. The
# assistant reply reuses PROBE_B_FINAL_ASSISTANT.
PROBE_RESUME_USER = (
    PROBE_A_USER.replace('"id":"80099184"', '"id":"85405b76"')
    .replace('"parentId":"ea78c826"', '"parentId":"410fc717"')
    .replace(
        '"text":"Reply with exactly the word PONG and nothing else."',
        '"text":"What was the marker string you found earlier? Answer with just the string."',
    )
)

# The fork's own turn (probe E): the new user prompt and reply, with the probe-A record layout.
FORK_NEW_USER = PROBE_A_USER.replace('"id":"80099184"', '"id":"f05cbc0e"').replace(
    '"parentId":"ea78c826"', '"parentId":"410fc717"'
).replace(
    '"text":"Reply with exactly the word PONG and nothing else."',
    '"text":"Now say goodbye in five words."',
)
FORK_NEW_ASSISTANT = PROBE_A_ASSISTANT.replace('"id":"410fc717"', '"id":"3a45fab2"').replace(
    '"parentId":"80099184"', '"parentId":"f05cbc0e"'
).replace('"text":"PONG"', '"text":"Goodbye now, see you later."')


def _load_only(tmp_path, lines, stem="9daf97ff-1d89-4407-8dbd-10425d9c2b04"):
    root = tmp_path / "projects"
    _write_transcript(root, "c-dev-scratch-cmdc-probe", stem, lines)
    provider = CommandCodeProvider(projects_root=root)
    ref = next(iter(provider.list_candidates()))
    return provider.load(ref)


def _user_text_record(text, record_id="a1b2c3d4", parent="ea78c826"):
    """A user message with arbitrary text, serialized with real UTF-8 (no \\u escapes)."""
    record = {
        "type": "message",
        "id": record_id,
        "parentId": parent,
        "timestamp": "2026-10-21T05:55:05.668Z",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }
    return json.dumps(record, ensure_ascii=False)


def test_load_drops_thinking_and_session_info(tmp_path):
    # Thinking block shape from research 3.4 (repo session). It sits before the text block.
    thinking_assistant = (
        '{"type":"message","id":"410fc717","parentId":"80099184","timestamp":"2026-10-21T05:55:05.668Z",'
        '"message":{"role":"assistant","content":['
        '{"type":"thinking","thinking":"The user wants one word.","signature":""},'
        '{"type":"text","text":"PONG"}],"meta":{"source":"model","createdAt":1792562105667,'
        '"messageId":"3623b66e-d771-4e15-a42a-a2ae3273a7e1"}},'
        '"usage":{"inputTokens":20178,"outputTokens":4,"cacheReadTokens":128,"cacheWriteTokens":0,"costUsd":0},'
        '"model":"minimax/minimax-m3.1-flash:free"}'
    )
    conv = _load_only(tmp_path, [PROBE_A_HEADER, PROBE_A_NAME, PROBE_A_USER, thinking_assistant])

    assert [(m.role, m.text) for m in conv.messages] == [
        (Role.USER, "Reply with exactly the word PONG and nothing else."),
        (Role.ASSISTANT, "PONG"),
    ]


def test_load_malformed_lines_are_skipped(tmp_path):
    lines = [
        PROBE_A_HEADER,
        '{"type":"message","id":',  # truncated JSON
        "{not json",
        "",  # blank line
        "   ",  # whitespace-only line
        "42",  # valid JSON, not an object
        '"x"',  # valid JSON string, not an object
        PROBE_A_USER,
        PROBE_A_ASSISTANT,
    ]
    conv = _load_only(tmp_path, lines)

    assert [(m.role, m.text) for m in conv.messages] == [
        (Role.USER, "Reply with exactly the word PONG and nothing else."),
        (Role.ASSISTANT, "PONG"),
    ]
    assert conv.project == "cmdc-probe"


def test_header_missing_session_record_gives_no_project(tmp_path):
    conv = _load_only(tmp_path, [PROBE_A_USER, PROBE_A_ASSISTANT])

    assert conv.project is None
    assert len(conv.messages) == 2


def test_header_cwd_missing_or_non_string_gives_no_project(tmp_path):
    missing = '{"type":"session","version":3,"id":"9daf97ff-1d89-4407-8dbd-10425d9c2b04"}'
    non_string = '{"type":"session","version":3,"id":"9daf97ff-1d89-4407-8dbd-10425d9c2b04","cwd":42}'

    assert _load_only(tmp_path / "a", [missing, PROBE_A_USER]).project is None
    assert _load_only(tmp_path / "b", [non_string, PROBE_A_USER]).project is None


def test_header_second_session_record_does_not_overwrite_cwd(tmp_path):
    second = PROBE_A_HEADER.replace("cmdc-probe", "some-other-dir")
    conv = _load_only(tmp_path, [PROBE_A_HEADER, second, PROBE_A_USER])

    assert conv.project == "cmdc-probe"


def test_load_trims_leading_non_user_records(tmp_path):
    # Assistant, tool_use, and tool_result records before the first user turn are dropped.
    conv = _load_only(
        tmp_path,
        [PROBE_A_HEADER, PROBE_A_ASSISTANT, PROBE_B_TOOL_USE, PROBE_B_TOOL_RESULT, PROBE_A_USER, PROBE_A_ASSISTANT],
    )

    assert [m.role for m in conv.messages] == [Role.USER, Role.ASSISTANT]
    assert conv.messages[0].text == "Reply with exactly the word PONG and nothing else."


def test_load_resume_keeps_both_user_turns_in_order(tmp_path):
    root = tmp_path / "projects"
    project = "c-dev-scratch-cmdc-probe"
    _write_transcript(
        root,
        project,
        RESUME_STEM,
        [PROBE_A_HEADER, PROBE_A_NAME, PROBE_A_USER, PROBE_A_ASSISTANT, PROBE_RESUME_USER, PROBE_B_FINAL_ASSISTANT],
    )
    # Sidecars next to the resumed transcript must not produce refs or messages.
    (root / project / f"{RESUME_STEM}.checkpoints.jsonl").write_text('{"turnNumber":1}\n{"turnNumber":2}\n', encoding="utf-8")
    (root / project / f"{RESUME_STEM}.meta.json").write_text('{"title":"probe-tool"}', encoding="utf-8")
    provider = CommandCodeProvider(projects_root=root)

    refs = list(provider.list_candidates())
    conv = provider.load(refs[0])

    assert [ref.conversation_id for ref in refs] == [RESUME_STEM]
    assert [(m.role, m.text) for m in conv.messages] == [
        (Role.USER, "Reply with exactly the word PONG and nothing else."),
        (Role.ASSISTANT, "PONG"),
        (Role.USER, "What was the marker string you found earlier? Answer with just the string."),
        (Role.ASSISTANT, "probe-marker-42"),
    ]


def test_fork_is_a_separate_conversation_with_parent_prefix_and_new_turn(tmp_path):
    root = tmp_path / "projects"
    project = "c-dev-scratch-cmdc-probe"
    # The fork copies the parent's two records (same ids), then adds its own
    # turn. In the real fork file (research 3.8) the copied records also carry
    # `sessionId`, which these copies omit. Only the copy-then-append layout is
    # confirmed; the sessionId field is not reproduced here.
    parent_path = _write_transcript(
        root, project, PARENT_STEM, [PROBE_PARENT_HEADER, PROBE_A_USER, PROBE_A_ASSISTANT]
    )
    _write_transcript(
        root,
        project,
        FORK_STEM,
        [PROBE_FORK_HEADER, PROBE_A_USER, PROBE_A_ASSISTANT, FORK_NEW_USER, FORK_NEW_ASSISTANT],
    )
    parent_bytes_before = parent_path.read_bytes()
    provider = CommandCodeProvider(projects_root=root)

    refs = {ref.conversation_id: ref for ref in provider.list_candidates()}
    fork = provider.load(refs[FORK_STEM])
    parent_bytes_after_load = parent_path.read_bytes()
    parent = provider.load(refs[PARENT_STEM])

    assert set(refs) == {PARENT_STEM, FORK_STEM}
    assert fork.ref.conversation_id == FORK_STEM
    assert [m.text for m in fork.messages] == [
        "Reply with exactly the word PONG and nothing else.",
        "PONG",
        "Now say goodbye in five words.",
        "Goodbye now, see you later.",
    ]
    assert [m.text for m in parent.messages] == [
        "Reply with exactly the word PONG and nothing else.",
        "PONG",
    ]
    assert parent_bytes_after_load == parent_bytes_before
    assert parent_path.read_bytes() == parent_bytes_before


def test_load_error_result_maps_to_tool_result(tmp_path):
    # Synthetic: no is_error record was observed in the probes (research 3.5, Q4).
    error_result = (
        '{"type":"message","id":"e1a0b0c0","parentId":"410fc717","timestamp":"2026-10-21T05:55:13.293Z",'
        '"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","is_error":true,'
        '"content":[{"type":"text","text":"boom"}]}]}}'
    )
    conv = _load_only(tmp_path, [PROBE_A_HEADER, PROBE_A_USER, error_result])

    assert [(m.role, m.text) for m in conv.messages] == [
        (Role.USER, "Reply with exactly the word PONG and nothing else."),
        (Role.TOOL_RESULT, "boom"),
    ]


def test_load_tool_result_without_content_does_not_raise(tmp_path):
    # Synthetic: a tool_result with no `content` key. It has no text, so nothing is emitted.
    no_content = (
        '{"type":"message","id":"e2b1c2d3","parentId":"410fc717","timestamp":"2026-10-21T05:55:13.293Z",'
        '"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"x"}]}}'
    )
    conv = _load_only(tmp_path, [PROBE_A_HEADER, PROBE_A_USER, no_content, PROBE_A_ASSISTANT])

    assert [m.role for m in conv.messages] == [Role.USER, Role.ASSISTANT]


def test_load_ide_context_prefix_is_kept_verbatim(tmp_path):
    # Text from research 3.3 (repo session). The `<ide-context>` prefix is kept,
    # as claude_code keeps it. Research Q7 (whether to strip it) is still open.
    ide_text = "<ide-context>\nFile: src\\chat_mother_forker\\__init__.py (python, line 47)\n</ide-context>\n\n\nTest"
    conv = _load_only(tmp_path, [PROBE_A_HEADER, _user_text_record(ide_text)])

    assert [(m.role, m.text) for m in conv.messages] == [(Role.USER, ide_text)]


def test_load_non_ascii_round_trips(tmp_path):
    user_text = "café — 日本語のテスト"
    assistant_text = "naïve résumé 你好"
    assistant = json.dumps(
        {
            "type": "message",
            "id": "410fc717",
            "parentId": "a1b2c3d4",
            "timestamp": "2026-10-21T05:55:05.668Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        },
        ensure_ascii=False,
    )
    conv = _load_only(tmp_path, [PROBE_A_HEADER, _user_text_record(user_text), assistant])

    assert [(m.role, m.text) for m in conv.messages] == [
        (Role.USER, user_text),
        (Role.ASSISTANT, assistant_text),
    ]


def test_checkpoint_line_in_tool_result_is_found(tmp_path):
    uuid = "27ebccde-2451-45c6-91b2-acc9156ef44e"
    checkpoint_line = f"CHAT CHECKPOINT UUID={uuid} SLUG=my-slug"
    checkpoint_result = json.dumps(
        {
            "type": "message",
            "id": "c4d5e6f7",
            "parentId": "8b820e70",
            "timestamp": "2026-10-21T05:55:13.293Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_checkpoint",
                        "content": [{"type": "text", "text": checkpoint_line}],
                    }
                ],
            },
        }
    )
    conv = _load_only(tmp_path, [PROBE_A_HEADER, PROBE_A_USER, PROBE_B_TOOL_USE, checkpoint_result])

    found = find_checkpoints(conv)

    assert [(c.uuid, c.slug) for c in found] == [(uuid, "my-slug")]


# --- A3: registration and hermetic end-to-end through the public API ---

TOOL_STEM = "3d0a7c1e-5b2f-4a9e-8d6c-7f1b2e3a4c5d"
MARKER = "zz-marker-9"


def _assistant_text_record(text, record_id="ad1e2f3a", parent="80099184"):
    record = {
        "type": "message",
        "id": record_id,
        "parentId": parent,
        "timestamp": "2026-10-21T05:55:14.322Z",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }
    return json.dumps(record, ensure_ascii=False)


def _tool_result_record(result_text, record_id="c3b32af1", parent="8b820e70"):
    record = {
        "type": "message",
        "id": record_id,
        "parentId": parent,
        "timestamp": "2026-10-21T05:55:13.293Z",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_01a15d0729e974599680b0cd",
                    "content": [{"type": "text", "text": result_text}],
                }
            ],
        },
    }
    return json.dumps(record, ensure_ascii=False)


def _write_probe_projects(root):
    """Write the parent, fork, resumed and tool sessions with ordered mtimes
    (parent oldest, fork newest). Returns the list of written paths."""
    project = "c-dev-scratch-cmdc-probe"
    paths = [
        (_write_transcript(root, project, PARENT_STEM, [PROBE_PARENT_HEADER, PROBE_A_USER, PROBE_A_ASSISTANT]), 1000),
        (_write_transcript(root, project, RESUME_STEM, [PROBE_A_HEADER, PROBE_A_USER, PROBE_A_ASSISTANT, PROBE_RESUME_USER, PROBE_B_FINAL_ASSISTANT]), 2000),
        (
            _write_transcript(
                root,
                "c-dev-github-chat-mother-forker",
                TOOL_STEM,
                [
                    PROBE_A_HEADER.replace("scratch", "github").replace("cmdc-probe", "chat-mother-forker"),
                    PROBE_A_USER,
                    PROBE_B_TOOL_USE,
                    PROBE_B_TOOL_RESULT,
                    PROBE_B_FINAL_ASSISTANT,
                ],
            ),
            3000,
        ),
        (_write_transcript(root, project, FORK_STEM, [PROBE_FORK_HEADER, PROBE_A_USER, PROBE_A_ASSISTANT, FORK_NEW_USER, FORK_NEW_ASSISTANT]), 4000),
    ]
    for path, mtime in paths:
        os.utime(path, (mtime, mtime))
    return [path for path, _ in paths]


def test_register_command_code_in_default_providers():
    providers = default_providers()

    assert any(p.name == "command_code" for p in providers)
    first = next(p for p in providers if p.name == "command_code")
    second = next(p for p in default_providers() if p.name == "command_code")
    assert first is not second


def test_e2e_find_chats_lists_sessions_newest_first_with_projects(tmp_path):
    root = tmp_path / "projects"
    _write_probe_projects(root)
    provider = CommandCodeProvider(projects_root=root)

    rows = find_chats(providers=[provider])

    assert [(r.key, r.project) for r in rows] == [
        (f"command_code:{FORK_STEM}", "cmdc-probe"),
        (f"command_code:{TOOL_STEM}", "chat-mother-forker"),
        (f"command_code:{RESUME_STEM}", "cmdc-probe"),
        (f"command_code:{PARENT_STEM}", "cmdc-probe"),
    ]
    assert all(r.database is None and r.locator for r in rows)


def test_e2e_load_and_fork_go_through_public_api(tmp_path):
    root = tmp_path / "projects"
    _write_probe_projects(root)
    provider = CommandCodeProvider(projects_root=root)

    transcript = load_chat(f"command_code:{RESUME_STEM}", providers=[provider])
    forked = fork_chat(f"command_code:{FORK_STEM}", providers=[provider])

    assert "Reply with exactly the word PONG" in transcript
    assert "What was the marker string you found earlier?" in transcript
    assert "probe-marker-42" in transcript
    assert "Now say goodbye in five words." in forked
    assert "Goodbye now, see you later." in forked


def test_e2e_search_matches_assistant_text_not_tool_result_text(tmp_path):
    root = tmp_path / "projects"
    project = "c-dev-scratch-cmdc-probe"
    tool_only_stem = "11111111-aaaa-4bbb-8ccc-000000000001"
    assistant_stem = "22222222-bbbb-4ccc-8ddd-000000000002"
    # Same marker in both: only in a TOOL_RESULT in the first, only in assistant text in the second.
    _write_transcript(
        root,
        project,
        tool_only_stem,
        [
            PROBE_A_HEADER,
            PROBE_A_USER,
            PROBE_B_TOOL_USE,
            _tool_result_record(f"Read 1/1 file, 1 line\n1: {MARKER}"),
            _assistant_text_record("Done."),
        ],
    )
    _write_transcript(
        root,
        project,
        assistant_stem,
        [PROBE_A_HEADER, PROBE_A_USER, _assistant_text_record(f"The marker is {MARKER}.")],
    )
    provider = CommandCodeProvider(projects_root=root)

    rows = find_chats(MARKER, providers=[provider])

    assert [r.key for r in rows] == [f"command_code:{assistant_stem}"]
    assert find_chats("zz-never-written", providers=[provider]) == []
