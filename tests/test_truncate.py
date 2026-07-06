from chat_mother_forker.truncate import (
    context_window,
    truncate_middle_list,
    truncate_middle_text,
    truncate_preview,
)


def test_truncate_middle_text_under_limit_is_unchanged():
    text = "short text"
    assert truncate_middle_text(text, max_chars=2000) == text


def test_truncate_middle_text_exactly_at_limit_is_unchanged():
    text = "x" * 50
    assert truncate_middle_text(text, max_chars=50) == text


def test_truncate_middle_text_drops_middle_keeps_head_and_tail():
    text = "HEAD" + ("m" * 200) + "TAIL"
    out = truncate_middle_text(text, max_chars=20)

    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "characters truncated" in out
    # Nothing from the dropped middle should leak into the marker line.
    assert "mmmmmmmmmm" not in out.split("\n")[1]


def test_truncate_middle_text_marker_reports_accurate_dropped_count():
    text = "a" * 1000
    out = truncate_middle_text(text, max_chars=100)
    lines = out.split("\n")
    marker = lines[1]

    head, tail = lines[0], lines[2]
    # marker looks like "[900 characters truncated]"
    dropped_reported = int(marker.strip("[]").split(" ")[0])
    assert dropped_reported == len(text) - len(head) - len(tail)
    # Sanity: total accounted-for length matches the original.
    assert len(head) + dropped_reported + len(tail) == len(text)


def test_truncate_middle_text_empty_string():
    assert truncate_middle_text("", max_chars=10) == ""


def test_truncate_preview_short_text_unchanged():
    assert truncate_preview("hello", max_chars=128) == "hello"


def test_truncate_preview_strips_whitespace_before_measuring():
    assert truncate_preview("   hello   ", max_chars=128) == "hello"


def test_truncate_preview_truncates_from_end_with_ellipsis():
    text = "y" * 150
    out = truncate_preview(text, max_chars=128)
    assert out == "y" * 128 + "..."
    assert len(out) == 131


def test_truncate_preview_exactly_at_limit_no_ellipsis():
    text = "z" * 128
    assert truncate_preview(text, max_chars=128) == text


def test_truncate_preview_default_max_chars_is_128():
    text = "y" * 150
    assert truncate_preview(text) == "y" * 128 + "..."


def test_truncate_preview_collapses_newlines():
    text = "line one\nline two\r\nline three"
    out = truncate_preview(text, max_chars=128)
    assert "\n" not in out
    assert "\r" not in out
    assert out == "line one line two  line three"


def test_truncate_middle_list_under_limit_returns_copy():
    items = [1, 2, 3]
    out = truncate_middle_list(items, max_items=5, marker_factory=lambda n: f"dropped-{n}")
    assert out == items
    assert out is not items  # must be a copy, not the same list object


def test_truncate_middle_list_drops_middle_keeps_first_and_last_half():
    items = list(range(10))
    out = truncate_middle_list(items, max_items=4, marker_factory=lambda n: f"dropped-{n}")
    assert out == [0, 1, "dropped-6", 8, 9]


def test_truncate_middle_list_odd_max_items_favors_symmetric_split():
    items = list(range(9))
    out = truncate_middle_list(items, max_items=5, marker_factory=lambda n: n)
    # keep = 5 // 2 = 2 items from each end.
    assert out[:2] == [0, 1]
    assert out[-2:] == [7, 8]
    assert out[2] == 9 - 2 - 2  # dropped count


def test_truncate_middle_list_marker_factory_receives_correct_count():
    items = list(range(100))
    seen_counts = []

    def factory(n):
        seen_counts.append(n)
        return "MARK"

    out = truncate_middle_list(items, max_items=10, marker_factory=factory)
    # keep = 10 // 2 = 5 items from each end, so 100 - 5 - 5 = 90 dropped.
    assert seen_counts == [90]
    assert out.count("MARK") == 1
    assert len(out) == 11  # 5 head + marker + 5 tail


def test_context_window_bolds_the_match():
    text = "hello widgets world"
    out = context_window(text, text.index("widgets"), len("widgets"), width=256)
    assert out == "hello **widgets** world"


def test_context_window_no_ellipsis_when_fully_contained():
    text = "short text with widgets in it"
    out = context_window(text, text.index("widgets"), len("widgets"), width=256)
    assert not out.startswith("...")
    assert not out.endswith("...")


def test_context_window_adds_ellipsis_when_clipped_on_both_sides():
    text = "x" * 500 + "widgets" + "y" * 500
    out = context_window(text, 500, len("widgets"), width=20)
    assert out.startswith("...")
    assert out.endswith("...")
    assert "**widgets**" in out


def test_context_window_no_leading_ellipsis_at_start_of_text():
    text = "widgets" + "y" * 500
    out = context_window(text, 0, len("widgets"), width=20)
    assert not out.startswith("...")
    assert out.endswith("...")
    assert "**widgets**" in out


def test_context_window_no_trailing_ellipsis_at_end_of_text():
    text = "x" * 500 + "widgets"
    out = context_window(text, 500, len("widgets"), width=20)
    assert out.startswith("...")
    assert not out.endswith("...")
    assert "**widgets**" in out


def test_context_window_respects_width_budget():
    text = "x" * 500 + "widgets" + "y" * 500
    out = context_window(text, 500, len("widgets"), width=50)
    # Stripped of ellipses and bold markers, the window shouldn't wildly
    # exceed the requested width.
    stripped = out.strip(".").replace("**", "")
    assert len(stripped) <= 50 + 2  # +/- rounding slack


def test_context_window_default_width_is_128():
    text = "x" * 200 + "widgets" + "y" * 200
    out = context_window(text, 200, len("widgets"))
    stripped = out.strip(".").replace("**", "")
    assert len(stripped) <= 128 + 2  # +/- rounding slack


def test_context_window_collapses_newlines():
    text = "line one\nwidgets here\r\nline three"
    idx = text.index("widgets")
    out = context_window(text, idx, len("widgets"), width=128)
    assert "\n" not in out
    assert "\r" not in out
    assert "**widgets**" in out
