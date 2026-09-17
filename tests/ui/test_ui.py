"""Unit tests for UI helper logic and title generation."""

from app.ui.chat import _generate_title


def test_generate_title_short() -> None:
    assert _generate_title("What was the revenue?") == "What was the revenue?"


def test_generate_title_long() -> None:
    long_question = "What was the total financial revenue generated during the third quarter of 2024?"
    title = _generate_title(long_question)
    assert len(title) <= 38
    assert title.endswith("...")
    assert not title.startswith(" ")
