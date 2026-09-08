"""Tests for the notebook-aligned lightweight extractor heuristics."""

from app.ingestion.extractor import (
    _looks_like_heading,
    classify_page_text,
)


def test_looks_like_heading_accepts_normal_heading() -> None:
    assert _looks_like_heading("Project Objective")
    assert _looks_like_heading("SCOPE")
    assert _looks_like_heading("Functional Requirements")


def test_classify_page_text_uses_notebook_list_and_paragraph_rules() -> None:
    text = "1. Introduction\nThis is the introductory paragraph."
    blocks = classify_page_text(text)
    assert len(blocks) == 2
    assert blocks[0] == ("list_item", "1. Introduction")
    assert blocks[1] == ("paragraph", "This is the introductory paragraph.")
