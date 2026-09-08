"""Lightweight PDF extractor porting the notebook pipeline.

Uses PyMuPDF for raster-image extraction, pypdf for text extraction,
manual heuristic structuring into a DoclingDocument, and Gemini vision
for image captions.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Callable

import pymupdf
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from pypdf import PdfReader

logger = logging.getLogger(__name__)

LIST_MARKER_RE = re.compile(
    r"^(?:[-*â€¢â€£â–ª]\s+|\(?\d{1,3}[.)]\s+|\([a-zA-Z]\)\s+|[a-zA-Z][.)]\s+)"
)
MID_LINE_LIST_SPLIT = re.compile(r"(?<=[.\)])\s+(?=\d{1,3}[.)]\s+[A-Z])")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9])")

IMAGE_PROMPT = """
You are preprocessing a visual element from a PDF for a retrieval-augmented generation system.

If the image contains readable text, transcribe the important text accurately.
If it is a diagram, chart, architecture diagram, screenshot, or infographic, describe the factual information needed to answer questions about it.
Preserve visible labels, names, numbers, relationships, arrows, axes, and important values.
Do not invent information.
Keep the result concise and retrieval-friendly.
Return plain text only.
"""


def normalize_text(text: str) -> str:
    """Normalize PDF text without destroying meaningful line structure."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def image_area_ratio(page: Any, xref: int) -> float:
    rects = page.get_image_rects(xref)
    if not rects:
        return 0.0
    page_area = max(page.rect.width * page.rect.height, 1.0)
    return max((r.width * r.height) / page_area for r in rects)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_candidate_images(
    pdf_path: str,
    min_image_width: int = 80,
    min_image_height: int = 80,
    min_image_area_ratio: float = 0.02,
) -> list[dict[str, Any]]:
    """Extract unique, useful raster images directly from PDF image XObjects."""
    pdf = pymupdf.open(pdf_path)
    candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_xrefs: set[int] = set()

    for page_index, page in enumerate(pdf):
        for info in page.get_images(full=True):
            xref = info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                extracted = pdf.extract_image(xref)
                image_bytes = extracted["image"]
                width = int(extracted.get("width", 0))
                height = int(extracted.get("height", 0))
                area_ratio = image_area_ratio(page, xref)

                if (
                    width < min_image_width
                    or height < min_image_height
                    or area_ratio < min_image_area_ratio
                ):
                    continue

                digest = sha256_bytes(image_bytes)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

                candidates.append({
                    "id": digest[:16],
                    "hash": digest,
                    "page": page_index + 1,
                    "xref": xref,
                    "width": width,
                    "height": height,
                    "area_ratio": area_ratio,
                    "extension": extracted.get("ext", "png"),
                    "bytes": image_bytes,
                })
            except Exception as error:
                logger.warning(
                    "Image extraction failed page=%d xref=%d: %s",
                    page_index + 1,
                    xref,
                    error,
                )

    pdf.close()
    return candidates


def describe_image_with_gemini(
    gemini_client: Any,
    cache: Any,
    image_bytes: bytes,
    mime_type: str,
    image_hash: str,
    rate_limiter: Any = None,
    caption_model: str = "gemini-2.0-flash-lite",
) -> tuple[str | None, str]:
    """Return a cached caption or ask Gemini to describe the image."""
    if cache is not None:
        cached = cache.get(image_hash)
        if cached:
            return cached, "cache"
    if gemini_client is None:
        return None, "no_api_key"
    if rate_limiter is not None:
        rate_limiter.wait()
    try:
        from google.genai import types

        response = gemini_client.models.generate_content(
            model=caption_model,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), IMAGE_PROMPT],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=512),
        )
        text = (response.text or "").strip()
        if not text:
            return None, "empty_response"
        if cache is not None:
            cache.set(image_hash, text)
        return text, "gemini"
    except Exception as error:
        logger.warning("Gemini image description error: %s", error)
        return None, "error"


def extract_page_text(page: Any) -> str:
    try:
        text = page.extract_text(extraction_mode="layout") or ""
        if text.strip():
            return text
    except Exception:
        pass
    return page.extract_text() or ""


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    words = line.split()

    if not words or len(words) > 8:
        return False
    if len(line) > 100:
        return False
    if line[-1] in ".,;:!?":
        return False
    if LIST_MARKER_RE.match(line):
        return False

    # Strong signals only. The old capitalization heuristic classified
    # ordinary English prose as headings.
    if line.isupper() and len(words) <= 8:
        return True

    heading_prefixes = (
        "project objective",
        "expected outcome",
        "functional requirements",
        "recommended technical approach",
        "implementation notes",
        "definition of done",
        "review criteria",
        "suggested milestones",
        "stretch ideas",
        "scope",
        "deployment",
        "windows",
    )
    return line.lower() in heading_prefixes


def _split_glued_list_items(line: str) -> list[str]:
    parts = MID_LINE_LIST_SPLIT.split(line)
    return [part.strip() for part in parts if part.strip()]


def classify_page_text(raw_text: str) -> list[tuple[str, str]]:
    """Split page text into the notebook's paragraph, heading, and list blocks."""
    lines = [line.strip() for line in raw_text.split("\n")]
    blocks: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(("paragraph", " ".join(buffer)))
            buffer.clear()

    for line in lines:
        if not line:
            flush()
            continue

        if LIST_MARKER_RE.match(line):
            flush()
            for item in (_split_glued_list_items(line) or [line]):
                blocks.append(("list_item", item))
            continue

        if _looks_like_heading(line):
            flush()
            blocks.append(("heading", line))
            continue

        buffer.append(line)

    flush()
    return blocks


def split_into_paragraphs(text: str) -> list[str]:
    blocks = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(blocks) > 1:
        return blocks

    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT.split(flat) if sentence.strip()]
    return sentences if sentences else ([text] if text.strip() else [])


def _add_structured_item(
    doc: DoclingDocument,
    kind: str,
    text: str,
    page_no: int,
    page_bbox: BoundingBox,
) -> None:
    prov = ProvenanceItem(page_no=page_no, bbox=page_bbox, charspan=(0, len(text)))

    if kind == "heading" and hasattr(doc, "add_heading"):
        try:
            doc.add_heading(text=text, prov=prov)
            return
        except Exception:
            pass

    if kind == "list_item" and hasattr(doc, "add_list_item"):
        try:
            doc.add_list_item(text=text, prov=prov)
            return
        except Exception:
            pass

    label = {
        "heading": DocItemLabel.SECTION_HEADER,
        "list_item": DocItemLabel.LIST_ITEM,
    }.get(kind, DocItemLabel.TEXT)
    doc.add_text(label=label, text=text, prov=prov)


def build_doc(
    pdf_path: str,
    image_candidates: list[dict[str, Any]],
    describe_fn: Callable[[bytes, str, str], tuple[str | None, str]],
) -> DoclingDocument:
    """Build a DoclingDocument from all page text and optional image summaries."""
    reader = PdfReader(pdf_path)
    doc = DoclingDocument(name=Path(pdf_path).stem)

    images_by_page: dict[int, list[dict[str, Any]]] = {}
    for image in image_candidates:
        images_by_page.setdefault(image["page"], []).append(image)

    for page_index, page in enumerate(reader.pages):
        page_no = page_index + 1
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)

        doc.add_page(page_no=page_no, size=Size(width=width, height=height))
        page_bbox = BoundingBox(
            l=0,
            t=0,
            r=width,
            b=height,
            coord_origin=CoordOrigin.TOPLEFT,
        )

        raw_text = normalize_text(extract_page_text(page))
        if raw_text:
            blocks = classify_page_text(raw_text)
            if not blocks:
                blocks = [("paragraph", paragraph) for paragraph in split_into_paragraphs(raw_text)]

            for kind, text in blocks:
                if text.strip():
                    _add_structured_item(doc, kind, text, page_no, page_bbox)

        for image in images_by_page.get(page_no, []):
            mime = f"image/{image['extension'].lower()}"
            if mime == "image/jpg":
                mime = "image/jpeg"

            description, _ = describe_fn(image["bytes"], mime, image["hash"])
            if not description:
                continue

            text = f"[Visual content on page {page_no}]\n{description}"
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                prov=ProvenanceItem(
                    page_no=page_no,
                    bbox=page_bbox,
                    charspan=(0, len(text)),
                ),
            )

    return doc


def extract_to_docling_document(
    pdf_path: str | Path,
    gemini_client: Any = None,
    cache: Any = None,
    rate_limiter: Any = None,
    max_gemini_calls: int = 20,
    caption_model: str = "gemini-2.0-flash-lite",
    min_image_width: int = 80,
    min_image_height: int = 80,
    min_image_area_ratio: float = 0.02,
) -> DoclingDocument:
    """Extract a PDF into a DoclingDocument using the notebook's direct flow."""
    pdf_path_str = str(pdf_path)
    image_candidates = extract_candidate_images(
        pdf_path_str,
        min_image_width=min_image_width,
        min_image_height=min_image_height,
        min_image_area_ratio=min_image_area_ratio,
    )

    calls_used = 0

    def describe_fn(image_bytes: bytes, mime: str, image_hash: str) -> tuple[str | None, str]:
        nonlocal calls_used

        # Cached captions are reusable even after the API-call budget is reached.
        if cache is not None and cache.get(image_hash):
            return describe_image_with_gemini(
                gemini_client=gemini_client,
                cache=cache,
                image_bytes=image_bytes,
                mime_type=mime,
                image_hash=image_hash,
                rate_limiter=rate_limiter,
                caption_model=caption_model,
            )
        if calls_used >= max_gemini_calls:
            return None, "call_budget_exhausted"

        text, status = describe_image_with_gemini(
            gemini_client=gemini_client,
            cache=cache,
            image_bytes=image_bytes,
            mime_type=mime,
            image_hash=image_hash,
            rate_limiter=rate_limiter,
            caption_model=caption_model,
        )
        if status == "gemini":
            calls_used += 1
        return text, status

    return build_doc(pdf_path_str, image_candidates, describe_fn=describe_fn)


def extract_document(file_path: str | Path, converter: Any = None) -> DoclingDocument:
    """Convenience / backward-compatible wrapper."""
    if hasattr(converter, "convert"):
        return converter.convert(Path(file_path)).document
    return extract_to_docling_document(file_path)
