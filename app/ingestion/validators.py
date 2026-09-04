"""In-memory validation for uploaded PDFs."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import clamd
from pypdf import PdfReader


MINIMUM_PAGE_COUNT = 50


class ValidationError(ValueError):
    """The uploaded file did not meet an ingestion requirement."""


def validate_pdf_format(file_bytes: bytes) -> None:
    try:
        import magic
        mime_type = magic.from_buffer(file_bytes, mime=True)
    except ImportError as error:
        raise ValidationError(
            "PDF format validation is unavailable: install libmagic1 (or libmagic locally)."
        ) from error
    if mime_type != "application/pdf":
        raise ValidationError(f"Expected a PDF file; detected {mime_type!r}.")


def read_pdf_page_count(file_bytes: bytes) -> int:
    try:
        reader = PdfReader(BytesIO(file_bytes), strict=True)
        return len(reader.pages)
    except Exception as error:
        raise ValidationError("PDF is corrupt or unreadable.") from error


def validate_page_count(page_count: int) -> None:
    if page_count < MINIMUM_PAGE_COUNT:
        raise ValidationError(
            f"PDF has {page_count} pages; at least {MINIMUM_PAGE_COUNT} pages are required."
        )


def scan_for_virus(file_bytes: bytes, host: str, port: int) -> None:
    try:
        # clamd 1.0.2 expects a readable binary stream, not raw bytes.
        result = clamd.ClamdNetworkSocket(host=host, port=port).instream(BytesIO(file_bytes))
    except Exception as error:
        raise ValidationError("Virus scanner is unavailable; upload cannot be accepted.") from error
    status, detail = next(iter(result.values()))
    if status != "OK":
        raise ValidationError(f"Malware detected: {detail}")


def content_hash(file_bytes: bytes) -> str:
    return sha256(file_bytes).hexdigest()
