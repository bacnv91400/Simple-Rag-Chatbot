from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from pypdf import PdfWriter

from app.ingestion.validators import (ValidationError, read_pdf_page_count, scan_for_virus,
    validate_page_count)


def test_page_count_rejects_short_pdf() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    data = BytesIO(); writer.write(data)
    assert read_pdf_page_count(data.getvalue()) == 1
    with pytest.raises(ValidationError, match="at least 50"):
        validate_page_count(1)


@patch("app.ingestion.validators.clamd.ClamdNetworkSocket")
def test_virus_scan_passes_a_binary_stream_to_clamd(network_socket: Mock) -> None:
    network_socket.return_value.instream.return_value = {"stream": ("OK", None)}
    scan_for_virus(b"content", "clamav", 3310)
    argument = network_socket.return_value.instream.call_args.args[0]
    assert argument.read() == b"content"
