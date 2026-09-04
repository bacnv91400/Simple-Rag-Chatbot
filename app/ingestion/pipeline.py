"""Orchestration for the validate, deduplicate, store, and persist flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import boto3

from app.config.settings import Settings
from app.db.client import DocumentRepository, create_repository
from app.ingestion.validators import (ValidationError, content_hash, read_pdf_page_count,
    scan_for_virus, validate_page_count, validate_pdf_format)
from app.storage.s3_client import storage_key_for, upload_if_absent


@dataclass(frozen=True)
class ProcessLogEntry:
    step: str
    status: str
    message: str


@dataclass
class ProcessUploadResult:
    success: bool
    logs: list[ProcessLogEntry] = field(default_factory=list)
    error: str | None = None
    upload: dict[str, Any] | None = None


def _s3_client(settings: Settings) -> Any:
    return boto3.client("s3", endpoint_url=settings.idrive_e2_endpoint,
        region_name=settings.idrive_e2_region, aws_access_key_id=settings.idrive_e2_access_key,
        aws_secret_access_key=settings.idrive_e2_secret_key)


def process_upload(filename: str, file_bytes: bytes, *, settings: Settings,
                   repository: DocumentRepository | None = None, s3_client: Any | None = None) -> ProcessUploadResult:
    """Process one file. No database mutation happens before all validation succeeds."""
    logs = [ProcessLogEntry("upload_received", "success", f"Received {filename}.")]
    missing = settings.missing_upload_settings()
    if missing:
        message = f"Missing required configuration: {', '.join(missing)}"
        logs.append(ProcessLogEntry("configuration", "failed", message))
        return ProcessUploadResult(False, logs, message)
    try:
        validate_pdf_format(file_bytes); logs.append(ProcessLogEntry("format_check", "success", "PDF signature verified."))
        page_count = read_pdf_page_count(file_bytes); logs.append(ProcessLogEntry("corrupt_check", "success", "PDF is readable."))
        validate_page_count(page_count); logs.append(ProcessLogEntry("page_count_check", "success", f"{page_count} pages."))
        if settings.virus_scan_enabled:
            scan_for_virus(file_bytes, settings.clamav_host, settings.clamav_port)
            logs.append(ProcessLogEntry("virus_scan", "success", "No malware detected."))
        else:
            logs.append(ProcessLogEntry("virus_scan", "skipped", "Virus scanning is disabled."))
    except ValidationError as error:
        step = "validation"
        if "pages" in str(error): step = "page_count_check"
        elif "corrupt" in str(error): step = "corrupt_check"
        elif "malware" in str(error).lower() or "scanner" in str(error).lower(): step = "virus_scan"
        elif "Expected" in str(error): step = "format_check"
        logs.append(ProcessLogEntry(step, "failed", str(error)))
        return ProcessUploadResult(False, logs, str(error))

    repository = repository or create_repository(settings.supabase_url or "", settings.supabase_service_role_key or "")
    file_hash = content_hash(file_bytes)
    try:
        existing = repository.find_physical_document(file_hash)
    except Exception as error:
        message = f"Duplicate lookup failed: {error}"
        logs.append(ProcessLogEntry("duplicate_check", "failed", message))
        return ProcessUploadResult(False, logs, message)
    duplicate = existing is not None
    logs.append(ProcessLogEntry("duplicate_check", "success", "Existing content reused." if duplicate else "New content."))
    key = existing["storage_key"] if existing else storage_key_for(file_hash)
    if not duplicate:
        try:
            key = upload_if_absent(s3_client or _s3_client(settings), settings.idrive_e2_bucket or "", file_hash, file_bytes)
            logs.append(ProcessLogEntry("uploading_storage", "success", f"Stored as {key}."))
        except Exception as error:
            message = f"Storage upload failed: {error}"
            logs.append(ProcessLogEntry("uploading_storage", "failed", message))
            return ProcessUploadResult(False, logs, message)
    else:
        logs.append(ProcessLogEntry("uploading_storage", "skipped", "Content is already stored."))
    try:
        upload = repository.save_successful_upload(file_hash=file_hash, storage_key=key,
            file_size_bytes=len(file_bytes), page_count=page_count, requested_filename=filename,
            is_duplicate_content=duplicate)
    except Exception as error:
        message = f"Metadata save failed: {error}"
        logs.append(ProcessLogEntry("saving_metadata", "failed", message))
        return ProcessUploadResult(False, logs, message)
    logs.extend([ProcessLogEntry("saving_metadata", "success", "Upload metadata saved."), ProcessLogEntry("success", "success", "Upload completed.")])
    return ProcessUploadResult(True, logs, upload=upload)
