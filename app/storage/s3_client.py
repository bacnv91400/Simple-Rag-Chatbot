"""IDrive e2 (S3-compatible) object storage operations."""

from __future__ import annotations


def storage_key_for(file_hash: str) -> str:
    return f"{file_hash}.pdf"


def upload_if_absent(s3_client: object, bucket: str, file_hash: str, file_bytes: bytes) -> str:
    """Store content-addressed bytes once, returning their stable object key."""
    key = storage_key_for(file_hash)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return key
    except s3_client.exceptions.ClientError as error:
        if error.response["Error"].get("Code") not in ("404", "NoSuchKey", "NotFound"):
            raise
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=file_bytes,
        ContentType="application/pdf",
        Metadata={"sha256": file_hash},
    )
    return key
