import boto3
from moto import mock_aws

from app.storage.s3_client import storage_key_for, upload_if_absent


@mock_aws
def test_upload_if_absent_is_content_addressed() -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    key = upload_if_absent(client, "test-bucket", "a" * 64, b"pdf bytes")
    assert key == storage_key_for("a" * 64)
    assert upload_if_absent(client, "test-bucket", "a" * 64, b"different") == key
    assert client.get_object(Bucket="test-bucket", Key=key)["Body"].read() == b"pdf bytes"
