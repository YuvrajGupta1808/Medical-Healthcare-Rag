from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

@lru_cache
def _get_s3_client():
    """Create and cache full initialized S3 client."""
    s = get_settings()
    # Ensure minio_secure is cast for correct schema
    endpoint = f"http://{s.minio_endpoint}" if not s.minio_secure else f"https://{s.minio_endpoint}"
    
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=s.minio_access_key,
        aws_secret_access_key=s.minio_secret_key,
        config=Config(signature_version="s3v4"),
    )

def ensure_bucket_exists() -> None:
    """Check if the bucket exists and create it otherwise on startup."""
    s = get_settings()
    client = _get_s3_client()
    try:
        client.head_bucket(Bucket=s.minio_bucket)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404" or error_code == 404:
            logger.info("Bucket '%s' not found, creating it...", s.minio_bucket)
            client.create_bucket(Bucket=s.minio_bucket)
        else:
            logger.error("Failed to head bucket '%s': %s", s.minio_bucket, e)
            raise e

def upload_file(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """
    Upload file bytes to MinIO and return a storage_ref string.
    
    Generates deterministic key based on SHA256 hashing of bytes.
    Returns: storage_ref format: "bucket_name/key" Use this to store in Weaviate.
    """
    ensure_bucket_exists()  # small safety check to ensure bucket is up
    s = get_settings()
    client = _get_s3_client()
    
    # generate key from hash
    hasher = hashlib.sha256(image_bytes)
    h = hasher.hexdigest()
    # ext extraction based on mime fallback
    ext = "jpg" if "jpeg" in mime_type else "png" if "png" in mime_type else "bin"
    key = f"images/{h}.{ext}"
    
    try:
        client.put_object(
            Bucket=s.minio_bucket,
            Key=key,
            Body=image_bytes,
            ContentType=mime_type,
        )
    except Exception as e:
        logger.error("Failed uploading to MinIO: %s", e)
        raise e
        
    logger.info("Uploaded item to storage_ref: %s/%s", s.minio_bucket, key)
    return f"{s.minio_bucket}/{key}"

def generate_signed_url(storage_ref: str, expires_in: int = 3600) -> str:
    """
    Generate presigned download URL for a given storage_ref or reference.
    Expected ref format: "bucket/key"
    """
    if "/" not in storage_ref:
        raise ValueError(f"Invalid storage_ref format '{storage_ref}'; expected 'bucket/key'")
        
    bucket, key = storage_ref.split("/", 1)
    # The frontend accesses the images via localhost on the host machine.
    # To prevent Signature errors ("SignatureDoesNotMatch") due to Host differences in the signed headers, 
    # we must explicitly generate the signature natively against the localhost endpoint, NOT the internal 'minio' network route.
    s = get_settings()
    endpoint = f"http://localhost:9000" if not s.minio_secure else f"https://localhost:9000"
    
    import boto3
    from botocore.client import Config
    presign_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=s.minio_access_key,
        aws_secret_access_key=s.minio_secret_key,
        config=Config(signature_version="s3v4"),
    )
    
    try:
        url = presign_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in
        )
        return url
    except Exception as e:
        logger.error("Failed to generate signed url for %s: %s", storage_ref, e)
        raise e
