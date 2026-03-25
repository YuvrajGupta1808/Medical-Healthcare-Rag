from __future__ import annotations

import logging
from urllib.parse import urlparse

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

def _get_collection_name() -> str:
    """Read collection name from environment settings."""
    return get_settings().weaviate_collection

def __getattr__(name: str):
    if name == "COLLECTION_NAME":
        return _get_collection_name()
    raise AttributeError(f"module {__name__} has no attribute {name}")
    
EXPECTED_PROPERTIES: dict[str, DataType] = {
    "chunk_id":      DataType.TEXT,
    "doc_id":        DataType.TEXT,
    "patient_id":    DataType.TEXT,
    "doc_title":     DataType.TEXT,
    "text":          DataType.TEXT,
    "modality_type": DataType.TEXT,
    "section":       DataType.TEXT,
    "caption":       DataType.TEXT,
    "storage_ref":   DataType.TEXT,
    "attachment_id": DataType.TEXT,
    "page":          DataType.INT,
}

_client: weaviate.WeaviateClient | None = None


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

def connect() -> None:
    """Connect to Weaviate. Call once at application startup."""
    global _client
    settings = get_settings()

    parsed = urlparse(settings.weaviate_url)
    host = parsed.hostname or "localhost"
    http_port = parsed.port or 8080

    # Use connect_to_local for localhost (avoids OIDC discovery issues)
    # Use connect_to_custom for remote/cloud instances
    if host in ("localhost", "127.0.0.1"):
        _client = weaviate.connect_to_local(
            host=host,
            port=http_port,
            grpc_port=50051,
            skip_init_checks=True,
        )
    else:
        http_secure = parsed.scheme == "https"
        auth = (
            weaviate.auth.Auth.api_key(settings.weaviate_api_key)
            if settings.weaviate_api_key
            else None
        )
        _client = weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=http_secure,
            auth_credentials=auth,
            skip_init_checks=True,
        )

    logger.info("Weaviate connected at %s", settings.weaviate_url)


def close() -> None:
    """Disconnect from Weaviate. Call at application shutdown."""
    global _client
    if _client and _client.is_connected():
        _client.close()
        logger.info("Weaviate connection closed")


def get_client() -> weaviate.WeaviateClient:
    """Return the active client. Raises if not connected."""
    if _client is None or not _client.is_connected():
        raise RuntimeError(
            "Weaviate client is not connected. "
            "Ensure connect() is called in the application lifespan."
        )
    return _client


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def ensure_schema() -> None:
    """
    Create the MedicalChunk collection if it does not exist, then validate
    the live schema against EXPECTED_PROPERTIES to detect schema drift.
    Idempotent — safe to call on every startup.
    """
    client = get_client()

    coll_name = _get_collection_name()

    if client.collections.exists(coll_name):
        logger.info("Collection '%s' already exists — validating schema", coll_name)
    else:
        client.collections.create(
            name=coll_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
            properties=[
                # Exact-match / filter fields
                Property(name="chunk_id",     data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="doc_id",        data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="patient_id",    data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="attachment_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="modality_type", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                # BM25-indexed text fields
                Property(name="doc_title", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
                Property(name="text",      data_type=DataType.TEXT, tokenization=Tokenization.WORD),
                Property(name="caption",   data_type=DataType.TEXT, tokenization=Tokenization.WORD),
                Property(name="section",   data_type=DataType.TEXT, tokenization=Tokenization.WORD),
                # Numeric
                Property(name="page", data_type=DataType.INT),
                # Object-storage pointer — not for semantic search
                Property(
                    name="storage_ref",
                    data_type=DataType.TEXT,
                    skip_vectorization=True,
                    tokenization=Tokenization.FIELD,
                ),
            ],
        )
        logger.info("Collection '%s' created", coll_name)

    missing = validate_schema()
    if missing:
        raise RuntimeError(
            f"Schema drift on '{coll_name}': missing properties {missing}. "
            "Drop and recreate the collection or run a migration."
        )
    logger.info("Schema for '%s' validated — all %d properties present",
                coll_name, len(EXPECTED_PROPERTIES))


def validate_schema() -> list[str]:
    """
    Compare the live MedicalChunk collection against EXPECTED_PROPERTIES.

    Returns:
        List of property names missing from the collection (empty = no drift).
    """
    client = get_client()
    coll_name = _get_collection_name()
    if not client.collections.exists(coll_name):
        return list(EXPECTED_PROPERTIES.keys())

    config = client.collections.get(coll_name).config.get()
    existing = {prop.name for prop in config.properties}
    missing = [name for name in EXPECTED_PROPERTIES if name not in existing]

    if missing:
        logger.error(
            "Schema drift! '%s' is missing properties: %s", coll_name, missing
        )
    return missing


def delete_document(doc_id: str) -> int:
    """
    Delete all chunks belonging to ``doc_id``.

    Called before re-ingesting a document to keep the store idempotent
    (prevents duplicate chunks from multiple ingest runs of the same file).

    Returns:
        Number of objects successfully deleted.
    """
    collection = get_client().collections.get(_get_collection_name())
    result = collection.data.delete_many(
        where=wvc.query.Filter.by_property("doc_id").equal(doc_id)
    )
    count: int = getattr(result, "successful", 0)
    logger.info("Deleted %d chunk(s) for doc_id=%s", count, doc_id)
    return count
