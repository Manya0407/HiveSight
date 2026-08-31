"""
OpenSearch vector indexer for HiveSight chunks.

This module is intentionally separate from the existing FAISS indexer so the
project can compare FAISS and OpenSearch during the migration.
"""

from typing import Any


def get_opensearch_client(config: dict):
    """Create an OpenSearch client from config."""
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:
        raise RuntimeError(
            "OpenSearch backend requires the 'opensearch-py' package. "
            "Install dependencies from requirements.txt."
        ) from exc

    os_config = config.get("vector_store", {}).get("opensearch", {})
    host = os_config.get("host", "http://localhost:9200")
    username = os_config.get("username")
    password = os_config.get("password")
    http_auth = (username, password) if username and password else None

    return OpenSearch(
        hosts=[host],
        http_auth=http_auth,
        verify_certs=os_config.get("verify_certs", False),
        timeout=os_config.get("timeout", 30),
    )


def ensure_index(client, index_name: str, vector_dim: int) -> None:
    """Create the OpenSearch index/mapping if missing."""
    if client.indices.exists(index=index_name):
        return

    body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": vector_dim,
                },
                "query_id": {"type": "keyword"},
                "application_id": {"type": "keyword"},
                "dag_id": {"type": "keyword"},
                "service": {"type": "keyword"},
                "source_type": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "severity": {"type": "keyword"},
                "timestamp": {
                    "type": "date",
                    "format": "yyyy-MM-dd HH:mm:ss,SSS||strict_date_optional_time||epoch_millis",
                },
                "category": {"type": "keyword"},
                "failure_stage": {"type": "keyword"},
                "exception_class": {"type": "keyword"},
                "error_signature": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "total_chunks": {"type": "integer"},
            }
        },
    }
    client.indices.create(index=index_name, body=body)


def build_document(metadata: dict, vector: Any) -> dict:
    """Combine metadata and embedding into one OpenSearch document."""
    doc = {
        key: value
        for key, value in metadata.items()
        if value is not None and value != ""
    }
    doc["embedding"] = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    return doc


def index_documents(
    client,
    index_name: str,
    metadata: list[dict],
    vectors,
    batch_size: int = 500,
) -> None:
    """Bulk index enriched chunk documents."""
    try:
        from opensearchpy.helpers import bulk
    except ImportError as exc:
        raise RuntimeError(
            "OpenSearch bulk indexing requires opensearch-py helpers."
        ) from exc

    actions = []
    total = len(metadata)
    for idx, record in enumerate(metadata):
        doc = build_document(record, vectors[idx])
        actions.append({
            "_op_type": "index",
            "_index": index_name,
            "_id": record.get("chunk_id", f"chunk_{idx:06d}"),
            "_source": doc,
        })
        if len(actions) >= batch_size:
            bulk(client, actions)
            actions = []

    if actions:
        bulk(client, actions)

    client.indices.refresh(index=index_name)
    print(f"  OpenSearch indexed documents: {total:,}")


def save_opensearch_index(
    config: dict,
    metadata: list[dict],
    vectors,
    vector_dim: int,
) -> None:
    """Create OpenSearch index and write chunk documents."""
    os_config = config.get("vector_store", {}).get("opensearch", {})
    index_name = os_config.get("index_name", "hivesight-log-chunks")
    client = get_opensearch_client(config)
    ensure_index(client, index_name, vector_dim)
    index_documents(
        client=client,
        index_name=index_name,
        metadata=metadata,
        vectors=vectors,
        batch_size=os_config.get("batch_size", 500),
    )
    print(f"  OpenSearch index ready: {index_name}")
