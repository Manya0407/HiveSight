"""
Basic log enrichment for chunk metadata.

This module intentionally performs conservative extraction only. Values are
set when they are directly present in the chunk text or can be mapped from
another chunk with the same query_id in the same batch.
"""

import re
from collections import defaultdict
from typing import Iterable


QUERY_ID_PATTERN = re.compile(r"(hive_\d{14}_[A-Za-z0-9]+)")
APPLICATION_ID_PATTERN = re.compile(r"(application_\d+_\d+)")
DAG_ID_PATTERN = re.compile(r"(dag_\d+_\d+_\d+)")
EXCEPTION_PATTERN = re.compile(
    r"\b([A-Za-z_$][A-Za-z0-9_.$]*(?:Exception|Error))\b"
)


def enrich_metadata_records(records: list[dict]) -> list[dict]:
    """Add conservative structured metadata to chunk dictionaries."""
    enriched = [enrich_metadata_record(record) for record in records]
    query_to_apps = _build_query_application_map(enriched)

    for record in enriched:
        if not record.get("application_id"):
            query_id = record.get("query_id")
            if query_id in query_to_apps:
                record["application_id"] = query_to_apps[query_id]

    return enriched


def enrich_metadata_record(record: dict) -> dict:
    """Return a copy of one record with basic enrichment fields."""
    enriched = record.copy()
    text = enriched.get("text", "") or ""

    query_id = enriched.get("query_id") or extract_query_id(text)
    application_id = extract_application_id(text)
    exception_class = extract_exception_class(text)

    enriched["query_id"] = query_id
    enriched["application_id"] = application_id
    enriched["dag_id"] = extract_dag_id(text)
    enriched["service"] = (
        enriched.get("service")
        or enriched.get("source_type")
        or infer_service(enriched.get("source_file", ""))
    )
    enriched["severity"] = enriched.get("severity") or enriched.get("log_level")
    enriched["exception_class"] = exception_class
    enriched["error_signature"] = extract_error_signature(text, exception_class)

    return enriched


def extract_query_id(text: str) -> str | None:
    match = QUERY_ID_PATTERN.search(text or "")
    return match.group(1) if match else None


def extract_application_id(text: str) -> str | None:
    match = APPLICATION_ID_PATTERN.search(text or "")
    return match.group(1) if match else None


def extract_dag_id(text: str) -> str | None:
    match = DAG_ID_PATTERN.search(text or "")
    return match.group(1) if match else None


def extract_exception_class(text: str) -> str | None:
    match = EXCEPTION_PATTERN.search(text or "")
    return match.group(1) if match else None


def extract_error_signature(text: str, exception_class: str | None = None) -> str | None:
    """Extract a compact, stable error signature from chunk text."""
    if not text:
        return None

    interesting_terms = [
        "FAILED:",
        "Exception",
        "Error",
        "NoClassDefFoundError",
        "ClassNotFoundException",
        "OutOfMemoryError",
        "Table not found",
        "Database does not exist",
        "Unable to kill query",
    ]
    for line in text.splitlines():
        if any(term in line for term in interesting_terms):
            return normalize_signature(line)

    if exception_class:
        return exception_class
    return None


def normalize_signature(value: str) -> str:
    signature = re.sub(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}", "", value)
    signature = re.sub(r"hive_\d{14}_[A-Za-z0-9]+", "hive_<query_id>", signature)
    signature = re.sub(r"application_\d+_\d+", "application_<id>", signature)
    signature = re.sub(r"dag_\d+_\d+_\d+", "dag_<id>", signature)
    signature = re.sub(r"\b\d+\b", "<num>", signature)
    signature = re.sub(r"\s+", " ", signature)
    return signature.strip(" :-")[:500]


def infer_service(source_file: str) -> str | None:
    text = (source_file or "").lower()
    if "hiveserver2" in text or "hs2" in text:
        return "HS2"
    if "metastore" in text or "hms" in text:
        return "HMS"
    if "yarn" in text or "nodemanager" in text or "resourcemanager" in text:
        return "YARN"
    if "hdfs" in text or "namenode" in text or "datanode" in text:
        return "HDFS"
    if "hbase" in text:
        return "HBASE"
    if "tez" in text:
        return "TEZ"
    if "ranger" in text:
        return "RANGER"
    if "atlas" in text:
        return "ATLAS"
    return None


def _build_query_application_map(records: Iterable[dict]) -> dict[str, str]:
    """Map query_id to application_id only when the mapping is unambiguous."""
    candidates: dict[str, set[str]] = defaultdict(set)
    for record in records:
        query_id = record.get("query_id")
        application_id = record.get("application_id")
        if query_id and application_id:
            candidates[query_id].add(application_id)

    return {
        query_id: next(iter(application_ids))
        for query_id, application_ids in candidates.items()
        if len(application_ids) == 1
    }
