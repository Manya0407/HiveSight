"""
Indexer: builds and persists FAISS vector index.
IndexFlatL2 for exact search at prototype scale.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Tuple


def build_index(
    vectors: np.ndarray,
    index_type: str,
    vector_dim: int
) -> faiss.Index:
    """Build FAISS index from vectors."""
    print(f"Building FAISS index ({index_type})...")
    print(f"  Vectors: {vectors.shape[0]:,} × {vectors.shape[1]}")

    if index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(vector_dim)
    elif index_type == "IndexFlatIP":
        # Inner product — use with normalized vectors for cosine sim
        index = faiss.IndexFlatIP(vector_dim)
    else:
        # Default to FlatL2
        index = faiss.IndexFlatL2(vector_dim)

    # Ensure float32
    vectors_f32 = vectors.astype(np.float32)
    index.add(vectors_f32)

    print(f"  Index built. Total vectors indexed: {index.ntotal:,}")
    return index


def save_index(
    index: faiss.Index,
    metadata: List[dict],
    output_dir: str,
    index_filename: str,
    metadata_filename: str
) -> None:
    """Persist FAISS index and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(output_dir, index_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"  FAISS index saved: {index_path}")
    print(f"  Index size: "
          f"{os.path.getsize(index_path) / 1024 / 1024:.2f} MB")

    # Save metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved: {metadata_path}")
    print(f"  Metadata size: "
          f"{os.path.getsize(metadata_path) / 1024 / 1024:.2f} MB")


def load_index(
    output_dir: str,
    index_filename: str,
    metadata_filename: str
) -> Tuple[faiss.Index, List[dict]]:
    """Load persisted FAISS index and metadata."""
    index_path = os.path.join(output_dir, index_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    index = faiss.read_index(index_path)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Index loaded: {index.ntotal:,} vectors")
    return index, metadata


def search_index(
    query_vector: np.ndarray,
    index: faiss.Index,
    metadata: List[dict],
    top_k: int = 5
) -> List[dict]:
    """
    Search FAISS index for top-K similar chunks.
    Returns list of metadata dicts with similarity scores.
    """
    query_f32 = query_vector.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_f32, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        result = metadata[idx].copy()
        result['similarity_score'] = float(dist)
        results.append(result)

    return results