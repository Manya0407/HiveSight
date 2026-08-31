"""
Embedder: generates 384-dim vectors using all-MiniLM-L6-v2.
Runs locally, zero API cost.
"""

import os
import json
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from phase1.chunker import LogChunk


def load_model(model_name: str) -> SentenceTransformer:
    """Load the embedding model."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Vector dimensions: "
          f"{model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(
    chunks: List[LogChunk],
    model: SentenceTransformer,
    batch_size: int
) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate embeddings for all chunks.

    Returns:
        vectors: numpy array of shape (n_chunks, 384)
        metadata: list of dicts with chunk metadata
    """
    texts = [chunk.text for chunk in chunks]
    metadata = []

    print(f"Embedding {len(chunks):,} chunks in batches of {batch_size}...")

    # Generate embeddings with progress bar
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # normalize for cosine similarity
    )

    # Build metadata list — stored alongside FAISS index
    for i, chunk in enumerate(chunks):
        metadata.append({
            "chunk_id": chunk.chunk_id,
            "query_id": chunk.query_id,
            "timestamp": chunk.timestamp,
            "log_level": chunk.log_level,
            "source_file": chunk.source_file,
            "source_type": chunk.source_type,
            "category": chunk.category,
            "failure_stage": chunk.failure_stage,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "text": chunk.text,           # store text for retrieval
            "vector_index": i             # position in FAISS index
        })

    print(f"Embedding complete. Shape: {vectors.shape}")
    return vectors, metadata