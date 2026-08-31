"""
Pipeline: orchestrates all Phase 1 steps.
Prefilter → Chunk → Embed → Index
"""

import os
import yaml
import time
from phase1.prefilter import run_prefilter
from phase1.chunker import chunk_entries
from phase1.embedder import load_model, embed_chunks
from phase1.enrichment import enrich_metadata_records
from phase1.indexer import build_index, save_index
from phase1.opensearch_indexer import save_opensearch_index


def run_pipeline(config_path: str = "config/config.yaml") -> None:
    """Run the complete Phase 1 pipeline."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("HIVE LOG ANALYZER - PHASE 1 PIPELINE")
    print("=" * 60)
    start_time = time.time()

    # ---- STEP 1: PRE-FILTER ----
    print("\n[STEP 1/4] PRE-FILTERING LOGS")
    print("-" * 40)
    entries = run_prefilter(config)

    if not entries:
        print("ERROR: No entries after pre-filtering. "
              "Check log file paths in config.")
        return

    # ---- STEP 2: CHUNK ----
    print("\n[STEP 2/4] CHUNKING ENTRIES")
    print("-" * 40)
    chunks = chunk_entries(entries, config)

    if not chunks:
        print("ERROR: No chunks produced. Check chunking config.")
        return

    # ---- STEP 3: EMBED ----
    print("\n[STEP 3/4] GENERATING EMBEDDINGS")
    print("-" * 40)
    embed_config = config['embedding']
    model = load_model(embed_config['model_name'])
    vectors, metadata = embed_chunks(
        chunks,
        model,
        embed_config['batch_size']
    )
    metadata = enrich_metadata_records(metadata)

    # ---- STEP 4: INDEX ----
    print("\n[STEP 4/4] BUILDING VECTOR INDEX")
    print("-" * 40)
    index_config = config['index']
    vector_store = config.get("vector_store", {})
    backend = vector_store.get("backend", "faiss")
    if backend == "opensearch":
        print("Using OpenSearch vector backend")
        try:
            save_opensearch_index(
                config=config,
                metadata=metadata,
                vectors=vectors,
                vector_dim=embed_config['vector_dim'],
            )
        except Exception as exc:
            if not vector_store.get("fallback_to_faiss", True):
                raise
            print(f"OpenSearch indexing unavailable ({exc}); falling back to FAISS")
            backend = "faiss"
            index = build_index(
                vectors,
                index_config['index_type'],
                embed_config['vector_dim']
            )
            save_index(
                index,
                metadata,
                index_config['output_dir'],
                index_config['faiss_index_file'],
                index_config['metadata_file']
            )
        if backend == "opensearch" and vector_store.get("also_write_faiss", False):
            index = build_index(
                vectors,
                index_config['index_type'],
                embed_config['vector_dim']
            )
            save_index(
                index,
                metadata,
                index_config['output_dir'],
                index_config['faiss_index_file'],
                index_config['metadata_file']
            )
    else:
        index = build_index(
            vectors,
            index_config['index_type'],
            embed_config['vector_dim']
        )
        save_index(
            index,
            metadata,
            index_config['output_dir'],
            index_config['faiss_index_file'],
            index_config['metadata_file']
        )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"Total time:     {elapsed:.1f}s")
    print(f"Log entries:    {len(entries):,}")
    print(f"Chunks indexed: {len(chunks):,}")
    print(f"Vector dims:    {vectors.shape[1]}")
    print(f"Index backend:   {backend}")
    print(f"Index location: {index_config['output_dir']}/")
    print("\nReady for Phase 2 — Query Auto-Debugger")