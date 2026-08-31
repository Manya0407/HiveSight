"""
Retriever: loads FAISS index and searches for similar chunks.
Uses same embedding model as Phase 1 to ensure vector space alignment.
"""

import json
import numpy as np
import faiss
from typing import List, Optional
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, config: dict):
        self.config = config
        index_config = config['index']
        embed_config = config['embedding']
        vector_store = config.get("vector_store", {})
        self.backend = vector_store.get("backend", "faiss")

        index_path = f"{index_config['output_dir']}/{index_config['faiss_index_file']}"
        metadata_path = f"{index_config['output_dir']}/{index_config['metadata_file']}"

        print(f"Loading embedding model: {embed_config['model_name']}")
        self.model = SentenceTransformer(embed_config['model_name'])

        self.top_k = config['retrieval']['top_k']
        self.candidate_depth = config.get('retrieval', {}).get(
            'candidate_depth',
            max(self.top_k, self.top_k * 16)
        )
        self.index = None
        self.metadata = []
        self.os_client = None
        self.os_index_name = None

        if self.backend == "opensearch":
            try:
                self._init_opensearch()
            except Exception as exc:
                if vector_store.get("fallback_to_faiss", True):
                    print(f"OpenSearch unavailable ({exc}); falling back to FAISS")
                    self.backend = "faiss"
                    self._init_faiss(index_path, metadata_path)
                else:
                    raise
        else:
            self._init_faiss(index_path, metadata_path)

        if self.backend == "faiss":
            print(f"Retriever ready. Index has {self.index.ntotal:,} vectors.")
        else:
            print(f"Retriever ready. OpenSearch index: {self.os_index_name}")

    def _init_faiss(self, index_path: str, metadata_path: str) -> None:
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def _init_opensearch(self) -> None:
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:
            raise RuntimeError(
                "OpenSearch backend requires the 'opensearch-py' package."
            ) from exc

        os_config = self.config.get("vector_store", {}).get("opensearch", {})
        host = os_config.get("host", "http://localhost:9200")
        username = os_config.get("username")
        password = os_config.get("password")
        http_auth = (username, password) if username and password else None

        self.os_index_name = os_config.get("index_name", "hivesight-log-chunks")
        self.os_client = OpenSearch(
            hosts=[host],
            http_auth=http_auth,
            verify_certs=os_config.get("verify_certs", False),
            timeout=os_config.get("timeout", 30),
        )
        if not self.os_client.indices.exists(index=self.os_index_name):
            raise RuntimeError(f"OpenSearch index not found: {self.os_index_name}")

    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed query using same model as indexing."""
        vector = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vector.astype(np.float32)

    def search(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None
    ) -> List[dict]:
        """
        Search FAISS index for top-K similar chunks.
        Optionally filter by category.
        """
        k = top_k or self.top_k
        query_vector = self.embed_query(query_text)
        preferred_category = filter_category or self._infer_preferred_category(query_text)

        if self.backend == "opensearch":
            candidates = self._search_opensearch(
                query_vector=query_vector,
                top_k=k,
                filter_category=filter_category,
            )
            return self._rank_results(candidates, preferred_category)[:k]

        # Retrieve extra candidates so metadata-aware reranking can suppress
        # broad/noisy matches while keeping semantic recall.
        search_k = min(self.index.ntotal, max(k, self.candidate_depth))
        distances, indices = self.index.search(query_vector, search_k)

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx].copy()
            chunk['similarity_score'] = float(dist)
            if filter_category:
                if chunk.get('category') == filter_category:
                    candidates.append(chunk)
            else:
                candidates.append(chunk)

        return self._rank_results(candidates, preferred_category)[:k]

    def search_by_query_id(self, query_id: str) -> List[dict]:
        """Get all chunks associated with a specific queryId."""
        if self.backend == "opensearch":
            return self._search_opensearch_by_query_id(query_id)
        return [
            chunk for chunk in self.metadata
            if chunk.get('query_id') == query_id
        ]

    def _search_opensearch(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter_category: Optional[str] = None,
    ) -> List[dict]:
        # Retrieve a wider candidate pool before the existing metadata-aware
        # reranker narrows results to top_k.
        search_k = max(top_k, self.candidate_depth)
        knn_query = {
            "vector": query_vector.reshape(-1).tolist(),
            "k": search_k,
        }

        if filter_category:
            body = {
                "size": search_k,
                "query": {
                    "bool": {
                        "must": [{"knn": {"embedding": knn_query}}],
                        "filter": [{"term": {"category": filter_category}}],
                    }
                },
            }
        else:
            body = {
                "size": search_k,
                "query": {"knn": {"embedding": knn_query}},
            }

        response = self.os_client.search(index=self.os_index_name, body=body)
        return [self._hit_to_chunk(hit) for hit in response["hits"]["hits"]]

    def _search_opensearch_by_query_id(self, query_id: str) -> List[dict]:
        body = {
            "size": self.top_k * 8,
            "query": {
                "term": {
                    "query_id": query_id
                }
            }
        }
        response = self.os_client.search(index=self.os_index_name, body=body)
        return [self._hit_to_chunk(hit) for hit in response["hits"]["hits"]]

    def _hit_to_chunk(self, hit: dict) -> dict:
        source = hit.get("_source", {}).copy()
        source.pop("embedding", None)
        source["similarity_score"] = float(hit.get("_score", 0.0))
        return source

    def _infer_preferred_category(self, query_text: str) -> Optional[str]:
        """Infer a likely failure category from exception/search text."""
        text = query_text.lower()
        optimization_terms = [
            "cbo failed",
            "cost based optimizer",
            "optimization failed",
            "reloptplanner",
            "invalid statistics",
        ]
        if any(term in text for term in optimization_terms):
            return "CBO_OPTIMIZATION"

        hms_terms = [
            "table not found",
            "database does not exist",
            "database not found",
            "nosuchobjectexception",
            "metaexception",
            "metastore",
            "hms",
        ]
        if any(term in text for term in hms_terms):
            return "HMS_LOOKUP"

        compilation_terms = [
            "invalid table alias",
            "invalid column reference",
            "column reference",
            "nonexistent_column",
            "parseexception",
            "syntax error",
            "cannot recognize input",
            "invalid function",
        ]
        if any(term in text for term in compilation_terms):
            return "COMPILATION"

        if any(term in text for term in ["lockexception", "transaction", "txn"]):
            return "CONCURRENCY"

        dag_submission_terms = [
            "sessionnotrunning",
            "tez am failed to start",
            "submitdag failed",
            "application master not running",
            "dag submission failed",
        ]
        if any(term in text for term in dag_submission_terms):
            return "TEZ_AM"

        execution_terms = [
            "outofmemoryerror",
            "java heap space",
            "unable to kill query locally or on remote servers",
            "task attempt failed",
            "task failed",
            "vertex failed",
            "hive runtime error",
            "during execution",
        ]
        if any(term in text for term in execution_terms):
            return "EXECUTION"

        return None

    def _rank_results(
        self,
        chunks: List[dict],
        preferred_category: Optional[str]
    ) -> List[dict]:
        """Rank by semantic distance plus simple metadata preferences."""
        level_priority = {"FATAL": 0, "ERROR": 1, "WARN": 2}

        def rank_key(chunk: dict) -> tuple:
            category = chunk.get("category", "")
            level = chunk.get("log_level", "WARN")

            preferred_penalty = (
                0 if preferred_category and category == preferred_category else 1
            )
            baseline_penalty = 1 if category == "BASELINE" else 0

            return (
                preferred_penalty,
                baseline_penalty,
                level_priority.get(level, 3),
                chunk.get("similarity_score", float("inf")),
            )

        return sorted(chunks, key=rank_key)