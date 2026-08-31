"""
Debugger: orchestrates the full Phase 2 RAG pipeline.
Query Auto-Debugger — from failed query to diagnosis in under 2 minutes.
"""

import os
import re
import yaml
from collections import Counter, defaultdict
from typing import Optional
from phase1.chunker import chunk_entries
from phase1.enrichment import enrich_metadata_records
from phase1.prefilter import prefilter_file
from phase2.retriever import Retriever
from phase2.prompt import build_prompt, build_exception_prompt, build_summary_prompt
from phase2.llm import LLMClient
from phase2.parser import parse_response, format_diagnosis


class HiveDebugger:
    def __init__(self, config_path: str = "config/config.yaml", verbose: bool = False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.verbose = verbose
        print("Initializing Hive Query Auto-Debugger...")
        self.retriever = Retriever(self.config)
        self.llm = LLMClient(self.config)
        print("Debugger ready.\n")

    def _log(self, message: str) -> None:
        """Print progress details only when verbose mode is enabled."""
        if self.verbose:
            print(message)

    def debug_by_query_id(self, query_id: str) -> dict:
        """
        Debug a failed query by its Hive query ID.
        Retrieves relevant chunks and generates diagnosis.
        """
        print(f"Debugging query: {query_id}")

        # First try to find chunks with this exact query ID
        exact_chunks = self.retriever.search_by_query_id(query_id)

        if exact_chunks:
            print(f"Found {len(exact_chunks)} chunks with exact query ID match")
            retrieved_chunks = exact_chunks[:self.config['retrieval']['top_k']]
        else:
            # Fall back to semantic search using query ID as text
            print("No exact match — falling back to semantic search")
            retrieved_chunks = self.retriever.search(
                f"query failure {query_id}"
            )

        return self._generate_diagnosis(
            retrieved_chunks=retrieved_chunks,
            query_id=query_id,
            question="What is the root cause of this Hive query failure?"
        )

    def debug_by_exception(self, exception_text: str) -> dict:
        """
        Debug using a raw exception or stack trace.
        No log file needed — exception text is embedded directly.
        """
        print(f"Analyzing exception: {exception_text[:100]}...")

        if self._looks_like_non_failure(exception_text):
            return {
                "failure_stage": "UNKNOWN",
                "root_cause": (
                    "The input indicates a successful query or explicitly says "
                    "there is no exception/error, so there is no failure to diagnose."
                ),
                "error_class": "None",
                "confidence": "LOW",
                "evidence": exception_text,
                "retrieved_chunks": []
            }

        retrieved_chunks = self.retriever.search(exception_text)
        prompt = build_exception_prompt(exception_text, retrieved_chunks)

        print(f"Retrieved {len(retrieved_chunks)} similar chunks from index")
        print("Calling LLM for diagnosis...")

        llm_response = self.llm.call(prompt)
        diagnosis = parse_response(llm_response, retrieved_chunks, exception_text)
        return diagnosis

    def debug_log_file(
        self,
        log_file: str,
        query_id: str = None,
        last_only: bool = False
    ) -> dict:
        """
        Debug a newly provided log file without rebuilding the FAISS index.
        Uploaded chunks are primary evidence; historical chunks add context.
        """
        print(f"Analyzing uploaded log file: {log_file}")

        if not os.path.exists(log_file):
            return {
                "failure_stage": "UNKNOWN",
                "root_cause": f"Log file not found: {log_file}",
                "error_class": "FileNotFoundError",
                "confidence": "LOW",
                "evidence": "",
                "retrieved_chunks": []
            }

        pf_config = self.config["prefilter"]
        entries = prefilter_file(
            file_path=log_file,
            source_type=self._infer_source_type(log_file),
            category="UPLOADED",
            failure_stage="UNKNOWN",
            context_before=pf_config["context_before"],
            context_after=pf_config["context_after"],
            noise_patterns=pf_config["noise_patterns"]
        )

        if not entries:
            return {
                "failure_stage": "UNKNOWN",
                "root_cause": (
                    "No WARN, ERROR, or FATAL evidence was found after "
                    "pre-filtering this log file."
                ),
                "error_class": "None",
                "confidence": "LOW",
                "evidence": "",
                "retrieved_chunks": []
            }

        chunks = chunk_entries(entries, self.config)
        uploaded_chunks = [
            self._chunk_to_metadata(chunk, log_file)
            for chunk in chunks
        ]
        uploaded_chunks = enrich_metadata_records(uploaded_chunks)
        uploaded_chunks = self._rank_uploaded_chunks(uploaded_chunks)

        if query_id:
            return self._debug_uploaded_query_id(log_file, uploaded_chunks, query_id)

        query_groups = self._group_chunks_by_query_id(uploaded_chunks)
        if last_only:
            return self._debug_last_uploaded_query(log_file, query_groups)

        if len(query_groups) > 1:
            return self._debug_uploaded_query_groups(log_file, query_groups)

        top_k = self.config["retrieval"]["top_k"]
        primary_chunks = uploaded_chunks[:top_k]
        search_text = "\n".join(chunk["text"] for chunk in primary_chunks[:3])
        historical_chunks = self.retriever.search(
            search_text,
            top_k=min(3, top_k)
        )

        context_chunks = primary_chunks + historical_chunks
        print(
            f"Using {len(primary_chunks)} uploaded chunks and "
            f"{len(historical_chunks)} historical chunks"
        )

        prompt = build_prompt(
            retrieved_chunks=context_chunks,
            user_question=(
                "Analyze this uploaded Hive log file. Explain what failed, "
                "where it failed, and why it failed."
            )
        )

        print("Calling LLM for diagnosis...")
        llm_response = self.llm.call(prompt)
        diagnosis = parse_response(llm_response, context_chunks, search_text)
        dominant_stage = self._dominant_stage(primary_chunks)
        if dominant_stage != "UNKNOWN":
            diagnosis["failure_stage"] = dominant_stage
        return diagnosis

    def _debug_last_uploaded_query(self, log_file: str, query_groups: dict) -> dict:
        """Debug the latest failed query found in an uploaded log file."""
        if not query_groups:
            return {
                "failure_stage": "UNKNOWN",
                "root_cause": (
                    f"No query IDs with failure evidence were found in {log_file}."
                ),
                "error_class": "Unknown",
                "confidence": "LOW",
                "evidence": "",
                "retrieved_chunks": []
            }

        query_id, chunks = max(
            query_groups.items(),
            key=lambda item: self._latest_chunk_timestamp(item[1])
        )

        print(f"Latest failed query ID in uploaded log: {query_id}")
        return self._debug_uploaded_query_id(log_file, chunks, query_id)

    def summarize_log_file(self, log_file: str) -> dict:
        """Summarize a full log file, not just one failure."""
        print(f"Summarizing log file: {log_file}")

        if not os.path.exists(log_file):
            return {
                "overall_status": "NO_FAILURE_EVIDENCE",
                "summary": f"Log file not found: {log_file}",
                "main_events": [],
                "failure_stages": ["UNKNOWN"],
                "query_ids": [],
                "top_errors": [],
                "recommended_next_step": "Check that the file path is correct.",
                "evidence_chunks": []
            }

        pf_config = self.config["prefilter"]
        entries = prefilter_file(
            file_path=log_file,
            source_type=self._infer_source_type(log_file),
            category="UPLOADED",
            failure_stage="UNKNOWN",
            context_before=pf_config["context_before"],
            context_after=pf_config["context_after"],
            noise_patterns=pf_config["noise_patterns"]
        )

        if not entries:
            return {
                "overall_status": "NO_FAILURE_EVIDENCE",
                "summary": (
                    "No WARN, ERROR, or FATAL evidence was found after "
                    "pre-filtering this log file."
                ),
                "main_events": [],
                "failure_stages": ["UNKNOWN"],
                "query_ids": [],
                "top_errors": [],
                "recommended_next_step": (
                    "If a failure was expected, verify the log path and inspect "
                    "INFO-level framework output."
                ),
                "evidence_chunks": []
            }

        chunks = chunk_entries(entries, self.config)
        uploaded_chunks = [
            self._chunk_to_metadata(chunk, log_file)
            for chunk in chunks
        ]
        uploaded_chunks = enrich_metadata_records(uploaded_chunks)
        ranked_chunks = self._rank_uploaded_chunks(uploaded_chunks)
        selected_chunks = ranked_chunks[: self.config["retrieval"]["top_k"]]
        stats = self._build_summary_stats(entries, uploaded_chunks)

        prompt = build_summary_prompt(log_file, stats, selected_chunks)
        print("Calling LLM for log summary...")
        llm_response = self.llm.call(prompt)
        summary = self._parse_summary_response(llm_response)
        summary = self._ground_summary_from_stats(summary, stats)
        summary["evidence_chunks"] = selected_chunks
        summary["stats"] = stats
        return summary

    def _debug_uploaded_query_id(
        self,
        log_file: str,
        uploaded_chunks: list,
        query_id: str
    ) -> dict:
        """Debug one query ID from a newly uploaded log file."""
        matching_chunks = [
            chunk for chunk in uploaded_chunks
            if chunk.get("query_id") == query_id
            or str(chunk.get("query_id", "")).startswith(query_id)
        ]

        if not matching_chunks:
            return {
                "failure_stage": "UNKNOWN",
                "root_cause": (
                    f"Query ID {query_id} was not found in uploaded log file "
                    f"{log_file} after pre-filtering."
                ),
                "error_class": "Unknown",
                "confidence": "LOW",
                "evidence": "",
                "retrieved_chunks": []
            }

        if not self._query_group_has_failure(matching_chunks):
            return {
                "failure_stage": "UNKNOWN",
                "root_cause": (
                    f"Query ID {query_id} appears in {log_file}, but the "
                    "uploaded evidence does not show a query failure marker "
                    "such as FAILED, HiveSQLException, or Error running hive query."
                ),
                "error_class": "None",
                "confidence": "LOW",
                "evidence": "",
                "retrieved_chunks": matching_chunks[: self.config["retrieval"]["top_k"]],
                "query_id": query_id
            }

        print(
            f"Found {len(matching_chunks)} uploaded chunks for query ID: "
            f"{query_id}"
        )

        top_k = self.config["retrieval"]["top_k"]
        primary_chunks = self._select_primary_query_chunks(matching_chunks, top_k)
        search_text = "\n".join(chunk["text"] for chunk in primary_chunks[:2])
        context_chunks = primary_chunks

        prompt = build_prompt(
            retrieved_chunks=context_chunks,
            user_question=(
                "Analyze this specific failed query from an uploaded Hive log. "
                "Explain what failed, where it failed, and why it failed."
            ),
            query_id=query_id
        )

        print("Calling LLM for uploaded query diagnosis...")
        llm_response = self.llm.call(prompt)
        diagnosis = parse_response(llm_response, context_chunks, search_text)
        diagnosis = self._ground_uploaded_diagnosis(
            diagnosis,
            primary_chunks,
            query_id
        )
        dominant_stage = self._dominant_stage(primary_chunks)
        if dominant_stage != "UNKNOWN":
            diagnosis["failure_stage"] = dominant_stage
        diagnosis["query_id"] = query_id
        return diagnosis

    def _debug_uploaded_query_groups(
        self,
        log_file: str,
        query_groups: dict
    ) -> dict:
        """Generate one diagnosis per failed query in an uploaded log."""
        max_queries = self.config.get("uploaded_logs", {}).get(
            "max_queries_per_file",
            10
        )
        ranked_groups = sorted(
            query_groups.items(),
            key=lambda item: self._query_group_rank_key(item[1])
        )
        selected_groups = ranked_groups[:max_queries]

        print(
            f"Found {len(query_groups)} query IDs with failure evidence; "
            f"diagnosing {len(selected_groups)}"
        )

        diagnoses = []
        for query_id, chunks in selected_groups:
            top_k = self.config["retrieval"]["top_k"]
            primary_chunks = self._select_primary_query_chunks(chunks, top_k)
            search_text = "\n".join(chunk["text"] for chunk in primary_chunks[:2])
            context_chunks = primary_chunks
            prompt = build_prompt(
                retrieved_chunks=context_chunks,
                user_question=(
                    "Analyze this failed query from an uploaded Hive log. "
                    "Explain what failed, where it failed, and why it failed."
                ),
                query_id=query_id
            )

            self._log(f"Calling LLM for uploaded query diagnosis: {query_id}")
            llm_response = self.llm.call(prompt)
            diagnosis = parse_response(llm_response, context_chunks, search_text)
            diagnosis = self._ground_uploaded_diagnosis(
                diagnosis,
                primary_chunks,
                query_id
            )
            dominant_stage = self._dominant_stage(primary_chunks)
            if dominant_stage != "UNKNOWN":
                diagnosis["failure_stage"] = dominant_stage
            diagnosis["query_id"] = query_id
            diagnoses.append(diagnosis)

        return {
            "multi_query": True,
            "log_file": log_file,
            "total_failed_queries": len(query_groups),
            "diagnosed_queries": len(diagnoses),
            "diagnoses": diagnoses
        }

    def _looks_like_non_failure(self, text: str) -> bool:
        """Detect inputs that explicitly describe success rather than failure."""
        normalized = text.lower()
        success_terms = [
            "completed successfully",
            "query succeeded",
            "query successful",
            "no exception",
            "no error",
            "no failure",
        ]
        return any(term in normalized for term in success_terms)

    def _infer_source_type(self, log_file: str) -> str:
        """Infer source system from file name for uploaded logs."""
        filename = os.path.basename(log_file).lower()
        if "metastore" in filename or "hms" in filename:
            return "HMS"
        if "tez" in filename:
            return "TEZ"
        return "HS2"

    def _chunk_to_metadata(self, chunk, log_file: str) -> dict:
        """Convert a Phase 1 LogChunk into Phase 2 prompt metadata."""
        category = self._infer_category(chunk.text)
        query_id = self._extract_failure_query_id(chunk.text) or chunk.query_id
        return {
            "chunk_id": chunk.chunk_id,
            "query_id": query_id,
            "timestamp": chunk.timestamp,
            "log_level": chunk.log_level,
            "source_file": os.path.basename(log_file),
            "source_type": chunk.source_type,
            "category": category,
            "failure_stage": self._category_to_stage(category),
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "text": chunk.text,
            "similarity_score": 0.0,
            "origin": "uploaded_log"
        }

    def _infer_category(self, text: str) -> str:
        """Infer broad failure category from uploaded log text."""
        normalized = text.lower()

        if any(term in normalized for term in [
            "assertionerror",
            "attributeerror",
            "failed tests:",
            "exit status: exitcode.tests_failed",
            "start of test error",
            "test failed in",
            "test \"",
            "assert 'fail' == 'pass'",
            "pytest",
        ]):
            return "TEST_FRAMEWORK"

        # Malformed resource command usage is a command/parse-level error,
        # not a concurrency failure.
        if any(term in normalized for term in [
            "deleteresourceprocessor",
            "addresourceprocessor",
            "usage: delete [file|jar|archive]",
            "usage: add [file|jar|archive]",
        ]):
            return "COMPILATION"

        if "cannot convert an acid table to non-acid" in normalized:
            return "EXECUTION"

        if any(term in normalized for term in [
            "lockexception",
            "commit is not allowed",
            "rollback is not allowed",
            "could not acquire the lock",
            "txnabortedexception",
            "start_transaction is not supported",
            "start transaction is not supported",
        ]):
            return "CONCURRENCY"

        if any(term in normalized for term in [
            "sessionnotrunning",
            "tez am failed to start",
            "submitdag failed",
            "application master not running",
            "dag submission failed",
        ]):
            return "TEZ_AM"

        if any(term in normalized for term in [
            "outofmemoryerror",
            "java heap space",
            "noclassdeffounderror",
            "classnotfoundexception",
            "root_input_init_failure",
            "vertex failed",
            "dag did not succeed",
            "unable to kill query locally or on remote servers",
            "task attempt failed",
            "task failed",
            "vertex failed",
            "hive runtime error",
        ]):
            return "EXECUTION"

        if any(term in normalized for term in [
            "parseexception",
            "invalid function",
            "invalid table alias",
            "invalid column reference",
            "cannot recognize input",
            "syntax error",
        ]):
            return "COMPILATION"

        if any(term in normalized for term in [
            "nosuchobjectexception",
            "invalidtableexception",
            "table not found",
            "database does not exist",
            "partition not found",
            "metaexception",
        ]):
            return "HMS_LOOKUP"

        if any(term in normalized for term in [
            "cbo failed",
            "cost based optimizer",
            "optimization failed",
            "reloptplanner",
            "invalid statistics",
        ]):
            return "CBO_OPTIMIZATION"

        if any(term in normalized for term in [
            "semanticexception",
            "invalid column reference",
            "invalid table alias",
        ]):
            return "COMPILATION"

        return "UPLOADED"

    def _extract_failure_query_id(self, text: str) -> Optional[str]:
        """Attach a chunk to the query nearest its actual failure line."""
        lines = text.splitlines()
        failure_terms = [
            " ERROR ",
            " FATAL ",
            "FAILED:",
            "HiveSQLException",
            "Exception:",
        ]
        query_pattern = re.compile(r"(hive_\d{14}_[A-Za-z0-9]+)")

        for idx, line in enumerate(lines):
            if not any(term in line for term in failure_terms):
                continue

            same_line_match = query_pattern.search(line)
            if same_line_match:
                return same_line_match.group(1)

            for prev in range(idx, max(-1, idx - 25), -1):
                match = query_pattern.search(lines[prev])
                if match:
                    return match.group(1)

        return None

    def _category_to_stage(self, category: str) -> str:
        """Map internal categories to user-facing failure stages."""
        return {
            "COMPILATION": "COMPILATION",
            "HMS_LOOKUP": "HMS_LOOKUP",
            "EXECUTION": "EXECUTION",
            "TEZ_AM": "DAG_SUBMISSION",
            "CBO_OPTIMIZATION": "OPTIMIZATION",
            "CONCURRENCY": "CONCURRENCY",
            "TEST_FRAMEWORK": "TEST_FRAMEWORK",
        }.get(category, "UNKNOWN")

    def _rank_uploaded_chunks(self, chunks: list) -> list:
        """Prioritize uploaded chunks most likely to contain root cause."""
        level_priority = {"FATAL": 0, "ERROR": 1, "WARN": 2}

        def rank_key(chunk: dict) -> tuple:
            category = chunk.get("category")
            text = chunk.get("text", "").lower()
            framework_priority = 0
            if any(term in text for term in [
                "noclassdeffounderror",
                "classnotfoundexception",
                "jacksonfeature",
                "root_input_init_failure",
                "dag did not succeed",
            ]):
                framework_priority = -4
            elif category == "TEST_FRAMEWORK":
                framework_priority = -3
            elif any(term in text for term in [
                "failed tests:",
                "assertionerror",
                "exit status: exitcode.tests_failed",
                "test failed in",
            ]):
                framework_priority = -2

            category_penalty = 1 if category == "UPLOADED" else 0
            return (
                framework_priority,
                level_priority.get(chunk.get("log_level", "WARN"), 3),
                category_penalty,
                chunk.get("timestamp", ""),
            )

        return sorted(chunks, key=rank_key)

    def _build_summary_stats(self, entries: list, chunks: list) -> dict:
        """Create deterministic counts for full-log summaries."""
        query_ids = sorted({
            entry.query_id for entry in entries
            if entry.query_id and entry.query_id != "null"
        })
        level_counts = Counter(entry.level for entry in entries)
        category_counts = Counter(chunk.get("category", "UNKNOWN") for chunk in chunks)
        stage_counts = Counter(
            self._category_to_stage(chunk.get("category", ""))
            for chunk in chunks
        )
        failure_types = self._build_failure_type_stats(chunks)

        top_errors = []
        for chunk in self._rank_uploaded_chunks(chunks):
            for line in chunk.get("text", "").splitlines():
                if "ERROR" in line or "FATAL" in line:
                    top_errors.append(line.strip())
                    break
            if len(top_errors) >= 10:
                break

        return {
            "total_prefiltered_entries": len(entries),
            "total_chunks": len(chunks),
            "level_counts": dict(level_counts),
            "category_counts": dict(category_counts),
            "stage_counts": dict(stage_counts),
            "query_count": len(query_ids),
            "query_ids": query_ids[:25],
            "failure_types": failure_types,
            "top_errors": top_errors,
        }

    def _build_failure_type_stats(self, chunks: list) -> list:
        """Aggregate repeated failure lines into user-visible failure types."""
        grouped = {}
        seen_signatures = set()

        for chunk in self._rank_uploaded_chunks(chunks):
            for line in chunk.get("text", "").splitlines():
                if not self._is_failure_inventory_line(line):
                    continue

                error_line = line.strip()
                signature = self._normalize_error_signature(error_line)
                if not signature:
                    continue

                # A chunk can overlap another chunk; do not count the same
                # repeated line/signature twice for the same query/timestamp.
                dedupe_key = (
                    signature,
                    chunk.get("query_id") or "",
                    chunk.get("timestamp") or "",
                    chunk.get("source_file") or "",
                )
                if dedupe_key in seen_signatures:
                    continue
                seen_signatures.add(dedupe_key)

                stage, error_class, _ = self._classify_failure_signal(error_line)
                if error_class == "Unknown":
                    error_class = self._infer_error_class(error_line)
                if error_class in {"NoClassDefFoundError", "ClassNotFoundException"}:
                    missing_class = self._extract_missing_class(error_line)
                    if missing_class:
                        error_class = f"{error_class}: {missing_class}"
                failure_type = error_class if error_class != "Unknown" else stage
                if failure_type in {"ERROR", "FATAL", "UNKNOWN"}:
                    failure_type = self._compact_signature_label(signature)

                item = grouped.setdefault(
                    failure_type,
                    {
                        "failure_type": failure_type,
                        "failure_stage": stage,
                        "occurrences": 0,
                        "queries": set(),
                        "first_seen": chunk.get("timestamp", ""),
                        "examples": [],
                        "signatures": Counter(),
                    }
                )

                item["occurrences"] += 1
                item["signatures"][signature] += 1
                query_id = chunk.get("query_id")
                if query_id and query_id != "null":
                    item["queries"].add(query_id)
                timestamp = chunk.get("timestamp", "")
                if timestamp and (
                    not item["first_seen"] or timestamp < item["first_seen"]
                ):
                    item["first_seen"] = timestamp
                if error_line not in item["examples"]:
                    item["examples"].append(error_line)

        failure_types = []
        for item in grouped.values():
            unique_signatures = len(item["signatures"])
            failure_types.append({
                "failure_type": item["failure_type"],
                "failure_stage": item["failure_stage"],
                "occurrences": item["occurrences"],
                "query_count": len(item["queries"]),
                "first_seen": item["first_seen"] or "N/A",
                "unique_signatures": unique_signatures,
                "example": item["examples"][0] if item["examples"] else "",
                "examples": item["examples"][:3],
            })

        return sorted(
            failure_types,
            key=lambda item: (
                -item["occurrences"],
                item["failure_type"],
                item["first_seen"],
            )
        )

    def _is_failure_inventory_line(self, line: str) -> bool:
        """Return True for lines worth listing in summary failure inventory."""
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith(">"):
            return False
        if set(stripped) <= {"#", "-", "_", "="}:
            return False
        lowered = line.lower()
        if "noinspection" in lowered or "end of test error" in lowered:
            return False
        if "start of test error" in lowered:
            return False
        if "ignoring any errors here" in lowered:
            return False
        if "pybroadexception" in lowered:
            return False
        if "capture_manager.py" in lowered and lowered.rstrip().endswith("- error -"):
            return False
        if "http_client.py" in lowered and "policy response" in lowered:
            return False
        if stripped.startswith("assert status,"):
            return False
        inventory_terms = [
            "error",
            "fatal",
            "failed:",
            "failed to",
            "exception",
            "assertionerror",
            "attributeerror",
            "noclassdeffounderror",
            "classnotfoundexception",
            "dag did not succeed",
            "vertex failed",
            "command execution failed",
        ]
        return any(term in lowered for term in inventory_terms)

    def _compact_signature_label(self, signature: str) -> str:
        """Create a readable group name for generic ERROR/FATAL lines."""
        label = signature
        label = re.sub(r"^.*?\|\s*", "", label)
        label = re.sub(r"^.*?-\s*(ERROR|FATAL)\s*-\s*", "", label)
        label = re.sub(r"^.*?(ERROR|FATAL)\s*:?\s*", "", label)
        label = label.strip(" -:")
        if not label:
            label = "Unclassified failure"
        return label[:140]

    def _normalize_error_signature(self, line: str) -> str:
        """Normalize an error line so repeated failures collapse together."""
        signature = re.sub(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}", "", line)
        signature = re.sub(r"hive_\d{14}_[A-Za-z0-9]+", "hive_<query_id>", signature)
        signature = re.sub(r"attempt_\d+_\d+_[A-Za-z]+_\d+_\d+", "attempt_<id>", signature)
        signature = re.sub(r"task_\d+_\d+_[A-Za-z]+_\d+", "task_<id>", signature)
        signature = re.sub(r"application_\d+_\d+", "application_<id>", signature)
        signature = re.sub(r"\b\d+\b", "<num>", signature)
        signature = re.sub(r"\s+", " ", signature)
        signature = signature.strip(" :-")
        if not signature or set(signature) <= {"#", "-", "_", "="}:
            return ""
        return signature

    def _infer_error_class(self, line: str) -> str:
        """Extract a Java/Python-style exception class when rules do not match."""
        match = re.search(r"\b([A-Za-z][A-Za-z0-9_]*(?:Exception|Error))\b", line)
        if match:
            return match.group(1)
        if "ERROR" in line:
            return "ERROR"
        if "FATAL" in line:
            return "FATAL"
        return "Unknown"

    def _extract_missing_class(self, line: str) -> str:
        """Extract the missing Java class from classpath errors."""
        patterns = [
            r"NoClassDefFoundError:\s*([A-Za-z0-9_.$/]+)",
            r"ClassNotFoundException:\s*([A-Za-z0-9_.$/]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1).strip()
        return ""

    def _ground_summary_from_stats(self, summary: dict, stats: dict) -> dict:
        """Use deterministic failure inventory to keep summary root cause grounded."""
        failure_types = stats.get("failure_types", [])
        primary = self._select_primary_failure_type(failure_types)
        if not primary:
            return summary

        summary["primary_failure"] = primary
        failure_type = primary.get("failure_type", "Unknown")
        lowered = failure_type.lower()

        if "noclassdeffounderror" in lowered or "classnotfoundexception" in lowered:
            root_line = (
                "Primary failure is a Java classpath/dependency issue: "
                f"{failure_type}. This caused Hive/Tez execution to fail; "
                "later command-execution and assertion errors are wrappers "
                "around this underlying failure."
            )
            summary["summary"] = root_line
            existing_events = summary.get("main_events", [])
            summary["main_events"] = [root_line] + [
                event for event in existing_events
                if "command execution failed" not in str(event).lower()
            ][:4]
            existing_errors = summary.get("top_errors", [])
            summary["top_errors"] = [failure_type] + [
                error for error in existing_errors
                if failure_type not in str(error)
            ][:4]
            summary["recommended_next_step"] = (
                "Inspect the Hive/Tez runtime classpath and Jackson dependency "
                "versions for the missing class shown in the primary failure."
            )

        return summary

    def _select_primary_failure_type(self, failure_types: list) -> dict:
        """Choose the most root-cause-like grouped failure."""
        if not failure_types:
            return {}

        def priority(item: dict) -> tuple:
            failure_type = item.get("failure_type", "").lower()
            occurrences = item.get("occurrences", 0)
            if "noclassdeffounderror" in failure_type or "classnotfoundexception" in failure_type:
                return (0, -occurrences)
            if "outofmemoryerror" in failure_type:
                return (1, -occurrences)
            if "hiveaccesscontrolexception" in failure_type:
                return (2, -occurrences)
            if "remoteexception" in failure_type:
                return (3, -occurrences)
            if "hiveexception" in failure_type:
                return (4, -occurrences)
            if "assertionerror" in failure_type:
                return (8, -occurrences)
            return (5, -occurrences)

        return sorted(failure_types, key=priority)[0]

    def _parse_summary_response(self, llm_response: str) -> dict:
        """Parse LLM summary JSON with a safe fallback."""
        import json
        import re

        parsed = None
        try:
            parsed = json.loads(llm_response.strip())
        except json.JSONDecodeError:
            start = llm_response.find("{")
            end = llm_response.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(llm_response[start:end + 1])
                except json.JSONDecodeError:
                    parsed = None

        if not parsed:
            parsed = self._extract_loose_summary_fields(llm_response)

        if not parsed:
            parsed = {
                "overall_status": "FAILED",
                "summary": re.sub(r"\s+", " ", llm_response.strip())[:1000],
                "main_events": [],
                "failure_stages": ["UNKNOWN"],
                "query_ids": [],
                "top_errors": [],
                "recommended_next_step": (
                    "Review the evidence chunks shown below for the most "
                    "important failures."
                ),
            }

        return {
            "overall_status": parsed.get("overall_status", "FAILED"),
            "summary": parsed.get("summary", ""),
            "main_events": parsed.get("main_events", []),
            "failure_stages": parsed.get("failure_stages", ["UNKNOWN"]),
            "query_ids": parsed.get("query_ids", []),
            "top_errors": parsed.get("top_errors", []),
            "recommended_next_step": parsed.get("recommended_next_step", ""),
        }

    def _extract_loose_summary_fields(self, text: str) -> dict:
        """Extract summary fields when the LLM returns JSON-like text."""
        import json
        import re

        parsed = {}

        for field in ["overall_status", "summary", "recommended_next_step"]:
            match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text, re.DOTALL)
            if match:
                parsed[field] = match.group(1).strip()

        for field in ["main_events", "failure_stages", "query_ids", "top_errors"]:
            match = re.search(rf'"{field}"\s*:\s*(\[[\s\S]*?\])', text)
            if not match:
                continue
            try:
                parsed[field] = json.loads(match.group(1))
            except json.JSONDecodeError:
                parsed[field] = re.findall(r'"([^"]+)"', match.group(1))

        return parsed or None

    def _select_primary_query_chunks(self, chunks: list, top_k: int) -> list:
        """Select the uploaded chunks that best represent one query failure."""
        ranked = self._rank_uploaded_chunks(chunks)
        error_chunks = [
            chunk for chunk in ranked
            if chunk.get("log_level") in ["ERROR", "FATAL"]
            and self._has_real_failure_signal(chunk.get("text", ""))
        ]
        primary_pool = error_chunks or ranked

        dominant_stage = self._dominant_stage(primary_pool)
        if dominant_stage != "UNKNOWN":
            primary_pool = [
                chunk for chunk in primary_pool
                if chunk.get("failure_stage") == dominant_stage
                or self._category_to_stage(chunk.get("category", "")) == dominant_stage
            ] or primary_pool

        return primary_pool[:top_k]

    def _has_real_failure_signal(self, text: str) -> bool:
        """Return True when a chunk contains concrete failure evidence."""
        normalized = text.lower()
        failure_terms = [
            "failed:",
            "error while compiling statement",
            "error running hive query",
            "exception",
            "unable to kill query locally or on remote servers",
            "table not found",
            "database does not exist",
            "invalid function",
            "invalid column reference",
            "cannot recognize input",
            "outofmemoryerror",
            "lockexception",
        ]
        success_terms = [
            "completed executing command",
            "\nok\n",
        ]
        has_failure = any(term in normalized for term in failure_terms)
        success_only = (
            any(term in normalized for term in success_terms)
            and not has_failure
        )
        return has_failure and not success_only

    def _ground_uploaded_diagnosis(
        self,
        diagnosis: dict,
        primary_chunks: list,
        query_id: str
    ) -> dict:
        """Keep uploaded-log diagnoses tied to exact evidence."""
        evidence_text = "\n".join(
            chunk.get("text", "") for chunk in primary_chunks
        )
        signal = self._extract_best_failure_signal(evidence_text)
        if not signal:
            diagnosis["confidence"] = "LOW"
            return diagnosis

        stage, error_class, root_cause = self._classify_failure_signal(signal)
        if stage != "UNKNOWN":
            diagnosis["failure_stage"] = stage
        if error_class != "Unknown":
            diagnosis["error_class"] = error_class
        if root_cause:
            diagnosis["root_cause"] = root_cause

        diagnosis["evidence"] = signal
        diagnosis["confidence"] = (
            "HIGH" if stage != "UNKNOWN" and error_class != "Unknown"
            else diagnosis.get("confidence", "LOW")
        )
        diagnosis["retrieved_chunks"] = primary_chunks
        diagnosis["query_id"] = query_id
        return diagnosis

    def _extract_best_failure_signal(self, text: str) -> str:
        """Pick the most actionable failure line from uploaded evidence."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        patterns = [
            "FAILED:",
            "Error while compiling statement",
            "Error running hive query",
            "Unable to kill query locally or on remote servers",
            "Invalid function",
            "Table not found",
            "Database does not exist",
            "cannot recognize input",
            "Usage: delete [FILE|JAR|ARCHIVE]",
            "Cannot convert an ACID table to non-ACID",
            "OutOfMemoryError",
            "NoClassDefFoundError",
            "ClassNotFoundException",
            "ROOT_INPUT_INIT_FAILURE",
            "Vertex failed",
            "DAG did not succeed",
            "LockException",
            "START_TRANSACTION is not supported",
            "NoSuchObjectException",
            "MetaException",
        ]

        for pattern in patterns:
            for line in lines:
                if pattern.lower() in line.lower():
                    return line

        for line in lines:
            if " ERROR " in line or " FATAL " in line:
                return line

        return ""

    def _classify_failure_signal(self, signal: str) -> tuple:
        """Return stage, error class, and a grounded root cause."""
        text = signal.lower()

        if "unable to kill query locally or on remote servers" in text:
            return (
                "EXECUTION",
                "HiveException",
                "The query failed while executing a KILL QUERY operation because "
                "Hive could not kill the target query locally or on remote servers."
            )

        if (
            "deleteresourceprocessor" in text
            or "usage: delete [file|jar|archive]" in text
        ):
            return (
                "COMPILATION",
                "CommandUsageError",
                "The query failed because the Hive delete resource command was "
                "used without the required FILE, JAR, or ARCHIVE argument."
            )

        if "cannot convert an acid table to non-acid" in text:
            return (
                "EXECUTION",
                "HiveException",
                "The DDL operation failed during execution because Hive does "
                "not allow converting an ACID table to non-ACID."
            )

        table_match = re.search(r"table not found '?([^';]+)'?", signal, re.I)
        if table_match:
            table_name = table_match.group(1).strip()
            return (
                "HMS_LOOKUP",
                "SemanticException",
                f"The query failed during metadata lookup because table "
                f"'{table_name}' was not found."
            )

        db_match = re.search(
            r"database (?:does not exist|not found):?\s*([^;]+)?",
            signal,
            re.I
        )
        if db_match:
            db_name = (db_match.group(1) or "").strip()
            suffix = f" '{db_name}'" if db_name else ""
            return (
                "HMS_LOOKUP",
                "SemanticException",
                f"The query failed during metadata lookup because database"
                f"{suffix} does not exist."
            )

        function_match = re.search(r"invalid function\s+([A-Za-z0-9_]+)", signal, re.I)
        if function_match:
            function_name = function_match.group(1)
            return (
                "COMPILATION",
                "SemanticException",
                f"The query failed during compilation because function "
                f"'{function_name}' is not valid or registered in Hive."
            )

        if "parseexception" in text or "cannot recognize input" in text:
            return (
                "COMPILATION",
                "ParseException",
                "The query failed during SQL parsing because Hive could not "
                "recognize the submitted statement syntax."
            )

        if "invalid column reference" in text or "invalid table alias" in text:
            return (
                "COMPILATION",
                "SemanticException",
                "The query failed during semantic analysis because the SQL "
                "references an invalid column or table alias."
            )

        if "outofmemoryerror" in text or "java heap space" in text:
            return (
                "EXECUTION",
                "OutOfMemoryError",
                "The query failed during execution because a task ran out of memory."
            )

        if "noclassdeffounderror" in text or "classnotfoundexception" in text:
            return (
                "EXECUTION",
                "NoClassDefFoundError" if "noclassdeffounderror" in text else "ClassNotFoundException",
                "The query failed during execution because a required Java class "
                "was missing from the runtime classpath."
            )

        if "lockexception" in text:
            return (
                "CONCURRENCY",
                "LockException",
                "The query failed because Hive could not acquire or complete "
                "the required transaction lock."
            )

        if (
            "start_transaction is not supported" in text
            or "start transaction is not supported" in text
        ):
            return (
                "CONCURRENCY",
                "IllegalStateException",
                "The query failed because the requested transaction operation "
                "is not supported in this Hive execution path."
            )

        if "nosuchobjectexception" in text or "metaexception" in text:
            return (
                "HMS_LOOKUP",
                "MetaException" if "metaexception" in text else "NoSuchObjectException",
                "The query failed while looking up metadata in the Hive metastore."
            )

        return "UNKNOWN", "Unknown", ""

    def _dominant_stage(self, chunks: list) -> str:
        """Infer a stable stage from uploaded chunks, preferring ERROR/FATAL."""
        stage_counts = defaultdict(int)
        for chunk in chunks:
            if chunk.get("log_level") not in ["ERROR", "FATAL"]:
                continue
            stage = self._category_to_stage(chunk.get("category", ""))
            if stage != "UNKNOWN":
                stage_counts[stage] += 1

        if not stage_counts:
            for chunk in chunks:
                stage = self._category_to_stage(chunk.get("category", ""))
                if stage != "UNKNOWN":
                    stage_counts[stage] += 1

        if not stage_counts:
            return "UNKNOWN"

        priority = {
            "HMS_LOOKUP": 0,
            "COMPILATION": 1,
            "EXECUTION": 2,
            "DAG_SUBMISSION": 3,
            "OPTIMIZATION": 4,
            "CONCURRENCY": 5,
        }
        return sorted(
            stage_counts.items(),
            key=lambda item: (-item[1], priority.get(item[0], 99))
        )[0][0]

    def _group_chunks_by_query_id(self, chunks: list) -> dict:
        """Group uploaded evidence by query ID, ignoring untagged chunks."""
        grouped = defaultdict(list)
        for chunk in chunks:
            query_id = chunk.get("query_id")
            if query_id and query_id != "null":
                grouped[query_id].append(chunk)
        return {
            query_id: query_chunks
            for query_id, query_chunks in grouped.items()
            if self._query_group_has_failure(query_chunks)
        }

    def _query_group_has_failure(self, chunks: list) -> bool:
        """Keep query groups that have query-failure markers, not benign ERROR logs."""
        text = "\n".join(chunk.get("text", "") for chunk in chunks).lower()
        failure_markers = [
            "failed:",
            "error while compiling statement",
            "error running hive query",
            "hivesqlexception",
            "ddltask failed",
            "unable to kill query locally or on remote servers",
            "outofmemoryerror",
            "lockexception",
            "sessionnotrunning",
        ]
        return any(marker in text for marker in failure_markers)

    def _query_group_rank_key(self, chunks: list) -> tuple:
        """Rank query groups by severity and latest evidence timestamp."""
        level_priority = {"FATAL": 0, "ERROR": 1, "WARN": 2}
        best_level = min(
            level_priority.get(chunk.get("log_level", "WARN"), 3)
            for chunk in chunks
        )
        timestamps = [
            chunk.get("timestamp", "")
            for chunk in chunks
            if chunk.get("timestamp")
        ]
        latest_timestamp = max(timestamps) if timestamps else ""
        return (best_level, latest_timestamp)

    def _latest_chunk_timestamp(self, chunks: list) -> str:
        """Return the latest timestamp represented by a group of chunks."""
        timestamps = [
            chunk.get("timestamp", "")
            for chunk in chunks
            if chunk.get("timestamp")
        ]
        return max(timestamps) if timestamps else ""

    def debug_last_failed(self) -> dict:
        """
        Debug the most recently failed query.
        Finds the last ERROR entry in the index.
        """
        print("Finding last failed query...")

        # Find chunks with ERROR level sorted by timestamp
        error_chunks = [
            chunk for chunk in self.retriever.metadata
            if chunk.get('log_level') in ['ERROR', 'FATAL']
            and chunk.get('query_id')
        ]

        if not error_chunks:
            print("No failed queries found in index")
            return {"error": "No failed queries found in knowledge base"}

        # Sort by timestamp to get most recent
        error_chunks.sort(
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )

        last_query_id = error_chunks[0].get('query_id')
        print(f"Last failed query ID: {last_query_id}")
        return self.debug_by_query_id(last_query_id)

    def _generate_diagnosis(
        self,
        retrieved_chunks: list,
        query_id: str = None,
        question: str = "What is the root cause of this failure?"
    ) -> dict:
        """Core RAG pipeline: retrieve → prompt → LLM → parse."""

        if not retrieved_chunks:
            return {
                "failure_stage": "UNKNOWN",
                "root_cause": "No relevant log evidence found in knowledge base",
                "error_class": "Unknown",
                "confidence": "LOW",
                "evidence": "",
                "retrieved_chunks": []
            }

        print(f"Retrieved {len(retrieved_chunks)} chunks from knowledge base")

        # Build prompt
        prompt = build_prompt(
            retrieved_chunks=retrieved_chunks,
            user_question=question,
            query_id=query_id
        )

        # Call LLM
        print("Calling LLM for diagnosis...")
        llm_response = self.llm.call(prompt)

        # Parse response
        diagnosis = parse_response(llm_response, retrieved_chunks)
        return diagnosis