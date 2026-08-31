"""
Prompt assembler: builds structured prompts for LLM reasoning.
Follows the format specified in the Technical Design Document.
"""

from typing import List


SYSTEM_PROMPT = """You are an expert Apache Hive query debugging assistant.

Your job is to analyze retrieved Hive log excerpts and identify the root cause of query failures.

RULES:
- Base your analysis ONLY on the provided log excerpts
- Do NOT speculate beyond what the logs show
- Always output valid JSON in the exact format specified
- Be specific — reference exact error messages and exception classes from the logs
- If the logs are insufficient to determine root cause, say so in the root_cause field

FAILURE STAGES:
- COMPILATION: Failed during SQL parsing or semantic analysis (SemanticException, ParseException)
- HMS_LOOKUP: Failed during metadata lookup (NoSuchObjectException, MetaException, partition errors)
- EXECUTION: Failed during task execution (OOM, TaskAttempt failed, DAG failed, UDF errors)
- DAG_SUBMISSION: Failed during Tez AM startup or DAG submission (SessionNotRunning, queue errors)
- OPTIMIZATION: Failed or degraded during CBO optimization (CBO failed, invalid statistics)
- CONCURRENCY: Failed due to lock contention or transaction conflicts (LockException, TxnAborted)
- UNKNOWN: Cannot determine from provided evidence

CLASSIFICATION GUIDANCE:
- If the evidence says NoSuchObjectException, table not found, database does not exist, MetaException, or metastore lookup failed, classify as HMS_LOOKUP.
- If Hive wraps a metadata lookup failure in SemanticException during compilation, prefer HMS_LOOKUP when the missing object is a database, table, or partition.
- If the evidence says invalid table alias, invalid column reference, ParseException, or SQL syntax error, classify as COMPILATION unless there is a clear missing table/database/partition lookup error.
- If the evidence says task failed, task attempt failed, vertex failed during execution, OutOfMemoryError, Java heap space, or Hive Runtime Error while processing rows, classify as EXECUTION.
- If the evidence says CBO failed, cost based optimizer, RelOptPlanner, invalid statistics, or Calcite optimization failure, classify as OPTIMIZATION.
- If the evidence says SessionNotRunning, Tez AM failed to start, submitDAG failed, application master not running, or DAG submission failed, classify as DAG_SUBMISSION.
- If the user input explicitly says the query completed successfully or there is no exception/error, classify as UNKNOWN with LOW confidence.
- Do not use UNKNOWN when the evidence clearly maps to one of the defined stages."""


def build_prompt(
    retrieved_chunks: List[dict],
    user_question: str,
    query_id: str = None
) -> str:
    """
    Build structured prompt with retrieved log chunks.
    Format matches the TDD specification.
    """
    context_sections = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        section = f"""[Chunk {i} | Source: {chunk.get('source_type', 'HS2')} | """
        section += f"Category: {chunk.get('category', 'UNKNOWN')} | "
        section += f"Level: {chunk.get('log_level', 'ERROR')} | "
        section += f"File: {chunk.get('source_file', 'unknown')}]"
        if chunk.get('timestamp'):
            section += f"\nTimestamp: {chunk['timestamp']}"
        if chunk.get('query_id'):
            section += f"\nQuery ID: {chunk['query_id']}"
        section += f"\n\n{chunk.get('text', '')}"
        context_sections.append(section)

    log_context = "\n\n" + ("=" * 60) + "\n\n".join(context_sections)

    query_context = ""
    if query_id:
        query_context = f"\nFailed Query ID: {query_id}\n"

    prompt = f"""{query_context}
LOG EXCERPTS FROM KNOWLEDGE BASE:
{log_context}

{"=" * 60}

QUESTION: {user_question}

Analyze the log excerpts above and respond with ONLY this JSON structure:
{{
    "failure_stage": "COMPILATION or HMS_LOOKUP or EXECUTION or DAG_SUBMISSION or OPTIMIZATION or CONCURRENCY or UNKNOWN",
    "root_cause": "Plain English explanation of why the query failed, referencing specific error messages",
    "error_class": "The main exception class (e.g. SemanticException, OutOfMemoryError)",
    "confidence": "HIGH or MEDIUM or LOW",
    "evidence": "The most important log line that led to your diagnosis"
}}"""

    return prompt


def build_exception_prompt(
    exception_text: str,
    retrieved_chunks: List[dict]
) -> str:
    """Build prompt for exception-only mode."""
    user_question = f"""An engineer encountered this exception:

{exception_text}

Based on the similar log excerpts retrieved from the knowledge base,
what is the root cause and failure stage?"""

    return build_prompt(retrieved_chunks, user_question)


def build_summary_prompt(log_file: str, summary_stats: dict, chunks: List[dict]) -> str:
    """Build a prompt for full-log summarization."""
    context_sections = []

    for i, chunk in enumerate(chunks, 1):
        section = f"""[Evidence {i} | Source: {chunk.get('source_type', 'HS2')} | """
        section += f"Category: {chunk.get('category', 'UNKNOWN')} | "
        section += f"Level: {chunk.get('log_level', 'ERROR')} | "
        section += f"File: {chunk.get('source_file', 'unknown')}]"
        if chunk.get('timestamp'):
            section += f"\nTimestamp: {chunk['timestamp']}"
        if chunk.get('query_id'):
            section += f"\nQuery ID: {chunk['query_id']}"
        section += f"\n\n{chunk.get('text', '')}"
        context_sections.append(section)

    log_context = "\n\n" + ("=" * 60) + "\n\n".join(context_sections)

    return f"""You are summarizing a Hive/service/test-run log for an engineer.

LOG FILE:
{log_file}

STRUCTURED COUNTS:
{summary_stats}

IMPORTANT LOG EVIDENCE:
{log_context}

Write a simple, high-level explanation of what happened in this log.
Do not suggest fixes. Focus on timeline, major failures/warnings, affected query IDs,
and what an engineer should inspect next.

Use the service implied by the log filename/path. Do not describe the log as Hive,
HS2, HMS, or query execution unless the filename/path/evidence clearly indicates a
Hive, HiveServer2, or Hive Metastore log.

If this is a pytest/Hive QE/test-framework log, prioritize the final test outcome,
failed test names, AssertionError/AttributeError, FAILED TESTS summary, and traceback
over earlier setup noise. Mention early setup/file errors as context only if they
appear relevant to the final failure.

If STRUCTURED COUNTS contains failure_types, use those grouped failure types to
identify the dominant root cause. In particular, mention classpath/dependency
exceptions such as NoClassDefFoundError or ClassNotFoundException in the summary
and main_events when they appear, rather than only reporting the wrapper
AssertionError or "command failed after retries".

Respond with ONLY this JSON structure:
{{
    "overall_status": "FAILED or WARNING_ONLY or NO_FAILURE_EVIDENCE",
    "summary": "Plain-English summary of what happened in the log",
    "main_events": ["Important events in order"],
    "failure_stages": ["Stages observed, such as COMPILATION, HMS_LOOKUP, EXECUTION, DAG_SUBMISSION, OPTIMIZATION, CONCURRENCY, UNKNOWN"],
    "query_ids": ["Important query IDs if present"],
    "top_errors": ["Most important error messages"],
    "recommended_next_step": "Evidence-grounded next debugging step, not an automated fix"
}}"""