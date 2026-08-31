"""
Parser: extracts structured diagnosis from LLM response.
Handles cases where LLM output is not perfectly formatted JSON.
"""

import json
import re


VALID_STAGES = {
    "COMPILATION", "HMS_LOOKUP", "EXECUTION",
    "DAG_SUBMISSION", "OPTIMIZATION", "CONCURRENCY", "UNKNOWN"
}

VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


def parse_response(
    llm_response: str,
    retrieved_chunks: list,
    query_text: str = ""
) -> dict:
    """
    Parse LLM response into structured diagnosis.
    Handles imperfect JSON output gracefully.
    """
    result = {
        "failure_stage": "UNKNOWN",
        "root_cause": "Could not determine from available evidence",
        "error_class": "Unknown",
        "confidence": "LOW",
        "evidence": "",
        "retrieved_chunks": retrieved_chunks,
        "raw_response": llm_response
    }

    # Try to extract JSON from response
    parsed = _extract_json(llm_response)
    if not parsed:
        parsed = _extract_loose_fields(llm_response)

    if parsed:
        # Validate and populate fields
        stage = parsed.get("failure_stage", "UNKNOWN").upper()
        result["failure_stage"] = stage if stage in VALID_STAGES else "UNKNOWN"

        result["root_cause"] = parsed.get(
            "root_cause",
            "Could not determine from available evidence"
        )

        result["error_class"] = parsed.get("error_class", "Unknown")

        confidence = parsed.get("confidence", "LOW").upper()
        result["confidence"] = confidence if confidence in VALID_CONFIDENCE else "LOW"

        result["evidence"] = parsed.get("evidence", "")

    else:
        # JSON parsing failed — extract what we can from raw text
        result["root_cause"] = _extract_plain_text(llm_response)
        result["confidence"] = "LOW"

    result["failure_stage"] = _normalize_failure_stage(
        result,
        retrieved_chunks,
        query_text
    )
    return result


def _extract_json(text: str) -> dict:
    """Try multiple strategies to extract JSON from LLM response."""

    # Strategy 1: direct JSON parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: find the first JSON-looking object in response
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: find ```json code block
    code_block = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 4: find ``` code block without json tag
    code_block2 = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if code_block2:
        try:
            return json.loads(code_block2.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _extract_loose_fields(text: str) -> dict:
    """Extract JSON-style key/value fields when braces are missing."""
    fields = {}
    for field in [
        "failure_stage",
        "root_cause",
        "error_class",
        "confidence",
        "evidence",
    ]:
        pattern = rf'"{field}"\s*:\s*"([^"]*)"'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            fields[field] = match.group(1).strip()
    return fields or None


def _extract_plain_text(text: str) -> str:
    """Extract meaningful content from non-JSON response."""
    lines = text.strip().split('\n')
    meaningful = [
        line.strip() for line in lines
        if line.strip() and not line.strip().startswith('{')
        and not line.strip().startswith('}')
    ]
    return ' '.join(meaningful[:3]) if meaningful else text[:200]


def _normalize_failure_stage(
    diagnosis: dict,
    retrieved_chunks: list,
    query_text: str = ""
) -> str:
    """Apply deterministic stage rules for common Hive failure classes."""
    combined_parts = [
        diagnosis.get("failure_stage", ""),
        diagnosis.get("root_cause", ""),
        diagnosis.get("error_class", ""),
        diagnosis.get("evidence", ""),
    ]
    primary_text = "\n".join(combined_parts).lower()
    query_text_lower = query_text.lower()
    combined_parts.append(query_text)
    combined_parts.extend(chunk.get("text", "") for chunk in retrieved_chunks[:3])
    text = "\n".join(combined_parts).lower()

    if any(
        term in query_text_lower or term in primary_text
        for term in ["lockexception", "transaction", "txn"]
    ):
        return "CONCURRENCY"

    dag_submission_terms = [
        "sessionnotrunning",
        "tez am failed to start",
        "submitdag failed",
        "application master not running",
        "dag submission failed",
    ]
    if any(term in query_text_lower for term in dag_submission_terms):
        return "DAG_SUBMISSION"

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
    if any(term in query_text_lower for term in execution_terms):
        return "EXECUTION"

    compilation_terms = [
        "invalid table alias",
        "invalid column reference",
        "column reference",
        "parseexception",
        "syntax error",
        "nonexistent_column",
        "cannot recognize input",
        "invalid function",
    ]
    if any(term in query_text_lower for term in compilation_terms):
        return "COMPILATION"

    hms_terms = [
        "nosuchobjectexception",
        "invalidtableexception",
        "table not found",
        "database does not exist",
        "database not found",
        "partition not found",
        "metadata lookup",
        "metaexception",
        "hms_lookup",
    ]
    if any(term in text for term in hms_terms):
        return "HMS_LOOKUP"

    optimization_terms = [
        "cbo",
        "cost based optimizer",
        "optimization failed",
        "reloptplanner",
        "invalid statistics",
    ]
    if any(term in query_text_lower for term in optimization_terms):
        return "OPTIMIZATION"

    if any(term in primary_text for term in compilation_terms):
        return "COMPILATION"

    if any(term in primary_text for term in optimization_terms):
        return "OPTIMIZATION"

    if any(term in primary_text for term in execution_terms):
        return "EXECUTION"

    if any(term in primary_text for term in dag_submission_terms):
        return "DAG_SUBMISSION"

    stage = diagnosis.get("failure_stage", "UNKNOWN").upper()
    return stage if stage in VALID_STAGES else "UNKNOWN"


def format_diagnosis(diagnosis: dict) -> str:
    """Format diagnosis for human-readable CLI output."""
    stage = diagnosis['failure_stage']
    confidence = diagnosis['confidence']

    # Color indicators
    stage_colors = {
        "COMPILATION": "🔴",
        "HMS_LOOKUP": "🟠",
        "EXECUTION": "🔴",
        "DAG_SUBMISSION": "🟠",
        "OPTIMIZATION": "🟡",
        "CONCURRENCY": "🟠",
        "UNKNOWN": "⚪"
    }
    confidence_colors = {
        "HIGH": "✅",
        "MEDIUM": "⚠️",
        "LOW": "❌"
    }

    icon = stage_colors.get(stage, "⚪")
    conf_icon = confidence_colors.get(confidence, "❌")

    output = f"""
{'=' * 60}
🔍 HIVE QUERY DIAGNOSIS
{'=' * 60}

{icon} FAILURE STAGE:  {stage}
{conf_icon} CONFIDENCE:     {confidence}
📛 ERROR CLASS:    {diagnosis['error_class']}

📋 ROOT CAUSE:
{diagnosis['root_cause']}

🔎 KEY EVIDENCE:
{diagnosis.get('evidence', 'See retrieved chunks below')}

📂 RETRIEVED EVIDENCE ({len(diagnosis['retrieved_chunks'])} chunks):"""

    for i, chunk in enumerate(diagnosis['retrieved_chunks'], 1):
        output += f"""
  [{i}] {chunk.get('source_file', 'unknown')} | {chunk.get('category', '')} | Score: {chunk.get('similarity_score', 0):.4f}
      Query ID: {chunk.get('query_id', 'N/A')}"""

    output += f"\n{'=' * 60}"
    return output