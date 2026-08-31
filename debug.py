import argparse
import sys
from datetime import datetime
from pathlib import Path
from phase2.debugger import HiveDebugger
from phase2.parser import format_diagnosis


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Hive Query Auto-Debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 debug.py --query-id hive_20260512065604_5ec86c10
  python3 debug.py --exception "SemanticException: Invalid column reference"
  python3 debug.py --log-file data/logs/failed_query.log
  python3 debug.py --log-file data/logs/failed_query.log --query-id hive_20260512065604_5ec86c10
  python3 debug.py --log-file data/logs/failed_query.log --last
  python3 debug.py --summarize-log data/logs/failed_query.log
  python3 debug.py --log-file data/logs/failed_query.log --output-file diagnosis.txt
  python3 debug.py --interactive
        """
    )

    parser.add_argument(
        '--query-id',
        type=str,
        help='Debug a specific Hive query by its query ID'
    )
    parser.add_argument(
        '--exception',
        type=str,
        help='Debug by pasting a raw exception or stack trace'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Debug a newly provided Hive log file without rebuilding the index'
    )
    parser.add_argument(
        '--summarize-log',
        type=str,
        help='Summarize a Hive log file in plain English'
    )
    parser.add_argument(
        '--last',
        action='store_true',
        help='With --log-file, debug the latest failed query in that uploaded log'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode — enter query IDs or exceptions one by one'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show step-by-step progress details'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Save the final terminal output to a text file'
    )
    parser.add_argument(
        '--no-prompts',
        action='store_true',
        help='Do not ask follow-up questions after printing results'
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([
        args.query_id,
        args.exception,
        args.log_file,
        args.summarize_log,
        args.last,
        args.interactive
    ]):
        parser.print_help()
        sys.exit(1)

    # Initialize debugger
    debugger = HiveDebugger(config_path=args.config, verbose=args.verbose)

    # Run appropriate mode
    output = None
    if args.log_file:
        diagnosis = debugger.debug_log_file(
            args.log_file,
            query_id=args.query_id,
            last_only=args.last
        )
        output = format_debug_result(diagnosis)
        output_label = "query_debug"

    elif args.query_id:
        diagnosis = debugger.debug_by_query_id(args.query_id)
        output = format_diagnosis(diagnosis)
        output_label = "query_debug"

    elif args.exception:
        diagnosis = debugger.debug_by_exception(args.exception)
        output = format_diagnosis(diagnosis)
        output_label = "exception_debug"

    elif args.summarize_log:
        summary = debugger.summarize_log_file(args.summarize_log)
        output = format_log_summary(summary)
        output_label = "log_summary"

    elif args.last:
        output = standalone_last_usage_message()
        output_label = "usage"

    elif args.interactive:
        run_interactive(debugger)
        return

    print(output)
    if args.output_file:
        save_output(args.output_file, output)
    elif not args.no_prompts:
        run_post_result_prompts(
            args=args,
            debugger=debugger,
            primary_output=output,
            primary_label=output_label
        )


def run_interactive(debugger: HiveDebugger):
    """Interactive debugging session."""
    print("\n" + "=" * 60)
    print("HIVE QUERY AUTO-DEBUGGER — INTERACTIVE MODE")
    print("=" * 60)
    print("Commands:")
    print("  q <query_id>    — debug by query ID")
    print("  e <exception>   — debug by exception text")
    print("  file <path>     — debug an uploaded/new log file")
    print("  summary <path>  — summarize a log file")
    print("  last            — debug last failed query")
    print("  exit            — quit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("debug> ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'exit':
                print("Goodbye.")
                break

            elif user_input.lower() == 'last':
                diagnosis = debugger.debug_last_failed()
                print(format_diagnosis(diagnosis))

            elif user_input.lower().startswith('q '):
                query_id = user_input[2:].strip()
                diagnosis = debugger.debug_by_query_id(query_id)
                print(format_diagnosis(diagnosis))

            elif user_input.lower().startswith('e '):
                exception_text = user_input[2:].strip()
                diagnosis = debugger.debug_by_exception(exception_text)
                print(format_diagnosis(diagnosis))

            elif user_input.lower().startswith('file '):
                log_file = user_input[5:].strip()
                diagnosis = debugger.debug_log_file(log_file)
                print(format_debug_result(diagnosis))

            elif user_input.lower().startswith('summary '):
                log_file = user_input[8:].strip()
                summary = debugger.summarize_log_file(log_file)
                print(format_log_summary(summary))

            else:
                # Treat bare input as exception text
                diagnosis = debugger.debug_by_exception(user_input)
                print(format_diagnosis(diagnosis))

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as ex:
            print(f"Error: {ex}")


def run_post_result_prompts(
    args,
    debugger: HiveDebugger,
    primary_output: str,
    primary_label: str
) -> None:
    """Offer optional summarization and report saving after a CLI run."""
    outputs = [(primary_label, primary_output)]

    if args.log_file and ask_yes_no("Run full log summarizer now?"):
        summary = debugger.summarize_log_file(args.log_file)
        summary_output = format_log_summary(summary)
        print(summary_output)
        outputs.append(("log_summary", summary_output))

    if ask_yes_no("Save/download the terminal output as text file(s)?"):
        for label, output in outputs:
            filename = default_output_filename(label, args)
            save_output(filename, output)


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask a yes/no question without crashing in non-interactive shells."""
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        response = input(prompt + suffix).strip().lower()
    except EOFError:
        return default

    if not response:
        return default
    return response in {"y", "yes"}


def default_output_filename(label: str, args) -> str:
    """Create a readable default filename for saved terminal output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    source = (
        args.log_file
        or args.summarize_log
        or args.query_id
        or "hive_debug"
    )
    stem = Path(source).stem.replace(" ", "_")
    return f"{stem}_{label}_{timestamp}.txt"


def format_debug_result(result: dict) -> str:
    """Format either a single diagnosis or multi-query result."""
    if result.get("multi_query"):
        return format_multi_query_diagnosis(result)
    return format_diagnosis(result)


def standalone_last_usage_message() -> str:
    """Explain that --last is scoped to uploaded logs."""
    return """
============================================================
USAGE
============================================================

The --last option now works with an uploaded log file.

Use:
  /opt/homebrew/bin/python3.11 debug.py --log-file hiveserver2.log --last

This finds the latest failed query inside that log file and debugs it.
============================================================"""


def format_multi_query_diagnosis(result: dict) -> str:
    """Format multiple uploaded-log query diagnoses."""
    output = f"""
============================================================
HIVE LOG MULTI-QUERY DIAGNOSIS
============================================================

LOG FILE: {result.get('log_file', 'unknown')}
FAILED QUERIES FOUND: {result.get('total_failed_queries', 0)}
QUERIES DIAGNOSED: {result.get('diagnosed_queries', 0)}
"""

    for i, diagnosis in enumerate(result.get("diagnoses", []), 1):
        output += f"""
------------------------------------------------------------
[{i}] QUERY ID: {diagnosis.get('query_id', 'N/A')}
FAILURE STAGE: {diagnosis.get('failure_stage', 'UNKNOWN')}
CONFIDENCE: {diagnosis.get('confidence', 'LOW')}
ERROR CLASS: {diagnosis.get('error_class', 'Unknown')}

ROOT CAUSE:
{diagnosis.get('root_cause', '')}

KEY EVIDENCE:
{diagnosis.get('evidence', '')}

RETRIEVED EVIDENCE ({len(diagnosis.get('retrieved_chunks', []))} chunks):"""
        for j, chunk in enumerate(diagnosis.get("retrieved_chunks", [])[:5], 1):
            output += f"""
  [{j}] {chunk.get('source_file', 'unknown')} | {chunk.get('category', '')} | {chunk.get('log_level', '')}
      Query ID: {chunk.get('query_id', 'N/A')}"""

    output += f"\n{'=' * 60}"
    return output


def format_log_summary(summary: dict) -> str:
    """Format a full-log summary for terminal/file output."""
    stages = ", ".join(summary.get("failure_stages", [])) or "UNKNOWN"
    main_events = summary.get("main_events", [])
    top_errors = summary.get("top_errors", [])
    query_ids = summary.get("query_ids", [])
    chunks = summary.get("evidence_chunks", [])
    stats = summary.get("stats", {})
    failure_types = stats.get("failure_types", [])

    output = f"""
============================================================
HIVE LOG SUMMARY
============================================================

OVERALL STATUS: {summary.get('overall_status', 'UNKNOWN')}
FAILURE STAGES: {stages}

SUMMARY:
{summary.get('summary', '')}

COUNTS:
- Prefiltered entries: {stats.get('total_prefiltered_entries', 'N/A')}
- Chunks analyzed: {stats.get('total_chunks', 'N/A')}
- Query IDs found: {stats.get('query_count', 'N/A')}
- Log levels: {stats.get('level_counts', {})}
- Categories: {stats.get('category_counts', {})}

MAIN EVENTS:"""

    if main_events:
        for event in main_events[:10]:
            output += f"\n- {event}"
    else:
        output += "\n- None reported"

    output += "\n\nFAILURE TYPES:"
    if failure_types:
        for failure in failure_types:
            query_count = failure.get("query_count", 0)
            query_label = "query" if query_count == 1 else "queries"
            output += (
                f"\n- {failure.get('failure_type', 'Unknown')}: "
                f"{failure.get('occurrences', 0)} occurrences, "
                f"{query_count} {query_label}, "
                f"stage {failure.get('failure_stage', 'UNKNOWN')}, "
                f"first seen {failure.get('first_seen', 'N/A')}, "
                f"{failure.get('unique_signatures', 0)} unique signatures"
            )
    else:
        output += "\n- None reported"

    output += "\n\nTOP ERROR EXAMPLES BY FAILURE TYPE:"
    if failure_types:
        for failure in failure_types:
            examples = failure.get("examples") or [failure.get("example", "")]
            for example in examples:
                if example:
                    output += (
                        f"\n- {failure.get('failure_type', 'Unknown')}: "
                        f"{format_summary_example(example)}"
                    )
    else:
        output += "\n- None reported"

    output += "\n\nTOP ERRORS:"
    if top_errors:
        for error in top_errors[:10]:
            output += f"\n- {error}"
    else:
        output += "\n- None reported"

    id_label = (
        "IMPORTANT QUERY IDS"
        if any(str(query_id).startswith("hive_") for query_id in query_ids)
        else "IMPORTANT TESTS / QUERY IDS"
    )
    output += f"\n\n{id_label}:"
    if query_ids:
        for query_id in query_ids[:15]:
            output += f"\n- {query_id}"
    else:
        output += "\n- None found"

    output += f"""

RECOMMENDED NEXT STEP:
{summary.get('recommended_next_step', '')}

EVIDENCE CHUNKS ({len(chunks)} shown):"""

    for i, chunk in enumerate(chunks[:5], 1):
        output += f"""
  [{i}] {chunk.get('source_file', 'unknown')} | {chunk.get('category', '')} | {chunk.get('log_level', '')}
      Query ID: {chunk.get('query_id', 'N/A')}"""

    output += f"\n{'=' * 60}"
    return output


def format_summary_example(example: str, max_length: int = 500) -> str:
    """Keep summary examples readable and avoid dumping huge log payloads."""
    import re

    cleaned = re.sub(r"trustStorePassword=[^;\\s]+", "trustStorePassword=<redacted>", example)
    cleaned = re.sub(r"password=[^,;\\s]+", "password=<redacted>", cleaned, flags=re.I)
    cleaned = re.sub(r"-u '[^']+'", "-u '<redacted>'", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_length:
        return cleaned[:max_length].rstrip() + " ... [truncated]"
    return cleaned


def save_output(path: str, output: str) -> None:
    """Save final CLI output to a text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(output)
        f.write("\n")
    print(f"\nSaved output to: {path}")


if __name__ == "__main__":
    main()