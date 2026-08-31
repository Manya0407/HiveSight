"""
Pre-filter: extracts WARN/ERROR/FATAL lines with context windows.
Discards ~90% of raw log content that is INFO-level noise.
"""

import re
import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm


@dataclass
class LogEntry:
    """A single pre-filtered log block with context."""
    raw_text: str           # full text of the block
    level: str              # WARN / ERROR / FATAL
    timestamp: str          # extracted from anchor line
    query_id: Optional[str] # extracted queryId if present
    source_file: str        # which log file this came from
    source_type: str        # HS2 or HMS
    category: str           # COMPILATION, EXECUTION, etc.
    failure_stage: str      # COMPILATION, HMS_LOOKUP, EXECUTION, etc.
    line_number: int        # line number of anchor line in source file


# Regex patterns
LEVEL_PATTERN = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})'
    r'(?:\s+|\|)(?P<level>WARN|ERROR|FATAL)(?:\s+|\|)'
)
TIMESTAMP_PATTERN = re.compile(
    r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})'
)
QUERY_ID_PATTERN = re.compile(
    r'queryId[=\s]+([a-zA-Z0-9_]+)'
)

FRAMEWORK_FAILURE_PATTERNS = [
    re.compile(r'=+\s*FAILURES\s*=+'),
    re.compile(r'START OF TEST ERROR'),
    re.compile(r'END OF TEST ERROR'),
    re.compile(r'\bAssertionError\b'),
    re.compile(r'\bAttributeError\b'),
    re.compile(r'Traceback \(most recent call last\)'),
    re.compile(r'TEST ".*" FAILED'),
    re.compile(r'\]\s+TEST FAILED in'),
    re.compile(r'FAILED TESTS:'),
    re.compile(r'PASSED:\s*\d+,\s*FAILED:\s*\d+'),
    re.compile(r'Exit Status:\s*ExitCode\.TESTS_FAILED'),
    re.compile(r"assert\s+'fail'\s*==\s*'pass'"),
    re.compile(r'NoClassDefFoundError'),
    re.compile(r'ClassNotFoundException'),
    re.compile(r'ROOT_INPUT_INIT_FAILURE'),
    re.compile(r'Vertex failed'),
    re.compile(r'DAG did not succeed'),
    re.compile(r'Command execution failed even after retries'),
]


def extract_query_id(text: str) -> Optional[str]:
    """Extract queryId from a log block."""
    match = QUERY_ID_PATTERN.search(text)
    return match.group(1) if match else None


def extract_timestamp(line: str) -> str:
    """Extract timestamp from a log line."""
    match = TIMESTAMP_PATTERN.match(line)
    return match.group(1) if match else ""


def is_noise(line: str, noise_patterns: List[str]) -> bool:
    """Return True if line matches a known noise pattern."""
    for pattern in noise_patterns:
        if pattern in line:
            return True
        try:
            if re.search(pattern, line):
                return True
        except re.error:
            # Treat invalid regex patterns as literal-only filters.
            continue
    return False


def is_framework_failure_anchor(line: str) -> bool:
    """Return True for pytest/test-framework failure lines."""
    return any(pattern.search(line) for pattern in FRAMEWORK_FAILURE_PATTERNS)


def prefilter_file(
    file_path: str,
    source_type: str,
    category: str,
    failure_stage: str,
    context_before: int,
    context_after: int,
    noise_patterns: List[str]
) -> List[LogEntry]:
    """
    Extract WARN/ERROR/FATAL lines with surrounding context
    from a single log file.
    """
    filename = os.path.basename(file_path)
    entries = []

    print(f"  Pre-filtering: {filename}")

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    total_lines = len(lines)
    anchor_indices = []

    # Find all WARN/ERROR/FATAL anchor lines
    for i, line in enumerate(lines):
        match = LEVEL_PATTERN.match(line)
        if match:
            level = match.group("level")
            # Skip noise even if WARN/ERROR
            if not is_noise(line, noise_patterns):
                anchor_indices.append((i, level, False))
        elif is_framework_failure_anchor(line):
            anchor_indices.append((i, "ERROR", True))

    print(f"    Total lines: {total_lines:,}")
    print(f"    Anchor lines found: {len(anchor_indices):,}")
    if total_lines:
        print(f"    Noise reduction: "
              f"{100 * (1 - len(anchor_indices)/total_lines):.1f}%")
    else:
        print("    Noise reduction: N/A (empty file)")

    # For each anchor, extract context window
    # Merge overlapping windows to avoid duplicates
    covered = set()

    for anchor_idx, level, is_framework_anchor in anchor_indices:
        before = max(context_before, 8) if is_framework_anchor else context_before
        after = max(context_after, 30) if is_framework_anchor else context_after
        start = max(0, anchor_idx - before)
        end = min(total_lines - 1, anchor_idx + after)

        # Collect lines in window
        window_lines = []
        for i in range(start, end + 1):
            window_lines.append(lines[i])
            covered.add(i)

        block_text = "".join(window_lines)
        timestamp = extract_timestamp(lines[anchor_idx])
        query_id = extract_query_id(block_text)

        entry = LogEntry(
            raw_text=block_text,
            level=level,
            timestamp=timestamp,
            query_id=query_id,
            source_file=filename,
            source_type=source_type,
            category=category,
            failure_stage=failure_stage,
            line_number=anchor_idx + 1
        )
        entries.append(entry)

    print(f"    Extracted {len(entries):,} log entries")
    return entries


def run_prefilter(config: dict) -> List[LogEntry]:
    """Run pre-filter on all configured log files."""
    all_entries = []

    pf_config = config['prefilter']
    log_config = config['logs']

    for file_config in log_config['files']:
        file_path = os.path.join(
            log_config['input_dir'],
            file_config['path']
        )

        if not os.path.exists(file_path):
            print(f"  WARNING: File not found: {file_path}")
            continue

        entries = prefilter_file(
            file_path=file_path,
            source_type=file_config['source'],
            category=file_config['category'],
            failure_stage=file_config['failure_stage'],
            context_before=pf_config['context_before'],
            context_after=pf_config['context_after'],
            noise_patterns=pf_config['noise_patterns']
        )
        all_entries.extend(entries)

    print(f"\nTotal entries after pre-filter: {len(all_entries):,}")
    return all_entries