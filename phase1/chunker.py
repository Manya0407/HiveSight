"""
Chunker: splits pre-filtered log entries into 300-500 token chunks.
Each chunk is tagged with query_id and metadata for retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from phase1.prefilter import LogEntry


@dataclass
class LogChunk:
    """A single embeddable chunk with full metadata."""
    chunk_id: str           # unique identifier
    text: str               # chunk content to embed
    query_id: Optional[str] # queryId if extractable
    timestamp: str          # from anchor line
    log_level: str          # WARN / ERROR / FATAL
    source_file: str        # originating log file
    source_type: str        # HS2 or HMS
    category: str           # failure category
    failure_stage: str      # lifecycle stage
    chunk_index: int        # position within entry
    total_chunks: int       # total chunks from this entry


def estimate_tokens(text: str) -> int:
    """
    Approximate token count.
    Rule of thumb: words * 1.3 for log content
    (log lines have many special chars that tokenize heavily)
    """
    words = len(text.split())
    return int(words * 1.3)


def split_into_chunks(
    text: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int
) -> List[str]:
    """
    Split text into overlapping chunks of target token size.
    Splits on newlines to preserve log line integrity.
    """
    lines = text.split('\n')
    chunks = []
    current_chunk_lines = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)

        # If adding this line exceeds max, save current chunk
        if current_tokens + line_tokens > max_tokens and current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            if estimate_tokens(chunk_text) >= min_tokens:
                chunks.append(chunk_text)

            # Overlap: keep last N tokens worth of lines
            overlap_lines = []
            overlap_count = 0
            for prev_line in reversed(current_chunk_lines):
                prev_tokens = estimate_tokens(prev_line)
                if overlap_count + prev_tokens <= overlap_tokens:
                    overlap_lines.insert(0, prev_line)
                    overlap_count += prev_tokens
                else:
                    break

            current_chunk_lines = overlap_lines + [line]
            current_tokens = overlap_count + line_tokens
        else:
            current_chunk_lines.append(line)
            current_tokens += line_tokens

    # Don't lose the last chunk
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines)
        # Include even if below min_tokens — it's the tail of an entry
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks if chunks else [text]


def chunk_entries(
    entries: List[LogEntry],
    config: dict
) -> List[LogChunk]:
    """Convert pre-filtered entries into embeddable chunks."""
    chunk_config = config['chunking']
    min_tokens = chunk_config['min_tokens']
    max_tokens = chunk_config['max_tokens']
    overlap_tokens = chunk_config['overlap_tokens']

    all_chunks = []
    chunk_counter = 0

    for entry in entries:
        text_chunks = split_into_chunks(
            entry.raw_text,
            min_tokens,
            max_tokens,
            overlap_tokens
        )

        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"chunk_{chunk_counter:06d}"
            chunk_counter += 1

            chunk = LogChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                query_id=entry.query_id,
                timestamp=entry.timestamp,
                log_level=entry.level,
                source_file=entry.source_file,
                source_type=entry.source_type,
                category=entry.category,
                failure_stage=entry.failure_stage,
                chunk_index=i,
                total_chunks=len(text_chunks)
            )
            all_chunks.append(chunk)

    print(f"Total chunks after chunking: {len(all_chunks):,}")
    return all_chunks