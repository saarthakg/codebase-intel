import re
import uuid
from typing import Optional

from app.models.schemas import ChunkMetadata

# Approximate token ratios: 1 token ≈ 4 chars
DEFAULT_CHUNK_SIZE_CHARS = 1600   # ~400 tokens
DEFAULT_OVERLAP_CHARS = 200       # ~50 tokens

# Quick regex pre-scans for symbols and imports per language
_PY_SYMBOL_RE = re.compile(r'^(?:def|class)\s+(\w+)', re.MULTILINE)
_PY_IMPORT_RE = re.compile(r'^(?:import|from)\s+([\w.]+)', re.MULTILINE)
_TS_SYMBOL_RE = re.compile(
    r'(?:^|\n)(?:export\s+)?(?:function|class|const|let|var)\s+(\w+)', re.MULTILINE
)
_TS_IMPORT_RE = re.compile(r"(?:import|from)\s+['\"]([^'\"]+)['\"]", re.MULTILINE)


def _prescan_symbols(content: str, language: str) -> tuple[list[str], list[str]]:
    """Quick regex scan for symbol names and import targets across the whole file."""
    if language == "python":
        symbols = _PY_SYMBOL_RE.findall(content)
        imports = _PY_IMPORT_RE.findall(content)
    elif language in ("typescript", "javascript"):
        symbols = _TS_SYMBOL_RE.findall(content)
        imports = _TS_IMPORT_RE.findall(content)
    else:
        symbols = []
        imports = []
    return list(dict.fromkeys(symbols)), list(dict.fromkeys(imports))  # dedup, preserve order


def _count_lines_before(content: str, offset: int) -> int:
    """Return number of newlines before `offset` (0-based), giving 1-based start line."""
    return content[:offset].count("\n")


def chunk_file(
    content: str,
    file_path: str,
    language: str,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[ChunkMetadata]:
    """Split file content into overlapping chunks with metadata."""
    if not content.strip():
        return []

    file_symbols, file_imports = _prescan_symbols(content, language)
    total = len(content)
    chunks: list[ChunkMetadata] = []
    start = 0

    while start < total:
        end = min(start + chunk_size_chars, total)

        # Avoid splitting mid-line: find nearest newline before `end`
        if end < total:
            newline_pos = content.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos + 1  # include the newline

        chunk_text = content[start:end]
        if not chunk_text.strip():
            start = end
            continue

        start_line = _count_lines_before(content, start) + 1  # 1-based
        end_line = start_line + chunk_text.count("\n")

        chunks.append(
            ChunkMetadata(
                chunk_id=str(uuid.uuid4()),
                file_path=file_path,
                language=language,
                start_line=start_line,
                end_line=end_line,
                symbols=file_symbols,
                imports=file_imports,
                content=chunk_text,
            )
        )

        # Advance with overlap: next chunk starts `overlap_chars` before current end
        next_start = end - overlap_chars
        # Snap to line boundary to avoid starting mid-line
        if next_start > start and next_start < total:
            newline_pos = content.find("\n", next_start)
            if newline_pos != -1 and newline_pos < end:
                next_start = newline_pos + 1
        start = max(next_start, end) if next_start <= start else next_start

        # Safety: always advance
        if start >= end:
            start = end

    return chunks
