import re
import uuid
from typing import TYPE_CHECKING, Optional

from app.models.schemas import ChunkMetadata

if TYPE_CHECKING:
    from app.core.symbols import ImportInfo, SymbolInfo

# Approximate token ratios: 1 token ≈ 4 chars
DEFAULT_CHUNK_SIZE_CHARS = 1600   # ~400 tokens
DEFAULT_OVERLAP_CHARS = 200       # ~50 tokens

# Quick regex pre-scans for symbols and imports, used when the caller has no
# parsed symbols to pass in.
_PY_SYMBOL_RE = re.compile(r'^[ \t]*(?:async[ \t]+)?(?:def|class)\s+(\w+)', re.MULTILINE)
_PY_IMPORT_RE = re.compile(r'^(?:import|from)\s+([\w.]+)', re.MULTILINE)
_TS_SYMBOL_RE = re.compile(
    r'(?:^|\n)(?:export\s+)?(?:function|class|const|let|var)\s+(\w+)', re.MULTILINE
)
_TS_IMPORT_RE = re.compile(r"(?:import|from)\s+['\"]([^'\"]+)['\"]", re.MULTILINE)


def _prescan_symbols(content: str, language: str) -> tuple[list[str], list[str]]:
    """Quick regex scan for symbol names and import targets in a piece of source."""
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
    symbols: Optional[list["SymbolInfo"]] = None,
    imports: Optional[list["ImportInfo"]] = None,
) -> list[ChunkMetadata]:
    """Split file content into overlapping chunks with metadata.

    Each chunk's `symbols`/`imports` list only what that chunk contains:
    symbols whose definition line falls inside it and imports on its lines.
    Pass the parsed `symbols`/`imports` for the file when available (qualified
    names, methods included); otherwise each chunk is regex-scanned.
    """
    if not content.strip():
        return []

    def _chunk_metadata(text: str, first: int, last: int) -> tuple[list[str], list[str]]:
        if symbols is None and imports is None:
            return _prescan_symbols(text, language)
        syms = [s.qualified_name for s in (symbols or []) if first <= s.start_line <= last]
        imps = [i.imported_module for i in (imports or []) if first <= i.line <= last]
        return list(dict.fromkeys(syms)), list(dict.fromkeys(imps))
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
        # A trailing newline doesn't start a new line of content.
        last_content_line = end_line - 1 if chunk_text.endswith("\n") else end_line
        chunk_symbols, chunk_imports = _chunk_metadata(chunk_text, start_line, last_content_line)

        chunks.append(
            ChunkMetadata(
                chunk_id=str(uuid.uuid4()),
                file_path=file_path,
                language=language,
                start_line=start_line,
                end_line=end_line,
                symbols=chunk_symbols,
                imports=chunk_imports,
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
