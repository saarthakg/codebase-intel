import re
import uuid
from dataclasses import dataclass, field
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

# Lines directly above a definition that belong to it.
_PY_ATTACHED_RE = re.compile(r'^\s*(?:@|#)')
_TS_ATTACHED_RE = re.compile(r'^\s*(?:@|//|/\*|\*)')


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


# ── Structure-aware splitting ─────────────────────────────────────────────────

@dataclass
class _Span:
    start: int                      # 1-based, inclusive
    end: int                        # 1-based, inclusive
    children: list["_Span"] = field(default_factory=list)


def _symbol_tree(symbols: list["SymbolInfo"], n_lines: int) -> list[_Span]:
    """Nest definition spans by containment; returns the top-level spans."""
    spans = sorted(
        (_Span(s.start_line, min(s.end_line, n_lines)) for s in symbols if s.end_line),
        key=lambda sp: (sp.start, -sp.end),
    )
    roots: list[_Span] = []
    stack: list[_Span] = []
    for span in spans:
        while stack and span.start > stack[-1].end:
            stack.pop()
        if stack and span.end <= stack[-1].end:
            if span.start != stack[-1].start or span.end != stack[-1].end:
                stack[-1].children.append(span)
                stack.append(span)
            # identical span (e.g. an @overload stub recorded twice): skip
        elif not stack:
            roots.append(span)
            stack.append(span)
        # else: partially overlapping span — can't nest it, ignore
    return roots


class _StructuredSplitter:
    """Cut a file into line ranges at definition boundaries, then pack them.

    A definition that fits the budget stays whole. One that doesn't is split
    into its own children (a class into its methods) with the code between
    them (class header, attributes) kept as separate pieces. A leaf that is
    still too big falls back to overlapping line windows. Finally, adjacent
    pieces are packed greedily up to the budget so small methods and
    top-level statements don't become fragment-sized chunks.
    """

    def __init__(self, lines: list[str], language: str, max_chars: int, overlap_chars: int):
        self.lines = lines
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.attached_re = _PY_ATTACHED_RE if language == "python" else _TS_ATTACHED_RE
        # prefix[i] = chars in lines[0:i], so range size is O(1)
        self.prefix = [0]
        for line in lines:
            self.prefix.append(self.prefix[-1] + len(line))

    def size(self, start: int, end: int) -> int:
        return self.prefix[end] - self.prefix[start - 1]

    def is_blank(self, start: int, end: int) -> bool:
        return all(not self.lines[i - 1].strip() for i in range(start, end + 1))

    def attached_start(self, start: int, floor: int) -> int:
        """Extend a definition upward over its decorators/comments."""
        while start - 1 >= floor and self.attached_re.match(self.lines[start - 2]):
            start -= 1
        return start

    def windows(self, start: int, end: int) -> list[tuple[int, int]]:
        """Overlapping line windows for a range with no usable structure."""
        out = []
        s = start
        while s <= end:
            e = s
            while e < end and self.size(s, e + 1) <= self.max_chars:
                e += 1
            out.append((s, e))
            if e >= end:
                break
            # step back roughly overlap_chars worth of lines, but always advance
            back = e
            while back > s + 1 and self.size(back, e) < self.overlap_chars:
                back -= 1
            s = max(back, s + 1)
        return out

    def split(self, start: int, end: int, children: list[_Span]) -> list[tuple[int, int]]:
        if self.size(start, end) <= self.max_chars:
            return [(start, end)]
        if not children:
            return self.windows(start, end)
        pieces: list[tuple[int, int]] = []
        cursor = start
        for child in children:
            child_start = self.attached_start(child.start, cursor)
            if cursor < child_start:
                pieces += self.split(cursor, child_start - 1, [])
            pieces += self.split(child_start, child.end, child.children)
            cursor = child.end + 1
        if cursor <= end:
            pieces += self.split(cursor, end, [])
        return pieces

    def pack(self, pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
        pieces = [p for p in pieces if not self.is_blank(*p)]
        packed: list[tuple[int, int]] = []
        for start, end in pieces:
            if packed:
                prev_start, prev_end = packed[-1]
                # only merge contiguous (or blank-separated) neighbours; overlapping
                # windows from an oversized leaf stay separate
                if prev_end < start and (prev_end + 1 == start or self.is_blank(prev_end + 1, start - 1)) \
                        and self.size(prev_start, end) <= self.max_chars:
                    packed[-1] = (prev_start, end)
                    continue
            packed.append((start, end))
        return packed


def _structured_ranges(
    content: str, language: str, symbols: list["SymbolInfo"], max_chars: int, overlap_chars: int
) -> list[tuple[int, int]]:
    lines = content.splitlines(keepends=True)
    splitter = _StructuredSplitter(lines, language, max_chars, overlap_chars)
    roots = _symbol_tree(symbols, len(lines))
    return splitter.pack(splitter.split(1, len(lines), roots))


def _window_ranges(content: str, max_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    lines = content.splitlines(keepends=True)
    splitter = _StructuredSplitter(lines, "unknown", max_chars, overlap_chars)
    ranges = []
    for start, end in splitter.windows(1, len(lines)):
        if not splitter.is_blank(start, end):
            ranges.append((start, end))
    return ranges


def chunk_file(
    content: str,
    file_path: str,
    language: str,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    symbols: Optional[list["SymbolInfo"]] = None,
    imports: Optional[list["ImportInfo"]] = None,
) -> list[ChunkMetadata]:
    """Split file content into chunks with metadata.

    When parsed `symbols` with line spans are passed (Python/TS/JS), chunks
    follow definition boundaries — see _StructuredSplitter. Otherwise (markdown,
    config, regex-fallback parses) the file is cut into overlapping line
    windows of ~`chunk_size_chars`.

    Each chunk's `symbols`/`imports` list only what that chunk contains:
    symbols whose definition line falls inside it and imports on its lines.
    Without parsed symbols/imports each chunk is regex-scanned instead.
    `start_line`/`end_line` are 1-based and inclusive of the last content line.
    """
    if not content.strip():
        return []

    structured = bool(symbols) and all(s.end_line for s in symbols) \
        and language in ("python", "typescript", "javascript")
    if structured:
        ranges = _structured_ranges(content, language, symbols, chunk_size_chars, overlap_chars)
    else:
        ranges = _window_ranges(content, chunk_size_chars, overlap_chars)

    lines = content.splitlines(keepends=True)
    chunks: list[ChunkMetadata] = []
    for start_line, end_line in ranges:
        text = "".join(lines[start_line - 1:end_line])
        if symbols is None and imports is None:
            chunk_symbols, chunk_imports = _prescan_symbols(text, language)
        else:
            chunk_symbols = list(dict.fromkeys(
                s.qualified_name for s in (symbols or []) if start_line <= s.start_line <= end_line
            ))
            chunk_imports = list(dict.fromkeys(
                i.imported_module for i in (imports or []) if start_line <= i.line <= end_line
            ))
        chunks.append(
            ChunkMetadata(
                chunk_id=str(uuid.uuid4()),
                file_path=file_path,
                language=language,
                start_line=start_line,
                end_line=end_line,
                symbols=chunk_symbols,
                imports=chunk_imports,
                content=text,
            )
        )
    return chunks
