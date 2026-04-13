import re
import warnings
from dataclasses import dataclass
from typing import Optional

# Suppress the FutureWarning from tree-sitter-languages about Language(path, name)
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

try:
    from tree_sitter_languages import get_parser as _get_parser
    _TS_AVAILABLE = True
except Exception:
    _TS_AVAILABLE = False


@dataclass
class SymbolInfo:
    name: str
    kind: str        # "function" | "class" | "method"
    start_line: int
    file_path: str


@dataclass
class ImportInfo:
    source_file: str
    imported_module: str
    is_relative: bool


# ── Regex fallback patterns ───────────────────────────────────────────────────

_PY_FUNC_RE = re.compile(r'^def\s+(\w+)\s*\(', re.MULTILINE)
_PY_CLASS_RE = re.compile(r'^class\s+(\w+)', re.MULTILINE)
_PY_IMPORT_RE = re.compile(r'^(?:import|from)\s+([\w.]+)', re.MULTILINE)
_TS_FUNC_RE = re.compile(
    r'(?:^|\n)\s*(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()',
    re.MULTILINE,
)
_TS_CLASS_RE = re.compile(r'(?:^|\n)\s*(?:export\s+)?class\s+(\w+)', re.MULTILINE)
_TS_IMPORT_RE = re.compile(r"(?:import|from)\s+['\"]([^'\"]+)['\"]", re.MULTILINE)


def _regex_extract_symbols(content: str, file_path: str, language: str) -> list[SymbolInfo]:
    symbols: list[SymbolInfo] = []
    lines = content.splitlines()

    def line_of(name: str, pattern: re.Pattern, kind: str) -> None:
        for m in pattern.finditer(content):
            matched_name = next((g for g in m.groups() if g), None) if m.groups() else m.group(1)
            if matched_name:
                line_num = content[:m.start()].count("\n") + 1
                symbols.append(SymbolInfo(name=matched_name, kind=kind, start_line=line_num, file_path=file_path))

    if language == "python":
        line_of("", _PY_FUNC_RE, "function")
        line_of("", _PY_CLASS_RE, "class")
    elif language in ("typescript", "javascript"):
        for m in _TS_FUNC_RE.finditer(content):
            name = m.group(1) or m.group(2)
            if name:
                line_num = content[:m.start()].count("\n") + 1
                symbols.append(SymbolInfo(name=name, kind="function", start_line=line_num, file_path=file_path))
        for m in _TS_CLASS_RE.finditer(content):
            line_num = content[:m.start()].count("\n") + 1
            symbols.append(SymbolInfo(name=m.group(1), kind="class", start_line=line_num, file_path=file_path))
    return symbols


def _regex_extract_imports(content: str, file_path: str, language: str) -> list[ImportInfo]:
    imports: list[ImportInfo] = []
    if language == "python":
        for m in _PY_IMPORT_RE.finditer(content):
            mod = m.group(1)
            imports.append(ImportInfo(source_file=file_path, imported_module=mod, is_relative=mod.startswith(".")))
    elif language in ("typescript", "javascript"):
        for m in _TS_IMPORT_RE.finditer(content):
            mod = m.group(1)
            imports.append(ImportInfo(source_file=file_path, imported_module=mod, is_relative=mod.startswith(".")))
    return imports


# ── tree-sitter extraction ────────────────────────────────────────────────────

def _walk_tree(node, types: set[str]):
    """Yield all descendant nodes matching `types`."""
    if node.type in types:
        yield node
    for child in node.children:
        yield from _walk_tree(child, types)


def _ts_extract_python(tree, content: str, file_path: str) -> tuple[list[SymbolInfo], list[ImportInfo]]:
    symbols: list[SymbolInfo] = []
    imports: list[ImportInfo] = []
    lines = content.encode("utf-8")

    for node in _walk_tree(tree.root_node, {"function_definition", "class_definition"}):
        name_node = node.child_by_field_name("name")
        if name_node:
            name = content[name_node.start_byte:name_node.end_byte]
            kind = "function" if node.type == "function_definition" else "class"
            line = node.start_point[0] + 1  # 0-based → 1-based
            symbols.append(SymbolInfo(name=name, kind=kind, start_line=line, file_path=file_path))

    for node in _walk_tree(tree.root_node, {"import_statement", "import_from_statement"}):
        text = content[node.start_byte:node.end_byte]
        for m in _PY_IMPORT_RE.finditer(text):
            mod = m.group(1)
            imports.append(ImportInfo(source_file=file_path, imported_module=mod, is_relative=mod.startswith(".")))

    return symbols, imports


def _ts_extract_typescript(tree, content: str, file_path: str) -> tuple[list[SymbolInfo], list[ImportInfo]]:
    symbols: list[SymbolInfo] = []
    imports: list[ImportInfo] = []

    func_types = {"function_declaration", "function_expression", "arrow_function"}
    class_types = {"class_declaration"}
    import_types = {"import_statement"}
    lexical_types = {"lexical_declaration", "variable_declaration"}

    for node in _walk_tree(tree.root_node, func_types | class_types | import_types | lexical_types):
        if node.type in func_types | class_types:
            name_node = node.child_by_field_name("name")
            if name_node:
                name = content[name_node.start_byte:name_node.end_byte]
                kind = "class" if node.type in class_types else "function"
                line = node.start_point[0] + 1
                symbols.append(SymbolInfo(name=name, kind=kind, start_line=line, file_path=file_path))

        elif node.type in lexical_types:
            # const foo = () => {} or const foo = function() {}
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    val_node = child.child_by_field_name("value")
                    if name_node and val_node and val_node.type in {"arrow_function", "function_expression"}:
                        name = content[name_node.start_byte:name_node.end_byte]
                        line = node.start_point[0] + 1
                        symbols.append(SymbolInfo(name=name, kind="function", start_line=line, file_path=file_path))

        elif node.type == "import_statement":
            text = content[node.start_byte:node.end_byte]
            for m in _TS_IMPORT_RE.finditer(text):
                mod = m.group(1)
                imports.append(ImportInfo(source_file=file_path, imported_module=mod, is_relative=mod.startswith(".")))

    return symbols, imports


# ── Public API ────────────────────────────────────────────────────────────────

def extract_symbols(content: str, file_path: str, language: str) -> list[SymbolInfo]:
    if not _TS_AVAILABLE or language not in ("python", "typescript", "javascript"):
        return _regex_extract_symbols(content, file_path, language)
    try:
        lang_name = "python" if language == "python" else "typescript"
        parser = _get_parser(lang_name)
        tree = parser.parse(content.encode("utf-8"))
        if language == "python":
            syms, _ = _ts_extract_python(tree, content, file_path)
        else:
            syms, _ = _ts_extract_typescript(tree, content, file_path)
        return syms
    except Exception:
        return _regex_extract_symbols(content, file_path, language)


def extract_imports(content: str, file_path: str, language: str) -> list[ImportInfo]:
    if not _TS_AVAILABLE or language not in ("python", "typescript", "javascript"):
        return _regex_extract_imports(content, file_path, language)
    try:
        lang_name = "python" if language == "python" else "typescript"
        parser = _get_parser(lang_name)
        tree = parser.parse(content.encode("utf-8"))
        if language == "python":
            _, imps = _ts_extract_python(tree, content, file_path)
        else:
            _, imps = _ts_extract_typescript(tree, content, file_path)
        return imps
    except Exception:
        return _regex_extract_imports(content, file_path, language)
