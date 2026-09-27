import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
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
    kind: str        # "function" | "class" | "method" | "interface" | "type" | "enum"
    start_line: int
    file_path: str
    end_line: Optional[int] = None
    # Dotted path of enclosing classes/functions, e.g. "HTTPAdapter.send".
    # Equal to `name` for top-level definitions.
    qualified_name: Optional[str] = None

    def __post_init__(self):
        if self.qualified_name is None:
            self.qualified_name = self.name


@dataclass
class ImportInfo:
    source_file: str
    # The module exactly as written: "os", "pkg.mod", ".", "..pkg.mod", "./util", "react"
    imported_module: str
    is_relative: bool
    # Python `from M import a, b` → ["a", "b"] (needed to resolve `from . import submodule`).
    # Empty for `import M` and for TS/JS.
    names: list[str] = field(default_factory=list)
    line: int = 0


@dataclass
class ReferenceInfo:
    """A usage of an identifier (not its definition site)."""
    name: str
    line: int


@dataclass
class FileAnalysis:
    symbols: list[SymbolInfo]
    imports: list[ImportInfo]
    references: list[ReferenceInfo]


# ── Regex fallback patterns ───────────────────────────────────────────────────

_PY_DEF_RE = re.compile(r'^([ \t]*)(?:async[ \t]+)?def[ \t]+(\w+)[ \t]*[\(\[]', re.MULTILINE)
_PY_CLASS_RE = re.compile(r'^([ \t]*)class[ \t]+(\w+)', re.MULTILINE)
_PY_IMPORT_RE = re.compile(r'^[ \t]*import[ \t]+([^\n#]+)', re.MULTILINE)
_PY_FROM_IMPORT_RE = re.compile(
    r'^[ \t]*from[ \t]+(\.+[\w.]*|[\w.]+)[ \t]+import[ \t]+(\([^)]*\)|[^\n#]+)', re.MULTILINE
)
_TS_FUNC_RE = re.compile(
    r'(?:^|\n)\s*(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()',
    re.MULTILINE,
)
_TS_CLASS_RE = re.compile(r'(?:^|\n)\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)', re.MULTILINE)
_TS_IMPORT_RE = re.compile(
    r"""(?:\bimport\s+(?:[^'";]*?\s+from\s+)?|\bexport\s+[^'";]*?\s+from\s+|\brequire\s*\(\s*|\bimport\s*\(\s*)['"]([^'"]+)['"]"""
)
_IDENT_RE = re.compile(r'\b[A-Za-z_]\w*\b')


def _line_at(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _regex_extract_symbols(content: str, file_path: str, language: str) -> list[SymbolInfo]:
    symbols: list[SymbolInfo] = []
    if language == "python":
        for m in _PY_CLASS_RE.finditer(content):
            symbols.append(SymbolInfo(m.group(2), "class", _line_at(content, m.start(2)), file_path))
        for m in _PY_DEF_RE.finditer(content):
            # Without a parse tree we can't tell a method from a nested function;
            # indentation is the best cheap signal.
            kind = "method" if m.group(1) else "function"
            symbols.append(SymbolInfo(m.group(2), kind, _line_at(content, m.start(2)), file_path))
    elif language in ("typescript", "javascript"):
        for m in _TS_FUNC_RE.finditer(content):
            name = m.group(1) or m.group(2)
            if name:
                symbols.append(SymbolInfo(name, "function", _line_at(content, m.start()), file_path))
        for m in _TS_CLASS_RE.finditer(content):
            symbols.append(SymbolInfo(m.group(1), "class", _line_at(content, m.start()), file_path))
    symbols.sort(key=lambda s: s.start_line)
    return symbols


def _split_import_names(text: str) -> list[str]:
    """'(a, b as c,\n d)' → ['a', 'b', 'd']"""
    text = re.sub(r'#[^\n]*', '', text).strip().strip("()")
    names = []
    for part in text.split(","):
        part = part.strip()
        if part:
            names.append(part.split()[0])
    return names


def _regex_extract_imports(content: str, file_path: str, language: str) -> list[ImportInfo]:
    imports: list[ImportInfo] = []
    if language == "python":
        for m in _PY_FROM_IMPORT_RE.finditer(content):
            mod = m.group(1)
            imports.append(ImportInfo(
                source_file=file_path, imported_module=mod, is_relative=mod.startswith("."),
                names=_split_import_names(m.group(2)), line=_line_at(content, m.start()),
            ))
        for m in _PY_IMPORT_RE.finditer(content):
            for mod in _split_import_names(m.group(1)):
                imports.append(ImportInfo(
                    source_file=file_path, imported_module=mod, is_relative=False,
                    line=_line_at(content, m.start()),
                ))
        imports.sort(key=lambda i: i.line)
    elif language in ("typescript", "javascript"):
        for m in _TS_IMPORT_RE.finditer(content):
            mod = m.group(1)
            imports.append(ImportInfo(
                source_file=file_path, imported_module=mod, is_relative=mod.startswith("."),
                line=_line_at(content, m.start()),
            ))
    return imports


def _regex_extract_references(content: str, definitions: list[SymbolInfo]) -> list[ReferenceInfo]:
    """Fallback: every identifier-looking token that isn't a definition site.

    Unlike the tree-sitter path this also matches words inside strings and
    comments; the pipeline later prunes names that aren't defined anywhere in
    the repo, which removes most of that noise.
    """
    def_sites = {(s.name, s.start_line) for s in definitions}
    refs: set[tuple[str, int]] = set()
    for lineno, line in enumerate(content.splitlines(), 1):
        for m in _IDENT_RE.finditer(line):
            if (m.group(0), lineno) not in def_sites:
                refs.add((m.group(0), lineno))
    return [ReferenceInfo(n, l) for n, l in sorted(refs, key=lambda r: (r[1], r[0]))]


# ── tree-sitter extraction ────────────────────────────────────────────────────

def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _grammar_for(file_path: str, language: str) -> str:
    """Pick the tree-sitter grammar from the file extension.

    .tsx needs the `tsx` grammar (the plain `typescript` grammar mis-parses JSX),
    and plain JS is parsed with the `javascript` grammar.
    """
    if language == "python":
        return "python"
    ext = Path(file_path).suffix.lower()
    if ext == ".tsx":
        return "tsx"
    if ext in (".js", ".jsx", ".mjs", ".cjs"):
        return "javascript"
    return "typescript"


def _ts_analyze_python(root, src: bytes, file_path: str) -> FileAnalysis:
    symbols: list[SymbolInfo] = []
    imports: list[ImportInfo] = []
    references: list[ReferenceInfo] = []
    def_name_bytes: set[int] = set()  # start_byte of each definition's name node

    # Iterative walk (deep expression trees can exceed Python's recursion limit).
    # Each entry: (node, enclosing qualified-name parts, innermost scope is a class)
    stack = [(root, (), False)]
    while stack:
        node, scope, in_class = stack.pop()
        child_scope, child_in_class = scope, in_class

        if node.type in ("function_definition", "class_definition"):
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = _text(name_node, src)
                if node.type == "class_definition":
                    kind = "class"
                else:
                    kind = "method" if in_class else "function"
                symbols.append(SymbolInfo(
                    name=name, kind=kind, file_path=file_path,
                    start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
                    qualified_name=".".join(scope + (name,)),
                ))
                def_name_bytes.add(name_node.start_byte)
                child_scope = scope + (name,)
                child_in_class = node.type == "class_definition"

        elif node.type == "import_statement":
            for child in node.children_by_field_name("name"):
                target = child.child_by_field_name("name") if child.type == "aliased_import" else child
                if target is not None:
                    imports.append(ImportInfo(
                        source_file=file_path, imported_module=_text(target, src),
                        is_relative=False, line=node.start_point[0] + 1,
                    ))

        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            if module_node is not None:
                module = re.sub(r"\s+", "", _text(module_node, src))
                names = []
                for child in node.children_by_field_name("name"):
                    target = child.child_by_field_name("name") if child.type == "aliased_import" else child
                    if target is not None:
                        names.append(_text(target, src))
                if any(c.type == "wildcard_import" for c in node.children):
                    names.append("*")
                imports.append(ImportInfo(
                    source_file=file_path, imported_module=module,
                    is_relative=module.startswith("."), names=names,
                    line=node.start_point[0] + 1,
                ))

        elif node.type == "identifier":
            if node.start_byte not in def_name_bytes:
                references.append(ReferenceInfo(_text(node, src), node.start_point[0] + 1))
            continue  # leaf

        for child in reversed(node.children):
            stack.append((child, child_scope, child_in_class))

    # A def's name node is visited after the def itself (it's a child), so the
    # skip-set is always populated in time; sort for stable output.
    symbols.sort(key=lambda s: (s.start_line, s.qualified_name))
    return FileAnalysis(symbols, imports, references)


_TS_CLASS_TYPES = {"class_declaration", "abstract_class_declaration", "class"}
_TS_FUNC_TYPES = {"function_declaration", "generator_function_declaration", "function_expression", "function"}
_TS_DECL_KINDS = {
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}
_TS_IDENT_TYPES = {"identifier", "property_identifier", "type_identifier", "shorthand_property_identifier"}


def _string_value(node, src: bytes) -> Optional[str]:
    if node is None or node.type not in ("string", "template_string"):
        return None
    return _text(node, src)[1:-1]


def _ts_analyze_typescript(root, src: bytes, file_path: str) -> FileAnalysis:
    symbols: list[SymbolInfo] = []
    imports: list[ImportInfo] = []
    references: list[ReferenceInfo] = []
    def_name_bytes: set[int] = set()

    def add_symbol(node, name_node, kind: str, scope: tuple) -> tuple:
        name = _text(name_node, src)
        symbols.append(SymbolInfo(
            name=name, kind=kind, file_path=file_path,
            start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            qualified_name=".".join(scope + (name,)),
        ))
        def_name_bytes.add(name_node.start_byte)
        return scope + (name,)

    def add_import(module: Optional[str], line_node) -> None:
        if module:
            imports.append(ImportInfo(
                source_file=file_path, imported_module=module,
                is_relative=module.startswith("."), line=line_node.start_point[0] + 1,
            ))

    stack = [(root, ())]
    while stack:
        node, scope = stack.pop()
        child_scope = scope
        name_node = node.child_by_field_name("name")

        if node.type in _TS_CLASS_TYPES and name_node is not None:
            child_scope = add_symbol(node, name_node, "class", scope)
        elif node.type == "method_definition" and name_node is not None:
            # Class methods are `method_definition` nodes, not `function_declaration` —
            # without this branch, every method on a TS/JS class was silently dropped.
            child_scope = add_symbol(node, name_node, "method", scope)
        elif node.type in _TS_FUNC_TYPES and name_node is not None:
            child_scope = add_symbol(node, name_node, "function", scope)
        elif node.type in _TS_DECL_KINDS and name_node is not None:
            child_scope = add_symbol(node, name_node, _TS_DECL_KINDS[node.type], scope)
        elif node.type == "variable_declarator":
            # const foo = () => {} or const foo = function() {}
            value = node.child_by_field_name("value")
            if name_node is not None and name_node.type == "identifier" and value is not None \
                    and value.type in {"arrow_function", "function_expression", "function"}:
                decl = node.parent if node.parent is not None else node
                child_scope = add_symbol(decl, name_node, "function", scope)
        elif node.type in ("import_statement", "export_statement"):
            add_import(_string_value(node.child_by_field_name("source"), src), node)
        elif node.type == "call_expression":
            # require('x') and dynamic import('x')
            fn = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if fn is not None and args is not None and (
                fn.type == "import" or (fn.type == "identifier" and _text(fn, src) == "require")
            ):
                first = next((a for a in args.children if a.type in ("string", "template_string")), None)
                add_import(_string_value(first, src), node)
        elif node.type in _TS_IDENT_TYPES:
            if node.start_byte not in def_name_bytes:
                references.append(ReferenceInfo(_text(node, src), node.start_point[0] + 1))
            continue

        for child in reversed(node.children):
            stack.append((child, child_scope))

    symbols.sort(key=lambda s: (s.start_line, s.qualified_name))
    return FileAnalysis(symbols, imports, references)


# ── Public API ────────────────────────────────────────────────────────────────

def _regex_analyze(content: str, file_path: str, language: str) -> FileAnalysis:
    symbols = _regex_extract_symbols(content, file_path, language)
    imports = _regex_extract_imports(content, file_path, language)
    references = _regex_extract_references(content, symbols) if symbols or imports else []
    return FileAnalysis(symbols, imports, references)


def analyze_file(content: str, file_path: str, language: str) -> FileAnalysis:
    """Extract symbols, imports and identifier references from one parse of a file.

    Falls back to regex extraction if tree-sitter is unavailable or fails.
    Non-code files (markdown, config, ...) yield an empty analysis.
    """
    if language not in ("python", "typescript", "javascript"):
        return FileAnalysis([], [], [])
    if not _TS_AVAILABLE:
        return _regex_analyze(content, file_path, language)
    try:
        parser = _get_parser(_grammar_for(file_path, language))
        src = content.encode("utf-8")
        tree = parser.parse(src)
        if language == "python":
            return _ts_analyze_python(tree.root_node, src, file_path)
        return _ts_analyze_typescript(tree.root_node, src, file_path)
    except Exception:
        return _regex_analyze(content, file_path, language)


def extract_symbols(content: str, file_path: str, language: str) -> list[SymbolInfo]:
    return analyze_file(content, file_path, language).symbols


def extract_imports(content: str, file_path: str, language: str) -> list[ImportInfo]:
    return analyze_file(content, file_path, language).imports
