import pytest
from unittest.mock import patch

from codebase_intel.core.symbols import (
    extract_symbols, extract_imports,
    _regex_extract_symbols, _regex_extract_imports,
    SymbolInfo, ImportInfo,
)

PYTHON_FIXTURE = """\
import os
from pathlib import Path
from auth import validate_token

class UserService:
    def get_user(self, user_id):
        return None

def submit_order(order_id: int) -> bool:
    return True

def _helper():
    pass
"""

TYPESCRIPT_FIXTURE = """\
import { useState } from 'react';
import axios from 'axios';

class ApiClient {
    baseUrl: string;
}

function fetchData(url: string): Promise<any> {
    return axios.get(url);
}

const processOrder = async (id: number) => {
    return id;
};
"""

TYPESCRIPT_CLASS_METHODS_FIXTURE = """\
class ApiClient {
    baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    async fetchUser(id: number): Promise<any> {
        return this.get(`/users/${id}`);
    }

    private get(path: string) {
        return fetch(this.baseUrl + path);
    }
}
"""


# ── Python extraction ─────────────────────────────────────────────────────────

def test_python_extract_functions():
    syms = extract_symbols(PYTHON_FIXTURE, "service.py", "python")
    names = [s.name for s in syms]
    assert "submit_order" in names
    assert "_helper" in names


def test_python_extract_class():
    syms = extract_symbols(PYTHON_FIXTURE, "service.py", "python")
    classes = [s for s in syms if s.kind == "class"]
    assert any(c.name == "UserService" for c in classes)


def test_python_extract_method():
    syms = extract_symbols(PYTHON_FIXTURE, "service.py", "python")
    names = [s.name for s in syms]
    assert "get_user" in names


def test_python_extract_imports():
    imps = extract_imports(PYTHON_FIXTURE, "service.py", "python")
    modules = [i.imported_module for i in imps]
    assert "os" in modules
    assert "pathlib" in modules
    assert "auth" in modules


def test_python_symbol_line_numbers():
    syms = extract_symbols(PYTHON_FIXTURE, "service.py", "python")
    submit = next(s for s in syms if s.name == "submit_order")
    # "def submit_order" is on line 9 in the fixture
    assert submit.start_line == 9


# ── TypeScript extraction ─────────────────────────────────────────────────────

def test_typescript_extract_function():
    syms = extract_symbols(TYPESCRIPT_FIXTURE, "api.ts", "typescript")
    names = [s.name for s in syms]
    assert "fetchData" in names


def test_typescript_extract_class():
    syms = extract_symbols(TYPESCRIPT_FIXTURE, "api.ts", "typescript")
    classes = [s for s in syms if s.kind == "class"]
    assert any(c.name == "ApiClient" for c in classes)


def test_typescript_extract_arrow_function():
    syms = extract_symbols(TYPESCRIPT_FIXTURE, "api.ts", "typescript")
    names = [s.name for s in syms]
    assert "processOrder" in names


def test_typescript_extract_class_methods():
    """Class methods are `method_definition` nodes, distinct from top-level functions —
    previously these were silently dropped by the tree-sitter extractor."""
    syms = extract_symbols(TYPESCRIPT_CLASS_METHODS_FIXTURE, "api.ts", "typescript")
    methods = {s.name for s in syms if s.kind == "method"}
    assert "fetchUser" in methods
    assert "get" in methods
    assert "constructor" in methods


def test_typescript_extract_imports():
    imps = extract_imports(TYPESCRIPT_FIXTURE, "api.ts", "typescript")
    modules = [i.imported_module for i in imps]
    assert "react" in modules
    assert "axios" in modules


# ── Regex fallback ────────────────────────────────────────────────────────────

def test_regex_fallback_python_symbols():
    syms = _regex_extract_symbols(PYTHON_FIXTURE, "service.py", "python")
    names = [s.name for s in syms]
    assert "submit_order" in names
    assert "UserService" in names


def test_regex_fallback_python_imports():
    imps = _regex_extract_imports(PYTHON_FIXTURE, "service.py", "python")
    modules = [i.imported_module for i in imps]
    assert "os" in modules
    assert "auth" in modules


def test_regex_fallback_typescript_symbols():
    syms = _regex_extract_symbols(TYPESCRIPT_FIXTURE, "api.ts", "typescript")
    names = [s.name for s in syms]
    assert "fetchData" in names or "processOrder" in names  # regex may catch one or both


def test_extract_symbols_falls_back_on_tree_sitter_error():
    """If tree-sitter raises, we fall back to regex without crashing."""
    with patch("codebase_intel.core.symbols._get_parser", side_effect=RuntimeError("no parser")):
        syms = extract_symbols(PYTHON_FIXTURE, "service.py", "python")
        names = [s.name for s in syms]
        assert "submit_order" in names


def test_extract_imports_falls_back_on_tree_sitter_error():
    with patch("codebase_intel.core.symbols._get_parser", side_effect=RuntimeError("no parser")):
        imps = extract_imports(PYTHON_FIXTURE, "service.py", "python")
        modules = [i.imported_module for i in imps]
        assert "os" in modules


# ── Qualified names, imports with names, references (single-parse analysis) ──

from codebase_intel.core.symbols import analyze_file

NESTED_PY_FIXTURE = """\
from . import certs, utils as u
from ..pkg.mod import (
    Alpha,
    Beta as B,
)
import os.path as osp, sys

class HTTPAdapter:
    def send(self):
        return certs.where()

    def close(self):
        pass

class Session:
    def send(self):
        return HTTPAdapter().send()

def top():
    def inner():
        pass
    return inner
"""


def test_python_methods_are_kind_method_with_qualified_names():
    syms = {s.qualified_name: s for s in extract_symbols(NESTED_PY_FIXTURE, "a.py", "python")}
    assert syms["HTTPAdapter.send"].kind == "method"
    assert syms["Session.send"].kind == "method"
    assert syms["top"].kind == "function"
    assert syms["top.inner"].kind == "function"  # nested function, not a method
    assert syms["HTTPAdapter"].end_line == 13


def test_python_same_name_methods_in_one_file_are_all_kept():
    """Two classes each defining `send` in the same file must both survive —
    previously the (name, file) key collapsed them into one row."""
    sends = [s for s in extract_symbols(NESTED_PY_FIXTURE, "a.py", "python") if s.name == "send"]
    assert {s.qualified_name for s in sends} == {"HTTPAdapter.send", "Session.send"}


def test_python_from_import_records_imported_names():
    """`from . import certs` must keep `certs`: it's the submodule being imported,
    and dropping it made the import resolve to the package __init__ instead."""
    imps = extract_imports(NESTED_PY_FIXTURE, "a.py", "python")
    by_module = {i.imported_module: i for i in imps}
    assert by_module["."].names == ["certs", "utils"]
    assert by_module["."].is_relative
    assert by_module["..pkg.mod"].names == ["Alpha", "Beta"]
    assert "os.path" in by_module and "sys" in by_module


def test_python_references_exclude_definition_sites():
    refs = analyze_file(NESTED_PY_FIXTURE, "a.py", "python").references
    adapter_lines = sorted(r.line for r in refs if r.name == "HTTPAdapter")
    assert adapter_lines == [17]  # the usage in Session.send, not the `class` line
    assert any(r.name == "certs" and r.line == 10 for r in refs)


def test_regex_fallback_keeps_from_import_names():
    imps = _regex_extract_imports(NESTED_PY_FIXTURE, "a.py", "python")
    by_module = {i.imported_module: i for i in imps}
    assert by_module["."].names == ["certs", "utils"]
    assert by_module["..pkg.mod"].names == ["Alpha", "Beta"]
    assert "os.path" in by_module


TSX_FIXTURE = """\
import { Button } from "./ui/button.component";
export * from "./types";
const fs = require("fs");
const lazy = () => import("./Lazy");

interface Props { label: string }
type Size = "s" | "m";
enum Color { Red }

export class Panel extends Base {
    render(p: Props) {
        return <Button label={p.label} />;
    }
}
"""


def test_tsx_uses_tsx_grammar_and_finds_all_import_forms():
    imps = [i.imported_module for i in extract_imports(TSX_FIXTURE, "panel.tsx", "typescript")]
    assert imps == ["./ui/button.component", "./types", "fs", "./Lazy"]


def test_typescript_declarations_and_qualified_methods():
    syms = {s.qualified_name: s.kind for s in extract_symbols(TSX_FIXTURE, "panel.tsx", "typescript")}
    assert syms["Panel"] == "class"
    assert syms["Panel.render"] == "method"
    assert syms["Props"] == "interface"
    assert syms["Size"] == "type"
    assert syms["Color"] == "enum"
    assert syms["lazy"] == "function"



# ── Test-file naming ──────────────────────────────────────────────────────────

# aliased: pytest would collect names starting with "test" as tests
from codebase_intel.core.definitions import tested_module_stem as module_stem, tests_named_for as named_tests


def test_module_stem():
    assert module_stem("tests/test_utils.py") == "utils"
    assert module_stem("pkg/thing_test.py") == "thing"
    assert module_stem("src/foo.spec.ts") == "foo"
    assert module_stem("src/foo.test.tsx") == "foo"
    assert module_stem("tests/testserver/server.py") is None   # helper, not a test of "server"
    assert module_stem("tests/conftest.py") is None
    assert module_stem("src/app.py") is None


def test_named_tests():
    files = ["tests/test_adapters.py", "tests/test_requests.py", "src/foo.test.ts", "src/requests/models.py"]
    assert named_tests("src/requests/adapters.py", files) == ["tests/test_adapters.py"]
    assert named_tests("src/requests/__init__.py", files) == ["tests/test_requests.py"]  # package name
    assert named_tests("src/foo.ts", files) == ["src/foo.test.ts"]
    assert named_tests("tests/test_adapters.py", files) == []
