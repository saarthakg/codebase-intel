import pytest
from unittest.mock import patch

from app.core.symbols import (
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
    with patch("app.core.symbols._get_parser", side_effect=RuntimeError("no parser")):
        syms = extract_symbols(PYTHON_FIXTURE, "service.py", "python")
        names = [s.name for s in syms]
        assert "submit_order" in names


def test_extract_imports_falls_back_on_tree_sitter_error():
    with patch("app.core.symbols._get_parser", side_effect=RuntimeError("no parser")):
        imps = extract_imports(PYTHON_FIXTURE, "service.py", "python")
        modules = [i.imported_module for i in imps]
        assert "os" in modules
