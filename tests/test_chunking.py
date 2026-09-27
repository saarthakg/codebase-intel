from app.core.chunking import chunk_file
from app.core.symbols import analyze_file


def _chunks(content: str, path: str = "m.py", language: str = "python", size: int = 400):
    analysis = analyze_file(content, path, language)
    return chunk_file(
        content, path, language, chunk_size_chars=size,
        symbols=analysis.symbols, imports=analysis.imports,
    )


def _method(name: str, body_lines: int) -> str:
    return f"    def {name}(self):\n" + "        x = 1\n" * body_lines + "\n"


def test_small_file_is_one_chunk_with_exact_line_range():
    src = "import os\n\ndef f():\n    return os.sep\n"
    [chunk] = _chunks(src)
    assert (chunk.start_line, chunk.end_line) == (1, 4)
    assert chunk.content == src


def test_chunks_align_to_definition_boundaries():
    """No chunk should start or end in the middle of a small function."""
    src = "".join(f"def f{i}():\n" + "    x = 1\n" * 8 + "\n" for i in range(10))
    chunks = _chunks(src, size=300)
    assert len(chunks) > 1
    for c in chunks:
        assert c.content.startswith("def f"), c.content[:20]
        assert c.content.rstrip().endswith("x = 1")


def test_oversized_class_is_split_into_methods_with_header_kept():
    src = "class Big:\n    \"\"\"Doc.\"\"\"\n    attr = 1\n\n" + "".join(_method(f"m{i}", 12) for i in range(4))
    chunks = _chunks(src, size=300)
    assert chunks[0].content.startswith("class Big:")
    method_chunks = [c for c in chunks if c.content.lstrip().startswith("def m")]
    assert {s for c in method_chunks for s in c.symbols} >= {"Big.m1", "Big.m2", "Big.m3"}


def test_decorators_and_comments_stay_with_their_definition():
    src = (
        "def a():\n" + "    x = 1\n" * 12 + "\n"
        "# explains b\n@decorator\ndef b():\n" + "    y = 2\n" * 12
    )
    chunks = _chunks(src, size=200)
    b_chunk = next(c for c in chunks if "b" in c.symbols)
    assert b_chunk.content.startswith("# explains b\n@decorator\ndef b():")


def test_oversized_function_falls_back_to_overlapping_windows_covering_it():
    src = "def huge():\n" + "".join(f"    v{i} = {i}\n" for i in range(200))
    chunks = _chunks(src, size=500)
    assert len(chunks) > 2
    covered = set()
    for c in chunks:
        assert len(c.content) <= 500
        covered |= set(range(c.start_line, c.end_line + 1))
    assert covered == set(range(1, 202))
    assert chunks[1].start_line <= chunks[0].end_line  # windows overlap


def test_end_line_is_last_content_line():
    """end_line used to be one past the last line whenever the chunk ended in a newline."""
    src = "\n".join(f"line {i}" for i in range(1, 400)) + "\n"
    chunks = chunk_file(src, "notes.txt", "unknown", chunk_size_chars=300)
    for c in chunks:
        assert c.content.count("\n") == c.end_line - c.start_line + 1
        assert c.content.splitlines()[0] == f"line {c.start_line}"
        assert c.content.splitlines()[-1] == f"line {c.end_line}"


def test_non_code_files_use_windows_and_skip_blank_chunks():
    src = "# Title\n\n" + "Some prose line.\n" * 100 + "\n" * 50
    chunks = chunk_file(src, "README.md", "markdown", chunk_size_chars=300)
    assert chunks
    assert all(c.content.strip() for c in chunks)


def test_typescript_class_split_into_methods():
    body = "        const x = 1;\n" * 12
    src = "export class Api {\n" + "".join(f"    m{i}() {{\n{body}    }}\n\n" for i in range(4)) + "}\n"
    chunks = _chunks(src, path="api.ts", language="typescript", size=350)
    assert len(chunks) > 1
    assert {s for c in chunks for s in c.symbols} >= {"Api", "Api.m0", "Api.m3"}


def test_without_parsed_symbols_chunks_are_regex_scanned():
    src = "def f():\n    pass\n\nclass K:\n    def g(self):\n        pass\n"
    [chunk] = chunk_file(src, "m.py", "python")
    assert chunk.symbols == ["f", "K", "g"]
