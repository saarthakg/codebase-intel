import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from app.models.schemas import ChunkMetadata

if TYPE_CHECKING:
    from app.core.symbols import ReferenceInfo, SymbolInfo


class MetadataStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._migrate_legacy_symbols()
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                repo_id     TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                language    TEXT NOT NULL,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER NOT NULL,
                symbols     TEXT NOT NULL,
                imports     TEXT NOT NULL,
                content     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS symbols (
                symbol_name    TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                repo_id        TEXT NOT NULL,
                file_path      TEXT NOT NULL,
                start_line     INTEGER NOT NULL,
                end_line       INTEGER,
                kind           TEXT,
                PRIMARY KEY (repo_id, file_path, qualified_name, start_line)
            );

            -- Identifier usages (not definition sites). One row per name per line.
            CREATE TABLE IF NOT EXISTS symbol_refs (
                repo_id     TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                line        INTEGER NOT NULL,
                PRIMARY KEY (repo_id, symbol_name, file_path, line)
            );

            CREATE TABLE IF NOT EXISTS edges (
                repo_id     TEXT NOT NULL,
                source_file TEXT NOT NULL,
                target_file TEXT NOT NULL,
                edge_type   TEXT NOT NULL,
                PRIMARY KEY (repo_id, source_file, target_file, edge_type)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_repo_file ON chunks (repo_id, file_path);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_name ON symbols (repo_id, symbol_name);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_qualified ON symbols (repo_id, qualified_name);
            CREATE INDEX IF NOT EXISTS idx_edges_repo_source ON edges (repo_id, source_file);
            CREATE INDEX IF NOT EXISTS idx_edges_repo_target ON edges (repo_id, target_file);
        """)
        self._conn.commit()

    def _migrate_legacy_symbols(self) -> None:
        """Upgrade a pre-qualified-name `symbols` table in place.

        The old table was keyed on (symbol_name, repo_id, file_path), which
        silently collapsed same-named symbols in one file (e.g. every
        `__init__`). Existing rows are carried over so /definition keeps
        working for already-ingested repos; re-ingesting fills in the rest.
        """
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(symbols)")}
        if not cols or "qualified_name" in cols:
            return
        self._conn.executescript("""
            ALTER TABLE symbols RENAME TO symbols_legacy;
            DROP INDEX IF EXISTS idx_symbols_repo_name;
        """)
        self._conn.executescript("""
            CREATE TABLE symbols (
                symbol_name    TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                repo_id        TEXT NOT NULL,
                file_path      TEXT NOT NULL,
                start_line     INTEGER NOT NULL,
                end_line       INTEGER,
                kind           TEXT,
                PRIMARY KEY (repo_id, file_path, qualified_name, start_line)
            );
            INSERT OR IGNORE INTO symbols
                (symbol_name, qualified_name, repo_id, file_path, start_line, end_line, kind)
                SELECT symbol_name, symbol_name, repo_id, file_path, COALESCE(start_line, 0), NULL, kind
                FROM symbols_legacy;
            DROP TABLE symbols_legacy;
        """)
        self._conn.commit()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def clear_repo(self, repo_id: str, commit: bool = True) -> None:
        """Delete all chunks/symbols/edges for a repo_id.

        Must be called before re-ingesting an already-indexed repo_id — chunk_ids
        are fresh UUIDs on every ingest, so without this, re-running /ingest on
        the same repo_id accumulates orphaned rows from every previous run
        instead of replacing them.
        """
        self._conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
        self._conn.execute("DELETE FROM symbols WHERE repo_id = ?", (repo_id,))
        self._conn.execute("DELETE FROM edges WHERE repo_id = ?", (repo_id,))
        self._conn.execute("DELETE FROM symbol_refs WHERE repo_id = ?", (repo_id,))
        if commit:
            self._conn.commit()

    def upsert_chunk(self, chunk: ChunkMetadata, repo_id: str) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, repo_id, file_path, language, start_line, end_line, symbols, imports, content)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk.chunk_id,
                repo_id,
                chunk.file_path,
                chunk.language,
                chunk.start_line,
                chunk.end_line,
                json.dumps(chunk.symbols),
                json.dumps(chunk.imports),
                chunk.content,
            ),
        )
        self._conn.commit()

    def add_chunks(self, chunks: list[ChunkMetadata], repo_id: str) -> None:
        """Bulk insert without committing — the caller commits once per ingest."""
        self._conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, repo_id, file_path, language, start_line, end_line, symbols, imports, content)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (c.chunk_id, repo_id, c.file_path, c.language, c.start_line, c.end_line,
                 json.dumps(c.symbols), json.dumps(c.imports), c.content)
                for c in chunks
            ],
        )

    def get_chunk(self, chunk_id: str) -> Optional[ChunkMetadata]:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_chunks_by_file(self, repo_id: str, file_path: str) -> list[ChunkMetadata]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE repo_id = ? AND file_path = ? ORDER BY start_line",
            (repo_id, file_path),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkMetadata:
        return ChunkMetadata(
            chunk_id=row["chunk_id"],
            file_path=row["file_path"],
            language=row["language"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            symbols=json.loads(row["symbols"]),
            imports=json.loads(row["imports"]),
            content=row["content"],
        )

    def upsert_symbol(
        self, name: str, repo_id: str, file_path: str, line: int, kind: str,
        qualified_name: Optional[str] = None, end_line: Optional[int] = None,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO symbols
               (symbol_name, qualified_name, repo_id, file_path, start_line, end_line, kind)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, qualified_name or name, repo_id, file_path, line, end_line, kind),
        )
        self._conn.commit()

    def add_symbols(self, repo_id: str, symbols: list["SymbolInfo"]) -> None:
        """Bulk insert without committing — the caller commits once per ingest."""
        self._conn.executemany(
            """INSERT OR REPLACE INTO symbols
               (symbol_name, qualified_name, repo_id, file_path, start_line, end_line, kind)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (s.name, s.qualified_name, repo_id, s.file_path, s.start_line, s.end_line, s.kind)
                for s in symbols
            ],
        )

    def find_symbol(self, repo_id: str, name: str) -> list[dict]:
        """Definitions matching `name`, ordered by file and line.

        A bare name ("send") matches every symbol with that name, top-level or
        nested. A dotted name ("HTTPAdapter.send") matches that qualified name,
        or any qualified name ending in it ("Outer.HTTPAdapter.send").
        """
        if "." in name:
            rows = self._conn.execute(
                """SELECT * FROM symbols
                   WHERE repo_id = ? AND (qualified_name = ? OR qualified_name LIKE ? ESCAPE '\\')
                   ORDER BY file_path, start_line""",
                (repo_id, name, "%." + _like_escape(name)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM symbols WHERE repo_id = ? AND symbol_name = ? ORDER BY file_path, start_line",
                (repo_id, name),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_references(self, repo_id: str, file_path: str, references: list["ReferenceInfo"]) -> None:
        """Bulk insert identifier usages without committing."""
        self._conn.executemany(
            "INSERT OR IGNORE INTO symbol_refs (repo_id, symbol_name, file_path, line) VALUES (?, ?, ?, ?)",
            [(repo_id, r.name, file_path, r.line) for r in references],
        )

    def prune_references(self, repo_id: str) -> None:
        """Drop usages of names that aren't defined anywhere in the repo.

        References are collected for every identifier (locals, parameters,
        stdlib names, ...); only usages of the repo's own symbols are ever
        queried, so the rest is dead weight.
        """
        self._conn.execute(
            """DELETE FROM symbol_refs WHERE repo_id = ? AND symbol_name NOT IN
               (SELECT DISTINCT symbol_name FROM symbols WHERE repo_id = ?)""",
            (repo_id, repo_id),
        )

    def find_references(self, repo_id: str, name: str) -> list[dict]:
        """Usages of `name` as {file_path, line}. For a dotted name, matches on
        its last component: references are recorded by identifier, not type."""
        bare = name.rsplit(".", 1)[-1]
        rows = self._conn.execute(
            """SELECT file_path, line FROM symbol_refs
               WHERE repo_id = ? AND symbol_name = ? ORDER BY file_path, line""",
            (repo_id, bare),
        ).fetchall()
        return [dict(r) for r in rows]

    def add_edge(self, repo_id: str, source: str, target: str, edge_type: str) -> None:
        """Insert an edge without committing."""
        self._conn.execute(
            """INSERT OR REPLACE INTO edges (repo_id, source_file, target_file, edge_type)
               VALUES (?, ?, ?, ?)""",
            (repo_id, source, target, edge_type),
        )

    def upsert_edge(
        self, repo_id: str, source: str, target: str, edge_type: str
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO edges (repo_id, source_file, target_file, edge_type)
               VALUES (?, ?, ?, ?)""",
            (repo_id, source, target, edge_type),
        )
        self._conn.commit()

    def get_edges_from(self, repo_id: str, file_path: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE repo_id = ? AND source_file = ?",
            (repo_id, file_path),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_edges_to(self, repo_id: str, file_path: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE repo_id = ? AND target_file = ?",
            (repo_id, file_path),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_chunks(self, repo_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE repo_id = ?", (repo_id,)
        ).fetchone()
        return row[0]

    def count_symbols(self, repo_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE repo_id = ?", (repo_id,)
        ).fetchone()
        return row[0]

    def count_edges(self, repo_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE repo_id = ?", (repo_id,)
        ).fetchone()
        return row[0]

    def count_references(self, repo_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM symbol_refs WHERE repo_id = ?", (repo_id,)
        ).fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
