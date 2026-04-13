import json
import sqlite3
from pathlib import Path
from typing import Optional

from app.models.schemas import ChunkMetadata


class MetadataStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
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
                symbol_name TEXT NOT NULL,
                repo_id     TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                start_line  INTEGER,
                kind        TEXT,
                PRIMARY KEY (symbol_name, repo_id, file_path)
            );

            CREATE TABLE IF NOT EXISTS edges (
                repo_id     TEXT NOT NULL,
                source_file TEXT NOT NULL,
                target_file TEXT NOT NULL,
                edge_type   TEXT NOT NULL,
                PRIMARY KEY (repo_id, source_file, target_file, edge_type)
            );
        """)
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
        self, name: str, repo_id: str, file_path: str, line: int, kind: str
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO symbols (symbol_name, repo_id, file_path, start_line, kind)
               VALUES (?, ?, ?, ?, ?)""",
            (name, repo_id, file_path, line, kind),
        )
        self._conn.commit()

    def find_symbol(self, repo_id: str, name: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM symbols WHERE repo_id = ? AND symbol_name = ?",
            (repo_id, name),
        ).fetchall()
        return [dict(r) for r in rows]

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

    def close(self) -> None:
        self._conn.close()
