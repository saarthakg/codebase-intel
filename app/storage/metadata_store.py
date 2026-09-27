import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from app.core.text import expand_identifiers
from app.models.schemas import ChunkMetadata

if TYPE_CHECKING:
    import numpy as np

    from app.core.history import CoChange
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
        fts_existed = self._table_exists("chunks_fts")
        self._conn.executescript("""
            -- Keyword index over chunks. Columns are search text only (camelCase
            -- identifiers expanded); the raw chunk lives in `chunks`.
            -- porter: "redirects"/"redirecting" match "redirect".
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                repo_id UNINDEXED,
                path,
                symbols,
                content,
                tokenize = "porter unicode61"
            );
        """)
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

            -- /ask answers keyed by a hash of (prompt version, backend, model,
            -- system prompt, full prompt). The prompt embeds the excerpt text,
            -- so a key can only match while the code it was answered from is
            -- unchanged; entries therefore survive re-ingest safely.
            CREATE TABLE IF NOT EXISTS answer_cache (
                cache_key   TEXT PRIMARY KEY,
                response    TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Git co-change: commits per file, and commits shared by file pairs.
            CREATE TABLE IF NOT EXISTS cochange_files (
                repo_id   TEXT NOT NULL,
                file_path TEXT NOT NULL,
                commits   INTEGER NOT NULL,
                PRIMARY KEY (repo_id, file_path)
            );
            CREATE TABLE IF NOT EXISTS cochange_pairs (
                repo_id TEXT NOT NULL,
                file_a  TEXT NOT NULL,
                file_b  TEXT NOT NULL,
                commits INTEGER NOT NULL,
                PRIMARY KEY (repo_id, file_a, file_b)
            );

            -- Embeddings keyed by (backend:model, sha256 of the exact embedded
            -- text). Re-ingesting unchanged code reuses vectors instead of
            -- re-embedding (~95% of ingest time). Not cleared by clear_repo;
            -- pruned to the current index after each successful ingest.
            CREATE TABLE IF NOT EXISTS embedding_cache (
                model     TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector    BLOB NOT NULL,
                PRIMARY KEY (model, text_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_repo_file ON chunks (repo_id, file_path);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_name ON symbols (repo_id, symbol_name);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_qualified ON symbols (repo_id, qualified_name);
            CREATE INDEX IF NOT EXISTS idx_edges_repo_source ON edges (repo_id, source_file);
            CREATE INDEX IF NOT EXISTS idx_edges_repo_target ON edges (repo_id, target_file);
        """)
        self._conn.commit()
        if not fts_existed:
            self._backfill_fts()

    def _table_exists(self, name: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone() is not None

    def _backfill_fts(self) -> None:
        """Index chunks from a DB created before keyword search existed."""
        rows = self._conn.execute("SELECT * FROM chunks").fetchall()
        if rows:
            self._insert_fts([(r["repo_id"], self._row_to_chunk(r)) for r in rows])
            self._conn.commit()

    def _insert_fts(self, items: list[tuple[str, ChunkMetadata]]) -> None:
        self._conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, repo_id, path, symbols, content) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    c.chunk_id, repo_id,
                    expand_identifiers(c.file_path),
                    expand_identifiers(" ".join(c.symbols)),
                    expand_identifiers(c.content),
                )
                for repo_id, c in items
            ],
        )

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
        self._conn.execute("DELETE FROM chunks_fts WHERE repo_id = ?", (repo_id,))
        self._conn.execute("DELETE FROM cochange_files WHERE repo_id = ?", (repo_id,))
        self._conn.execute("DELETE FROM cochange_pairs WHERE repo_id = ?", (repo_id,))
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
        self._insert_fts([(repo_id, c) for c in chunks])

    def keyword_search(
        self, repo_id: str, terms: list[str], limit: int, per_file_cap: Optional[int] = None
    ) -> list[tuple[str, float]]:
        """BM25 search: (chunk_id, score) best-first, score higher = better.

        Terms are OR-ed: natural-language questions rarely have every word in
        one chunk. Symbol-name matches weigh more than path, path more than body.
        `per_file_cap` keeps at most that many chunks from any one file.
        """
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        # Over-fetch when capping so the list can still fill up to `limit`.
        fetch = limit * 4 if per_file_cap else limit
        rows = self._conn.execute(
            """SELECT f.chunk_id, c.file_path, bm25(chunks_fts, 0, 0, 2.0, 3.0, 1.0) AS rank
               FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.chunk_id
               WHERE chunks_fts MATCH ? AND f.repo_id = ?
               ORDER BY rank LIMIT ?""",
            (match, repo_id, fetch),
        ).fetchall()
        results: list[tuple[str, float]] = []
        per_file: dict[str, int] = {}
        for r in rows:
            if per_file_cap:
                if per_file.get(r["file_path"], 0) >= per_file_cap:
                    continue
                per_file[r["file_path"]] = per_file.get(r["file_path"], 0) + 1
            results.append((r["chunk_id"], -r["rank"]))  # bm25() is lower-is-better
            if len(results) >= limit:
                break
        return results

    def chunks_defining(self, repo_id: str, names: list[str]) -> list[str]:
        """chunk_ids of chunks that contain the definition of any of `names`
        (bare or qualified), ordered source-before-tests then by path/line."""
        if not names:
            return []
        marks = ",".join("?" * len(names))
        rows = self._conn.execute(
            f"""SELECT DISTINCT c.chunk_id, c.file_path, c.start_line FROM symbols s
                JOIN chunks c ON c.repo_id = s.repo_id AND c.file_path = s.file_path
                     AND s.start_line BETWEEN c.start_line AND c.end_line
                WHERE s.repo_id = ? AND (s.symbol_name IN ({marks}) OR s.qualified_name IN ({marks}))
                ORDER BY c.file_path, c.start_line""",
            (repo_id, *names, *names),
        ).fetchall()
        from app.core.definitions import is_test_path
        ids = sorted(rows, key=lambda r: is_test_path(r["file_path"]))
        return list(dict.fromkeys(r["chunk_id"] for r in ids))

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

    def indexed_files(self, repo_id: str) -> list[str]:
        rows = self._conn.execute(
            """SELECT file_path FROM chunks WHERE repo_id = ?
               UNION SELECT file_path FROM symbols WHERE repo_id = ?
               UNION SELECT source_file FROM edges WHERE repo_id = ?
               UNION SELECT target_file FROM edges WHERE repo_id = ?""",
            (repo_id,) * 4,
        ).fetchall()
        return sorted(r[0] for r in rows)

    def all_edges(self, repo_id: str) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT source_file, target_file FROM edges WHERE repo_id = ? AND edge_type = 'import'",
            (repo_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def symbols_in_file(self, repo_id: str, file_path: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM symbols WHERE repo_id = ? AND file_path = ? ORDER BY start_line",
            (repo_id, file_path),
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

    def save_cochange(self, repo_id: str, cochange: "CoChange") -> None:
        """Store co-change stats without committing (part of the ingest transaction)."""
        files, pairs = cochange.to_rows()
        self._conn.executemany(
            "INSERT OR REPLACE INTO cochange_files (repo_id, file_path, commits) VALUES (?, ?, ?)",
            [(repo_id, f, n) for f, n in files],
        )
        self._conn.executemany(
            "INSERT OR REPLACE INTO cochange_pairs (repo_id, file_a, file_b, commits) VALUES (?, ?, ?, ?)",
            [(repo_id, a, b, n) for a, b, n in pairs],
        )

    def load_cochange(self, repo_id: str) -> "CoChange":
        from app.core.history import CoChange
        files = self._conn.execute(
            "SELECT file_path, commits FROM cochange_files WHERE repo_id = ?", (repo_id,)
        ).fetchall()
        pairs = self._conn.execute(
            "SELECT file_a, file_b, commits FROM cochange_pairs WHERE repo_id = ?", (repo_id,)
        ).fetchall()
        return CoChange.from_rows([tuple(r) for r in files], [tuple(r) for r in pairs])

    def get_cached_embeddings(self, model: str, text_hashes: list[str]) -> dict[str, "np.ndarray"]:
        import numpy as np
        found: dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(text_hashes))
        for i in range(0, len(unique), 500):  # stay under SQLite's variable limit
            batch = unique[i:i + 500]
            rows = self._conn.execute(
                f"SELECT text_hash, vector FROM embedding_cache WHERE model = ? "
                f"AND text_hash IN ({','.join('?' * len(batch))})",
                (model, *batch),
            ).fetchall()
            for r in rows:
                found[r["text_hash"]] = np.frombuffer(r["vector"], dtype=np.float32)
        return found

    def put_cached_embeddings(self, model: str, vectors: dict[str, "np.ndarray"]) -> None:
        """Store vectors without committing (part of the ingest transaction)."""
        import numpy as np
        self._conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (model, text_hash, vector) VALUES (?, ?, ?)",
            [(model, h, np.asarray(v, dtype=np.float32).tobytes()) for h, v in vectors.items()],
        )

    def prune_embedding_cache(self, model: str, keep: set[str]) -> None:
        """Drop cached vectors not used by the current index (other models included)."""
        self._conn.execute("CREATE TEMP TABLE IF NOT EXISTS _keep_hashes (h TEXT PRIMARY KEY)")
        self._conn.execute("DELETE FROM _keep_hashes")
        self._conn.executemany("INSERT OR IGNORE INTO _keep_hashes (h) VALUES (?)", [(h,) for h in keep])
        self._conn.execute(
            "DELETE FROM embedding_cache WHERE model != ? OR text_hash NOT IN (SELECT h FROM _keep_hashes)",
            (model,),
        )

    def get_cached_answer(self, cache_key: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT response FROM answer_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return json.loads(row["response"]) if row else None

    def put_cached_answer(self, cache_key: str, response: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO answer_cache (cache_key, response) VALUES (?, ?)",
            (cache_key, json.dumps(response)),
        )
        self._conn.commit()

    def symbol_exists(self, repo_id: str, name: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM symbols WHERE repo_id = ? AND (symbol_name = ? OR qualified_name = ?) LIMIT 1",
            (repo_id, name, name),
        ).fetchone() is not None

    def file_exists(self, repo_id: str, path: str) -> bool:
        """True if an indexed file is `path` or ends with it ("adapters.py" matches
        "src/requests/adapters.py")."""
        return self._conn.execute(
            """SELECT 1 FROM chunks WHERE repo_id = ?
               AND (file_path = ? OR file_path LIKE ? ESCAPE '\\') LIMIT 1""",
            (repo_id, path, "%/" + _like_escape(path)),
        ).fetchone() is not None

    def count_references(self, repo_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM symbol_refs WHERE repo_id = ?", (repo_id,)
        ).fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
