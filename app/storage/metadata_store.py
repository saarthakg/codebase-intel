import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.core.history import CoChange
    from app.core.symbols import ReferenceInfo, SymbolInfo


class SchemaVersionError(RuntimeError):
    pass


class MetadataStore:
    # Schema version, kept in SQLite's user_version. To change a table, bump it
    # and append a step to _MIGRATIONS (below the class): step i takes a DB from
    # version i to i + 1, so existing databases upgrade in place on open.
    SCHEMA_VERSION = 3

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def schema_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        version = self.schema_version()
        if version > self.SCHEMA_VERSION:
            raise SchemaVersionError(
                f"{self.db_path} was written by a newer version of codebase-intel "
                f"(schema {version}, this one reads up to {self.SCHEMA_VERSION}). Upgrade, or re-index."
            )
        for step in _MIGRATIONS[version:self.SCHEMA_VERSION]:
            step(self)
            version += 1
            self._conn.execute(f"PRAGMA user_version = {version}")
            self._conn.commit()

    def _v1_baseline(self) -> None:
        """Version 1: the tables as of the versioned schema. Databases from
        before versioning (user_version 0) may lack any of these, or have the
        pre-qualified-name symbols table, so everything is create-if-missing.
        (Tables removed since are dropped by later steps.)"""
        self._migrate_legacy_symbols()
        self._conn.executescript("""
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

            -- Method references with the receiver's inferred type (app/core/typeinfer.py):
            -- receiver is a repo class name, '' (external type) or '?' (unknown).
            -- One row per (file, method name, receiver).
            CREATE TABLE IF NOT EXISTS method_refs (
                repo_id     TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                name        TEXT NOT NULL,
                receiver    TEXT NOT NULL,
                PRIMARY KEY (repo_id, name, receiver, file_path)
            );
            -- Class inheritance, for resolving method dispatch.
            CREATE TABLE IF NOT EXISTS class_bases (
                repo_id     TEXT NOT NULL,
                class_name  TEXT NOT NULL,
                base_name   TEXT NOT NULL,
                PRIMARY KEY (repo_id, class_name, base_name)
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

            CREATE INDEX IF NOT EXISTS idx_symbols_repo_name ON symbols (repo_id, symbol_name);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_qualified ON symbols (repo_id, qualified_name);
            CREATE INDEX IF NOT EXISTS idx_edges_repo_source ON edges (repo_id, source_file);
            CREATE INDEX IF NOT EXISTS idx_edges_repo_target ON edges (repo_id, target_file);
        """)
        self._conn.commit()

    def _v2_drop_embedding_cache(self) -> None:
        """Version 2: dropped the embedding cache (vectors were reused from the
        previous index instead)."""
        self._conn.execute("DROP TABLE IF EXISTS embedding_cache")
        self._conn.commit()

    def _v3_impact_only(self) -> None:
        """Version 3: search and /ask were removed, taking the code chunks, their
        keyword index and the answer cache with them. Indexed files get their
        own table instead of being read off the chunks."""
        self._conn.executescript("""
            DROP TABLE IF EXISTS chunks_fts;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS answer_cache;
            CREATE TABLE IF NOT EXISTS files (
                repo_id   TEXT NOT NULL,
                file_path TEXT NOT NULL,
                language  TEXT NOT NULL,
                PRIMARY KEY (repo_id, file_path)
            );
        """)
        self._conn.commit()
        self._conn.execute("VACUUM")  # give the space back

    def _migrate_legacy_symbols(self) -> None:
        """Upgrade a pre-qualified-name `symbols` table in place.

        The old table was keyed on (symbol_name, repo_id, file_path), which
        silently collapsed same-named symbols in one file (e.g. every
        `__init__`). Existing rows are carried over; re-indexing fills in the rest.
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
        """Delete everything stored for a repo_id, before re-indexing it."""
        for table in ("files", "symbols", "edges", "symbol_refs", "method_refs", "class_bases",
                      "cochange_files", "cochange_pairs"):
            self._conn.execute(f"DELETE FROM {table} WHERE repo_id = ?", (repo_id,))
        if commit:
            self._conn.commit()

    def add_files(self, repo_id: str, files: list[tuple[str, str]]) -> None:
        """(file_path, language) rows, without committing."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO files (repo_id, file_path, language) VALUES (?, ?, ?)",
            [(repo_id, f, lang) for f, lang in files],
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
        """Bulk insert without committing — the caller commits once per index."""
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

    def add_method_refs(self, repo_id: str, file_path: str, refs) -> None:
        """Store typed method references (typeinfer.AttrRef) without committing."""
        self._conn.executemany(
            "INSERT OR IGNORE INTO method_refs (repo_id, file_path, name, receiver) VALUES (?, ?, ?, ?)",
            [(repo_id, file_path, r.name, r.receiver) for r in refs],
        )

    def add_class_bases(self, repo_id: str, pairs: list[tuple[str, str]]) -> None:
        self._conn.executemany(
            "INSERT OR IGNORE INTO class_bases (repo_id, class_name, base_name) VALUES (?, ?, ?)",
            [(repo_id, c, b) for c, b in pairs],
        )

    def has_method_refs(self, repo_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM method_refs WHERE repo_id = ? LIMIT 1", (repo_id,)
        ).fetchone() is not None

    def method_ref_files(self, repo_id: str, name: str, receivers: list[str]) -> set[str]:
        if not receivers:
            return set()
        marks = ",".join("?" * len(receivers))
        rows = self._conn.execute(
            f"SELECT DISTINCT file_path FROM method_refs WHERE repo_id = ? AND name = ? AND receiver IN ({marks})",
            (repo_id, name, *receivers),
        ).fetchall()
        return {r[0] for r in rows}

    def class_bases_map(self, repo_id: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for c, b in self._conn.execute(
            "SELECT class_name, base_name FROM class_bases WHERE repo_id = ?", (repo_id,)
        ).fetchall():
            out.setdefault(c, []).append(b)
        return out

    def indexed_files(self, repo_id: str) -> list[str]:
        rows = self._conn.execute(
            """SELECT file_path FROM files WHERE repo_id = ?
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

    def upsert_edge(self, repo_id: str, source: str, target: str, edge_type: str) -> None:
        self.add_edge(repo_id, source, target, edge_type)
        self._conn.commit()

    def count_symbols(self, repo_id: str) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM symbols WHERE repo_id = ?", (repo_id,)).fetchone()[0]

    def count_references(self, repo_id: str) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM symbol_refs WHERE repo_id = ?", (repo_id,)).fetchone()[0]

    def save_cochange(self, repo_id: str, cochange: "CoChange") -> None:
        """Store co-change stats without committing (part of the index transaction)."""
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

    def close(self) -> None:
        self._conn.close()


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_MIGRATIONS = [MetadataStore._v1_baseline, MetadataStore._v2_drop_embedding_cache, MetadataStore._v3_impact_only]
