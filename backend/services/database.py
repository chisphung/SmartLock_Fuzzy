"""
database.py – SQLite persistence layer for SmartLock event logs.

Tables
------
access_logs     : Every door-unlock or deny decision (face + keypad).
keypad_events   : Raw keypad events (digits entered, buffer cleared, lockout, …).
system_logs     : General system / application events (startup, errors, camera, …).

All timestamps are stored as Unix epoch floats (REAL) for precision and
simplicity; the API layer converts them to ISO-8601 strings on the way out.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger("smartlock.database")

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS access_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    source          TEXT    NOT NULL,   -- 'face' | 'keypad' | 'api'
    action          TEXT    NOT NULL,   -- 'unlock' | 'deny' | 'otp' | 'lockout'
    person_name     TEXT,               -- recognised face name, NULL for keypad
    confidence      REAL,               -- fuzzy model confidence (0-100), NULL for keypad
    security_risk   REAL,               -- fuzzy security risk (0-1), NULL for keypad
    illumination    REAL,               -- lighting level, NULL for keypad
    facial_angle    REAL,               -- pose angle, NULL for keypad
    details         TEXT,               -- human-readable decision detail
    extra_json      TEXT                -- arbitrary JSON for future fields
);

CREATE TABLE IF NOT EXISTS keypad_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    event_type      TEXT    NOT NULL,   -- 'digit_entered' | 'password_correct' | 'password_wrong'
                                        --   | 'lockout' | 'buffer_cleared' | 'key_ignored'
    message         TEXT,
    buffer_length   INTEGER,
    remaining       INTEGER,
    failed_attempts INTEGER,
    lockout_seconds REAL,
    extra_json      TEXT
);

CREATE TABLE IF NOT EXISTS system_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    level           TEXT    NOT NULL DEFAULT 'INFO',  -- 'DEBUG'|'INFO'|'WARNING'|'ERROR'|'CRITICAL'
    component       TEXT    NOT NULL,                 -- e.g. 'camera'|'hardware'|'fuzzy'|'api'
    message         TEXT    NOT NULL,
    extra_json      TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_access_logs_ts      ON access_logs   (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_access_logs_source  ON access_logs   (source);
CREATE INDEX IF NOT EXISTS idx_access_logs_action  ON access_logs   (action);
CREATE INDEX IF NOT EXISTS idx_keypad_events_ts    ON keypad_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_keypad_events_type  ON keypad_events (event_type);
CREATE INDEX IF NOT EXISTS idx_system_logs_ts      ON system_logs   (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_level   ON system_logs   (level);
CREATE INDEX IF NOT EXISTS idx_system_logs_comp    ON system_logs   (component);
"""

# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------

_DEFAULT_DB_NAME = "smartlock_logs.db"


def _resolve_db_path() -> Path:
    import os

    env_path = os.environ.get("SMARTLOCK_DB_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # Default: <repo_root>/backend/logs/smartlock_logs.db
    default = Path(__file__).resolve().parents[1] / "logs" / _DEFAULT_DB_NAME
    default.parent.mkdir(parents=True, exist_ok=True)
    return default


# ---------------------------------------------------------------------------
# LogDatabase
# ---------------------------------------------------------------------------


class LogDatabase:
    """
    Thread-safe SQLite log database.

    One instance is created at module level (``db``) and shared across the
    whole application.  All public methods are safe to call from any thread.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or _resolve_db_path()
        self._local = threading.local()  # per-thread connection
        self._lock = threading.Lock()
        self._init_schema()
        logger.info(f"[Database] SQLite log database at: {self._path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return (or create) the per-thread SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
                timeout=10.0,
            )
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._connect()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(_DDL)
        conn.commit()
        logger.debug("[Database] Schema initialised.")

    @staticmethod
    def _to_json(obj: Any) -> str | None:
        if obj is None:
            return None
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return None

    @staticmethod
    def _from_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        # Parse extra_json back to dict
        if d.get("extra_json"):
            try:
                d["extra_json"] = json.loads(d["extra_json"])
            except Exception:
                pass
        return d

    # ------------------------------------------------------------------
    # access_logs
    # ------------------------------------------------------------------

    def log_access(
        self,
        *,
        source: str,
        action: str,
        person_name: str | None = None,
        confidence: float | None = None,
        security_risk: float | None = None,
        illumination: float | None = None,
        facial_angle: float | None = None,
        details: str | None = None,
        extra: dict | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Insert a record into ``access_logs``. Returns the new row id."""
        ts = timestamp or time.time()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO access_logs
                    (timestamp, source, action, person_name, confidence,
                     security_risk, illumination, facial_angle, details, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    source,
                    action,
                    person_name,
                    confidence,
                    security_risk,
                    illumination,
                    facial_angle,
                    details,
                    self._to_json(extra),
                ),
            )
            row_id = cur.lastrowid
        logger.debug(f"[Database] access_log id={row_id} source={source} action={action}")
        return row_id

    def get_access_logs(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        source: str | None = None,
        action: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict]:
        """Query access_logs with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if source:
            clauses.append("source = ?")
            params.append(source)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params += [limit, offset]

        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM access_logs {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params,
            )
            return [self._from_row(r) for r in cur.fetchall()]

    def count_access_logs(
        self,
        *,
        source: str | None = None,
        action: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM access_logs {where}", params)
            return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # keypad_events
    # ------------------------------------------------------------------

    def log_keypad_event(
        self,
        *,
        event_type: str,
        message: str | None = None,
        buffer_length: int | None = None,
        remaining: int | None = None,
        failed_attempts: int | None = None,
        lockout_seconds: float | None = None,
        extra: dict | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Insert a record into ``keypad_events``. Returns the new row id."""
        ts = timestamp or time.time()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO keypad_events
                    (timestamp, event_type, message, buffer_length, remaining,
                     failed_attempts, lockout_seconds, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    event_type,
                    message,
                    buffer_length,
                    remaining,
                    failed_attempts,
                    lockout_seconds,
                    self._to_json(extra),
                ),
            )
            row_id = cur.lastrowid
        logger.debug(f"[Database] keypad_event id={row_id} type={event_type}")
        return row_id

    def get_keypad_events(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params += [limit, offset]
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM keypad_events {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params,
            )
            return [self._from_row(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # system_logs
    # ------------------------------------------------------------------

    def log_system(
        self,
        *,
        component: str,
        message: str,
        level: str = "INFO",
        extra: dict | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Insert a record into ``system_logs``. Returns the new row id."""
        ts = timestamp or time.time()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO system_logs
                    (timestamp, level, component, message, extra_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, level.upper(), component, message, self._to_json(extra)),
            )
            row_id = cur.lastrowid
        logger.debug(f"[Database] system_log id={row_id} [{level}] {component}: {message[:60]}")
        return row_id

    def get_system_logs(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        level: str | None = None,
        component: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if level:
            clauses.append("level = ?")
            params.append(level.upper())
        if component:
            clauses.append("component = ?")
            params.append(component)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params += [limit, offset]
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM system_logs {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params,
            )
            return [self._from_row(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Summary / stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return aggregate counts for the dashboard."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM access_logs")
            total_access = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM access_logs WHERE action = 'unlock'")
            total_unlock = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM access_logs WHERE action = 'deny'")
            total_deny = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM access_logs WHERE action = 'lockout'")
            total_lockout = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM keypad_events")
            total_keypad = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM system_logs WHERE level IN ('ERROR','CRITICAL')")
            total_errors = cur.fetchone()[0]

            # Last 24 hours
            since_24h = time.time() - 86400
            cur.execute(
                "SELECT COUNT(*) FROM access_logs WHERE timestamp >= ?", (since_24h,)
            )
            access_24h = cur.fetchone()[0]

        return {
            "access_logs": {
                "total": total_access,
                "unlocks": total_unlock,
                "denials": total_deny,
                "lockouts": total_lockout,
                "last_24h": access_24h,
            },
            "keypad_events": {"total": total_keypad},
            "system_logs": {"total_errors": total_errors},
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_old_logs(self, older_than_days: int = 30) -> dict:
        """Delete logs older than *older_than_days* days. Returns counts deleted."""
        cutoff = time.time() - older_than_days * 86400
        deleted: dict[str, int] = {}
        with self._cursor() as cur:
            cur.execute("DELETE FROM access_logs WHERE timestamp < ?", (cutoff,))
            deleted["access_logs"] = cur.rowcount
            cur.execute("DELETE FROM keypad_events WHERE timestamp < ?", (cutoff,))
            deleted["keypad_events"] = cur.rowcount
            cur.execute("DELETE FROM system_logs WHERE timestamp < ?", (cutoff,))
            deleted["system_logs"] = cur.rowcount
        logger.info(f"[Database] Purged logs older than {older_than_days} days: {deleted}")
        return deleted

    def close(self) -> None:
        """Close the current thread's connection (call on shutdown)."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

db = LogDatabase()
