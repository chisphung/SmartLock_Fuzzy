"""
logs.py – REST API router for SmartLock log database queries.

Endpoints
---------
GET  /api/v1/logs/access          – paginated access log query
GET  /api/v1/logs/keypad          – paginated keypad event query
GET  /api/v1/logs/system          – paginated system log query
GET  /api/v1/logs/stats           – aggregate statistics
POST /api/v1/logs/purge           – delete old records
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.database import db

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _ts_to_iso(ts: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()


def _format_row(row: dict) -> dict:
    """Convert Unix timestamp to ISO-8601 in every row before returning."""
    out = dict(row)
    if "timestamp" in out and isinstance(out["timestamp"], (int, float)):
        out["timestamp_iso"] = _ts_to_iso(out["timestamp"])
    return out


# ---------------------------------------------------------------------------
# Access Logs
# ---------------------------------------------------------------------------


@router.get("/access", summary="Query access logs")
async def get_access_logs(
    limit: int = Query(default=50, ge=1, le=500, description="Max rows to return"),
    offset: int = Query(default=0, ge=0),
    source: Optional[Literal["face", "keypad", "api"]] = Query(
        default=None, description="Filter by unlock source"
    ),
    action: Optional[Literal["unlock", "deny", "otp", "lockout"]] = Query(
        default=None, description="Filter by action taken"
    ),
    since: Optional[float] = Query(
        default=None, description="Unix epoch – only rows AFTER this time"
    ),
    until: Optional[float] = Query(
        default=None, description="Unix epoch – only rows BEFORE this time"
    ),
):
    """
    Return paginated access log records.

    Each record contains face-recognition scores (confidence, security_risk,
    illumination, facial_angle) for face events, and basic metadata for
    keypad events.
    """
    rows = db.get_access_logs(
        limit=limit,
        offset=offset,
        source=source,
        action=action,
        since=since,
        until=until,
    )
    total = db.count_access_logs(source=source, action=action, since=since, until=until)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": [_format_row(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# Keypad Events
# ---------------------------------------------------------------------------


@router.get("/keypad", summary="Query keypad events")
async def get_keypad_events(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    event_type: Optional[str] = Query(
        default=None,
        description=(
            "Filter by event type: digit_entered | password_correct | "
            "password_wrong | lockout | buffer_cleared | key_ignored"
        ),
    ),
    since: Optional[float] = Query(default=None),
    until: Optional[float] = Query(default=None),
):
    """Return paginated keypad event records."""
    rows = db.get_keypad_events(
        limit=limit,
        offset=offset,
        event_type=event_type,
        since=since,
        until=until,
    )
    return {
        "limit": limit,
        "offset": offset,
        "data": [_format_row(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# System Logs
# ---------------------------------------------------------------------------


@router.get("/system", summary="Query system logs")
async def get_system_logs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = Query(
        default=None, description="Filter by log level"
    ),
    component: Optional[str] = Query(
        default=None, description="Filter by component (e.g. camera, hardware, fuzzy)"
    ),
    since: Optional[float] = Query(default=None),
    until: Optional[float] = Query(default=None),
):
    """Return paginated system / application log records."""
    rows = db.get_system_logs(
        limit=limit,
        offset=offset,
        level=level,
        component=component,
        since=since,
        until=until,
    )
    return {
        "limit": limit,
        "offset": offset,
        "data": [_format_row(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@router.get("/stats", summary="Aggregate log statistics")
async def get_log_stats():
    """
    Return aggregate counts: total unlocks, denials, lockouts, errors, and
    a 24-hour access count.
    """
    return db.get_stats()


# ---------------------------------------------------------------------------
# Purge (maintenance)
# ---------------------------------------------------------------------------


class PurgeRequest(BaseModel):
    older_than_days: int = Field(
        default=30, ge=1, le=3650, description="Delete logs older than this many days"
    )


@router.post("/purge", summary="Delete old log records")
async def purge_old_logs(request: PurgeRequest):
    """
    Permanently delete log records older than ``older_than_days`` days.
    Use with caution – this operation cannot be undone.
    """
    if request.older_than_days < 1:
        raise HTTPException(status_code=400, detail="older_than_days must be >= 1")
    deleted = db.purge_old_logs(older_than_days=request.older_than_days)
    return {
        "message": f"Deleted records older than {request.older_than_days} days.",
        "deleted": deleted,
    }
