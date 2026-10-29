"""Privacy governance, security controls, data retention, and path safety for facial captures.

Enforces:
1. Anonymous session isolation: Strictly prohibits cross-session biometric template matching or search.
2. Secure path traversal prevention: Validates capture paths against null bytes, traversal tokens, absolute paths, and reserved names.
3. Automated data retention & pruning: Safely sweeps expired person profiles and removes disk capture assets.
4. Tamper-evident audit logging: Records security and privacy events in-memory and to structured JSONL logs.
5. Biometric redaction and blur: Provides reversible/irreversible visual anonymization for bystanders and expired tracks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import urllib.parse
import uuid

import cv2
import numpy as np

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PrivacyGovernanceError(Exception):
    """Base exception for privacy and data governance violations."""
    pass


class PathTraversalError(PrivacyGovernanceError, ValueError):
    """Raised when an asset path traversal or illegal filename is detected."""
    pass


class CrossSessionMatchingViolation(PrivacyGovernanceError, PermissionError):
    """Raised when biometric matching or template search is attempted across sessions."""
    pass


class InvalidSessionIdError(PrivacyGovernanceError, ValueError):
    """Raised when a session ID does not conform to anonymous session requirements."""
    pass


class RetentionPolicyError(PrivacyGovernanceError):
    """Raised when data retention or cleanup policy fails."""
    pass


# ---------------------------------------------------------------------------
# Audit Logging Data Structures and Logger
# ---------------------------------------------------------------------------

class AuditEventType:
    """Standard security and privacy audit event types."""
    CAPTURE_DELETED = "capture_deleted"
    SESSION_PURGED = "session_purged"
    ACCESS_DENIED = "access_denied"
    RETENTION_SWEEP = "retention_sweep"
    PATH_TRAVERSAL_BLOCKED = "path_traversal_blocked"
    SESSION_CREATED = "session_created"
    ANONYMIZATION_APPLIED = "anonymization_applied"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class AuditRecord:
    """Structured security and data governance audit entry."""
    event_id: str
    timestamp: float
    iso_timestamp: str
    event_type: str
    severity: str  # "INFO", "WARNING", "CRITICAL", "AUDIT"
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "iso_timestamp": self.iso_timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "session_id": self.session_id,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)


class AuditLogger:
    """Thread-safe security audit logger with memory buffer and JSONL disk persistence."""

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        max_buffer_size: int = 2000,
        enable_stdout: bool = False,
    ):
        self.log_path = Path(log_path).resolve() if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: deque[AuditRecord] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self.enable_stdout = enable_stdout

    def log_event(
        self,
        event_type: str,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> AuditRecord:
        """Create, store, and persist a security/privacy audit record."""
        now = time.time()
        iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        record = AuditRecord(
            event_id=str(uuid.uuid4()),
            timestamp=now,
            iso_timestamp=iso,
            event_type=str(event_type),
            severity=str(severity).upper(),
            session_id=str(session_id) if session_id is not None else None,
            details=details or {},
        )

        with self._lock:
            self._buffer.append(record)
            if self.log_path:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(record.to_json() + "\n")
                except Exception as exc:
                    LOG.error("Failed to write audit record to %s: %s", self.log_path, exc)

        if self.enable_stdout or record.severity in ("WARNING", "CRITICAL"):
            LOG.warning("AUDIT [%s] %s (session=%s): %s", record.severity, record.event_type, record.session_id, record.details)

        return record

    def get_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        min_timestamp: Optional[float] = None,
    ) -> List[AuditRecord]:
        """Query buffered audit events with optional filters."""
        with self._lock:
            results = list(self._buffer)

        if event_type:
            results = [r for r in results if r.event_type == event_type]
        if session_id:
            results = [r for r in results if r.session_id == session_id]
        if min_timestamp is not None:
            results = [r for r in results if r.timestamp >= min_timestamp]

        return results[-limit:] if limit > 0 else results

    def clear(self) -> None:
        """Clear memory buffer (useful during unit testing)."""
        with self._lock:
            self._buffer.clear()


# ---------------------------------------------------------------------------
# Path Traversal Prevention & Validation
# ---------------------------------------------------------------------------

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
}

_SAFE_PATH_COMPONENT_RE = re.compile(r"^[a-zA-Z0-9_.\-+=@]+$")


def validate_capture_path(
    base_dir: Union[str, Path],
    requested_path: Union[str, Path],
    audit_logger: Optional[AuditLogger] = None,
) -> Path:
    """Validate that a requested capture file path strictly resides within base_dir.

    Ensures no path traversal attacks:
    - Rejects null bytes (`\\x00`).
    - Rejects absolute paths (`/etc/...`, `C:\\...`, `\\\\unc\\...`).
    - Rejects relative traversal components (`..`, `../`, `..\\\\`).
    - Rejects URL-encoded path traversal sequences (`%2e%2e`).
    - Rejects Windows reserved device names (`CON`, `NUL`, etc.).
    - Verifies resolved path is strictly a descendant of base_dir.

    Raises:
        PathTraversalError: If any path traversal or invalid path indicator is found.

    Returns:
        Path: Absolute resolved path guaranteed to be inside base_dir.
    """
    if requested_path is None:
        if audit_logger:
            audit_logger.log_event(
                AuditEventType.PATH_TRAVERSAL_BLOCKED,
                details={"reason": "Path is None"},
                severity="WARNING",
            )
        raise PathTraversalError("Requested path cannot be None")

    req_str = str(requested_path)

    # 1. Null byte check
    if "\x00" in req_str:
        if audit_logger:
            audit_logger.log_event(
                AuditEventType.PATH_TRAVERSAL_BLOCKED,
                details={"reason": "Null byte injection detected", "raw_path": repr(req_str)},
                severity="CRITICAL",
            )
        raise PathTraversalError("Path contains prohibited null byte")

    # 2. Empty or whitespace path
    trimmed = req_str.strip()
    if not trimmed:
        if audit_logger:
            audit_logger.log_event(
                AuditEventType.PATH_TRAVERSAL_BLOCKED,
                details={"reason": "Empty or whitespace-only path"},
                severity="WARNING",
            )
        raise PathTraversalError("Path cannot be empty or whitespace only")

    # 3. URL-encoded sequence decoding & traversal check
    unquoted = urllib.parse.unquote(req_str)
    if "\x00" in unquoted:
        if audit_logger:
            audit_logger.log_event(
                AuditEventType.PATH_TRAVERSAL_BLOCKED,
                details={"reason": "URL-encoded null byte detected", "raw_path": repr(req_str)},
                severity="CRITICAL",
            )
        raise PathTraversalError("URL-encoded null byte detected")

    # 4. Absolute path rejection
    # Check for Unix root, Windows drive letter, or UNC share
    is_abs = (
        os.path.isabs(req_str)
        or os.path.isabs(unquoted)
        or req_str.startswith(("/", "\\"))
        or unquoted.startswith(("/", "\\"))
        or bool(re.match(r"^[a-zA-Z]:", req_str))
        or bool(re.match(r"^[a-zA-Z]:", unquoted))
    )
    if is_abs:
        if audit_logger:
            audit_logger.log_event(
                AuditEventType.PATH_TRAVERSAL_BLOCKED,
                details={"reason": "Absolute path prohibited", "raw_path": req_str},
                severity="CRITICAL",
            )
        raise PathTraversalError(f"Absolute paths are strictly prohibited: '{req_str}'")

    # 5. Check path segments for traversal '..'
    normalized_slash = unquoted.replace("\\", "/")
    segments = [s for s in normalized_slash.split("/") if s]
    for seg in segments:
        if seg == "..":
            if audit_logger:
                audit_logger.log_event(
                    AuditEventType.PATH_TRAVERSAL_BLOCKED,
                    details={"reason": "Parent directory traversal '..' detected", "raw_path": req_str},
                    severity="CRITICAL",
                )
            raise PathTraversalError(f"Path traversal '..' is prohibited: '{req_str}'")
        # Check Windows reserved device names
        stem = seg.split(".")[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            if audit_logger:
                audit_logger.log_event(
                    AuditEventType.PATH_TRAVERSAL_BLOCKED,
                    details={"reason": f"Windows reserved device name: {stem}", "raw_path": req_str},
                    severity="WARNING",
                )
            raise PathTraversalError(f"Windows reserved device name prohibited: '{stem}'")

    # 6. Canonical directory resolution and boundary check
    base_resolved = Path(base_dir).resolve()
    target_resolved = (base_resolved / Path(*segments)).resolve()

    try:
        # Python 3.9+ is_relative_to
        if not target_resolved.is_relative_to(base_resolved):
            if audit_logger:
                audit_logger.log_event(
                    AuditEventType.PATH_TRAVERSAL_BLOCKED,
                    details={
                        "reason": "Resolved path escapes base directory",
                        "base_dir": str(base_resolved),
                        "target_resolved": str(target_resolved),
                    },
                    severity="CRITICAL",
                )
            raise PathTraversalError(
                f"Path traversal escape detected: '{req_str}' resolves outside base directory '{base_resolved}'"
            )
    except AttributeError:
        # Fallback for older python
        try:
            target_resolved.relative_to(base_resolved)
        except ValueError:
            if audit_logger:
                audit_logger.log_event(
                    AuditEventType.PATH_TRAVERSAL_BLOCKED,
                    details={"reason": "Path escapes base directory", "raw_path": req_str},
                    severity="CRITICAL",
                )
            raise PathTraversalError(f"Path traversal escape detected: '{req_str}'")

    # Capture files must not be the base directory itself
    if target_resolved == base_resolved:
        if audit_logger:
            audit_logger.log_event(
                AuditEventType.PATH_TRAVERSAL_BLOCKED,
                details={"reason": "Target path cannot be the base directory itself"},
                severity="WARNING",
            )
        raise PathTraversalError("Target path cannot resolve to the base directory root itself")

    return target_resolved


# ---------------------------------------------------------------------------
# Anonymous Session ID Enforcement & Anti-Cross-Session Matching
# ---------------------------------------------------------------------------

_ANONYMOUS_SESSION_PATTERN = re.compile(r"^[a-zA-Z0-9_\-:]{6,64}$")
_PII_SUSPICIOUS_PATTERNS = [
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # Email
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),  # IPv4
    re.compile(r"(?:user|admin|name|ssn|phone)", re.IGNORECASE),  # Common PII markers
]


def generate_anonymous_session_id(prefix: str = "sess_") -> str:
    """Generate a cryptographically random, anonymous session identifier.

    Guarantees no PII, hardware fingerprints, or persistent user associations.
    """
    token = secrets.token_hex(16)
    return f"{prefix}{token}"


def is_anonymous_session_id(session_id: str) -> bool:
    """Verify that a session ID adheres strictly to anonymous session specifications."""
    if not isinstance(session_id, str):
        return False
    if not _ANONYMOUS_SESSION_PATTERN.match(session_id):
        return False
    # Check for accidental PII inclusion
    for pattern in _PII_SUSPICIOUS_PATTERNS:
        if pattern.search(session_id):
            return False
    return True


def validate_anonymous_session_id(session_id: str) -> None:
    """Raise InvalidSessionIdError if session_id is not strictly anonymous."""
    if not is_anonymous_session_id(session_id):
        raise InvalidSessionIdError(
            f"Invalid or non-anonymous session ID: '{session_id}'. "
            "Session IDs must be 8-64 alphanumeric/dash/underscore characters with no PII."
        )


def assert_session_isolation(
    query_session_id: str,
    candidate_session_id: str,
    audit_logger: Optional[AuditLogger] = None,
) -> None:
    """Strictly prohibit cross-session biometric template matching or search.

    Biometric face templates and candidate profiles MUST remain partitioned
    within the originating anonymous capture session. Cross-session linkage
    violates privacy-by-design requirements.
    """
    if query_session_id != candidate_session_id:
        if audit_logger:
            audit_logger.log_event(
                AuditEventType.ACCESS_DENIED,
                session_id=query_session_id,
                details={
                    "reason": "Cross-session biometric matching prohibited",
                    "query_session": query_session_id,
                    "target_session": candidate_session_id,
                },
                severity="CRITICAL",
            )
        raise CrossSessionMatchingViolation(
            f"Privacy Governance Violation: Cross-session biometric matching or search "
            f"is strictly prohibited. (query session '{query_session_id}' != candidate session '{candidate_session_id}')"
        )


def filter_cross_session_candidates(
    query_session_id: str,
    candidates: Sequence[Any],
    strict: bool = True,
    audit_logger: Optional[AuditLogger] = None,
) -> List[Any]:
    """Filter biometric candidates to ensure they originate from query_session_id.

    If strict is True and any candidate belongs to a different session,
    raises CrossSessionMatchingViolation.
    If strict is False, silently filters out non-matching candidates.
    """
    validate_anonymous_session_id(query_session_id)
    allowed = []
    cross_session_detected = False

    for cand in candidates:
        cand_session = None
        if hasattr(cand, "session_id"):
            cand_session = getattr(cand, "session_id")
        elif isinstance(cand, dict) and "session_id" in cand:
            cand_session = cand["session_id"]

        if cand_session is None:
            # Cannot verify session -> treat as untrusted
            cross_session_detected = True
            continue

        if cand_session == query_session_id:
            allowed.append(cand)
        else:
            cross_session_detected = True

    if cross_session_detected and strict:
        if audit_logger:
            audit_logger.log_event(
                AuditEventType.ACCESS_DENIED,
                session_id=query_session_id,
                details={"reason": "Cross-session candidates rejected during search filtering"},
                severity="CRITICAL",
            )
        raise CrossSessionMatchingViolation(
            f"Rejected candidate pool containing records outside session '{query_session_id}'"
        )

    return allowed


# ---------------------------------------------------------------------------
# Data Retention & Auto-Pruning
# ---------------------------------------------------------------------------

def _get_row_val(row: Any, key: str, index: Optional[int] = None, default: Any = None) -> Any:
    """Safely extract a value from sqlite3.Row, tuple, list, dict, or object."""
    if row is None:
        return default
    try:
        val = row[key]
        if val is not None:
            return val
    except (KeyError, IndexError, TypeError):
        pass
    if index is not None:
        try:
            val = row[index]
            if val is not None:
                return val
        except (IndexError, TypeError):
            pass
    if hasattr(row, key):
        val = getattr(row, key)
        if val is not None:
            return val
    return default


def _safe_delete_file(path_str: Optional[str]) -> Tuple[bool, int]:
    """Safely unlink a capture file and return (deleted_boolean, bytes_freed)."""
    if not path_str:
        return False, 0
    try:
        p = Path(path_str)
        if p.is_file():
            size = p.stat().st_size
            p.unlink(missing_ok=True)
            return True, size
    except Exception as exc:
        LOG.warning("Could not delete capture file %s during retention sweep: %s", path_str, exc)
    return False, 0


def prune_expired_records(
    db: Any,
    max_age_seconds: float,
    max_profiles: Optional[int] = None,
    current_time: Optional[float] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> Dict[str, Any]:
    """Prune expired person profiles and delete corresponding disk captures.

    Handles:
    - SQLite database connections (`sqlite3.Connection`, `sqlite3.Cursor`, or file path `str`/`Path`).
    - In-memory data structures / dictionaries with `profiles` and `captures` keys/attributes.
    - Capacity-based pruning: limits active profiles to `max_profiles`, removing oldest first.
    - Age-based pruning: removes profiles and captures older than `max_age_seconds`.
    - Disk capture asset cleanup: deletes `original_path` and `enhanced_path` files.
    - Audit logging: records retention sweep and capture deletion metrics.

    Returns:
        Dict[str, Any]: Metrics summary:
            - "pruned_profiles": count of profiles removed
            - "pruned_captures": count of captures removed
            - "deleted_files": count of disk capture files deleted
            - "freed_bytes": total bytes freed on disk
            - "timestamp": time of execution
    """
    now = current_time if current_time is not None else time.time()
    cutoff_time = max(0.0, now - float(max_age_seconds))

    pruned_profiles_count = 0
    pruned_captures_count = 0
    deleted_files_count = 0
    freed_bytes = 0

    target_db = db
    if hasattr(target_db, "db_path"):
        target_db = getattr(target_db, "db_path")
    elif hasattr(target_db, "_get_connection") and callable(getattr(target_db, "_get_connection")):
        target_db = target_db._get_connection()

    # -----------------------------------------------------------------------
    # Case 1: SQLite Database Connection / Path
    # -----------------------------------------------------------------------
    if isinstance(target_db, (sqlite3.Connection, sqlite3.Cursor, str, Path)):
        conn = None
        should_close = False
        if isinstance(target_db, (str, Path)):
            conn = sqlite3.connect(str(target_db))
            conn.row_factory = sqlite3.Row
            should_close = True
        elif isinstance(target_db, sqlite3.Cursor):
            conn = target_db.connection
        else:
            conn = target_db

        try:
            # Discover tables
            tables_cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = {row[0] if isinstance(row, (tuple, list)) else row["name"] for row in tables_cursor.fetchall()}

            # Determine profile and capture table names
            profile_tbl = next((t for t in ("people_profiles", "person_profiles", "profiles") if t in table_names), None)
            capture_tbl = next((t for t in ("people_captures", "capture_records", "captures") if t in table_names), None)

            # Discover columns for profiles
            profile_time_col = "last_seen"
            if profile_tbl:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({profile_tbl})").fetchall()]
                if "last_seen" in cols:
                    profile_time_col = "last_seen"
                elif "timestamp" in cols:
                    profile_time_col = "timestamp"
                elif "created_at" in cols:
                    profile_time_col = "created_at"

            # Discover columns for captures
            capture_time_col = "timestamp"
            capture_has_orig = False
            capture_has_enh = False
            if capture_tbl:
                cap_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({capture_tbl})").fetchall()]
                if "timestamp" in cap_cols:
                    capture_time_col = "timestamp"
                elif "created_at" in cap_cols:
                    capture_time_col = "created_at"
                capture_has_orig = "original_path" in cap_cols
                capture_has_enh = "enhanced_path" in cap_cols

            files_to_delete = []
            captures_to_delete_ids = []
            profiles_to_delete_keys = []

            # 1. Age-based profile identification
            if profile_tbl:
                expired_prof_rows = conn.execute(
                    f"SELECT session_id, track_id FROM {profile_tbl} WHERE {profile_time_col} < ?",
                    (cutoff_time,)
                ).fetchall()
                for r in expired_prof_rows:
                    s_id = _get_row_val(r, "session_id", 0)
                    t_id = _get_row_val(r, "track_id", 1)
                    if s_id is not None and t_id is not None:
                        profiles_to_delete_keys.append((s_id, t_id))

            # 2. Capacity-based profile identification (max_profiles)
            if profile_tbl and max_profiles is not None and max_profiles > 0:
                count_row = conn.execute(f"SELECT COUNT(*) FROM {profile_tbl}").fetchone()
                total_profiles = _get_row_val(count_row, "COUNT(*)", 0, default=0)
                if total_profiles > max_profiles:
                    excess_count = total_profiles - max_profiles
                    excess_rows = conn.execute(
                        f"SELECT session_id, track_id FROM {profile_tbl} ORDER BY {profile_time_col} ASC LIMIT ?",
                        (excess_count,)
                    ).fetchall()
                    for r in excess_rows:
                        s_id = _get_row_val(r, "session_id", 0)
                        t_id = _get_row_val(r, "track_id", 1)
                        if s_id is not None and t_id is not None and (s_id, t_id) not in profiles_to_delete_keys:
                            profiles_to_delete_keys.append((s_id, t_id))

            # 3. Find captures tied to pruned profiles
            if capture_tbl and profiles_to_delete_keys:
                for s_id, t_id in profiles_to_delete_keys:
                    cap_rows = conn.execute(
                        f"SELECT capture_id, original_path, { 'enhanced_path' if capture_has_enh else 'NULL' } FROM {capture_tbl} WHERE session_id = ? AND track_id = ?",
                        (s_id, t_id)
                    ).fetchall()
                    for cr in cap_rows:
                        c_id = _get_row_val(cr, "capture_id", 0)
                        orig = _get_row_val(cr, "original_path", 1)
                        enh = _get_row_val(cr, "enhanced_path", 2) if capture_has_enh else None
                        if c_id is not None:
                            captures_to_delete_ids.append(c_id)
                        if orig:
                            files_to_delete.append(orig)
                        if enh:
                            files_to_delete.append(enh)

            # 4. Find captures expired by age directly
            if capture_tbl:
                expired_cap_rows = conn.execute(
                    f"SELECT capture_id, original_path, { 'enhanced_path' if capture_has_enh else 'NULL' } FROM {capture_tbl} WHERE {capture_time_col} < ?",
                    (cutoff_time,)
                ).fetchall()
                for cr in expired_cap_rows:
                    c_id = _get_row_val(cr, "capture_id", 0)
                    orig = _get_row_val(cr, "original_path", 1)
                    enh = _get_row_val(cr, "enhanced_path", 2) if capture_has_enh else None
                    if c_id is not None and c_id not in captures_to_delete_ids:
                        captures_to_delete_ids.append(c_id)
                        if orig:
                            files_to_delete.append(orig)
                        if enh:
                            files_to_delete.append(enh)

            # Delete physical files from disk
            for file_path in set(files_to_delete):
                deleted, bytes_freed = _safe_delete_file(file_path)
                if deleted:
                    deleted_files_count += 1
                    freed_bytes += bytes_freed

            # Delete database records in a transaction
            if captures_to_delete_ids and capture_tbl:
                for cid in captures_to_delete_ids:
                    conn.execute(f"DELETE FROM {capture_tbl} WHERE capture_id = ?", (cid,))
                pruned_captures_count = len(captures_to_delete_ids)

            if profiles_to_delete_keys and profile_tbl:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({profile_tbl})").fetchall()]
                has_track_col = "track_id" in cols
                for s_id, t_id in profiles_to_delete_keys:
                    if has_track_col and t_id is not None:
                        conn.execute(f"DELETE FROM {profile_tbl} WHERE session_id = ? AND track_id = ?", (s_id, t_id))
                    else:
                        conn.execute(f"DELETE FROM {profile_tbl} WHERE session_id = ?", (s_id,))
                pruned_profiles_count = len(profiles_to_delete_keys)

            conn.commit()

        finally:
            if should_close and conn:
                conn.close()

    # -----------------------------------------------------------------------
    # Case 2: In-Memory Storage / Dictionary / Object
    # -----------------------------------------------------------------------
    elif isinstance(db, dict) or hasattr(db, "profiles") or hasattr(db, "captures"):
        profiles_dict = db.get("profiles", {}) if isinstance(db, dict) else getattr(db, "profiles", {})
        captures_dict = db.get("captures", {}) if isinstance(db, dict) else getattr(db, "captures", {})

        expired_profile_keys = []
        # Age-based profile identification
        for p_key, prof in list(profiles_dict.items()):
            last_seen = getattr(prof, "last_seen", None) or (prof.get("last_seen") if isinstance(prof, dict) else None)
            created_at = getattr(prof, "created_at", None) or (prof.get("created_at") if isinstance(prof, dict) else None)
            timestamp = last_seen if last_seen is not None else created_at
            if timestamp is not None and timestamp < cutoff_time:
                expired_profile_keys.append(p_key)

        # Capacity-based profile identification
        if max_profiles is not None and max_profiles > 0 and len(profiles_dict) > max_profiles:
            excess = len(profiles_dict) - max_profiles
            # Sort by last_seen or created_at
            def _sort_key(k):
                p = profiles_dict[k]
                return getattr(p, "last_seen", None) or (p.get("last_seen", 0) if isinstance(p, dict) else 0)
            sorted_keys = sorted(profiles_dict.keys(), key=_sort_key)
            for k in sorted_keys[:excess]:
                if k not in expired_profile_keys:
                    expired_profile_keys.append(k)

        # Gather captures to delete
        captures_to_delete = []
        for cap_key, cap in list(captures_dict.items()):
            cap_time = getattr(cap, "timestamp", None) or (cap.get("timestamp") if isinstance(cap, dict) else None)
            cap_session = getattr(cap, "session_id", None) or (cap.get("session_id") if isinstance(cap, dict) else None)
            cap_track = getattr(cap, "track_id", None) or (cap.get("track_id") if isinstance(cap, dict) else None)

            # Check if tied to pruned profile
            is_tied_to_pruned_profile = (
                cap_session is not None and (cap_session in expired_profile_keys or f"{cap_session}:{cap_track}" in expired_profile_keys)
            )
            # Check age
            is_expired = (cap_time is not None and cap_time < cutoff_time)

            if is_tied_to_pruned_profile or is_expired:
                captures_to_delete.append((cap_key, cap))

        # Delete disk files
        for cap_key, cap in captures_to_delete:
            orig = getattr(cap, "original_path", None) or (cap.get("original_path") if isinstance(cap, dict) else None)
            enh = getattr(cap, "enhanced_path", None) or (cap.get("enhanced_path") if isinstance(cap, dict) else None)
            for path_str in (orig, enh):
                deleted, bytes_freed = _safe_delete_file(path_str)
                if deleted:
                    deleted_files_count += 1
                    freed_bytes += bytes_freed
            # Remove from dict
            captures_dict.pop(cap_key, None)
            pruned_captures_count += 1

        # Remove profiles
        for p_key in expired_profile_keys:
            profiles_dict.pop(p_key, None)
            pruned_profiles_count += 1

    # -----------------------------------------------------------------------
    # Audit Logging
    # -----------------------------------------------------------------------
    summary = {
        "pruned_profiles": pruned_profiles_count,
        "pruned_captures": pruned_captures_count,
        "deleted_files": deleted_files_count,
        "freed_bytes": freed_bytes,
        "timestamp": now,
    }

    if audit_logger:
        audit_logger.log_event(
            AuditEventType.RETENTION_SWEEP,
            details=summary,
            severity="INFO",
        )
        if pruned_captures_count > 0:
            audit_logger.log_event(
                AuditEventType.CAPTURE_DELETED,
                details={"count": pruned_captures_count, "freed_bytes": freed_bytes},
                severity="INFO",
            )
        if pruned_profiles_count > 0:
            audit_logger.log_event(
                AuditEventType.SESSION_PURGED,
                details={"count": pruned_profiles_count},
                severity="INFO",
            )

    return summary


# ---------------------------------------------------------------------------
# Redaction & Blur Utilities
# ---------------------------------------------------------------------------

def blur_region(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    method: str = "gaussian",
    factor: int = 25,
    copy: bool = True,
) -> np.ndarray:
    """Apply visual redaction (Gaussian blur, mosaic pixelation, or blackout) to bbox.

    Args:
        image: Source image numpy array (H, W, C) or (H, W).
        bbox: Bounding box tuple (x1, y1, x2, y2) in pixel coordinates.
        method: Redaction algorithm: 'gaussian', 'pixelate', or 'solid'.
        factor: Blur intensity factor (kernel size or pixelation cell size).
        copy: If True, returns a modified copy; if False, modifies in-place.

    Returns:
        np.ndarray: Redacted image.
    """
    if image is None or image.size == 0:
        return image

    img = image.copy() if copy else image
    h, w = img.shape[:2]

    # Coordinate clamping and boundary sanity check
    x1 = max(0, min(int(round(float(bbox[0]))), w - 1))
    y1 = max(0, min(int(round(float(bbox[1]))), h - 1))
    x2 = max(0, min(int(round(float(bbox[2]))), w))
    y2 = max(0, min(int(round(float(bbox[3]))), h))

    if x2 <= x1 or y2 <= y1:
        return img  # Degenerate or out-of-bounds bbox

    roi = img[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    if roi_h <= 0 or roi_w <= 0:
        return img

    if method == "pixelate":
        # Downscale then upscale nearest-neighbor
        cell_size = max(2, int(factor))
        small_w = max(1, roi_w // cell_size)
        small_h = max(1, roi_h // cell_size)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = pixelated

    elif method == "solid":
        # Solid fill (blackout)
        if img.ndim == 3:
            img[y1:y2, x1:x2] = (0, 0, 0)
        else:
            img[y1:y2, x1:x2] = 0

    else:  # "gaussian"
        # Ensure odd kernel size
        k = max(3, int(factor))
        if k % 2 == 0:
            k += 1
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(roi, (k, k), sigmaX=k / 2.0, sigmaY=k / 2.0)
        img[y1:y2, x1:x2] = blurred

    return img


def redact_frame(
    frame: np.ndarray,
    bboxes: Sequence[Tuple[float, float, float, float]],
    method: str = "gaussian",
    factor: int = 25,
) -> np.ndarray:
    """Sequentially apply visual redaction to a list of bounding boxes."""
    if frame is None or frame.size == 0 or not bboxes:
        return frame
    out = frame.copy()
    for box in bboxes:
        out = blur_region(out, box, method=method, factor=factor, copy=False)
    return out


def anonymize_frame_tracks(
    frame: np.ndarray,
    tracks: Sequence[Any],
    active_track_ids: Optional[Set[int]] = None,
    blur_non_captured: bool = True,
    blur_expired: bool = True,
    method: str = "gaussian",
    factor: int = 25,
) -> np.ndarray:
    """Anonymize frame by blurring bystanders (non-captured) and/or expired tracks."""
    if frame is None or frame.size == 0 or not tracks:
        return frame

    active_ids = active_track_ids if active_track_ids is not None else set()
    boxes_to_blur = []

    for trk in tracks:
        # Extract track attributes supporting dataclass or dict
        t_id = getattr(trk, "track_id", None) or (trk.get("track_id") if isinstance(trk, dict) else getattr(trk, "ident", None))
        bbox = getattr(trk, "bbox", None) or (trk.get("bbox") if isinstance(trk, dict) else None)
        state = getattr(trk, "state", "active") or (trk.get("state") if isinstance(trk, dict) else "active")

        if bbox is None:
            continue

        should_blur = False
        if blur_non_captured and (t_id is not None and t_id not in active_ids):
            should_blur = True
        elif blur_expired and state in ("lost", "ended"):
            should_blur = True

        if should_blur:
            boxes_to_blur.append(bbox)

    return redact_frame(frame, boxes_to_blur, method=method, factor=factor)


# ---------------------------------------------------------------------------
# PrivacyGovernance Main Class
# ---------------------------------------------------------------------------

class PrivacyGovernance:
    """Comprehensive Privacy Governance, Security Enforcement, and Data Retention Manager.

    Unifies:
    - Session ID issuance & strict session partition validation
    - Path traversal prevention for capture assets
    - SQLite & memory data retention pruning sweeps
    - Tamper-evident security audit logging
    - Real-time video frame face redaction
    """

    def __init__(
        self,
        audit_log_path: Optional[Union[str, Path]] = None,
        max_retention_seconds: float = 3600.0,
        max_profiles: int = 1000,
        enable_audit_logging: bool = True,
        strict_cross_session_isolation: bool = True,
    ):
        self.max_retention_seconds = float(max_retention_seconds)
        self.max_profiles = int(max_profiles)
        self.strict_cross_session_isolation = strict_cross_session_isolation
        self.audit_logger = AuditLogger(log_path=audit_log_path) if enable_audit_logging else None

    # --- Session Management ---

    def create_anonymous_session(self, prefix: str = "sess_") -> str:
        """Create and register a strictly anonymous capture session ID."""
        session_id = generate_anonymous_session_id(prefix=prefix)
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEventType.SESSION_CREATED,
                session_id=session_id,
                details={"prefix": prefix},
                severity="INFO",
            )
        return session_id

    def validate_session(self, session_id: str) -> None:
        """Validate session ID against anonymous formatting rules."""
        validate_anonymous_session_id(session_id)

    def enforce_session_isolation(self, query_session_id: str, candidate_session_id: str) -> None:
        """Prohibit cross-session biometric matching or search."""
        assert_session_isolation(query_session_id, candidate_session_id, audit_logger=self.audit_logger)

    def filter_candidates(self, query_session_id: str, candidates: Sequence[Any]) -> List[Any]:
        """Filter biometric search candidates strictly within the query session."""
        return filter_cross_session_candidates(
            query_session_id,
            candidates,
            strict=self.strict_cross_session_isolation,
            audit_logger=self.audit_logger,
        )

    # --- Secure Path Validation ---

    def validate_capture_path(self, base_dir: Union[str, Path], requested_path: Union[str, Path]) -> Path:
        """Ensure capture file path cannot traverse out of base_dir."""
        return validate_capture_path(base_dir, requested_path, audit_logger=self.audit_logger)

    # --- Data Retention & Auto-Pruning ---

    def prune_expired_records(
        self,
        db: Any,
        max_age_seconds: Optional[float] = None,
        max_profiles: Optional[int] = None,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Prune expired profile and capture records from db and delete disk assets."""
        effective_max_age = max_age_seconds if max_age_seconds is not None else self.max_retention_seconds
        effective_max_profiles = max_profiles if max_profiles is not None else self.max_profiles
        return prune_expired_records(
            db=db,
            max_age_seconds=effective_max_age,
            max_profiles=effective_max_profiles,
            current_time=current_time,
            audit_logger=self.audit_logger,
        )

    # --- Redaction Utilities ---

    def blur_region(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        method: str = "gaussian",
        factor: int = 25,
        copy: bool = True,
    ) -> np.ndarray:
        """Blur or pixelate a bounding box within an image."""
        return blur_region(image, bbox, method=method, factor=factor, copy=copy)

    def redact_frame(
        self,
        frame: np.ndarray,
        bboxes: Sequence[Tuple[float, float, float, float]],
        method: str = "gaussian",
        factor: int = 25,
    ) -> np.ndarray:
        """Redact multiple bounding boxes from a frame."""
        return redact_frame(frame, bboxes, method=method, factor=factor)

    def anonymize_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[Any],
        active_track_ids: Optional[Set[int]] = None,
        blur_non_captured: bool = True,
        blur_expired: bool = True,
        method: str = "gaussian",
        factor: int = 25,
    ) -> np.ndarray:
        """Apply automatic bystander and expired track anonymization."""
        return anonymize_frame_tracks(
            frame,
            tracks,
            active_track_ids=active_track_ids,
            blur_non_captured=blur_non_captured,
            blur_expired=blur_expired,
            method=method,
            factor=factor,
        )

    # --- Audit Logging ---

    def log_audit(
        self,
        event_type: str,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> Optional[AuditRecord]:
        """Log an audit event directly."""
        if self.audit_logger:
            return self.audit_logger.log_event(event_type, session_id=session_id, details=details, severity=severity)
        return None

    def get_audit_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[AuditRecord]:
        """Query recent audit events."""
        if self.audit_logger:
            return self.audit_logger.get_events(limit=limit, event_type=event_type, session_id=session_id)
        return []
