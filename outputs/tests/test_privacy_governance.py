"""Unit tests for Privacy Governance, Security Controls, and Data Retention."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid

import numpy as np
import pytest

from services.privacy_governance import (
    PrivacyGovernance,
    AuditLogger,
    AuditEventType,
    AuditRecord,
    PathTraversalError,
    CrossSessionMatchingViolation,
    InvalidSessionIdError,
    validate_capture_path,
    generate_anonymous_session_id,
    is_anonymous_session_id,
    validate_anonymous_session_id,
    assert_session_isolation,
    filter_cross_session_candidates,
    prune_expired_records,
    blur_region,
    redact_frame,
    anonymize_frame_tracks,
)
from services.face_contracts import PersonTrack, CaptureRecord, PersonProfile


# ===========================================================================
# 1. Path Traversal Prevention Tests
# ===========================================================================

def test_validate_capture_path_valid(tmp_path):
    base_dir = tmp_path / "captures"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Valid simple filename
    p1 = validate_capture_path(base_dir, "face_001.jpg")
    assert p1 == (base_dir / "face_001.jpg").resolve()

    # Valid subfolder
    p2 = validate_capture_path(base_dir, "sess_123/track_456_frontal.jpg")
    assert p2 == (base_dir / "sess_123" / "track_456_frontal.jpg").resolve()

    # Valid safe relative path
    p3 = validate_capture_path(str(base_dir), "exports/archive.tar.gz")
    assert p3 == (base_dir / "exports" / "archive.tar.gz").resolve()


@pytest.mark.parametrize("malicious_path", [
    "../secret.txt",
    "../../etc/passwd",
    "sess_1/../../shadow",
    "..\\..\\windows\\system32\\cmd.exe",
    "..",
    ".",
    "",
    "   ",
    "/etc/shadow",
    "\\Windows\\System32",
    "C:\\Windows\\System32\\calc.exe",
    "D:\\captures\\external.jpg",
    "\\\\attacker_server\\share\\exploit.jpg",
    "face.jpg\x00.png",
    "\x00evil.jpg",
    "sub/\x00/file.jpg",
    "..%2f..%2fsecret.txt",
    "face%00.jpg",
    "CON",
    "con.jpg",
    "NUL",
    "nul.png",
    "PRN",
    "AUX.jpg",
    "COM1",
    "LPT1",
])
def test_validate_capture_path_blocks_malicious_inputs(tmp_path, malicious_path):
    base_dir = tmp_path / "captures"
    base_dir.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger()

    with pytest.raises((PathTraversalError, ValueError)):
        validate_capture_path(base_dir, malicious_path, audit_logger=logger)

    events = logger.get_events(event_type=AuditEventType.PATH_TRAVERSAL_BLOCKED)
    assert len(events) >= 1
    assert events[0].severity in ("WARNING", "CRITICAL")


def test_validate_capture_path_none_rejected(tmp_path):
    base_dir = tmp_path / "captures"
    base_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PathTraversalError):
        validate_capture_path(base_dir, None)


# ===========================================================================
# 2. Anonymous Session ID Generation & Enforcement Tests
# ===========================================================================

def test_generate_anonymous_session_id():
    # Verify prefix and randomness
    s1 = generate_anonymous_session_id("sess_")
    s2 = generate_anonymous_session_id("sess_")
    assert s1.startswith("sess_")
    assert s2.startswith("sess_")
    assert s1 != s2
    assert len(s1) >= 20

    # Test uniqueness over 100 generations
    ids = {generate_anonymous_session_id() for _ in range(100)}
    assert len(ids) == 100


def test_validate_anonymous_session_id():
    valid_id = generate_anonymous_session_id()
    assert is_anonymous_session_id(valid_id) is True
    validate_anonymous_session_id(valid_id)  # Does not raise

    # Rejection of invalid or PII-laden strings
    invalid_ids = [
        "",
        "short",
        "user_john_doe@example.com",
        "192.168.1.100",
        "admin:password",
        "../../traversal",
        "session\x00null",
        "session with spaces",
        "session!@#$%^&*",
        "sess_" + "a" * 100,  # Exceeds max length
    ]
    for bad_id in invalid_ids:
        assert is_anonymous_session_id(bad_id) is False
        with pytest.raises(InvalidSessionIdError):
            validate_anonymous_session_id(bad_id)


def test_cross_session_matching_strictly_prohibited():
    logger = AuditLogger()
    sess_a = generate_anonymous_session_id()
    sess_b = generate_anonymous_session_id()

    # Same session allowed
    assert_session_isolation(sess_a, sess_a, audit_logger=logger)

    # Cross-session matching MUST raise CrossSessionMatchingViolation
    with pytest.raises(CrossSessionMatchingViolation) as exc_info:
        assert_session_isolation(sess_a, sess_b, audit_logger=logger)

    assert "strictly prohibited" in str(exc_info.value)

    # Verify audit event logged
    events = logger.get_events(event_type=AuditEventType.ACCESS_DENIED)
    assert len(events) == 1
    assert events[0].session_id == sess_a
    assert events[0].details["query_session"] == sess_a
    assert events[0].details["target_session"] == sess_b


def test_filter_cross_session_candidates():
    sess_a = generate_anonymous_session_id()
    sess_b = generate_anonymous_session_id()
    logger = AuditLogger()

    candidates = [
        {"session_id": sess_a, "track_id": 1},
        {"session_id": sess_b, "track_id": 2},
        {"session_id": sess_a, "track_id": 3},
    ]

    # Strict mode raises on foreign candidate
    with pytest.raises(CrossSessionMatchingViolation):
        filter_cross_session_candidates(sess_a, candidates, strict=True, audit_logger=logger)

    # Non-strict mode silently partitions candidates
    allowed = filter_cross_session_candidates(sess_a, candidates, strict=False, audit_logger=logger)
    assert len(allowed) == 2
    assert all(c["session_id"] == sess_a for c in allowed)


# ===========================================================================
# 3. Audit Logger Tests
# ===========================================================================

def test_audit_logger_in_memory_and_queries():
    logger = AuditLogger(max_buffer_size=10)
    sess_id = generate_anonymous_session_id()

    r1 = logger.log_event(AuditEventType.SESSION_CREATED, session_id=sess_id, details={"meta": "test"})
    r2 = logger.log_event(AuditEventType.ACCESS_DENIED, session_id=sess_id, severity="CRITICAL")
    r3 = logger.log_event(AuditEventType.RETENTION_SWEEP, details={"swept": 5})

    assert isinstance(r1, AuditRecord)
    assert r1.event_type == AuditEventType.SESSION_CREATED
    assert r1.severity == "INFO"
    assert r1.session_id == sess_id
    assert r1.iso_timestamp is not None
    assert uuid.UUID(r1.event_id)  # Valid UUID

    # Test filtering by event_type
    events = logger.get_events(event_type=AuditEventType.ACCESS_DENIED)
    assert len(events) == 1
    assert events[0].severity == "CRITICAL"

    # Test filtering by session_id
    sess_events = logger.get_events(session_id=sess_id)
    assert len(sess_events) == 2

    # Test clear
    logger.clear()
    assert len(logger.get_events()) == 0


def test_audit_logger_disk_jsonl_persistence(tmp_path):
    log_file = tmp_path / "audit" / "security_audit.jsonl"
    logger = AuditLogger(log_path=log_file)
    sess_id = generate_anonymous_session_id()

    logger.log_event(AuditEventType.SESSION_CREATED, session_id=sess_id, details={"ip": "anonymous"})
    logger.log_event(AuditEventType.CAPTURE_DELETED, session_id=sess_id, details={"file": "cap1.jpg"})

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    record_0 = json.loads(lines[0])
    assert record_0["event_type"] == AuditEventType.SESSION_CREATED
    assert record_0["session_id"] == sess_id

    record_1 = json.loads(lines[1])
    assert record_1["event_type"] == AuditEventType.CAPTURE_DELETED


def test_audit_logger_thread_safety():
    logger = AuditLogger()
    errors = []

    def worker(worker_id):
        try:
            for i in range(50):
                logger.log_event(AuditEventType.RETENTION_SWEEP, details={"worker": worker_id, "iter": i})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    events = logger.get_events(limit=500)
    assert len(events) == 250


# ===========================================================================
# 4. Data Retention & Auto-Pruning Tests
# ===========================================================================

def _setup_test_sqlite_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE person_profiles (
            session_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            first_seen REAL,
            last_seen REAL,
            duration REAL,
            status TEXT,
            PRIMARY KEY (session_id, track_id)
        );
        CREATE TABLE capture_records (
            capture_id TEXT PRIMARY KEY,
            track_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            category TEXT,
            quality_score REAL,
            original_path TEXT NOT NULL,
            enhanced_path TEXT
        );
    """)
    conn.commit()
    conn.close()


def test_retention_policy_prunes_expired_profiles_and_cleans_files(tmp_path):
    db_file = tmp_path / "people_test.sqlite3"
    captures_dir = tmp_path / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    _setup_test_sqlite_db(db_file)

    now = 1000.0
    sess_id = generate_anonymous_session_id()

    # Create dummy capture files on disk
    old_cap_orig = captures_dir / "cap_old_orig.jpg"
    old_cap_orig.write_bytes(b"OLD_ORIGINAL_IMAGE_DATA_123456789")
    old_cap_enh = captures_dir / "cap_old_enh.jpg"
    old_cap_enh.write_bytes(b"OLD_ENHANCED_IMAGE_DATA_123456789")

    fresh_cap_orig = captures_dir / "cap_fresh_orig.jpg"
    fresh_cap_orig.write_bytes(b"FRESH_ORIGINAL_IMAGE_DATA")

    # Insert into database
    conn = sqlite3.connect(str(db_file))
    # Profile 1: EXPIRED (last_seen = 800.0, age = 200s)
    conn.execute(
        "INSERT INTO person_profiles VALUES (?, ?, ?, ?, ?, ?)",
        (sess_id, 1, 750.0, 800.0, 50.0, "ended")
    )
    # Capture 1: EXPIRED (tied to track 1, timestamp = 780.0)
    conn.execute(
        "INSERT INTO capture_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("cap_old", 1, sess_id, 780.0, "frontal", 85.0, str(old_cap_orig), str(old_cap_enh))
    )

    # Profile 2: FRESH (last_seen = 990.0, age = 10s)
    conn.execute(
        "INSERT INTO person_profiles VALUES (?, ?, ?, ?, ?, ?)",
        (sess_id, 2, 980.0, 990.0, 10.0, "active")
    )
    # Capture 2: FRESH (tied to track 2, timestamp = 985.0)
    conn.execute(
        "INSERT INTO capture_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("cap_fresh", 2, sess_id, 985.0, "frontal", 92.0, str(fresh_cap_orig), None)
    )
    conn.commit()
    conn.close()

    logger = AuditLogger()

    # Sweep with max_age_seconds = 100 (cutoff = 900.0)
    result = prune_expired_records(
        db=db_file,
        max_age_seconds=100.0,
        current_time=now,
        audit_logger=logger,
    )

    assert result["pruned_profiles"] == 1
    assert result["pruned_captures"] == 1
    assert result["deleted_files"] == 2
    assert result["freed_bytes"] > 0

    # Verify physical file deletion
    assert not old_cap_orig.exists()
    assert not old_cap_enh.exists()
    assert fresh_cap_orig.exists()

    # Verify database contents
    conn = sqlite3.connect(str(db_file))
    profiles = conn.execute("SELECT track_id FROM person_profiles").fetchall()
    captures = conn.execute("SELECT capture_id FROM capture_records").fetchall()
    conn.close()

    assert len(profiles) == 1
    assert profiles[0][0] == 2
    assert len(captures) == 1
    assert captures[0][0] == "cap_fresh"

    # Verify audit event
    events = logger.get_events(event_type=AuditEventType.RETENTION_SWEEP)
    assert len(events) == 1
    assert events[0].details["pruned_profiles"] == 1
    assert events[0].details["deleted_files"] == 2


def test_retention_policy_capacity_pruning(tmp_path):
    db_file = tmp_path / "capacity_test.sqlite3"
    captures_dir = tmp_path / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    _setup_test_sqlite_db(db_file)

    now = 1000.0
    sess_id = generate_anonymous_session_id()

    conn = sqlite3.connect(str(db_file))
    # Insert 5 profiles with different last_seen times
    for i in range(1, 6):
        cap_file = captures_dir / f"cap_{i}.jpg"
        cap_file.write_bytes(b"X" * 100)
        conn.execute(
            "INSERT INTO person_profiles VALUES (?, ?, ?, ?, ?, ?)",
            (sess_id, i, now - 50 + i * 5, now - 40 + i * 5, 10.0, "active")
        )
        conn.execute(
            "INSERT INTO capture_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (f"cap_{i}", i, sess_id, now - 40 + i * 5, "frontal", 80.0, str(cap_file), None)
        )
    conn.commit()
    conn.close()

    # Prune with max_profiles = 2 (should prune oldest 3: track 1, 2, 3)
    result = prune_expired_records(
        db=db_file,
        max_age_seconds=3600.0,  # Age alone wouldn't expire them
        max_profiles=2,
        current_time=now,
    )

    assert result["pruned_profiles"] == 3
    assert result["pruned_captures"] == 3
    assert result["deleted_files"] == 3

    # Check database remaining
    conn = sqlite3.connect(str(db_file))
    remaining = [r[0] for r in conn.execute("SELECT track_id FROM person_profiles ORDER BY track_id ASC").fetchall()]
    conn.close()
    assert remaining == [4, 5]


def test_retention_policy_in_memory_dict():
    sess_id = generate_anonymous_session_id()
    now = 500.0

    in_mem_db = {
        "profiles": {
            f"{sess_id}:1": {"session_id": sess_id, "track_id": 1, "last_seen": 200.0},  # Old
            f"{sess_id}:2": {"session_id": sess_id, "track_id": 2, "last_seen": 480.0},  # Fresh
        },
        "captures": {
            "cap1": {"capture_id": "cap1", "session_id": sess_id, "track_id": 1, "timestamp": 200.0},
            "cap2": {"capture_id": "cap2", "session_id": sess_id, "track_id": 2, "timestamp": 480.0},
        }
    }

    result = prune_expired_records(in_mem_db, max_age_seconds=100.0, current_time=now)
    assert result["pruned_profiles"] == 1
    assert result["pruned_captures"] == 1
    assert f"{sess_id}:1" not in in_mem_db["profiles"]
    assert f"{sess_id}:2" in in_mem_db["profiles"]
    assert "cap1" not in in_mem_db["captures"]
    assert "cap2" in in_mem_db["captures"]


# ===========================================================================
# 5. Visual Redaction & Blur Tests
# ===========================================================================

def test_blur_region_gaussian():
    # Create image with high-contrast sharp rectangle
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 255  # White square inside region

    bbox = (20.0, 20.0, 80.0, 80.0)
    blurred = blur_region(img, bbox, method="gaussian", factor=21, copy=True)

    assert blurred.shape == img.shape
    assert blurred.dtype == img.dtype
    # The edges should be smoothed, differing from original
    assert not np.array_equal(blurred[20:80, 20:80], img[20:80, 20:80])
    # Outside ROI should remain unchanged
    assert np.array_equal(blurred[:10, :10], img[:10, :10])


def test_blur_region_pixelate():
    img = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    bbox = (20.0, 20.0, 80.0, 80.0)
    pixelated = blur_region(img, bbox, method="pixelate", factor=10)

    assert pixelated.shape == img.shape
    assert not np.array_equal(pixelated[20:80, 20:80], img[20:80, 20:80])


def test_blur_region_solid():
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    bbox = (10.0, 10.0, 40.0, 40.0)
    blackout = blur_region(img, bbox, method="solid")

    assert np.all(blackout[10:40, 10:40] == 0)
    assert np.all(blackout[:5, :5] == 200)


def test_blur_region_out_of_bounds_safety():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    # Beyond image dimensions
    out1 = blur_region(img, (-50.0, -50.0, 200.0, 200.0))
    assert out1.shape == (50, 50, 3)

    # Inverted bbox (x2 < x1)
    out2 = blur_region(img, (40.0, 40.0, 20.0, 20.0))
    assert np.array_equal(out2, img)


def test_anonymize_frame_tracks():
    frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    # Track 1 is active & captured (keep clear)
    # Track 2 is active but non-captured bystander (should be blurred)
    # Track 3 is lost/expired (should be blurred)
    sess_id = generate_anonymous_session_id()
    t1 = PersonTrack(track_id=1, session_id=sess_id, first_seen=0, last_seen=10, bbox=(10, 10, 30, 30), confidence=0.9, state="active")
    t2 = PersonTrack(track_id=2, session_id=sess_id, first_seen=0, last_seen=10, bbox=(40, 10, 60, 30), confidence=0.9, state="active")
    t3 = PersonTrack(track_id=3, session_id=sess_id, first_seen=0, last_seen=5, bbox=(70, 10, 90, 30), confidence=0.9, state="lost")

    anonymized = anonymize_frame_tracks(
        frame=frame,
        tracks=[t1, t2, t3],
        active_track_ids={1},  # Only track 1 is allowed
        blur_non_captured=True,
        blur_expired=True,
        method="solid",
    )

    # Track 1 area should NOT be blacked out
    assert np.all(anonymized[10:30, 10:30] == 150)
    # Track 2 area SHOULD be blacked out (non-captured)
    assert np.all(anonymized[10:30, 40:60] == 0)
    # Track 3 area SHOULD be blacked out (lost/expired)
    assert np.all(anonymized[10:30, 70:90] == 0)


# ===========================================================================
# 6. PrivacyGovernance Class Integration Test
# ===========================================================================

def test_privacy_governance_full_lifecycle(tmp_path):
    audit_file = tmp_path / "privacy_audit.jsonl"
    gov = PrivacyGovernance(
        audit_log_path=audit_file,
        max_retention_seconds=60.0,
        max_profiles=5,
        enable_audit_logging=True,
    )

    # 1. Create anonymous session
    session_id = gov.create_anonymous_session()
    gov.validate_session(session_id)

    # 2. Path validation
    base_captures = tmp_path / "captures"
    base_captures.mkdir()
    valid_path = gov.validate_capture_path(base_captures, f"{session_id}/face.jpg")
    assert valid_path.parent == (base_captures / session_id).resolve()

    # Malicious path
    with pytest.raises(PathTraversalError):
        gov.validate_capture_path(base_captures, "../../etc/shadow")

    # 3. Biometric isolation
    foreign_session = gov.create_anonymous_session()
    with pytest.raises(CrossSessionMatchingViolation):
        gov.enforce_session_isolation(session_id, foreign_session)

    # 4. Pruning
    db_file = tmp_path / "test_gov.sqlite3"
    _setup_test_sqlite_db(db_file)
    summary = gov.prune_expired_records(db_file)
    assert "pruned_profiles" in summary

    # 5. Check audit events
    events = gov.get_audit_events()
    assert len(events) >= 3
    event_types = {e.event_type for e in events}
    assert AuditEventType.SESSION_CREATED in event_types
    assert AuditEventType.PATH_TRAVERSAL_BLOCKED in event_types
    assert AuditEventType.ACCESS_DENIED in event_types


def test_retention_policy_people_profiles_and_captures_schema(tmp_path):
    """Test pruning with the exact people_profiles and people_captures tables."""
    db_file = tmp_path / "people_repo_test.sqlite3"
    captures_dir = tmp_path / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    conn.executescript("""
        CREATE TABLE people_profiles (
            session_id TEXT PRIMARY KEY,
            track_id INTEGER NOT NULL,
            camera_id TEXT NOT NULL,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            duration REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'active',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE people_captures (
            capture_id TEXT PRIMARY KEY,
            track_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            category TEXT NOT NULL,
            quality_score REAL NOT NULL,
            pose_category TEXT NOT NULL,
            original_path TEXT NOT NULL,
            enhanced_path TEXT,
            created_at REAL NOT NULL
        );
    """)

    now = 2000.0
    sess_old = "cam1:1000:1"
    sess_fresh = "cam1:1990:2"

    cap_old_orig = captures_dir / "cap_old_orig.jpg"
    cap_old_orig.write_bytes(b"OLD_IMG")
    cap_fresh_orig = captures_dir / "cap_fresh_orig.jpg"
    cap_fresh_orig.write_bytes(b"FRESH_IMG")

    conn.execute(
        "INSERT INTO people_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sess_old, 1, "cam1", 1000.0, 1100.0, 100.0, "ended", 1000.0, 1100.0)
    )
    conn.execute(
        "INSERT INTO people_captures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("c_old", 1, sess_old, 1050.0, "frontal", 80.0, "frontal", str(cap_old_orig), None, 1050.0)
    )
    conn.execute(
        "INSERT INTO people_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sess_fresh, 2, "cam1", 1900.0, 1990.0, 90.0, "active", 1900.0, 1990.0)
    )
    conn.execute(
        "INSERT INTO people_captures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("c_fresh", 2, sess_fresh, 1950.0, "frontal", 95.0, "frontal", str(cap_fresh_orig), None, 1950.0)
    )
    conn.commit()
    conn.close()

    result = prune_expired_records(db_file, max_age_seconds=300.0, current_time=now)
    assert result["pruned_profiles"] == 1
    assert result["pruned_captures"] == 1
    assert result["deleted_files"] == 1

    assert not cap_old_orig.exists()
    assert cap_fresh_orig.exists()


def test_anonymous_session_colon_format():
    """Verify camera:epoch:track format is accepted as anonymous session ID."""
    colon_id = "cam1:1729000000:42"
    assert is_anonymous_session_id(colon_id) is True
    validate_anonymous_session_id(colon_id)

