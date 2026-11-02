"""Comprehensive tests for People Facial Capture and Tracking REST API endpoints."""
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pytest
from flask import Flask

from services.face_contracts import (
    AnalysisState,
    CaptureRecord,
    PersonProfile,
    ProcessingMetrics,
)
from services.people_routes import (
    InMemoryPeopleOrchestrator,
    InMemoryProfileRepository,
    init_people_routes,
    people_bp,
    set_orchestrator,
    set_repository,
)


# Minimal valid 1x1 JPEG bytes for testing
MINIMAL_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
)

ENHANCED_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x80\xff\xd9"
)


@pytest.fixture
def test_setup(tmp_path: Path):
    """Fixture providing a configured Flask test app, test_client, repository, and orchestrator."""
    captures_dir = tmp_path / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["CAPTURES_BASE_DIR"] = str(captures_dir.resolve())

    repo = InMemoryProfileRepository(captures_dir=captures_dir)
    orch = InMemoryPeopleOrchestrator(repository=repo)

    # Initialize and register blueprint with injected dependencies
    init_people_routes(app, orchestrator=orch, repository=repo, captures_dir=captures_dir)

    client = app.test_client()
    yield app, client, repo, orch, captures_dir

    # Reset module-level singletons
    set_orchestrator(None)
    set_repository(None)


# ============================================================================
# 1. State and Control Start/Stop Endpoints
# ============================================================================


def test_state_endpoint_idle(test_setup):
    """Verify GET /api/people/state returns initial AnalysisState, counts, and metrics."""
    app, client, repo, orch, _ = test_setup

    res = client.get("/api/people/state")
    assert res.status_code == 200
    data = res.get_json()

    # Core state fields
    assert data["is_running"] is False
    assert data["status_message"] == "Idle"
    assert data["error"] is None
    assert "started_at" in data

    # Structured state & active counts
    assert "state" in data
    assert "active_counts" in data
    assert data["active_counts"]["active_tracks"] == 0
    assert data["active_counts"]["total_profiles"] == 0
    assert data["active_counts"]["total_captures"] == 0

    # Metrics container
    assert "metrics" in data


def test_control_start_and_stop(test_setup):
    """Verify POST /api/people/control start and stop actions."""
    app, client, repo, orch, _ = test_setup

    # Start analysis
    res_start = client.post("/api/people/control", json={"action": "start"})
    assert res_start.status_code == 200
    data_start = res_start.get_json()
    assert data_start["status"] == "ok"
    assert data_start["action"] == "start"
    assert data_start["is_running"] is True
    assert data_start["state"]["is_running"] is True

    # Check updated state
    res_state = client.get("/api/people/state")
    assert res_state.get_json()["is_running"] is True

    # Stop analysis
    res_stop = client.post("/api/people/control", json={"action": "stop"})
    assert res_stop.status_code == 200
    data_stop = res_stop.get_json()
    assert data_stop["status"] == "ok"
    assert data_stop["action"] == "stop"
    assert data_stop["is_running"] is False
    assert data_stop["state"]["is_running"] is False

    # Check state is now stopped
    res_state2 = client.get("/api/people/state")
    assert res_state2.get_json()["is_running"] is False


def test_control_validation_errors(test_setup):
    """Verify POST /api/people/control input validation and error responses."""
    app, client, repo, orch, _ = test_setup

    # Missing / non-JSON body
    res_non_json = client.post("/api/people/control", data="not json", content_type="text/plain")
    assert res_non_json.status_code == 400
    assert "error" in res_non_json.get_json()

    # Empty JSON object
    res_empty = client.post("/api/people/control", json={})
    assert res_empty.status_code == 400
    assert "action" in res_empty.get_json()["error"]

    # Unsupported action
    res_invalid_action = client.post("/api/people/control", json={"action": "restart"})
    assert res_invalid_action.status_code == 400
    assert "action" in res_invalid_action.get_json()["error"]

    # Action of wrong type
    res_num_action = client.post("/api/people/control", json={"action": 123})
    assert res_num_action.status_code == 400


# ============================================================================
# 2. Profile Listing with Pagination and Filters
# ============================================================================


def _create_mock_profile(
    session_id: str,
    track_id: int,
    first_seen: float,
    last_seen: float,
    status: str = "active",
    quality_score: float = 75.0,
    captures_dir: Optional[Path] = None,
) -> PersonProfile:
    """Helper to construct PersonProfile with optional dummy capture records."""
    cap = None
    if captures_dir:
        img_path = str(captures_dir / f"{session_id}_front.jpg")
        cap = CaptureRecord(
            capture_id=f"cap_{session_id}",
            track_id=track_id,
            session_id=session_id,
            timestamp=last_seen,
            category="frontal",
            quality_score=quality_score,
            pose_category="frontal",
            yaw=0.0,
            pitch=0.0,
            original_path=img_path,
            crop_width=120,
            crop_height=120,
        )

    return PersonProfile(
        session_id=session_id,
        track_id=track_id,
        camera_id="cam_main",
        first_seen=first_seen,
        last_seen=last_seen,
        duration=round(last_seen - first_seen, 2),
        status=status,
        best_capture=cap,
        captures_by_category={"frontal": cap} if cap else {},
        capture_history=[cap] if cap else [],
        total_observations=10,
        best_quality_score=quality_score,
    )


def test_profiles_listing_empty(test_setup):
    """Verify GET /api/people/profiles on an empty repository."""
    app, client, repo, orch, _ = test_setup

    res = client.get("/api/people/profiles")
    assert res.status_code == 200
    data = res.get_json()
    assert data["profiles"] == []
    assert data["total"] == 0
    assert data["limit"] == 50
    assert data["offset"] == 0


def test_profiles_listing_with_pagination_and_filters(test_setup):
    """Verify pagination, sorting, and status/quality filtering."""
    app, client, repo, orch, captures_dir = test_setup

    # Seed 6 profiles with varied timestamps and qualities
    profiles = [
        _create_mock_profile("s1", 1, 100.0, 110.0, "active", 85.0),
        _create_mock_profile("s2", 2, 105.0, 120.0, "active", 92.0),
        _create_mock_profile("s3", 3, 110.0, 115.0, "lost", 60.0),
        _create_mock_profile("s4", 4, 115.0, 130.0, "completed", 95.0),
        _create_mock_profile("s5", 5, 120.0, 125.0, "active", 70.0),
        _create_mock_profile("s6", 6, 125.0, 140.0, "completed", 88.0),
    ]
    for p in profiles:
        repo.save_profile(p)

    # Test default listing (returns all 6, total 6)
    res_all = client.get("/api/people/profiles")
    assert res_all.status_code == 200
    data_all = res_all.get_json()
    assert len(data_all["profiles"]) == 6
    assert data_all["total"] == 6

    # Test pagination: limit=2, offset=0
    res_p1 = client.get("/api/people/profiles?limit=2&offset=0")
    data_p1 = res_p1.get_json()
    assert len(data_p1["profiles"]) == 2
    assert data_p1["total"] == 6
    assert data_p1["limit"] == 2
    assert data_p1["offset"] == 0

    # Test pagination: limit=2, offset=2
    res_p2 = client.get("/api/people/profiles?limit=2&offset=2")
    data_p2 = res_p2.get_json()
    assert len(data_p2["profiles"]) == 2
    # Ensure distinct items across pages
    ids_p1 = {p["session_id"] for p in data_p1["profiles"]}
    ids_p2 = {p["session_id"] for p in data_p2["profiles"]}
    assert ids_p1.isdisjoint(ids_p2)

    # Test status filter
    res_active = client.get("/api/people/profiles?status=active")
    data_active = res_active.get_json()
    assert data_active["total"] == 3
    assert all(p["status"] == "active" for p in data_active["profiles"])

    res_completed = client.get("/api/people/profiles?status=completed")
    data_completed = res_completed.get_json()
    assert data_completed["total"] == 2
    assert all(p["status"] == "completed" for p in data_completed["profiles"])

    # Test min_quality filter
    res_high_qual = client.get("/api/people/profiles?min_quality=90.0")
    data_high_qual = res_high_qual.get_json()
    assert data_high_qual["total"] == 2
    assert all(p["best_quality_score"] >= 90.0 for p in data_high_qual["profiles"])

    # Test sort_by=-best_quality_score
    res_sorted_qual = client.get("/api/people/profiles?sort_by=-best_quality_score")
    scores = [p["best_quality_score"] for p in res_sorted_qual.get_json()["profiles"]]
    assert scores == sorted(scores, reverse=True)

    # Test sort_by=first_seen (ascending)
    res_sorted_first = client.get("/api/people/profiles?sort_by=first_seen")
    first_seen_vals = [p["first_seen"] for p in res_sorted_first.get_json()["profiles"]]
    assert first_seen_vals == sorted(first_seen_vals)


def test_profiles_query_validation(test_setup):
    """Verify input validation on /api/people/profiles query parameters."""
    app, client, repo, orch, _ = test_setup

    # Invalid limits
    assert client.get("/api/people/profiles?limit=0").status_code == 400
    assert client.get("/api/people/profiles?limit=-5").status_code == 400
    assert client.get("/api/people/profiles?limit=5000").status_code == 400
    assert client.get("/api/people/profiles?limit=abc").status_code == 400

    # Invalid offsets
    assert client.get("/api/people/profiles?offset=-1").status_code == 400
    assert client.get("/api/people/profiles?offset=xyz").status_code == 400

    # Invalid status
    assert client.get("/api/people/profiles?status=unknown_status").status_code == 400

    # Invalid min_quality
    assert client.get("/api/people/profiles?min_quality=-1.0").status_code == 400
    assert client.get("/api/people/profiles?min_quality=101.0").status_code == 400
    assert client.get("/api/people/profiles?min_quality=bad").status_code == 400

    # Invalid sort_by field
    assert client.get("/api/people/profiles?sort_by=not_a_field").status_code == 400


# ============================================================================
# 3. Single Profile Lookup & Delete
# ============================================================================


def test_single_profile_lookup_and_not_found(test_setup):
    """Verify GET /api/people/profiles/<session_id> for existing and nonexistent profiles."""
    app, client, repo, orch, captures_dir = test_setup

    # 404 on nonexistent profile
    res_missing = client.get("/api/people/profiles/nonexistent_session")
    assert res_missing.status_code == 404
    assert "error" in res_missing.get_json()

    # Create detailed profile with capture history
    profile = _create_mock_profile("sess_101", 101, 200.0, 215.0, "active", 88.5, captures_dir)
    # Add a second capture angle to history
    second_cap = CaptureRecord(
        capture_id="cap_sess_101_left",
        track_id=101,
        session_id="sess_101",
        timestamp=205.0,
        category="left_angle",
        quality_score=82.0,
        pose_category="left_angle",
        yaw=-25.0,
        pitch=2.0,
        original_path=str(captures_dir / "sess_101_left.jpg"),
        crop_width=110,
        crop_height=110,
    )
    profile.captures_by_category["left_angle"] = second_cap
    profile.capture_history.append(second_cap)
    repo.save_profile(profile)

    # 200 on existing profile
    res_found = client.get("/api/people/profiles/sess_101")
    assert res_found.status_code == 200
    data = res_found.get_json()
    assert data["session_id"] == "sess_101"
    assert data["track_id"] == 101
    assert data["best_capture"]["capture_id"] == "cap_sess_101"
    assert "frontal" in data["captures_by_category"]
    assert "left_angle" in data["captures_by_category"]
    assert len(data["capture_history"]) == 2
    assert data["capture_history"][1]["capture_id"] == "cap_sess_101_left"

    # 400 on illegal session_id characters
    res_bad = client.get("/api/people/profiles/..%2Fbad")
    assert res_bad.status_code in (400, 404)


def test_delete_profile(test_setup):
    """Verify DELETE /api/people/profiles/<session_id> removes profile and associated captures."""
    app, client, repo, orch, captures_dir = test_setup

    # Create dummy capture file on disk
    img_file = captures_dir / "sess_del.jpg"
    img_file.write_bytes(MINIMAL_JPEG)

    cap = CaptureRecord(
        capture_id="cap_del",
        track_id=200,
        session_id="sess_del",
        timestamp=10.0,
        category="frontal",
        quality_score=90.0,
        pose_category="frontal",
        yaw=0.0,
        pitch=0.0,
        original_path=str(img_file),
        crop_width=100,
        crop_height=100,
    )
    repo.save_capture(cap)

    profile = PersonProfile(
        session_id="sess_del",
        track_id=200,
        camera_id="cam_main",
        first_seen=10.0,
        last_seen=20.0,
        duration=10.0,
        best_capture=cap,
        captures_by_category={"frontal": cap},
        capture_history=[cap],
    )
    repo.save_profile(profile)

    # Delete profile
    res_del = client.delete("/api/people/profiles/sess_del")
    assert res_del.status_code == 200
    assert res_del.get_json()["deleted"] is True

    # Subsequent lookup should return 404
    assert client.get("/api/people/profiles/sess_del").status_code == 404

    # Deleting again should return 404
    assert client.delete("/api/people/profiles/sess_del").status_code == 404


# ============================================================================
# 4. Image Retrieval and Path Traversal Protection
# ============================================================================


def test_image_retrieval_success(test_setup):
    """Verify GET /api/people/captures/<capture_id>/image serves original and enhanced crops."""
    app, client, repo, orch, captures_dir = test_setup

    # Create physical files in captures dir
    orig_file = captures_dir / "test_orig.jpg"
    enh_file = captures_dir / "test_enh.jpg"
    orig_file.write_bytes(MINIMAL_JPEG)
    enh_file.write_bytes(ENHANCED_JPEG)

    cap = CaptureRecord(
        capture_id="cap_img_1",
        track_id=1,
        session_id="sess_img_1",
        timestamp=50.0,
        category="frontal",
        quality_score=85.0,
        pose_category="frontal",
        yaw=0.0,
        pitch=0.0,
        original_path=str(orig_file),
        crop_width=100,
        crop_height=100,
        enhanced_path=str(enh_file),
        is_enhanced=True,
    )
    repo.save_capture(cap)

    # 1. Fetch original image
    res_orig = client.get("/api/people/captures/cap_img_1/image?type=original")
    assert res_orig.status_code == 200
    assert res_orig.headers["Content-Type"] == "image/jpeg"
    assert "public" in res_orig.headers.get("Cache-Control", "")
    assert "max-age=" in res_orig.headers.get("Cache-Control", "")
    assert "ETag" in res_orig.headers
    assert res_orig.data == MINIMAL_JPEG

    # 2. Fetch enhanced image
    res_enh = client.get("/api/people/captures/cap_img_1/image?type=enhanced")
    assert res_enh.status_code == 200
    assert res_enh.headers["Content-Type"] == "image/jpeg"
    assert res_enh.data == ENHANCED_JPEG


def test_image_retrieval_missing_and_errors(test_setup):
    """Verify 404 and 400 responses on missing captures or invalid requests."""
    app, client, repo, orch, captures_dir = test_setup

    # Capture not found
    assert client.get("/api/people/captures/missing_cap_id/image").status_code == 404

    # Invalid type query parameter
    assert client.get("/api/people/captures/cap_1/image?type=unsupported").status_code == 400

    # Enhanced requested but not available
    orig_file = captures_dir / "orig_only.jpg"
    orig_file.write_bytes(MINIMAL_JPEG)
    cap_no_enh = CaptureRecord(
        capture_id="cap_no_enh",
        track_id=2,
        session_id="sess_2",
        timestamp=60.0,
        category="frontal",
        quality_score=80.0,
        pose_category="frontal",
        yaw=0.0,
        pitch=0.0,
        original_path=str(orig_file),
        crop_width=100,
        crop_height=100,
        enhanced_path=None,
        is_enhanced=False,
    )
    repo.save_capture(cap_no_enh)
    res_enh = client.get("/api/people/captures/cap_no_enh/image?type=enhanced")
    assert res_enh.status_code == 404

    # File missing on disk
    cap_file_missing = CaptureRecord(
        capture_id="cap_file_missing",
        track_id=3,
        session_id="sess_3",
        timestamp=70.0,
        category="frontal",
        quality_score=80.0,
        pose_category="frontal",
        yaw=0.0,
        pitch=0.0,
        original_path=str(captures_dir / "does_not_exist.jpg"),
        crop_width=100,
        crop_height=100,
    )
    repo.save_capture(cap_file_missing)
    assert client.get("/api/people/captures/cap_file_missing/image").status_code == 404


def test_image_path_traversal_security(test_setup, tmp_path: Path):
    """Verify that path traversal attempts in capture_id or file paths are rejected."""
    app, client, repo, orch, captures_dir = test_setup

    # Secret file outside captures directory
    secret_file = tmp_path / "system_secret.txt"
    secret_file.write_text("SUPER_SECRET_TOKEN")

    # 1. Path traversal in capture_id
    res_traversal_id = client.get("/api/people/captures/..%2f..%2fsecret/image")
    assert res_traversal_id.status_code in (400, 404)

    res_traversal_dots = client.get("/api/people/captures/../secret/image")
    assert res_traversal_dots.status_code in (400, 404)

    # 2. CaptureRecord pointing outside captures base directory
    cap_malicious = CaptureRecord(
        capture_id="cap_malicious",
        track_id=99,
        session_id="sess_99",
        timestamp=99.0,
        category="frontal",
        quality_score=99.0,
        pose_category="frontal",
        yaw=0.0,
        pitch=0.0,
        original_path=str(secret_file),
        crop_width=100,
        crop_height=100,
    )
    repo.save_capture(cap_malicious)

    res_forbidden = client.get("/api/people/captures/cap_malicious/image")
    # Must be rejected as 403 Forbidden or 400 Bad Request
    assert res_forbidden.status_code in (400, 403)
    assert "traversal" in res_forbidden.get_json()["error"].lower() or "denied" in res_forbidden.get_json()["error"].lower()

    # 3. CaptureRecord with relative traversal path
    cap_rel_traversal = CaptureRecord(
        capture_id="cap_rel_traversal",
        track_id=98,
        session_id="sess_98",
        timestamp=98.0,
        category="frontal",
        quality_score=98.0,
        pose_category="frontal",
        yaw=0.0,
        pitch=0.0,
        original_path="../system_secret.txt",
        crop_width=100,
        crop_height=100,
    )
    repo.save_capture(cap_rel_traversal)

    res_rel_forbidden = client.get("/api/people/captures/cap_rel_traversal/image")
    assert res_rel_forbidden.status_code in (400, 403)


# ============================================================================
# 5. Metrics & Configuration Endpoints
# ============================================================================


def test_metrics_endpoint(test_setup):
    """Verify GET /api/people/metrics latency, FPS, memory, and pipeline stats."""
    app, client, repo, orch, _ = test_setup

    # Set mock metrics on orchestrator
    orch.metrics = ProcessingMetrics(
        frame_sequence=150,
        acquisition_fps=29.9,
        detection_ms=12.5,
        tracking_ms=4.2,
        face_detection_ms=8.1,
        quality_eval_ms=2.3,
        enhancement_ms=5.0,
        active_tracks=3,
        queue_depth=1,
        dropped_frames=0,
    )

    res = client.get("/api/people/metrics")
    assert res.status_code == 200
    data = res.get_json()

    # Latency structure
    assert "latency_ms" in data
    assert data["latency_ms"]["detection"] == 12.5
    assert data["latency_ms"]["tracking"] == 4.2
    assert data["latency_ms"]["face_detection"] == 8.1
    assert data["latency_ms"]["quality_eval"] == 2.3
    assert data["latency_ms"]["enhancement"] == 5.0
    assert data["latency_ms"]["total_pipeline_ms"] == round(12.5 + 4.2 + 8.1 + 2.3 + 5.0, 2)

    # FPS structure
    assert "fps" in data
    assert data["fps"]["acquisition_fps"] == 29.9
    assert data["fps"]["processing_fps"] > 0

    # Pipeline structure
    assert "pipeline" in data
    assert data["pipeline"]["frame_sequence"] == 150
    assert data["pipeline"]["active_tracks"] == 3

    # Memory structure
    assert "memory" in data
    assert "rss_mb" in data["memory"]
    assert "vms_mb" in data["memory"]
    assert "percent" in data["memory"]


def test_config_endpoint_success_and_validation(test_setup):
    """Verify POST /api/people/config updates settings and validates parameters."""
    app, client, repo, orch, _ = test_setup

    # 1. Valid configuration update
    valid_payload = {
        "face_confidence_threshold": 0.85,
        "quality_threshold": 72.0,
        "enhancement_enabled": False,
        "max_captures_per_person": 8,
    }
    res_update = client.post("/api/people/config", json=valid_payload)
    assert res_update.status_code == 200
    data_update = res_update.get_json()
    assert data_update["status"] == "ok"
    assert data_update["config"]["face_confidence_threshold"] == 0.85
    assert data_update["config"]["quality_threshold"] == 72.0
    assert data_update["config"]["enhancement_enabled"] is False
    assert data_update["config"]["max_captures_per_person"] == 8

    # Check orchestrator holds the updated config
    assert orch.get_config()["face_confidence_threshold"] == 0.85

    # 2. Validation: Empty or non-JSON payload
    assert client.post("/api/people/config", json={}).status_code == 400
    assert client.post("/api/people/config", data="plain", content_type="text/plain").status_code == 400

    # 3. Validation: Confidence thresholds (must be 0.0 to 1.0 numeric float)
    assert client.post("/api/people/config", json={"face_confidence_threshold": 1.5}).status_code == 400
    assert client.post("/api/people/config", json={"face_confidence_threshold": -0.1}).status_code == 400
    assert client.post("/api/people/config", json={"face_confidence_threshold": "high"}).status_code == 400
    assert client.post("/api/people/config", json={"face_confidence_threshold": True}).status_code == 400

    # 4. Validation: Quality thresholds (must be 0.0 to 100.0 numeric float)
    assert client.post("/api/people/config", json={"quality_threshold": -1.0}).status_code == 400
    assert client.post("/api/people/config", json={"quality_threshold": 105.0}).status_code == 400
    assert client.post("/api/people/config", json={"quality_threshold": "good"}).status_code == 400

    # 5. Validation: Enhancement toggle (must be boolean)
    assert client.post("/api/people/config", json={"enhancement_enabled": "yes"}).status_code == 400
    assert client.post("/api/people/config", json={"enhancement_enabled": 1}).status_code == 400

    # 6. Validation: Positive integer settings
    assert client.post("/api/people/config", json={"max_captures_per_person": 0}).status_code == 400
    assert client.post("/api/people/config", json={"max_captures_per_person": -3}).status_code == 400
    assert client.post("/api/people/config", json={"max_captures_per_person": 3.5}).status_code == 400


# ============================================================================
# 6. Dependency Injection & Mocking Verification
# ============================================================================


def test_dependency_injection_with_custom_mocks():
    """Verify that custom external repository and orchestrator mocks can be injected."""
    class CustomMockRepository:
        def __init__(self):
            self.calls = []

        def list_profiles(self, **kwargs):
            self.calls.append(("list_profiles", kwargs))
            return [], 0

        def get_profile(self, session_id):
            self.calls.append(("get_profile", session_id))
            return None

    class CustomMockOrchestrator:
        def __init__(self):
            self.state = AnalysisState(status_message="CustomMockRunning", is_running=True)

        def get_state(self):
            return self.state

        def get_metrics(self):
            return ProcessingMetrics(acquisition_fps=45.0)

    app = Flask(__name__)
    mock_repo = CustomMockRepository()
    mock_orch = CustomMockOrchestrator()

    init_people_routes(app, orchestrator=mock_orch, repository=mock_repo)

    client = app.test_client()
    res_state = client.get("/api/people/state")
    assert res_state.status_code == 200
    assert res_state.get_json()["status_message"] == "CustomMockRunning"
    assert res_state.get_json()["is_running"] is True

    res_profiles = client.get("/api/people/profiles")
    assert res_profiles.status_code == 200
    assert len(mock_repo.calls) == 1
    assert mock_repo.calls[0][0] == "list_profiles"


def test_interoperability_with_sqlite_profile_repository(tmp_path: Path):
    """Verify interoperability with the concrete SQLite-backed ProfileRepository."""
    try:
        from services.profile_repository import ProfileRepository
    except ImportError:
        pytest.skip("ProfileRepository not available")

    db_file = tmp_path / "people_test.sqlite3"
    captures_dir = tmp_path / "captures_sqlite"
    captures_dir.mkdir(parents=True, exist_ok=True)

    repo = ProfileRepository(db_path=db_file, storage_dir=captures_dir)
    orch = InMemoryPeopleOrchestrator(repository=repo)

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["CAPTURES_BASE_DIR"] = str(captures_dir.resolve())
    init_people_routes(app, orchestrator=orch, repository=repo, captures_dir=captures_dir)
    client = app.test_client()

    # Save a profile using the SQLite repository
    profile = PersonProfile(
        session_id="cam1:100:1",
        track_id=1,
        camera_id="cam1",
        first_seen=10.0,
        last_seen=25.0,
        duration=15.0,
        status="active",
        best_quality_score=91.5,
    )
    repo.save_profile(profile)

    # List profiles via API
    res_list = client.get("/api/people/profiles")
    assert res_list.status_code == 200
    data_list = res_list.get_json()
    assert data_list["total"] == 1
    assert len(data_list["profiles"]) == 1
    assert data_list["profiles"][0]["session_id"] == "cam1:100:1"

    # Single profile lookup via API
    res_single = client.get("/api/people/profiles/cam1:100:1")
    assert res_single.status_code == 200
    assert res_single.get_json()["session_id"] == "cam1:100:1"

    # Delete profile via API
    res_del = client.delete("/api/people/profiles/cam1:100:1")
    assert res_del.status_code == 200
    assert res_del.get_json()["deleted"] is True

    # Check 404 after deletion
    assert client.get("/api/people/profiles/cam1:100:1").status_code == 404

    repo.close()


def test_interoperability_with_analysis_lifecycle_controller():
    """Verify interoperability with the concrete AnalysisLifecycleController."""
    try:
        from services.analysis_lifecycle import AnalysisLifecycleController
    except ImportError:
        pytest.skip("AnalysisLifecycleController not available")

    controller = AnalysisLifecycleController()
    app = Flask(__name__)
    app.config["TESTING"] = True
    init_people_routes(app, orchestrator=controller)
    client = app.test_client()

    # Initial state
    res_init = client.get("/api/people/state")
    assert res_init.status_code == 200
    assert res_init.get_json()["is_running"] is False

    # Start analysis
    res_start = client.post("/api/people/control", json={"action": "start"})
    assert res_start.status_code == 200
    assert res_start.get_json()["is_running"] is True

    # Confirm running via state
    res_running = client.get("/api/people/state")
    assert res_running.status_code == 200
    assert res_running.get_json()["is_running"] is True

    # Stop analysis
    res_stop = client.post("/api/people/control", json={"action": "stop"})
    assert res_stop.status_code == 200
    assert res_stop.get_json()["is_running"] is False

    # Confirm stopped via state
    res_stopped = client.get("/api/people/state")
    assert res_stopped.status_code == 200
    assert res_stopped.get_json()["is_running"] is False


