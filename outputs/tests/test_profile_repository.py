"""Comprehensive unit tests for ProfileRepository.

Verifies:
- Profile creation, update, and retrieval
- CaptureRecord persistence and association with profile
- Multi-angle capture grouping and quality score upgrading
- Path validation, directory traversal prevention, and atomic disk writes
- Listing with filtering, search, sorting, and pagination
- System aggregate statistics
- Cascade deletion removing both DB records and disk files
- Retention pruning by age and top N rank
- Concurrent multi-threaded safety with SQLite and disk storage
"""
from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from services.face_contracts import CaptureCandidate, CaptureRecord, FacePose, PersonProfile
from services.profile_repository import ProfileRepository


@pytest.fixture
def repo(tmp_path: Path):
    """Fixture providing isolated ProfileRepository with temporary DB and storage folder."""
    db_file = tmp_path / "traffic_test.sqlite3"
    storage_folder = tmp_path / "captures"
    repository = ProfileRepository(db_path=db_file, storage_dir=storage_folder)
    yield repository
    repository.close()


def create_dummy_crop(width: int = 120, height: int = 140, color: tuple = (100, 150, 200)) -> np.ndarray:
    """Generate a synthetic test image crop array."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    cv2.circle(img, (width // 2, height // 2), 20, (255, 255, 255), -1)
    return img


# =============================================================================
# 1. Profile Creation, Update & Retrieval
# =============================================================================

def test_profile_lifecycle(repo: ProfileRepository):
    session_id = repo.format_session_id("cam_gate", "session_100", 42)
    assert session_id == "cam_gate:session_100:42"

    cam, epoch, track_id = repo.parse_session_id(session_id)
    assert cam == "cam_gate"
    assert epoch == "session_100"
    assert track_id == 42

    # Create profile
    profile = PersonProfile(
        session_id=session_id,
        track_id=42,
        camera_id="cam_gate",
        first_seen=100.0,
        last_seen=100.0,
        duration=0.0,
        status="active",
        metadata={"age_group": "adult", "glasses": False},
    )
    saved = repo.save_profile(profile)
    assert saved.session_id == session_id

    # Retrieve
    retrieved = repo.get_profile(session_id)
    assert retrieved is not None
    assert retrieved.session_id == session_id
    assert retrieved.track_id == 42
    assert retrieved.camera_id == "cam_gate"
    assert retrieved.first_seen == 100.0
    assert retrieved.last_seen == 100.0
    assert retrieved.duration == 0.0
    assert retrieved.status == "active"
    assert retrieved.metadata.get("age_group") == "adult"
    assert retrieved.metadata.get("glasses") is False

    # Update status and timestamps
    ok = repo.update_profile_status(session_id, status="completed", last_seen=125.5)
    assert ok is True

    updated = repo.get_profile(session_id)
    assert updated is not None
    assert updated.status == "completed"
    assert updated.last_seen == 125.5
    assert updated.duration == 25.5

    # Test get_or_create_profile
    existing = repo.get_or_create_profile(session_id, track_id=42, camera_id="cam_gate")
    assert existing.session_id == session_id
    assert existing.status == "completed"

    new_sess = "cam_gate:session_100:99"
    created = repo.get_or_create_profile(new_sess, track_id=99, camera_id="cam_gate", timestamp=200.0)
    assert created.session_id == new_sess
    assert created.first_seen == 200.0


# =============================================================================
# 2. Capture Persistence & Profile Association
# =============================================================================

def test_capture_persistence_and_association(repo: ProfileRepository):
    session_id = "cam_lobby:sess_1:10"
    crop_orig = create_dummy_crop(100, 120, (50, 100, 150))
    crop_enh = create_dummy_crop(200, 240, (150, 100, 50))

    capture = CaptureRecord(
        capture_id="cap_001",
        track_id=10,
        session_id=session_id,
        timestamp=105.0,
        category="frontal",
        quality_score=85.5,
        pose_category="frontal",
        yaw=2.5,
        pitch=-1.0,
        original_path="",
        crop_width=0,
        crop_height=0,
    )

    # Save capture with original crop
    saved_cap = repo.save_capture(capture, original_image=crop_orig)
    assert saved_cap.capture_id == "cap_001"
    assert saved_cap.crop_width == 100
    assert saved_cap.crop_height == 120
    assert saved_cap.file_size_bytes > 0
    assert Path(saved_cap.original_path).exists()

    # Verify disk file is directly accessible
    orig_path = repo.get_capture_image_path("cap_001", image_type="original")
    assert orig_path is not None
    assert orig_path.exists()

    orig_bytes = repo.get_capture_image_bytes("cap_001", image_type="original")
    assert orig_bytes is not None and len(orig_bytes) > 0

    orig_arr = repo.get_capture_image_array("cap_001", image_type="original")
    assert orig_arr is not None
    assert orig_arr.shape == (120, 100, 3)

    # Attach enhanced crop
    enhanced = repo.attach_enhanced_image("cap_001", crop_enh, enhancement_method="gfpgan")
    assert enhanced is not None
    assert enhanced.is_enhanced is True
    assert enhanced.enhancement_method == "gfpgan"
    assert Path(enhanced.enhanced_path).exists()

    enh_arr = repo.get_capture_image_array("cap_001", image_type="enhanced")
    assert enh_arr is not None
    assert enh_arr.shape == (240, 200, 3)

    # Check profile association
    prof = repo.get_profile(session_id)
    assert prof is not None
    assert prof.best_capture is not None
    assert prof.best_capture.capture_id == "cap_001"
    assert prof.best_quality_score == 85.5
    assert "frontal" in prof.captures_by_category
    assert len(prof.capture_history) == 1

    # Test candidate persistence
    candidate = CaptureCandidate(
        candidate_id="cand_002",
        track_id=10,
        timestamp=110.0,
        category="high_res",
        quality_score=89.0,
        pose=FacePose(yaw=1.0, pitch=0.5, roll=0.0, category="frontal"),
        crop_original=create_dummy_crop(150, 150),
        frame_sequence=45,
        bbox_in_frame=(10.0, 10.0, 160.0, 160.0),
    )
    saved_cand = repo.save_candidate(session_id, candidate)
    assert saved_cand.capture_id == "cand_002"
    assert saved_cand.category == "high_res"
    assert saved_cand.quality_score == 89.0

    prof2 = repo.get_profile(session_id)
    assert prof2 is not None
    assert prof2.best_capture.capture_id == "cand_002"
    assert prof2.best_quality_score == 89.0
    assert len(prof2.capture_history) == 2


# =============================================================================
# 3. Multi-Angle Captures & Quality Upgrade
# =============================================================================

def test_multi_angle_and_quality_upgrade(repo: ProfileRepository):
    session_id = "cam_test:s1:77"
    crop = create_dummy_crop(80, 80)

    # Add 3 captures with different angles
    cap_frontal = CaptureRecord(
        capture_id="cap_f1", track_id=77, session_id=session_id, timestamp=10.0,
        category="frontal", quality_score=72.0, pose_category="frontal", yaw=0.0, pitch=0.0,
        original_path="", crop_width=80, crop_height=80,
    )
    cap_left = CaptureRecord(
        capture_id="cap_l1", track_id=77, session_id=session_id, timestamp=12.0,
        category="left_angle", quality_score=78.0, pose_category="left_angle", yaw=-25.0, pitch=2.0,
        original_path="", crop_width=80, crop_height=80,
    )
    cap_right = CaptureRecord(
        capture_id="cap_r1", track_id=77, session_id=session_id, timestamp=14.0,
        category="right_angle", quality_score=68.0, pose_category="right_angle", yaw=25.0, pitch=-2.0,
        original_path="", crop_width=80, crop_height=80,
    )

    repo.save_capture(cap_frontal, original_image=crop)
    repo.save_capture(cap_left, original_image=crop)
    repo.save_capture(cap_right, original_image=crop)

    prof = repo.get_profile(session_id)
    assert prof is not None
    assert len(prof.captures_by_category) == 3
    assert prof.captures_by_category["frontal"].capture_id == "cap_f1"
    assert prof.captures_by_category["left_angle"].capture_id == "cap_l1"
    assert prof.captures_by_category["right_angle"].capture_id == "cap_r1"
    # cap_l1 has highest score so far (78.0)
    assert prof.best_capture.capture_id == "cap_l1"
    assert prof.best_quality_score == 78.0

    # Add a superior frontal capture (quality 94.0)
    cap_frontal_hq = CaptureRecord(
        capture_id="cap_f2", track_id=77, session_id=session_id, timestamp=16.0,
        category="frontal", quality_score=94.0, pose_category="frontal", yaw=1.0, pitch=0.0,
        original_path="", crop_width=80, crop_height=80,
    )
    repo.save_capture(cap_frontal_hq, original_image=crop)

    prof_updated = repo.get_profile(session_id)
    assert prof_updated is not None
    # Frontal should be upgraded to cap_f2
    assert prof_updated.captures_by_category["frontal"].capture_id == "cap_f2"
    assert prof_updated.captures_by_category["frontal"].quality_score == 94.0
    # Overall best should now be cap_f2
    assert prof_updated.best_capture.capture_id == "cap_f2"
    assert prof_updated.best_quality_score == 94.0
    assert len(prof_updated.capture_history) == 4


# =============================================================================
# 4. Path Validation & Directory Traversal Security
# =============================================================================

def test_disk_path_validation_and_security(repo: ProfileRepository):
    # Safe sanitization of directory traversal session IDs
    unsafe_session = "../../../etc/passwd"
    sanitized = repo.sanitize_session_id(unsafe_session)
    assert ".." not in sanitized
    assert "/" not in sanitized
    assert "\\" not in sanitized

    session_dir = repo.get_session_dir(unsafe_session)
    assert repo.storage_dir in session_dir.parents or session_dir.parent == repo.storage_dir

    # Filename traversal attempt
    unsafe_file = "../../secret.txt"
    sanitized_file = repo.sanitize_filename(unsafe_file)
    assert ".." not in sanitized_file

    # validate_safe_path
    safe_path = repo.storage_dir / "safe_file.jpg"
    assert repo.validate_safe_path(safe_path) == safe_path.resolve()

    outside_path = Path("/etc/shadow") if not hasattr(outside_path := Path("C:/Windows/System32"), "exists") else outside_path
    with pytest.raises(ValueError, match="outside allowed storage"):
        repo.validate_safe_path(outside_path)

    # Atomic write cleans up temporary files
    crop = create_dummy_crop(50, 50)
    written = repo.write_image_atomic("secure_sess", "test.jpg", crop)
    assert written.exists()
    assert written.name == "test.jpg"
    # Ensure no lingering .tmp files in folder
    tmp_files = list(written.parent.glob("*.tmp"))
    assert len(tmp_files) == 0


# =============================================================================
# 5. Listing, Filtering, Sorting & Pagination
# =============================================================================

def test_list_profiles_filtering_and_sorting(repo: ProfileRepository):
    crop = create_dummy_crop(60, 60)
    cameras = ["cam_north", "cam_south"]
    statuses = ["active", "lost", "completed"]

    for i in range(12):
        cam = cameras[i % 2]
        status = statuses[i % 3]
        session_id = f"{cam}:epoch1:{i + 1}"
        prof = PersonProfile(
            session_id=session_id,
            track_id=i + 1,
            camera_id=cam,
            first_seen=100.0 + i * 10,
            last_seen=100.0 + i * 10 + (i * 2),
            duration=float(i * 2),
            status=status,
            total_observations=i + 1,
            best_quality_score=60.0 + (i * 3),
        )
        repo.save_profile(prof)

        # Add capture for each
        cap = CaptureRecord(
            capture_id=f"cap_list_{i}",
            track_id=i + 1,
            session_id=session_id,
            timestamp=prof.last_seen,
            category="overall",
            quality_score=prof.best_quality_score,
            pose_category="frontal",
            yaw=0.0,
            pitch=0.0,
            original_path="",
            crop_width=60,
            crop_height=60,
        )
        repo.save_capture(cap, original_image=crop)

    # 1. Total count
    assert repo.count_profiles() == 12

    # 2. Sort by recent (last_seen desc)
    profiles_recent = repo.list_profiles(page=1, page_size=12, sort_by="recent", order="desc")
    assert len(profiles_recent) == 12
    assert profiles_recent[0].last_seen >= profiles_recent[-1].last_seen

    # 3. Sort by duration asc
    profiles_duration = repo.list_profiles(page=1, page_size=12, sort_by="duration", order="asc")
    assert profiles_duration[0].duration <= profiles_duration[-1].duration

    # 4. Sort by quality desc
    profiles_qual = repo.list_profiles(page=1, page_size=12, sort_by="quality", order="desc")
    assert profiles_qual[0].best_quality_score >= profiles_qual[-1].best_quality_score

    # 5. Filter by camera
    north_profiles = repo.list_profiles(camera_id="cam_north")
    assert all(p.camera_id == "cam_north" for p in north_profiles)
    assert len(north_profiles) == 6

    # 6. Filter by status
    completed_profiles = repo.list_profiles(status="completed")
    assert all(p.status == "completed" for p in completed_profiles)
    assert len(completed_profiles) == 4

    # 7. Filter by min_quality
    hq_profiles = repo.list_profiles(min_quality=80.0)
    assert all(p.best_quality_score >= 80.0 for p in hq_profiles)

    # 8. Search by camera substring
    search_cam = repo.list_profiles(search="south")
    assert len(search_cam) == 6

    # 9. Search by numeric track_id
    search_track = repo.list_profiles(search="5")
    assert any(p.track_id == 5 for p in search_track)

    # 10. Pagination
    p1 = repo.list_profiles(page=1, page_size=5, sort_by="recent", order="desc")
    p2 = repo.list_profiles(page=2, page_size=5, sort_by="recent", order="desc")
    p3 = repo.list_profiles(page=3, page_size=5, sort_by="recent", order="desc")
    assert len(p1) == 5
    assert len(p2) == 5
    assert len(p3) == 2
    # Ensure disjoint sets
    s1 = {p.session_id for p in p1}
    s2 = {p.session_id for p in p2}
    assert s1.isdisjoint(s2)

    # 11. Pagination response envelope
    paginated_resp = repo.get_profiles_paginated(page=2, page_size=5)
    assert paginated_resp["total"] == 12
    assert paginated_resp["page"] == 2
    assert paginated_resp["page_size"] == 5
    assert paginated_resp["total_pages"] == 3
    assert len(paginated_resp["items"]) == 5

    # 12. Captures listing
    caps = repo.list_captures(limit=5)
    assert len(caps) == 5


# =============================================================================
# 6. Statistics
# =============================================================================

def test_statistics(repo: ProfileRepository):
    crop = create_dummy_crop(50, 50)
    repo.save_profile(PersonProfile("cam:1:1", 1, "cam", 10.0, 20.0, 10.0, "active", total_observations=3))
    repo.save_profile(PersonProfile("cam:1:2", 2, "cam", 10.0, 30.0, 20.0, "lost", total_observations=2))
    repo.save_profile(PersonProfile("cam:1:3", 3, "cam", 10.0, 40.0, 30.0, "completed", total_observations=5))

    cap1 = CaptureRecord("cap_s1", 1, "cam:1:1", 15.0, "frontal", 80.0, "frontal", 0.0, 0.0, "", 50, 50)
    cap2 = CaptureRecord("cap_s2", 2, "cam:1:2", 25.0, "left_angle", 90.0, "left_angle", -20.0, 0.0, "", 50, 50)
    repo.save_capture(cap1, original_image=crop)
    repo.save_capture(cap2, original_image=crop)
    repo.attach_enhanced_image("cap_s1", crop)

    stats = repo.get_statistics()
    assert stats["total_profiles"] == 3
    assert stats["active_tracks"] == 1
    assert stats["lost_tracks"] == 1
    assert stats["completed_tracks"] == 1
    assert stats["saved_captures"] == 2
    assert stats["enhanced_captures"] == 1
    assert stats["avg_quality_score"] == 85.0
    assert stats["avg_duration"] == 20.0
    # Each save_capture increments observations (3 + 1 + 2 + 1 + 5 = 12)
    assert stats["total_observations"] == 12
    assert stats["captures_by_category"].get("frontal") == 1
    assert stats["captures_by_category"].get("left_angle") == 1


# =============================================================================
# 7. Cascade Deletion (Database & Filesystem)
# =============================================================================

def test_cascade_deletion(repo: ProfileRepository):
    session_id = "cam_del:sess_1:88"
    crop = create_dummy_crop(100, 100)

    cap1 = CaptureRecord("cap_d1", 88, session_id, 10.0, "frontal", 80.0, "frontal", 0.0, 0.0, "", 100, 100)
    cap2 = CaptureRecord("cap_d2", 88, session_id, 12.0, "high_res", 92.0, "frontal", 0.0, 0.0, "", 100, 100)

    repo.save_capture(cap1, original_image=crop)
    repo.save_capture(cap2, original_image=crop)
    repo.attach_enhanced_image("cap_d1", crop)

    # Verify files exist on disk
    path_orig1 = repo.get_capture_image_path("cap_d1", "original")
    path_enh1 = repo.get_capture_image_path("cap_d1", "enhanced")
    path_orig2 = repo.get_capture_image_path("cap_d2", "original")
    session_dir = repo.get_session_dir(session_id)

    assert path_orig1 is not None and path_orig1.exists()
    assert path_enh1 is not None and path_enh1.exists()
    assert path_orig2 is not None and path_orig2.exists()
    assert session_dir.exists()

    # 1. Delete single capture cap_d2 (which was the best capture)
    ok_del_cap = repo.delete_capture("cap_d2")
    assert ok_del_cap is True
    assert not path_orig2.exists()
    assert repo.get_capture("cap_d2") is None

    # Profile should now fall back to cap_d1 as best
    prof_after_cap_del = repo.get_profile(session_id)
    assert prof_after_cap_del is not None
    assert prof_after_cap_del.best_capture.capture_id == "cap_d1"
    assert prof_after_cap_del.best_quality_score == 80.0

    # 2. Delete entire profile
    ok_del_prof = repo.delete_profile(session_id)
    assert ok_del_prof is True

    # DB records gone
    assert repo.get_profile(session_id) is None
    assert repo.get_capture("cap_d1") is None

    # Disk folder and files completely cleaned up
    assert not path_orig1.exists()
    assert not path_enh1.exists()
    assert not session_dir.exists()


# =============================================================================
# 8. Retention Pruning
# =============================================================================

def test_pruning_retention(repo: ProfileRepository):
    crop = create_dummy_crop(50, 50)
    now = time.time()

    # 1. Active profile (old)
    p_act = PersonProfile("p_act", 1, "c", now - 5000, now - 4000, 1000, "active", best_quality_score=70.0)
    repo.save_profile(p_act)
    cap_act = CaptureRecord("cap_act", 1, "p_act", now - 4000, "frontal", 70.0, "frontal", 0, 0, "", 50, 50)
    repo.save_capture(cap_act, original_image=crop)

    # 2. Completed profile (old)
    p_old = PersonProfile("p_old", 2, "c", now - 5000, now - 4000, 1000, "completed", best_quality_score=60.0)
    repo.save_profile(p_old)
    cap_old = CaptureRecord("cap_old", 2, "p_old", now - 4000, "frontal", 60.0, "frontal", 0, 0, "", 50, 50)
    repo.save_capture(cap_old, original_image=crop)

    # 3. Completed profile (recent)
    p_rec = PersonProfile("p_rec", 3, "c", now - 100, now - 10, 90, "completed", best_quality_score=95.0)
    repo.save_profile(p_rec)
    cap_rec = CaptureRecord("cap_rec", 3, "p_rec", now - 10, "frontal", 95.0, "frontal", 0, 0, "", 50, 50)
    repo.save_capture(cap_rec, original_image=crop)

    # Prune older than 1000 seconds, keep active
    pruned = repo.prune_profiles(older_than_seconds=1000, keep_active=True)
    assert pruned == 1  # Only p_old was pruned!

    assert repo.get_profile("p_old") is None
    assert repo.get_profile("p_act") is not None
    assert repo.get_profile("p_rec") is not None

    # Prune keeping top 1 by quality
    pruned_top = repo.prune_profiles(keep_top_n=1, keep_active=False)
    assert pruned_top >= 1
    # p_rec has highest score (95.0), so it must survive
    assert repo.get_profile("p_rec") is not None


# =============================================================================
# 9. SQLite Concurrent Multi-Threading Safety
# =============================================================================

def test_sqlite_thread_safety(repo: ProfileRepository):
    num_threads = 8
    operations_per_thread = 5
    crop = create_dummy_crop(40, 40)

    def worker(thread_idx: int):
        for op in range(operations_per_thread):
            sess_id = f"thread_cam:{thread_idx}:{op}"
            prof = PersonProfile(
                session_id=sess_id,
                track_id=op,
                camera_id=f"thread_cam_{thread_idx}",
                first_seen=100.0 + op,
                last_seen=100.0 + op + 2,
                duration=2.0,
                status="active",
                total_observations=op + 1,
                best_quality_score=75.0 + op,
            )
            repo.save_profile(prof)

            cap = CaptureRecord(
                capture_id=f"cap_th_{thread_idx}_{op}",
                track_id=op,
                session_id=sess_id,
                timestamp=100.0 + op + 2,
                category="frontal",
                quality_score=75.0 + op,
                pose_category="frontal",
                yaw=0.0,
                pitch=0.0,
                original_path="",
                crop_width=40,
                crop_height=40,
            )
            repo.save_capture(cap, original_image=crop)

            # Perform reads during concurrent writes
            p = repo.get_profile(sess_id)
            assert p is not None
            assert p.session_id == sess_id

            # Update status
            repo.update_profile_status(sess_id, status="completed")

        return thread_idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            assert 0 <= res < num_threads

    # Verify all records exist and match expectations
    expected_count = num_threads * operations_per_thread
    stats = repo.get_statistics()
    assert stats["total_profiles"] == expected_count
    assert stats["saved_captures"] == expected_count
    assert stats["completed_tracks"] == expected_count
