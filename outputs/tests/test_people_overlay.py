"""Unit tests verifying PeopleOverlay module, integration, and interface compliance."""
import json
import os
import re
from pathlib import Path

from services.face_contracts import (
    PersonTrack, FacePose, FaceQualityResult, FaceObservation, CaptureRecord
)


OUTPUTS_DIR = Path(__file__).resolve().parents[1]


def test_people_overlay_file_exists_and_valid():
    js_path = OUTPUTS_DIR / "static/js/people_overlay.js"
    assert js_path.exists(), "people_overlay.js must exist in static/js"
    content = js_path.read_text(encoding="utf-8")
    assert "class PeopleOverlay" in content, "PeopleOverlay class must be defined"
    assert "drawPersonBox" in content, "drawPersonBox must be defined"
    assert "drawFaceBox" in content, "drawFaceBox must be defined"
    assert "drawPoseBadge" in content, "drawPoseBadge must be defined"
    assert "drawQualityBadge" in content, "drawQualityBadge must be defined"
    assert "drawLandmarks" in content, "drawLandmarks must be defined"
    assert "drawCapturePulse" in content, "drawCapturePulse must be defined"
    assert "normalizeBbox" in content, "normalizeBbox must be defined"


def test_index_html_includes_people_overlay():
    html_path = OUTPUTS_DIR / "templates/index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "people_overlay.js" in content, "index.html must reference people_overlay.js"


def test_overlay_js_integrates_people_overlay():
    overlay_path = OUTPUTS_DIR / "static/js/overlay.js"
    assert overlay_path.exists()
    content = overlay_path.read_text(encoding="utf-8")
    assert "PeopleOverlay" in content, "overlay.js must reference PeopleOverlay"
    assert "this.peopleOverlay" in content, "VehicleOverlay must maintain a peopleOverlay reference"


def test_face_contracts_schema_compatibility_with_overlay():
    """Verify that serialization outputs from face_contracts match what PeopleOverlay expects."""
    pose = FacePose(yaw=-22.0, pitch=4.0, roll=0.0, category="left_angle", confidence=0.9)
    quality = FaceQualityResult(
        score=82.0, sharpness=110.0, motion_blur=0.05, resolution_score=85.0,
        lighting_score=80.0, pose_score=80.0, size_score=80.0, is_acceptable=True
    )
    obs = FaceObservation(
        observation_id="obs-10", track_id=14, timestamp=10.5,
        bbox=(0.2, 0.25, 0.35, 0.45), confidence=0.96,
        landmarks=[(0.24, 0.30), (0.31, 0.30), (0.28, 0.35), (0.25, 0.40), (0.30, 0.40)],
        pose=pose, quality=quality
    )
    track = PersonTrack(
        track_id=14, session_id="session:14", first_seen=5.0, last_seen=10.5,
        bbox=(0.15, 0.2, 0.4, 0.8), confidence=0.92, hits=18, state="active", confirmed=True
    )

    t_dict = track.to_dict()
    obs_dict = obs.to_dict()

    # Emulated frontend payload
    person_payload = {
        **t_dict,
        "id": t_dict["track_id"],
        "face": obs_dict
    }

    # Person properties
    assert person_payload["id"] == 14
    assert person_payload["duration"] == 5.5
    assert person_payload["hits"] == 18
    assert person_payload["state"] == "active"

    # Face properties
    face = person_payload["face"]
    assert len(face["bbox"]) == 4
    assert len(face["landmarks"]) == 5
    assert face["pose"]["yaw"] == -22.0
    assert face["pose"]["category"] == "left_angle"
    assert face["quality"]["score"] == 82.0


def test_captured_badge_contract():
    record = CaptureRecord(
        capture_id="cap_99", track_id=14, session_id="session:14", timestamp=10.5,
        category="frontal", quality_score=88.5, pose_category="frontal", yaw=0.0, pitch=0.0,
        original_path="captures/cap_99.jpg", crop_width=100, crop_height=100
    )
    r_dict = record.to_dict()
    assert r_dict["capture_id"] == "cap_99"
    assert r_dict["quality_score"] == 88.5
    assert r_dict["pose_category"] == "frontal"


if __name__ == "__main__":
    test_people_overlay_file_exists_and_valid()
    test_index_html_includes_people_overlay()
    test_overlay_js_integrates_people_overlay()
    test_face_contracts_schema_compatibility_with_overlay()
    test_captured_badge_contract()
    print("All python overlay contract verification tests passed successfully!")
