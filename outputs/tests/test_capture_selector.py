"""Comprehensive unit tests for Intelligent Capture Selector."""
from __future__ import annotations

import numpy as np
import pytest

from services.capture_selector import (
    CATEGORY_FRONTAL,
    CATEGORY_HIGH_RES,
    CATEGORY_LEFT_ANGLE,
    CATEGORY_OVERALL,
    CATEGORY_RIGHT_ANGLE,
    CaptureDecision,
    CaptureSelector,
    CaptureSelectorConfig,
    determine_pose_category,
)
from services.face_contracts import (
    CaptureCandidate,
    FaceObservation,
    FacePose,
    FaceQualityResult,
)


def _make_obs(
    track_id: int = 1,
    timestamp: float = 10.0,
    quality_score: float = 60.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    crop_size: tuple[int, int] = (80, 80),
    is_acceptable: bool = True,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 180.0, 180.0),
) -> FaceObservation:
    """Helper to construct a realistic FaceObservation."""
    crop = np.zeros((crop_size[0], crop_size[1], 3), dtype=np.uint8)
    return FaceObservation(
        observation_id=f"obs_{track_id}_{timestamp}",
        track_id=track_id,
        timestamp=timestamp,
        bbox=bbox,
        confidence=0.95,
        pose=FacePose(yaw=yaw, pitch=pitch, roll=0.0),
        quality=FaceQualityResult(
            score=quality_score,
            sharpness=100.0,
            motion_blur=0.05,
            resolution_score=80.0,
            lighting_score=85.0,
            pose_score=85.0,
            size_score=80.0,
            is_acceptable=is_acceptable,
        ),
        face_crop=crop,
    )


class TestCandidateUpgrades:
    """Requirement 8: Candidate selection upgrades as higher quality frame arrives."""

    def test_candidate_selection_upgrades_with_higher_quality(self):
        selector = CaptureSelector(
            CaptureSelectorConfig(
                quality_delta_threshold=4.0,
                rate_limit_seconds=0.8,
            )
        )

        # Frame 1: initial baseline
        obs1 = _make_obs(track_id=1, timestamp=1.0, quality_score=60.0, yaw=0.0)
        decisions1 = selector.evaluate_observation(obs1)
        frontal_d1 = next(d for d in decisions1 if d.category == CATEGORY_FRONTAL)
        assert frontal_d1.is_selected is True
        assert frontal_d1.reason == "initial_capture"
        assert selector.get_best_capture(1, CATEGORY_FRONTAL).quality_score == 60.0

        # Frame 2: higher quality frame (+8 points, exceeds 4.0 delta)
        obs2 = _make_obs(track_id=1, timestamp=2.0, quality_score=68.0, yaw=0.0)
        decisions2 = selector.evaluate_observation(obs2)
        frontal_d2 = next(d for d in decisions2 if d.category == CATEGORY_FRONTAL)
        assert frontal_d2.is_selected is True
        assert frontal_d2.reason == "quality_upgrade"
        assert frontal_d2.quality_delta == 8.0
        assert selector.get_best_capture(1, CATEGORY_FRONTAL).quality_score == 68.0

        # Frame 3: even higher quality frame arrives
        obs3 = _make_obs(track_id=1, timestamp=3.0, quality_score=85.0, yaw=0.0)
        decisions3 = selector.evaluate_observation(obs3)
        frontal_d3 = next(d for d in decisions3 if d.category == CATEGORY_FRONTAL)
        assert frontal_d3.is_selected is True
        assert frontal_d3.reason == "quality_upgrade"
        assert selector.get_best_capture(1, CATEGORY_FRONTAL).quality_score == 85.0


class TestCategorySeparation:
    """Requirement 8: Category separation (frontal vs left_angle vs right_angle)."""

    def test_pose_yaw_classification(self):
        # Frontal: ±15 degrees
        assert determine_pose_category(FacePose(yaw=0.0, pitch=0.0, roll=0.0)) == CATEGORY_FRONTAL
        assert determine_pose_category(FacePose(yaw=15.0, pitch=0.0, roll=0.0)) == CATEGORY_FRONTAL
        assert determine_pose_category(FacePose(yaw=-15.0, pitch=0.0, roll=0.0)) == CATEGORY_FRONTAL

        # Left angle: -45 to -15 degrees
        assert determine_pose_category(FacePose(yaw=-15.5, pitch=0.0, roll=0.0)) == CATEGORY_LEFT_ANGLE
        assert determine_pose_category(FacePose(yaw=-30.0, pitch=0.0, roll=0.0)) == CATEGORY_LEFT_ANGLE
        assert determine_pose_category(FacePose(yaw=-45.0, pitch=0.0, roll=0.0)) == CATEGORY_LEFT_ANGLE

        # Right angle: +15 to +45 degrees
        assert determine_pose_category(FacePose(yaw=15.5, pitch=0.0, roll=0.0)) == CATEGORY_RIGHT_ANGLE
        assert determine_pose_category(FacePose(yaw=30.0, pitch=0.0, roll=0.0)) == CATEGORY_RIGHT_ANGLE
        assert determine_pose_category(FacePose(yaw=45.0, pitch=0.0, roll=0.0)) == CATEGORY_RIGHT_ANGLE

        # Profile / beyond 45 degrees
        assert determine_pose_category(FacePose(yaw=60.0, pitch=0.0, roll=0.0)) is None
        assert determine_pose_category(FacePose(yaw=-60.0, pitch=0.0, roll=0.0)) is None

    def test_category_separation_across_turns(self):
        selector = CaptureSelector(CaptureSelectorConfig(quality_delta_threshold=4.0))

        # Person looks frontal (yaw = 2.0)
        obs_frontal = _make_obs(track_id=1, timestamp=1.0, quality_score=70.0, yaw=2.0)
        selector.evaluate_observation(obs_frontal)

        # Person turns left (yaw = -25.0)
        obs_left = _make_obs(track_id=1, timestamp=2.0, quality_score=65.0, yaw=-25.0)
        selector.evaluate_observation(obs_left)

        # Person turns right (yaw = +30.0)
        obs_right = _make_obs(track_id=1, timestamp=3.0, quality_score=68.0, yaw=30.0)
        selector.evaluate_observation(obs_right)

        captures = selector.get_best_captures(1)
        assert CATEGORY_FRONTAL in captures
        assert CATEGORY_LEFT_ANGLE in captures
        assert CATEGORY_RIGHT_ANGLE in captures
        assert CATEGORY_OVERALL in captures

        # Check values are retained distinctly
        assert captures[CATEGORY_FRONTAL].quality_score == 70.0
        assert captures[CATEGORY_FRONTAL].pose.yaw == 2.0

        assert captures[CATEGORY_LEFT_ANGLE].quality_score == 65.0
        assert captures[CATEGORY_LEFT_ANGLE].pose.yaw == -25.0

        assert captures[CATEGORY_RIGHT_ANGLE].quality_score == 68.0
        assert captures[CATEGORY_RIGHT_ANGLE].pose.yaw == 30.0

        # "overall" should hold the highest quality across all categories
        assert captures[CATEGORY_OVERALL].quality_score == 70.0


class TestDeltaThresholdAntiJitter:
    """Requirement 8: Delta threshold prevents minor fluctuations from triggering captures."""

    def test_minor_jitter_does_not_replace_capture(self):
        selector = CaptureSelector(CaptureSelectorConfig(quality_delta_threshold=4.0))

        # Initial frame
        obs1 = _make_obs(track_id=1, timestamp=1.0, quality_score=70.0, yaw=0.0)
        selector.evaluate_observation(obs1)
        assert selector.get_best_capture(1, CATEGORY_FRONTAL).quality_score == 70.0

        # Minor fluctuations (+1.0, +2.5, +3.8, -1.0)
        for t, jitter_score in [(2.0, 71.0), (3.0, 72.5), (4.0, 73.8), (5.0, 69.0)]:
            obs_jitter = _make_obs(track_id=1, timestamp=t, quality_score=jitter_score, yaw=0.0)
            decisions = selector.evaluate_observation(obs_jitter)
            frontal_d = next(d for d in decisions if d.category == CATEGORY_FRONTAL)
            assert frontal_d.is_selected is False
            assert frontal_d.reason == "insufficient_quality_delta"
            # Selected capture remains at 70.0
            assert selector.get_best_capture(1, CATEGORY_FRONTAL).quality_score == 70.0

        # Sufficient increase: delta = 74.5 - 70.0 = 4.5 >= 4.0
        obs_good = _make_obs(track_id=1, timestamp=6.0, quality_score=74.5, yaw=0.0)
        decisions_good = selector.evaluate_observation(obs_good)
        frontal_good = next(d for d in decisions_good if d.category == CATEGORY_FRONTAL)
        assert frontal_good.is_selected is True
        assert frontal_good.reason == "quality_upgrade"
        assert selector.get_best_capture(1, CATEGORY_FRONTAL).quality_score == 74.5


class TestBoundedCandidateBuffer:
    """Requirement 8: Bounded buffer per track prevents memory exhaustion."""

    def test_buffer_stays_within_bound_and_retains_best(self):
        buf_size = 5
        selector = CaptureSelector(CaptureSelectorConfig(buffer_size=buf_size))

        scores = [20.0, 50.0, 45.0, 80.0, 60.0, 95.0, 30.0, 75.0, 90.0, 10.0]
        for seq, score in enumerate(scores):
            obs = _make_obs(track_id=1, timestamp=float(seq), quality_score=score)
            selector.evaluate_observation(obs, frame_sequence=seq)

            buf = selector.get_candidate_buffer(1)
            # Buffer capacity MUST NEVER exceed buffer_size
            assert len(buf) <= buf_size

        buf = selector.get_candidate_buffer(1)
        assert len(buf) == buf_size

        # Verify the top 5 scores from `scores` are present in buffer
        # Top 5 from [20, 50, 45, 80, 60, 95, 30, 75, 90, 10] are: 95, 90, 80, 75, 60
        expected_top_5 = [95.0, 90.0, 80.0, 75.0, 60.0]
        actual_scores = [c.quality_score for c in buf]
        assert actual_scores == expected_top_5


class TestHighResolutionCategory:
    """Requirement 3 & 4: high_res category behavior with area increase and quality."""

    def test_high_res_upgrades_on_significant_area_increase(self):
        selector = CaptureSelector(
            CaptureSelectorConfig(
                resolution_delta_ratio=0.15,  # 15% increase required
                high_res_min_quality=40.0,
            )
        )

        # Frame 1: 50x50 crop (area 2500), quality 60.0
        obs1 = _make_obs(track_id=1, timestamp=1.0, quality_score=60.0, crop_size=(50, 50))
        decisions1 = selector.evaluate_observation(obs1)
        high_res_d1 = next(d for d in decisions1 if d.category == CATEGORY_HIGH_RES)
        assert high_res_d1.is_selected is True
        assert high_res_d1.candidate.crop_original.shape[:2] == (50, 50)

        # Frame 2: 52x52 crop (area 2704, +8% area). Insufficient area increase (< 15%)
        obs2 = _make_obs(track_id=1, timestamp=2.0, quality_score=60.0, crop_size=(52, 52))
        decisions2 = selector.evaluate_observation(obs2)
        high_res_d2 = next(d for d in decisions2 if d.category == CATEGORY_HIGH_RES)
        assert high_res_d2.is_selected is False
        assert selector.get_best_capture(1, CATEGORY_HIGH_RES).crop_original.shape[:2] == (50, 50)

        # Frame 3: 80x80 crop (area 6400, +156% area). Significant increase!
        obs3 = _make_obs(track_id=1, timestamp=3.0, quality_score=58.0, crop_size=(80, 80))
        decisions3 = selector.evaluate_observation(obs3)
        high_res_d3 = next(d for d in decisions3 if d.category == CATEGORY_HIGH_RES)
        assert high_res_d3.is_selected is True
        assert high_res_d3.reason == "resolution_upgrade"
        assert selector.get_best_capture(1, CATEGORY_HIGH_RES).crop_original.shape[:2] == (80, 80)

    def test_high_res_rejects_unacceptable_quality(self):
        selector = CaptureSelector(CaptureSelectorConfig(high_res_min_quality=40.0))

        # Huge crop (200x200), but poor quality (score 25.0, below threshold 40.0)
        obs_bad = _make_obs(track_id=1, timestamp=1.0, quality_score=25.0, crop_size=(200, 200))
        decisions = selector.evaluate_observation(obs_bad)
        # High res category should not even be applicable or selected
        assert not any(d.category == CATEGORY_HIGH_RES and d.is_selected for d in decisions)
        assert selector.get_best_capture(1, CATEGORY_HIGH_RES) is None


class TestDiskRateLimiting:
    """Requirement 6: Rate-limit disk capture events (0.8s interval, >15pt jump override)."""

    def test_rate_limiting_with_time_and_jump_override(self):
        selector = CaptureSelector(
            CaptureSelectorConfig(
                quality_delta_threshold=4.0,
                rate_limit_seconds=0.8,
                quality_jump_threshold=15.0,
            )
        )

        # Frame 1 at t=10.0: Initial capture writes to disk
        obs1 = _make_obs(track_id=1, timestamp=10.0, quality_score=50.0)
        d1 = selector.evaluate_observation(obs1)
        frontal_d1 = next(d for d in d1 if d.category == CATEGORY_FRONTAL)
        assert frontal_d1.is_selected is True
        assert frontal_d1.should_save_disk is True

        # Frame 2 at t=10.2 (elapsed 0.2s < 0.8s): Quality delta +5.0 (55.0 vs 50.0)
        # Upgrades in memory, but rate-limited for disk
        obs2 = _make_obs(track_id=1, timestamp=10.2, quality_score=55.0)
        d2 = selector.evaluate_observation(obs2)
        frontal_d2 = next(d for d in d2 if d.category == CATEGORY_FRONTAL)
        assert frontal_d2.is_selected is True
        assert frontal_d2.should_save_disk is False  # Suppressed by rate limit

        # Frame 3 at t=10.4 (elapsed 0.4s < 0.8s): Quality jumps to 72.0!
        # Jump over last saved (50.0) is 22.0 > 15.0 points!
        # Jump override triggers disk write
        obs3 = _make_obs(track_id=1, timestamp=10.4, quality_score=72.0)
        d3 = selector.evaluate_observation(obs3)
        frontal_d3 = next(d for d in d3 if d.category == CATEGORY_FRONTAL)
        assert frontal_d3.is_selected is True
        assert frontal_d3.should_save_disk is True
        assert "jump_override" in frontal_d3.reason

        # Frame 4 at t=10.6 (elapsed 0.2s from 10.4): Score 77.0 (+5.0 delta, jump 5.0 <= 15)
        # Rate limited
        obs4 = _make_obs(track_id=1, timestamp=10.6, quality_score=77.0)
        d4 = selector.evaluate_observation(obs4)
        frontal_d4 = next(d for d in d4 if d.category == CATEGORY_FRONTAL)
        assert frontal_d4.is_selected is True
        assert frontal_d4.should_save_disk is False

        # Frame 5 at t=11.3 (elapsed 0.9s >= 0.8s from 10.4): Score 82.0 (+5.0 delta)
        # Rate limit elapsed -> disk write allowed
        obs5 = _make_obs(track_id=1, timestamp=11.3, quality_score=82.0)
        d5 = selector.evaluate_observation(obs5)
        frontal_d5 = next(d for d in d5 if d.category == CATEGORY_FRONTAL)
        assert frontal_d5.is_selected is True
        assert frontal_d5.should_save_disk is True


class TestBriefAppearances:
    """Requirement 7: Handle people appearing only briefly."""

    def test_brief_appearance_retains_highest_quality_seen(self):
        selector = CaptureSelector(CaptureSelectorConfig(quality_delta_threshold=4.0, rate_limit_seconds=0.8))

        # Person only appears for 2 frames within 0.1s
        obs1 = _make_obs(track_id=10, timestamp=5.0, quality_score=50.0)
        selector.evaluate_observation(obs1)

        # Higher quality frame arrives 0.05s later (rate-limited for disk write)
        obs2 = _make_obs(track_id=10, timestamp=5.05, quality_score=62.0)
        decisions2 = selector.evaluate_observation(obs2)
        frontal_d2 = next(d for d in decisions2 if d.category == CATEGORY_FRONTAL)
        assert frontal_d2.should_save_disk is False  # rate limited during live stream

        # Track ends abruptly (person walked out of frame)
        final_captures = selector.finalize_track(10)
        assert CATEGORY_FRONTAL in final_captures
        # Highest quality seen (62.0) was preserved!
        assert final_captures[CATEGORY_FRONTAL].quality_score == 62.0

        # finalize_track_decisions generates disk save for the unwritten upgrade
        decisions = selector.finalize_track_decisions(10)
        frontal_final = next(d for d in decisions if d.category == CATEGORY_FRONTAL)
        assert frontal_final.should_save_disk is True
        assert frontal_final.reason == "track_finalized"

    def test_brief_appearance_with_low_score_still_retained(self):
        selector = CaptureSelector(CaptureSelectorConfig(min_quality_threshold=50.0))

        # Person appears for 1 blurry frame (quality 28.0 < 50.0)
        obs = _make_obs(track_id=20, timestamp=1.0, quality_score=28.0)
        selector.evaluate_observation(obs)

        # Under strict threshold, no category selected yet
        assert selector.get_best_captures(20) == {}

        # But track finalization with fallback preserves the best capture seen
        final_captures = selector.finalize_track(20, fallback_to_any=True)
        assert CATEGORY_OVERALL in final_captures
        assert final_captures[CATEGORY_OVERALL].quality_score == 28.0


class TestContinuousEvaluationAndMemoryManagement:
    """Requirements 2 & memory lifecycle tests."""

    def test_continuous_evaluation_multi_track(self):
        selector = CaptureSelector()

        # Two distinct tracks tracked concurrently
        for t in range(5):
            obs_t1 = _make_obs(track_id=1, timestamp=float(t), quality_score=50.0 + t * 5)
            obs_t2 = _make_obs(track_id=2, timestamp=float(t), quality_score=40.0 + t * 6)
            selector.process_frame([obs_t1, obs_t2], frame_sequence=t)

        best1 = selector.get_best_capture(1, CATEGORY_OVERALL)
        best2 = selector.get_best_capture(2, CATEGORY_OVERALL)

        assert best1 is not None and best1.quality_score == 70.0
        assert best2 is not None and best2.quality_score == 64.0

    def test_prune_stale_tracks_and_reset(self):
        selector = CaptureSelector()

        obs1 = _make_obs(track_id=1, timestamp=10.0)
        obs2 = _make_obs(track_id=2, timestamp=50.0)
        selector.evaluate_observation(obs1)
        selector.evaluate_observation(obs2)

        assert len(selector.active_track_ids) == 2

        # Prune tracks older than 20 seconds relative to t=50.0 (track 1 is 40s old)
        pruned = selector.prune_stale_tracks(max_age_seconds=20.0, current_time=50.0)
        assert pruned == 1
        assert selector.active_track_ids == [2]

        # Reset
        selector.reset()
        assert len(selector.active_track_ids) == 0
        assert selector.get_stats()["total_observations"] == 0

    def test_evaluate_candidate_directly(self):
        selector = CaptureSelector()
        pose = FacePose(yaw=0.0, pitch=0.0, roll=0.0)
        crop = np.zeros((60, 60, 3), dtype=np.uint8)

        candidate = CaptureCandidate(
            candidate_id="direct_cand_1",
            track_id=99,
            timestamp=12.0,
            category=CATEGORY_FRONTAL,
            quality_score=75.0,
            pose=pose,
            crop_original=crop,
            frame_sequence=1,
            bbox_in_frame=(10.0, 10.0, 70.0, 70.0),
        )

        decision = selector.evaluate_candidate(candidate)
        assert decision.is_selected is True
        assert decision.should_save_disk is True
        assert selector.get_best_capture(99, CATEGORY_FRONTAL).quality_score == 75.0


class TestEdgeCasesAndIntegration:
    """Additional edge cases, image crop handling, and data contract validations."""

    def test_create_candidate_from_frame_image_and_normalized_bbox(self):
        selector = CaptureSelector()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Make a distinctive region for the bbox (100, 50) to (200, 150)
        frame[50:150, 100:200] = 255

        obs = FaceObservation(
            observation_id="obs_crop_test",
            track_id=42,
            timestamp=25.0,
            bbox=(100.0, 50.0, 200.0, 150.0),
            confidence=0.98,
            pose=FacePose(yaw=5.0, pitch=-2.0, roll=1.0),
            quality=FaceQualityResult(
                score=78.0,
                sharpness=120.0,
                motion_blur=0.02,
                resolution_score=85.0,
                lighting_score=90.0,
                pose_score=90.0,
                size_score=85.0,
                is_acceptable=True,
            ),
            face_crop=None,  # Not pre-cropped
        )

        candidate = selector.create_candidate(
            observation=obs,
            category=CATEGORY_FRONTAL,
            frame_sequence=10,
            frame_image=frame,
            frame_shape=(480, 640),
        )

        assert candidate.crop_original.shape == (150, 150, 3)
        assert np.all(candidate.crop_original[25:125,25:125] == 255)
        assert candidate.normalized_bbox is not None
        # Normalized (x1/640, y1/480, x2/640, y2/480)
        nx1, ny1, nx2, ny2 = candidate.normalized_bbox
        assert pytest.approx(nx1, rel=1e-3) == 100.0 / 640.0
        assert pytest.approx(ny1, rel=1e-3) == 50.0 / 480.0
        assert pytest.approx(nx2, rel=1e-3) == 200.0 / 640.0
        assert pytest.approx(ny2, rel=1e-3) == 150.0 / 480.0

    def test_profile_views_do_not_qualify_for_angles_but_qualify_for_overall(self):
        selector = CaptureSelector(CaptureSelectorConfig(min_quality_threshold=40.0))

        # Extreme yaw = 65.0 degrees (profile view)
        obs_profile = _make_obs(track_id=5, timestamp=1.0, quality_score=65.0, yaw=65.0)
        decisions = selector.evaluate_observation(obs_profile)

        selected_cats = [d.category for d in decisions if d.is_selected]
        # Should qualify for "overall" and "high_res", but NOT frontal, left_angle, or right_angle
        assert CATEGORY_OVERALL in selected_cats
        assert CATEGORY_FRONTAL not in selected_cats
        assert CATEGORY_LEFT_ANGLE not in selected_cats
        assert CATEGORY_RIGHT_ANGLE not in selected_cats

    def test_multi_track_rate_limiting_independence(self):
        selector = CaptureSelector(CaptureSelectorConfig(rate_limit_seconds=1.0))

        # Track 1 writes at t=0.0
        obs1_t1 = _make_obs(track_id=1, timestamp=0.0, quality_score=60.0)
        d1 = selector.evaluate_observation(obs1_t1)
        assert next(d for d in d1 if d.category == CATEGORY_FRONTAL).should_save_disk is True

        # Track 2 writes at t=0.2 (even though t=0.2 is < 1.0s from Track 1's write)
        obs1_t2 = _make_obs(track_id=2, timestamp=0.2, quality_score=60.0)
        d2 = selector.evaluate_observation(obs1_t2)
        assert next(d for d in d2 if d.category == CATEGORY_FRONTAL).should_save_disk is True

        # Track 1 again at t=0.3 (0.3s < 1.0s from t=0.0) -> Rate limited!
        obs2_t1 = _make_obs(track_id=1, timestamp=0.3, quality_score=66.0)
        d3 = selector.evaluate_observation(obs2_t1)
        assert next(d for d in d3 if d.category == CATEGORY_FRONTAL).should_save_disk is False

    def test_decision_to_dict_and_stats(self):
        selector = CaptureSelector()
        obs = _make_obs(track_id=7, timestamp=1.0, quality_score=75.5)
        decisions = selector.evaluate_observation(obs)
        d = decisions[0]

        d_dict = d.to_dict()
        assert d_dict["track_id"] == 7
        assert d_dict["category"] == CATEGORY_OVERALL
        assert d_dict["quality_score"] == 75.5
        assert d_dict["is_selected"] is True
        assert "candidate_id" in d_dict

        stats = selector.get_stats()
        assert stats["active_tracks_count"] == 1
        assert stats["total_observations"] == 1
        assert stats["total_disk_events"] >= 1
        assert 7 in stats["tracks"]

    def test_clear_track_removes_state(self):
        selector = CaptureSelector()
        obs = _make_obs(track_id=3, timestamp=1.0, quality_score=70.0)
        selector.evaluate_observation(obs)
        assert selector.get_best_capture(3, CATEGORY_OVERALL) is not None

        selector.clear_track(3)
        assert selector.get_best_capture(3, CATEGORY_OVERALL) is None
        assert selector.get_candidate_buffer(3) == []


class TestExtractAndFormatCandidate:
    """Requirement 2: Portrait crops with generous proportions and small face super-resolution."""

    def test_extract_and_format_candidate_portrait_proportions(self):
        selector = CaptureSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Face bbox: x1=200, y1=100, x2=300, y2=200 (bw=100, bh=100)
        obs = _make_obs(track_id=1, bbox=(200.0, 100.0, 300.0, 200.0))
        obs.face_crop = None

        candidate = selector._extract_and_format_candidate(
            observation=obs,
            category=CATEGORY_FRONTAL,
            frame_sequence=1,
            frame_image=frame,
            frame_shape=(480, 640),
            headroom=0.35,
            chin=0.25,
            sides=0.30,
            auto_enhance_small=False,
        )

        # Expected dimensions:
        # y1 = 100 - 35 = 65, y2 = 200 + 25 = 225 -> height = 160
        # x1 = 200 - 30 = 170, x2 = 300 + 30 = 330 -> width = 160
        assert candidate.crop_original.shape == (160, 160, 3)

    def test_extract_and_format_candidate_small_face_auto_enhance(self):
        selector = CaptureSelector()
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Small face bbox: 50x50
        obs = _make_obs(track_id=1, bbox=(100.0, 100.0, 150.0, 150.0))
        obs.face_crop = None

        candidate = selector._extract_and_format_candidate(
            observation=obs,
            category=CATEGORY_OVERALL,
            frame_sequence=1,
            frame_image=frame,
            frame_shape=(480, 640),
            min_dim=256,
            auto_enhance_small=True,
        )

        assert candidate.crop_original is not None
        assert candidate.crop_original.shape[:2] == (80,80)
        assert np.array_equal(candidate.crop_original,frame[82:162,85:165])
        assert candidate.needs_super_resolution is True

    def test_extract_and_format_candidate_marking_super_resolution(self):
        selector = CaptureSelector()
        # Small crop without auto_enhance_small
        obs = _make_obs(track_id=1, crop_size=(60, 60))
        candidate = selector._extract_and_format_candidate(
            observation=obs,
            category=CATEGORY_OVERALL,
            auto_enhance_small=False,
            min_dim=256,
        )
        assert candidate.needs_super_resolution is True

