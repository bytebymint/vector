"""Full-Scale People Facial Capture & Tracking System Orchestrator.

Orchestrates end-to-end integration across all subsystems:
- VideoPipeline (frame acquisition & master-resolution preservation)
- PersonDetector & PersonTracker (detection & multi-object tracking)
- FaceDetector & FacePoseEstimator (face localization, landmarks & 3D head-pose)
- FaceQualityAssessor / FaceQualityEvaluator (0-100 multi-factor quality scoring)
- CaptureSelector (intelligent best-frame decision engine)
- FaceEnhancer (background async super-resolution upscaling)
- ProfileRepository (SQLite storage of anonymous profiles & disk captures)
- PeopleEventEmitter (real-time WebSocket broadcasts)
- AnalysisLifecycleController (state transitions & worker coordination)
- PipelineScheduler (concurrency, priority, and backpressure)
- PrivacyGovernance (retention enforcement and safe path validation)
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import uuid
from dataclasses import replace

import cv2
import numpy as np

# Contracts
from services.face_contracts import (
    AnalysisState,
    CaptureCandidate,
    CaptureRecord,
    FaceObservation,
    FacePose,
    FaceQualityResult,
    FramePacket,
    PersonDetection,
    PersonProfile,
    PersonTrack,
    ProcessingMetrics,
)

# Subsystems
from services.video_pipeline import VideoPipeline, extract_crop
from services.person_detector import PersonDetector, PersonDetectorConfig
from services.person_tracker import PersonTracker
from services.face_detector import FaceDetector, FaceDetectorConfig
from services.face_coverage import FaceCoverage
from services.face_worker import FaceWorkerClient
from services.face_verifier import FaceVerifier
from services.face_pose import FacePoseEstimator
from services.face_quality import (
    FaceQualityAssessor,
    QualityThresholds,
    QualityWeights,
)
from services.capture_selector import (
    CATEGORY_FRONTAL,
    CATEGORY_HIGH_RES,
    CATEGORY_LEFT_ANGLE,
    CATEGORY_OVERALL,
    CATEGORY_RIGHT_ANGLE,
    CaptureDecision,
    CaptureSelector,
    CaptureSelectorConfig,
)
from services.face_enhancer import FaceEnhancer, EnhancementConfig, EnhancementResult
from services.profile_repository import ProfileRepository
from services.people_events import PeopleEventEmitter
from services.analysis_lifecycle import (
    AnalysisLifecycleController,
    AnalysisLifecycleState,
    WorkerSpec,
)
from services.pipeline_scheduler import (
    PipelineScheduler,
    PipelineSchedulerConfig,
    TaskPriority,
)
from services.privacy_governance import PrivacyGovernance

LOG = logging.getLogger(__name__)

# Alias for interface conformance
FaceQualityEvaluator = FaceQualityAssessor


class PeopleSystemOrchestrator:
    """Central orchestrator wiring the full people facial capture & tracking pipeline."""

    def __init__(
        self,
        video_pipeline: Optional[VideoPipeline] = None,
        person_detector: Optional[PersonDetector] = None,
        person_tracker: Optional[PersonTracker] = None,
        face_detector: Optional[FaceDetector] = None,
        face_pose_estimator: Optional[FacePoseEstimator] = None,
        face_quality_evaluator: Optional[FaceQualityAssessor] = None,
        capture_selector: Optional[CaptureSelector] = None,
        face_enhancer: Optional[FaceEnhancer] = None,
        profile_repository: Optional[ProfileRepository] = None,
        people_event_emitter: Optional[PeopleEventEmitter] = None,
        lifecycle_controller: Optional[AnalysisLifecycleController] = None,
        pipeline_scheduler: Optional[PipelineScheduler] = None,
        privacy_governance: Optional[PrivacyGovernance] = None,
        stream: Optional[Any] = None,
        storage: Optional[Any] = None,
        models_dir: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        socketio: Optional[Any] = None,
        camera_id: str = "camera_main",
        config: Optional[Dict[str, Any]] = None,
        auto_start: bool = False,
        shared_inference: bool = False,
        face_verifier: Optional[Any] = None,
    ):
        self.lock = threading.RLock()
        self.camera_id = str(camera_id)
        self.stream = stream
        self.storage = storage
        self.shared_inference = shared_inference
        self._shared_condition = threading.Condition()
        self._shared_latest = None
        self._shared_latest_mono: float = 0.0
        self._cached_body_detections: List[PersonDetection] = []
        self._cached_body_ts: float = 0.0

        # Resolve paths
        self.models_dir = (
            Path(models_dir).resolve()
            if models_dir
            else (Path(__file__).resolve().parents[1] / "models").resolve()
        )
        self.data_dir = (
            Path(data_dir).resolve()
            if data_dir
            else (Path(__file__).resolve().parents[1] / "data").resolve()
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.captures_dir = self.data_dir / "captures"
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "people.sqlite3"

        # Runtime configuration
        self._config: Dict[str, Any] = {
            "detector_model": "yolo26n",
            "person_confidence_threshold": 0.10,
            "face_confidence_threshold": 0.65,
            "quality_threshold": 55.0,
            "min_face_pixels": 32,
            "enhancement_enabled": True,
            "enhancement_scale": 2,
            "enhancement_model": "espcn",
            "max_captures_per_person": 5,
            "selective_face_cadence": 1,
            "governance_retention_seconds": 3600.0,
            "governance_prune_interval_frames": 100,
        }
        if config:
            self._config.update(config)

        # Wire subsystems (lazy-initialize with production defaults if not injected)
        self.privacy_governance = privacy_governance or PrivacyGovernance(
            audit_log_path=self.data_dir / "audit.jsonl",
            max_retention_seconds=float(self._config.get("governance_retention_seconds", 3600.0)),
        )

        self.profile_repository = profile_repository or ProfileRepository(
            db_path=self.db_path,
            storage_dir=self.captures_dir,
        )

        self.video_pipeline = video_pipeline or VideoPipeline(
            source_id=self.camera_id,
            buffer_size=2,
        )

        self.person_detector = person_detector or PersonDetector(
            models_dir=self.models_dir,
            config=PersonDetectorConfig(
                model_name=str(self._config.get("detector_model", "yolo26n")),
                confidence_threshold=float(self._config.get("person_confidence_threshold", 0.20)),
                fallback_to_mock=False,
            ),
        )

        self.person_tracker = person_tracker or PersonTracker(
            session_id=f"{self.camera_id}:session_0",
            lifetime=2.5,
            confirm_hits=2,
            high_score_threshold=float(self._config.get("person_confidence_threshold", 0.20)),
        )

        # Face detector: search for YuNet model or auto-fallback to Haar
        yunet_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        dynamic_yunet = self.models_dir / 'face_detection_yunet_2026may.onnx'
        face_runtime = 'opencv'
        if dynamic_yunet.exists():
            yunet_path, face_runtime = dynamic_yunet, 'onnxruntime'
        dnn_target = cv2.dnn.DNN_TARGET_CPU
        # Benchmarked on this Intel GPU: ~145 ms for multi-scale analysis vs
        # ~430 ms on CPU. The detector retries on CPU if OpenCL fails.
        if os.environ.get('TRAFFIC_FACE_DNN_TARGET', 'cpu').lower() == 'opencl' and cv2.dnn.DNN_TARGET_OPENCL_FP16 in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_OPENCV):
            dnn_target = cv2.dnn.DNN_TARGET_OPENCL_FP16
        cache_dir = self.data_dir / 'opencl_cache'
        cache_dir.mkdir(exist_ok=True)
        os.environ.setdefault('OPENCV_OCL4DNN_CONFIG_PATH', str(cache_dir))
        self.face_detector = face_detector or FaceDetector(
            FaceDetectorConfig(
                model_path=str(yunet_path) if yunet_path.exists() else None,
                backend="auto",
                score_threshold=float(self._config.get("face_confidence_threshold", 0.25)),
                roi_score_threshold=float(self._config.get("face_confidence_threshold", 0.75)),
                default_mode="full_frame",
                max_inference_dim=1280,
                fixed_input_size=0,
                min_person_width=16,
                min_person_height=32,
                extract_face_crops=True,
                dnn_target=dnn_target,
                runtime=face_runtime,
            )
        )
        self.face_coverage = FaceCoverage(self.face_detector)
        self.face_verifier = face_verifier or FaceVerifier(self.models_dir)
        self.face_worker = FaceWorkerClient(self.models_dir) if self.stream is not None and self.shared_inference else None
        self._capture_epoch = None
        self.analysis_focus = 'people'
        self._focus_generation = 0

        self.face_pose_estimator = face_pose_estimator or FacePoseEstimator(method='geometric')

        self.face_quality_evaluator = face_quality_evaluator or FaceQualityAssessor(
            QualityThresholds(
                min_score=float(self._config["quality_threshold"]),
                min_width=int(self._config["min_face_pixels"]),
                min_height=int(self._config["min_face_pixels"]),
                min_sharpness=5.0,
                max_motion_blur=0.80,
            )
        )

        self.capture_selector = capture_selector or CaptureSelector(
            CaptureSelectorConfig(
                buffer_size=int(self._config.get("max_captures_per_person", 5)),
                quality_delta_threshold=3.5,
                min_quality_threshold=float(self._config["quality_threshold"]),
                high_res_min_quality=70.0,
                high_res_min_pixels=160,
                allow_best_available=False,
                require_verified_face=True,
                auto_enhance_small_faces=False,
                min_source_pixels=int(self._config['min_face_pixels']),
            )
        )

        self.face_enhancer = face_enhancer or FaceEnhancer(
            config=EnhancementConfig(
                model_name=str(self._config.get("enhancement_model", "espcn")),
                scale=int(self._config.get("enhancement_scale", 2)),
                models_dir=self.models_dir,
                save_to_disk=False,  # Persistence handled atomically by ProfileRepository
            ),
            models_dir=self.models_dir,
            auto_start=True,
        )

        self.pipeline_scheduler = pipeline_scheduler or PipelineScheduler(
            config=PipelineSchedulerConfig(
                worker_threads=2,
                auto_start=True,
                stationary_recheck_interval_s=0.25,
                max_faces_per_frame=3,
            )
        )

        self.people_event_emitter = people_event_emitter or PeopleEventEmitter(
            socketio=socketio,
            namespace="/",
            max_fps=30.0,
        )
        if socketio is not None and self.people_event_emitter.socketio is None:
            self.people_event_emitter.attach_socketio(socketio)

        # Lifecycle controller binds all subsystems
        self.lifecycle_controller = lifecycle_controller or AnalysisLifecycleController(
            stream_manager=self.stream,
            detector=self.person_detector,
            tracker=self.person_tracker,
            face_evaluator=self.face_quality_evaluator,
            capture_selector=self.capture_selector,
            enhancer=self.face_enhancer,
            profile_storage=self.profile_repository,
        )

        # Telemetry & State
        self.metrics = ProcessingMetrics()
        self._fps_window: deque[float] = deque(maxlen=60)
        self._last_frame_time: Optional[float] = None
        self._frame_counter: int = 0
        self._last_snapshot: Dict[str, Any] = {
            "sequence": 0,
            "timestamp": 0.0,
            "tracks": [],
            "faces": [],
            "active_count": 0,
            "unique_count": 0,
        }

        # Asynchronous prioritized face perception buffers
        self._in_flight_face_future: Optional[Future] = None
        self._cached_face_tracks: List[Any] = []
        self._cached_face_observations: List[Any] = []
        self._cached_coverage_details: Dict[str, Any] = {}

        # Background worker management
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if auto_start:
            self.start()

    # -------------------------------------------------------------------------
    # Properties & Interface Conformance
    # -------------------------------------------------------------------------

    @property
    def repository(self) -> ProfileRepository:
        """Expose profile repository for people_routes and storage operations."""
        return self.profile_repository

    @property
    def snapshot(self) -> Dict[str, Any]:
        """Return the latest frame snapshot for websocket and overlay rendering."""
        with self.lock:
            return dict(self._last_snapshot)

    @property
    def enhancement_enabled(self) -> bool:
        """Check if super-resolution enhancement is enabled in config."""
        return bool(self._config.get("enhancement_enabled", False))

    @property
    def current_epoch(self) -> str:
        """Return current analysis session epoch."""
        return self.lifecycle_controller.current_epoch or "default_session"

    # -------------------------------------------------------------------------
    # Lifecycle Operations
    # -------------------------------------------------------------------------

    def start(self, epoch_id: Optional[str] = None) -> bool:
        """Start or resume the people analysis engine and frame acquisition."""
        with self.lock:
            epoch = epoch_id or f"session_{int(time.time())}"
            started = self.lifecycle_controller.start_analysis(epoch_id=epoch)
            if not started and not self.lifecycle_controller.is_running:
                return False

            self._stop_event.clear()

            # Start background super-resolution enhancer if stopped
            if self.face_enhancer and not self.face_enhancer.is_running:
                self.face_enhancer.start()

            # Start pipeline scheduler if stopped
            if self.pipeline_scheduler and not self.pipeline_scheduler.is_running:
                self.pipeline_scheduler.start()

            # Launch autonomous worker thread if stream is available
            if self.stream is not None and (self._worker_thread is None or not self._worker_thread.is_alive()):
                self._worker_thread = threading.Thread(
                    target=self._autonomous_stream_loop,
                    name="people-analysis-orchestrator",
                    daemon=True,
                )
                self._worker_thread.start()

            LOG.info("PeopleSystemOrchestrator started with epoch %s", epoch)
            return True

    def stop(self, drain_timeout: float = 2.0) -> bool:
        """Safely halt people analysis, drain pending captures, and flush state."""
        with self.lock:
            self._stop_event.set()
            if self.face_worker:
                self.face_worker.close()
            stopped = self.lifecycle_controller.stop_analysis(drain_timeout=drain_timeout)

            # Wait for background stream thread to exit
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=max(0.5, drain_timeout / 2.0))
                self._worker_thread = None

            LOG.info("PeopleSystemOrchestrator stopped successfully")
            return True

    def is_running(self) -> bool:
        """Return True if analysis is currently active and processing frames."""
        return self.lifecycle_controller.is_running

    def get_state(self) -> AnalysisState:
        """Return comprehensive system operational state."""
        with self.lock:
            state = self.lifecycle_controller.get_state()
            state.active_tracks_count = int(self._last_snapshot.get('active_count', 0))
            if hasattr(self.profile_repository, "count_profiles"):
                state.total_profiles_count = self.profile_repository.count_profiles()
            if hasattr(self.profile_repository, "count_captures"):
                state.total_captures_count = self.profile_repository.count_captures()
            if hasattr(self.face_enhancer, "_task_queue"):
                state.pending_enhancements = self.face_enhancer._task_queue.qsize()

            state.fps = round(float(self.metrics.acquisition_fps), 1)
            if self.stream is not None and time.time()-float(self._last_snapshot.get('timestamp') or 0)>3:
                state.active_tracks_count=0
                state.fps=0.
                state.status_message='Waiting for fresh face analysis'
            if self.analysis_focus == 'traffic':
                state.active_tracks_count=0
                state.fps=0.
                state.status_message='Standby while traffic view has processing priority'
            return state

    def set_focus(self, mode: str) -> None:
        with self.lock:
            if mode != self.analysis_focus and mode == 'people':
                self._focus_generation += 1
                self._fps_window.clear()
                self.metrics.acquisition_fps=0.
            self.analysis_focus=mode

    def get_metrics(self) -> ProcessingMetrics:
        """Return current performance latency and throughput metrics."""
        with self.lock:
            self.metrics.active_tracks = int(self._last_snapshot.get('active_count', 0))
            if hasattr(self.pipeline_scheduler, "analysis_queue_depth"):
                self.metrics.queue_depth = self.pipeline_scheduler.analysis_queue_depth
            return self.metrics

    def get_config(self) -> Dict[str, Any]:
        """Return runtime configuration dictionary."""
        with self.lock:
            return dict(self._config)

    def configure(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically update runtime configuration parameters."""
        if not isinstance(updates, dict):
            raise ValueError("Expected JSON dictionary for configuration")

        with self.lock:
            self._config.update(updates)

            if "person_confidence_threshold" in updates:
                conf = float(updates["person_confidence_threshold"])
                self.person_detector.config.confidence_threshold = conf
                self.person_tracker.high_score_threshold = conf

            if "face_confidence_threshold" in updates:
                f_conf = float(updates["face_confidence_threshold"])
                self.face_detector.config.score_threshold = f_conf
                self.face_detector.config.roi_score_threshold = f_conf

            if "quality_threshold" in updates:
                q_th = float(updates["quality_threshold"])
                self.face_quality_evaluator.thresholds.min_score = q_th
                self.capture_selector.config.min_quality_threshold = q_th

            if "min_face_pixels" in updates:
                pixels = max(16, min(512, int(updates["min_face_pixels"])))
                self.face_quality_evaluator.thresholds.min_width = pixels
                self.face_quality_evaluator.thresholds.min_height = pixels
                self.capture_selector.config.min_source_pixels = pixels

            if "enhancement_scale" in updates and self.face_enhancer:
                self.face_enhancer.config.scale = int(updates["enhancement_scale"])

            if "enhancement_model" in updates and self.face_enhancer:
                self.face_enhancer.config.model_name = str(updates["enhancement_model"])

            LOG.info("PeopleSystemOrchestrator updated configuration: %s", updates)
            return dict(self._config)

    # -------------------------------------------------------------------------
    # Frame Processing Loop (Synchronous & Pipeline Core)
    # -------------------------------------------------------------------------

    def submit_shared_frame(self, packet: Any, boxes: Any, detection_ms: float = 0.0) -> None:
        """Latest-only mailbox: reuse one YOLO pass and its exact native frame."""
        if not self.is_running():
            return
        rows = boxes.data.cpu().numpy() if hasattr(boxes.data, 'cpu') else np.asarray(boxes.data)
        detections = [PersonDetection(tuple(map(float, row[:4])), float(row[4])) for row in rows]
        with self._shared_condition:
            self._shared_latest = (packet, detections, detection_ms)
            self._shared_latest_mono = time.monotonic()
            self._shared_condition.notify()

    def process_frame(self, packet: Any, shared_detections=None, shared_detection_ms=0.0, async_face: Optional[bool] = None, fast_only: bool = False) -> Dict[str, Any]:
        """Execute complete multi-person tracking and facial capture on a frame packet.

        Workflow:
        1. Parse and standardize FramePacket contract.
        2. Detect persons and associate with stable anonymous tracks.
        3. Prioritize tracks for facial evaluation via PipelineScheduler.
        4. Detect faces, estimate 3D head-pose, and calculate 0-100 quality scores.
        5. Evaluate capture decisions (overall, frontal, angular, high_res).
        6. Persist selected captures to ProfileRepository.
        7. Enqueue asynchronous super-resolution enhancement.
        8. Emit real-time WebSocket events.
        9. Enforce privacy governance and metrics collection.

        Args:
            packet: FramePacket contract instance or raw numpy BGR image array.

        Returns:
            Dict[str, Any] containing frame analysis results, tracks, faces, and decisions.
        """
        t_frame_start = time.perf_counter()

        # Standardize input to FramePacket contract
        if not isinstance(packet, FramePacket):
            if packet is None:
                return {"error": "None packet received", "tracks": [], "faces": []}
            if isinstance(packet, np.ndarray):
                h, w = packet.shape[:2]
                self._frame_counter += 1
                frame_packet = FramePacket(
                    sequence=self._frame_counter,
                    timestamp=time.time(),
                    image=packet,
                    source_id=self.camera_id,
                    epoch=self.current_epoch,
                    width=w,
                    height=h,
                )
            elif hasattr(packet, "image"):
                # Handle StreamManager FramePacket
                h, w = packet.image.shape[:2]
                frame_packet = FramePacket(
                    sequence=getattr(packet, "sequence", self._frame_counter + 1),
                    timestamp=getattr(packet, "timestamp", time.time()),
                    image=packet.image,
                    source_id=self.camera_id,
                    epoch=getattr(packet, "epoch", self.current_epoch),
                    width=w,
                    height=h,
                    pts=packet.media_time if getattr(packet, "media_time", None) is not None else getattr(packet, "pts", None),
                )
            else:
                return {"error": "Unrecognized packet format", "tracks": [], "faces": []}
        else:
            frame_packet = packet

        # Publish to video pipeline ring buffer to avoid redundant decoding
        if self.video_pipeline and frame_packet.image is not None:
            try:
                self.video_pipeline.ring_buffer.append(frame_packet)
            except Exception as err:
                LOG.debug("RingBuffer append notice: %s", err)

        self.metrics.frame_sequence = frame_packet.sequence

        # Skip frame if lifecycle is paused / not running
        if not self.is_running():
            return {
                "sequence": frame_packet.sequence,
                "timestamp": frame_packet.timestamp,
                "status": "stopped",
                "tracks": [],
                "faces": [],
            }

        # Check backpressure / scheduler frame skip
        if shared_detections is None and self.pipeline_scheduler and self.pipeline_scheduler.should_skip_analysis_frame(frame_packet.sequence):
            self.metrics.dropped_frames += 1
            return {
                "sequence": frame_packet.sequence,
                "timestamp": frame_packet.timestamp,
                "status": "dropped_backpressure",
                "tracks": [],
                "faces": [],
            }

        # ---------------------------------------------------------------------
        # Stage 1: Person Detection & Tracking
        # ---------------------------------------------------------------------
        t_det_0 = time.perf_counter()
        detections: List[PersonDetection] = []
        try:
            detections = shared_detections if shared_detections is not None else self.person_detector.detect(frame_packet.image)
        except Exception as exc:
            LOG.warning("Person detection failed: %s", exc)
        det_ms = shared_detection_ms if shared_detections is not None else (time.perf_counter() - t_det_0) * 1000.0
        self.metrics.detection_ms = det_ms
        self.pipeline_scheduler.record_stage_latency("detection", det_ms)

        t_trk_0 = time.perf_counter()
        active_tracks: List[PersonTrack] = []
        try:
            active_tracks = self.person_tracker.update(
                detections=detections,
                timestamp=frame_packet.timestamp,
            )
        except Exception as exc:
            LOG.warning("Person tracker update failed: %s", exc)
        trk_ms = (time.perf_counter() - t_trk_0) * 1000.0
        self.metrics.tracking_ms = trk_ms
        self.pipeline_scheduler.record_stage_latency("tracking", trk_ms)

        # ---------------------------------------------------------------------
        # Stage 2: Independent multi-scale face discovery and spatial tracking
        # ---------------------------------------------------------------------
        t_face_0 = time.perf_counter()
        face_observations: List[FaceObservation] = []
        face_tracks = []
        face_error = None
        coverage_details = {}
        face_epoch = f'{frame_packet.epoch}:{self.current_epoch}:{self._focus_generation}:verified-v2'

        try:
            if self._capture_epoch != face_epoch:
                self.face_coverage.reset(face_epoch)
                self.capture_selector.reset()
                self._capture_epoch = face_epoch
                self._in_flight_face_future = None
                self._cached_face_tracks.clear()
                self._cached_face_observations.clear()
                self._cached_coverage_details.clear()

            # This method already runs in a dedicated background thread/process.
            # Results must stay paired with their originating FramePacket; replaying
            # cached coordinates over a newer image creates background captures.
            if self.face_worker:
                if fast_only:
                    try:
                        face_tracks, face_observations, coverage_details = self.face_worker.analyze(
                            frame_packet, active_tracks, face_epoch, self.get_config(), fast_only=True)
                    except TypeError:
                        face_tracks, face_observations, coverage_details = self.face_worker.analyze(
                            frame_packet, active_tracks, face_epoch, self.get_config())
                else:
                    face_tracks, face_observations, coverage_details = self.face_worker.analyze(
                        frame_packet, active_tracks, face_epoch, self.get_config())
            else:
                if fast_only:
                    try:
                        raw_observations = self.face_coverage.detect(
                            frame_packet, active_tracks, fast_only=True)
                    except TypeError:
                        raw_observations = self.face_coverage.detect(frame_packet, active_tracks)
                else:
                    raw_observations = self.face_coverage.detect(frame_packet, active_tracks)
                face_observations = self.face_verifier.validate(frame_packet.image, raw_observations)
                face_tracks, face_observations = self.face_coverage.update(
                    face_observations, frame_packet.timestamp, face_epoch)
                coverage_details = self.face_coverage.diagnostics
            face_observations = [o for o in face_observations
                                 if o.frame_sequence == frame_packet.sequence
                                 and abs(o.timestamp-frame_packet.timestamp)<1e-4
                                 and not o.observation_id.startswith('face_anchor_')]
            valid_ids={o.track_id for o in face_observations}
            face_tracks=[t for t in face_tracks if t.track_id in valid_ids]
        except Exception as exc:
            face_error = str(exc)
            LOG.exception("Face discovery failed")
        face_ms = (time.perf_counter() - t_face_0) * 1000.0
        self.metrics.face_detection_ms = face_ms
        self.pipeline_scheduler.record_stage_latency("face_detection", face_ms)

        # ---------------------------------------------------------------------
        # Stage 4: Landmarks, 3D Head-Pose & Quality Scoring
        # ---------------------------------------------------------------------
        t_qual_0 = time.perf_counter()
        for obs in face_observations:
            # Estimate head pose from landmarks
            if obs.landmarks and obs.pose is None:
                try:
                    obs.pose = self.face_pose_estimator.estimate_observation(obs)
                except Exception as exc:
                    LOG.debug("Pose estimation fallback: %s", exc)

            # Ensure high-resolution face crop is available
            if obs.face_crop is None and frame_packet.image is not None:
                try:
                    obs.face_crop = extract_crop(frame_packet, obs.bbox, margin=0.15)
                except Exception:
                    pass

            # Multi-factor quality scoring (0-100)
            if obs.quality is None:
                try:
                    # Score the actual face, excluding padding, hair and background.
                    # Saving still uses the untouched padded crop from the master.
                    x1, y1, x2, y2 = obs.bbox
                    h, w = frame_packet.image.shape[:2]
                    tight = frame_packet.image[max(0,int(y1)):min(h,int(np.ceil(y2))),
                                               max(0,int(x1)):min(w,int(np.ceil(x2)))]
                    evaluated = replace(obs, face_crop=tight,
                                        landmarks=[(x-max(0,int(x1)),y-max(0,int(y1))) for x,y in obs.landmarks])
                    obs.quality = self.face_quality_evaluator.assess_observation(evaluated)
                except Exception as exc:
                    LOG.debug("Quality assessment fallback: %s", exc)
        qual_ms = (time.perf_counter() - t_qual_0) * 1000.0
        self.metrics.quality_eval_ms = qual_ms
        self.pipeline_scheduler.record_stage_latency("quality_eval", qual_ms)

        # ---------------------------------------------------------------------
        # Stage 5: Intelligent Capture Selection
        # ---------------------------------------------------------------------
        decisions: List[CaptureDecision] = []
        confirmed_ids = {t.track_id for t in face_tracks if t.confirmed}
        capture_observations = [o for o in face_observations if o.track_id in confirmed_ids
                                and o.quality is not None and o.confidence >= .5
                                and min(o.bbox[2]-o.bbox[0],o.bbox[3]-o.bbox[1]) >= int(self._config['min_face_pixels'])]
        if capture_observations:
            try:
                decisions = self.capture_selector.process_frame(
                    observations=capture_observations,
                    frame_sequence=frame_packet.sequence,
                    frame_image=frame_packet.image,
                    frame_shape=(frame_packet.height, frame_packet.width),
                )
            except Exception as exc:
                LOG.warning("Capture selector processing failed: %s", exc)

        # Filter candidates via privacy governance
        saved_captures: List[CaptureRecord] = []
        for dec in decisions:
            if dec.should_save_disk and dec.candidate:
                candidate = dec.candidate
                session_id = self.profile_repository.format_session_id(
                    self.camera_id, face_epoch, dec.track_id
                )

                # Governance validation
                try:
                    self.privacy_governance.validate_session(session_id)
                except Exception as g_err:
                    LOG.warning("Privacy governance rejected session %s: %s", session_id, g_err)
                    continue

                # Atomically persist candidate to SQLite & disk
                try:
                    record = self.profile_repository.save_candidate(
                        session_id=session_id,
                        candidate=candidate,
                    )
                    saved_captures.append(record)

                    # Retrieve updated profile
                    profile = self.profile_repository.get_profile(session_id)

                    # Emit WebSocket capture and profile updates
                    if self.people_event_emitter:
                        self.people_event_emitter.emit_capture(record.to_dict())
                        if profile:
                            self.people_event_emitter.emit_profile_update(profile.to_dict())

                    # Submit asynchronous super-resolution enhancement task
                    if self.enhancement_enabled and self.face_enhancer and record:
                        self._dispatch_async_enhancement(record, candidate.crop_original, session_id)

                except Exception as p_err:
                    LOG.error("Failed to persist capture candidate: %s", p_err)

        # ---------------------------------------------------------------------
        # Stage 6: Real-Time Frame Broadcasting & Snapshot
        # ---------------------------------------------------------------------
        # Measure completed analysis throughput; skipped frames cannot inflate FPS.
        self._fps_window.append(time.monotonic())
        if len(self._fps_window) > 1:
            elapsed = self._fps_window[-1] - self._fps_window[0]
            self.metrics.acquisition_fps = (len(self._fps_window) - 1) / elapsed if elapsed > 0 else 0.0

        frame_summary = {
            "sequence": frame_packet.sequence,
            "timestamp": frame_packet.timestamp,
            "pts": frame_packet.pts,
            "fps": round(float(self.metrics.acquisition_fps), 1),
            "width": frame_packet.width,
            "height": frame_packet.height,
            "epoch": face_epoch,
            "tracking_kind": "face",
            "active_count": len(confirmed_ids),
            "body_count": len(active_tracks),
            "tracks": [dict(t.to_dict(), tracking_kind="face") for t in face_tracks],
            "faces": [obs.to_dict() for obs in face_observations],
            "coverage": dict(coverage_details,
                             confirmed_faces=len(confirmed_ids),
                             pending_confirmation=len(face_tracks)-len(confirmed_ids),
                             capture_eligible=len(capture_observations), error=face_error),
            "captures_saved": len(saved_captures),
            "total_latency_ms": round((time.perf_counter() - t_frame_start) * 1000.0, 2),
        }

        with self.lock:
            self._last_snapshot = frame_summary

        if self.people_event_emitter:
            try:
                self.people_event_emitter.emit_frame(frame_summary)
            except Exception as e_err:
                LOG.debug("WebSocket frame emit notice: %s", e_err)

        # ---------------------------------------------------------------------
        # Stage 7: Periodic Governance & Cache Maintenance
        # ---------------------------------------------------------------------
        if frame_packet.sequence % int(self._config.get("governance_prune_interval_frames", 100)) == 0:
            self._run_maintenance_tasks()

        total_latency = (time.perf_counter() - t_frame_start) * 1000.0
        self.pipeline_scheduler.record_frame_latency(total_latency)
        return frame_summary

    # -------------------------------------------------------------------------
    # Asynchronous Enhancement Dispatch
    # -------------------------------------------------------------------------

    def _dispatch_async_enhancement(
        self,
        record: CaptureRecord,
        crop: np.ndarray,
        session_id: str,
    ) -> None:
        """Submit background super-resolution enhancement task without blocking acquisition."""
        def on_enhancement_complete(result: EnhancementResult) -> None:
            if not result.success or result.enhanced_image is None:
                LOG.debug("Enhancement skipped/fallback for capture %s", record.capture_id)
                return

            try:
                updated_record = self.profile_repository.attach_enhanced_image(
                    capture_id=record.capture_id,
                    enhanced_crop=result.enhanced_image,
                    enhancement_method=result.method,
                )
                if updated_record and self.people_event_emitter:
                    self.people_event_emitter.emit_enhancement_ready(updated_record.to_dict())
                    updated_profile = self.profile_repository.get_profile(session_id)
                    if updated_profile:
                        self.people_event_emitter.emit_profile_update(updated_profile.to_dict())
            except Exception as err:
                LOG.error("Failed to attach enhanced image for capture %s: %s", record.capture_id, err)

        try:
            self.face_enhancer.submit(
                image=crop,
                capture_record=record,
                callback=on_enhancement_complete,
            )
        except Exception as exc:
            LOG.warning("Could not submit enhancement task: %s", exc)

    # -------------------------------------------------------------------------
    # Maintenance Tasks (Privacy Governance & Cache Pruning)
    # -------------------------------------------------------------------------

    def _run_maintenance_tasks(self) -> None:
        """Periodic maintenance: prune expired captures, clean stale track caches."""
        try:
            if self.privacy_governance and self.profile_repository:
                self.privacy_governance.prune_expired_records(self.profile_repository)

            if self.capture_selector:
                self.capture_selector.prune_stale_tracks(max_age_seconds=60.0)

            if self.pipeline_scheduler:
                active_ids = {t.track_id for t in self.person_tracker.active_tracks}
                self.pipeline_scheduler.prune_stale_tracks(active_track_ids=active_ids)
        except Exception as err:
            LOG.debug("Maintenance pruning notice: %s", err)

    # -------------------------------------------------------------------------
    # Autonomous Stream Acquisition Loop
    # -------------------------------------------------------------------------

    def _autonomous_stream_loop(self) -> None:
        """Background thread consuming incoming frames with zero delay and smart dual-cadence."""
        last_seq = -1
        last_state_emit = 0.0
        fast_step_counter = 0
        seen_focus_gen = self._focus_generation

        while not self._stop_event.is_set():
            if self.analysis_focus == 'traffic':
                self._stop_event.wait(0.05)
                continue

            if self._focus_generation != seen_focus_gen:
                seen_focus_gen = self._focus_generation
                fast_step_counter = 0

            # 1. Wait directly on StreamManager's FrameMailbox (< 0.5ms wakeup on frame arrival)
            frames_box = getattr(self.stream, "frames", None) if self.stream is not None else None
            packet = None
            if frames_box is not None and hasattr(frames_box, "wait"):
                packet = frames_box.wait(last_seq, timeout=0.04)
            elif frames_box is not None:
                candidate = getattr(frames_box, "latest", None)
                if candidate is not None and getattr(candidate, "sequence", -1) > last_seq:
                    packet = candidate
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.005)

            # 2. Drain shared YOLO detections non-blockingly without delaying frame tracking
            shared_packet = None
            fresh_detections: Optional[List[PersonDetection]] = None
            fresh_ms = 0.0
            shared_mono = 0.0
            with self._shared_condition:
                if self._shared_latest is not None:
                    shared_packet, fresh_detections, fresh_ms = self._shared_latest
                    shared_mono = self._shared_latest_mono or time.monotonic()
                    self._shared_latest = None

            now_mono = time.monotonic()
            if fresh_detections is not None:
                mono_age = (now_mono - shared_mono) if shared_mono > 0 else 0.0
                pkt_age = (time.time() - getattr(shared_packet, "timestamp", time.time())) if shared_packet is not None else 0.0
                if mono_age <= 0.35 and pkt_age <= 0.6:
                    self._cached_body_detections = list(fresh_detections)
                    self._cached_body_ts = now_mono
                if packet is None and shared_packet is not None and getattr(shared_packet, "sequence", -1) > last_seq:
                    packet = shared_packet

            if packet is not None:
                seq = getattr(packet, "sequence", -1)
                if seq > last_seq:
                    if last_seq >= 0:
                        self.metrics.dropped_frames += max(0, seq - last_seq - 1)
                    last_seq = seq

                    exact_match = (
                        fresh_detections is not None
                        and shared_packet is not None
                        and getattr(shared_packet, "sequence", None) == seq
                    )
                    shared_detections: List[PersonDetection] = list(fresh_detections) if exact_match else []
                    shared_ms: float = float(fresh_ms) if exact_match else 0.0

                    has_active_faces = len(self._last_snapshot.get("tracks", [])) > 0
                    use_fast = bool(has_active_faces and (fast_step_counter % 6 != 0))

                    try:
                        if use_fast:
                            fast_step_counter += 1
                            self.process_frame(
                                packet,
                                shared_detections,
                                shared_ms,
                                async_face=False,
                                fast_only=True,
                            )
                        else:
                            fast_step_counter = 1
                            if exact_match:
                                body_hints_or_shared = shared_detections
                            elif (time.monotonic() - self._cached_body_ts) <= 0.35 and self._cached_body_detections:
                                body_hints_or_shared = list(self._cached_body_detections)
                            else:
                                body_hints_or_shared = []
                            self.process_frame(
                                packet,
                                body_hints_or_shared,
                                shared_ms,
                                async_face=False,
                                fast_only=False,
                            )
                    except Exception as exc:
                        LOG.error("Unhandled error in orchestrator frame loop: %s", exc)

            # Periodic state emission (every 1s)
            now = time.monotonic()
            if now - last_state_emit >= 1.0:
                if self.people_event_emitter:
                    try:
                        self.people_event_emitter.emit_state(self.get_state().to_dict())
                    except Exception:
                        pass
                last_state_emit = now
