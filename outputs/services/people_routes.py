"""People Facial Capture and Tracking REST API Endpoints.

Provides the Flask Blueprint `people_bp` for monitoring and controlling the real-time
multi-person facial analysis engine, retrieving person profiles and captures, and
adjusting runtime pipeline configuration.
"""
from __future__ import annotations

import copy
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    make_response,
    request,
    send_file,
)

try:
    from .face_contracts import (
        AnalysisState,
        CaptureRecord,
        PersonProfile,
        ProcessingMetrics,
    )
except ImportError:
    from services.face_contracts import (  # type: ignore
        AnalysisState,
        CaptureRecord,
        PersonProfile,
        ProcessingMetrics,
    )

LOG = logging.getLogger(__name__)

# Flask Blueprint with standard URL prefix
people_bp = Blueprint("people", __name__, url_prefix="/api/people")


# ============================================================================
# In-Memory Default Implementations for Standalone Operation & Dependency Injection
# ============================================================================


class InMemoryProfileRepository:
    """Thread-safe in-memory profile and capture repository."""

    def __init__(self, captures_dir: Optional[Union[str, Path]] = None) -> None:
        self.profiles: Dict[str, PersonProfile] = {}
        self.captures: Dict[str, CaptureRecord] = {}
        self.captures_dir = (
            Path(captures_dir).resolve() if captures_dir else (Path.cwd() / "data" / "captures").resolve()
        )
        self.storage_dir = self.captures_dir

    def save_profile(self, profile: PersonProfile) -> PersonProfile:
        self.profiles[profile.session_id] = profile
        return profile

    def get_profile(self, session_id: str) -> Optional[PersonProfile]:
        return self.profiles.get(session_id)

    def delete_profile(self, session_id: str) -> bool:
        if session_id not in self.profiles:
            return False

        profile = self.profiles.pop(session_id)
        # Delete associated captures from repository & disk if within captures_dir
        caps_to_clean: List[CaptureRecord] = list(profile.capture_history)
        caps_to_clean.extend(profile.captures_by_category.values())
        if profile.best_capture:
            caps_to_clean.append(profile.best_capture)

        for cap in caps_to_clean:
            self.captures.pop(cap.capture_id, None)
            for path_str in (cap.original_path, cap.enhanced_path):
                if path_str:
                    try:
                        p = Path(path_str).resolve()
                        if p.is_file() and p.is_relative_to(self.captures_dir):
                            p.unlink(missing_ok=True)
                    except Exception as err:
                        LOG.debug("Could not unlink capture file %s: %s", path_str, err)

        return True

    def list_profiles(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        sort_by: Optional[str] = None,
        min_quality: Optional[float] = None,
    ) -> Tuple[List[PersonProfile], int]:
        items = list(self.profiles.values())

        if status:
            items = [p for p in items if p.status == status]

        if min_quality is not None:
            items = [p for p in items if (p.best_quality_score or 0.0) >= min_quality]

        if sort_by:
            descending = sort_by.startswith("-")
            field_name = sort_by.lstrip("-").lstrip("+")

            def _sort_key(p: PersonProfile) -> Any:
                val = getattr(p, field_name, 0)
                return val if val is not None else 0

            try:
                items.sort(key=_sort_key, reverse=descending)
            except Exception:
                pass
        else:
            items.sort(key=lambda p: p.last_seen or 0.0, reverse=True)

        total = len(items)
        paginated = items[offset : offset + limit]
        return paginated, total

    def save_capture(self, capture: CaptureRecord) -> CaptureRecord:
        self.captures[capture.capture_id] = capture
        return capture

    def get_capture(self, capture_id: str) -> Optional[CaptureRecord]:
        return self.captures.get(capture_id)

    def delete_capture(self, capture_id: str) -> bool:
        return self.captures.pop(capture_id, None) is not None

    def count_profiles(self, **kwargs: Any) -> int:
        status = kwargs.get("status")
        min_quality = kwargs.get("min_quality")
        items = list(self.profiles.values())
        if status:
            items = [p for p in items if p.status == status]
        if min_quality is not None:
            items = [p for p in items if (p.best_quality_score or 0.0) >= min_quality]
        return len(items)

    def count_captures(self) -> int:
        return len(self.captures)


class InMemoryPeopleOrchestrator:
    """Default orchestrator managing AnalysisState, metrics, and runtime settings."""

    def __init__(self, repository: Optional[Any] = None) -> None:
        self.repository = repository or InMemoryProfileRepository()
        self.state = AnalysisState()
        self.metrics = ProcessingMetrics()
        self.config: Dict[str, Any] = {
            "face_confidence_threshold": 0.6,
            "quality_threshold": 65.0,
            "enhancement_enabled": True,
            "tracker_confidence_threshold": 0.5,
            "max_captures_per_person": 5,
        }

    def get_state(self) -> AnalysisState:
        if hasattr(self.repository, "count_profiles"):
            try:
                self.state.total_profiles_count = self.repository.count_profiles()
            except Exception:
                pass
        if hasattr(self.repository, "count_captures"):
            try:
                self.state.total_captures_count = self.repository.count_captures()
            except Exception:
                pass
        return self.state

    def start(self) -> bool:
        self.state.is_running = True
        self.state.started_at = time.time()
        self.state.stopped_at = None
        self.state.status_message = "Running"
        self.state.error = None
        return True

    def start_analysis(self, epoch_id: Optional[str] = None) -> bool:
        return self.start()

    def stop(self) -> bool:
        self.state.is_running = False
        self.state.stopped_at = time.time()
        self.state.status_message = "Stopped"
        return True

    def stop_analysis(self, drain_timeout: Optional[float] = None) -> bool:
        return self.stop()

    def is_running(self) -> bool:
        return self.state.is_running

    def get_metrics(self) -> ProcessingMetrics:
        return self.metrics

    def get_config(self) -> Dict[str, Any]:
        return dict(self.config)

    def configure(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        self.config.update(updates)
        return dict(self.config)


# Module-level instances for fallback/injection
_module_repository: Optional[Any] = None
_module_orchestrator: Optional[Any] = None


def set_orchestrator(orchestrator: Any) -> None:
    """Inject an orchestrator service instance at the module level."""
    global _module_orchestrator
    _module_orchestrator = orchestrator


def set_repository(repository: Any) -> None:
    """Inject a profile repository instance at the module level."""
    global _module_repository
    _module_repository = repository


def get_orchestrator() -> Any:
    """Retrieve active orchestrator from Flask current_app or fallback instance."""
    if current_app:
        orch = current_app.extensions.get("people_orchestrator")
        if orch is not None:
            return orch
    global _module_orchestrator
    if _module_orchestrator is None:
        _module_orchestrator = InMemoryPeopleOrchestrator(get_repository())
    return _module_orchestrator


def get_repository() -> Any:
    """Retrieve active profile repository from Flask current_app or fallback instance."""
    if current_app:
        repo = current_app.extensions.get("people_repository")
        if repo is not None:
            return repo
    global _module_repository
    if _module_repository is None:
        _module_repository = InMemoryProfileRepository()
    return _module_repository


def init_people_routes(
    app: Any,
    orchestrator: Optional[Any] = None,
    repository: Optional[Any] = None,
    captures_dir: Optional[Union[str, Path]] = None,
) -> Blueprint:
    """Register and bind people_bp to a Flask application with optional dependency injection."""
    if repository is not None:
        app.extensions["people_repository"] = repository
    if orchestrator is not None:
        app.extensions["people_orchestrator"] = orchestrator
    if captures_dir is not None:
        app.config["CAPTURES_BASE_DIR"] = str(Path(captures_dir).resolve())

    # Avoid duplicate blueprint registration
    if "people" not in app.blueprints:
        app.register_blueprint(people_bp)
    return people_bp


# ============================================================================
# Security & Path Traversal Helpers
# ============================================================================


def _is_safe_identifier(val: str) -> bool:
    """Validate that an ID string contains no path traversal sequences or illegal chars."""
    if not val or not isinstance(val, str):
        return False
    # Reject directory traversal indicators and null bytes
    forbidden = ("..", "/", "\\", "\0")
    if any(seq in val for seq in forbidden):
        return False
    # Reject Windows drive letter paths like C:
    if len(val) >= 2 and val[1] == ":" and val[0].isalpha():
        return False
    return True


def _get_allowed_captures_dir() -> Path:
    """Return the absolute base directory allowed for serving capture image files."""
    if current_app and "CAPTURES_BASE_DIR" in current_app.config:
        return Path(current_app.config["CAPTURES_BASE_DIR"]).resolve()
    repo = get_repository()
    if hasattr(repo, "captures_dir") and repo.captures_dir:
        return Path(repo.captures_dir).resolve()
    if hasattr(repo, "storage_dir") and repo.storage_dir:
        return Path(repo.storage_dir).resolve()
    if current_app:
        return (Path(current_app.root_path) / "data" / "captures").resolve()
    return (Path.cwd() / "data" / "captures").resolve()


def _resolve_and_verify_image_path(path_str: str) -> Optional[Path]:
    """Verify that path_str points to a path within the allowed captures base directory."""
    if not path_str or not isinstance(path_str, str):
        return None
    # Reject explicit relative traversal patterns
    if ".." in path_str:
        return None

    base_dir = _get_allowed_captures_dir()
    try:
        candidate = Path(path_str)
        resolved = (base_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        # Ensure the resolved file path resides strictly inside base_dir
        resolved.relative_to(base_dir)
        return resolved
    except (ValueError, RuntimeError):
        return None


def _fetch_profiles_and_total(
    repo: Any,
    limit: int,
    offset: int,
    status: Optional[str],
    sort_by: Optional[str],
    min_quality: Optional[float],
) -> Tuple[List[Any], int]:
    """Fetch profiles and total count adapting across repository implementations."""
    sig = inspect.signature(repo.list_profiles)
    params = sig.parameters

    if "limit" in params and "offset" in params:
        result = repo.list_profiles(
            limit=limit,
            offset=offset,
            status=status,
            sort_by=sort_by,
            min_quality=min_quality,
        )
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        elif isinstance(result, list):
            total = (
                repo.count_profiles(status=status, min_quality=min_quality)
                if hasattr(repo, "count_profiles")
                else len(result)
            )
            return result, total

    # Handle ProfileRepository (using page and page_size)
    page = (offset // limit) + 1 if limit > 0 else 1
    page_size = limit

    order = "desc"
    clean_sort = "recent"
    if sort_by:
        if sort_by.startswith("-"):
            order = "desc"
            clean_sort = sort_by[1:]
        elif sort_by.startswith("+"):
            order = "asc"
            clean_sort = sort_by[1:]
        else:
            order = "asc"
            clean_sort = sort_by

    total = 0
    if hasattr(repo, "count_profiles"):
        try:
            total = repo.count_profiles(status=status, min_quality=min_quality)
        except Exception:
            total = 0

    profiles = repo.list_profiles(
        page=page,
        page_size=page_size,
        status=status,
        sort_by=clean_sort,
        order=order,
        min_quality=min_quality,
        include_captures=False,
    )
    return profiles, total


# ============================================================================
# API Endpoints
# ============================================================================


@people_bp.get("/state")
def get_people_state() -> Response:
    """GET /api/people/state: Return current AnalysisState, active counts, and metrics."""
    try:
        orch = get_orchestrator()
        state = orch.get_state() if hasattr(orch, "get_state") else getattr(orch, "state", None)
        if state is None:
            state = AnalysisState()
        state_dict = state.to_dict() if hasattr(state, "to_dict") else dict(state)

        metrics = orch.get_metrics() if hasattr(orch, "get_metrics") else getattr(orch, "metrics", None)
        metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else (dict(metrics) if metrics else {})

        active_counts = {
            "active_tracks": state_dict.get("active_tracks_count", 0),
            "total_profiles": state_dict.get("total_profiles_count", 0),
            "total_captures": state_dict.get("total_captures_count", 0),
            "pending_enhancements": state_dict.get("pending_enhancements", 0),
        }

        response_payload = {
            **state_dict,
            "state": state_dict,
            "active_counts": active_counts,
            "metrics": metrics_dict,
            "coverage": getattr(orch, 'snapshot', {}).get('coverage', {}),
        }
        return jsonify(response_payload)
    except Exception as exc:
        LOG.exception("Error retrieving people state: %s", exc)
        return jsonify({"error": f"Failed to retrieve state: {str(exc)}"}), 500


@people_bp.post("/control")
def control_analysis() -> Response:
    """POST /api/people/control: Start or Stop the facial analysis pipeline."""
    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a valid JSON object"}), 400

    action = payload.get("action")
    if not isinstance(action, str) or action not in ("start", "stop"):
        return jsonify({"error": "Field 'action' must be either 'start' or 'stop'"}), 400

    try:
        orch = get_orchestrator()
        if action == "start":
            if hasattr(orch, "start_analysis") and callable(orch.start_analysis):
                orch.start_analysis()
            elif hasattr(orch, "start") and callable(orch.start):
                orch.start()
        else:
            if hasattr(orch, "stop_analysis") and callable(orch.stop_analysis):
                orch.stop_analysis()
            elif hasattr(orch, "stop") and callable(orch.stop):
                orch.stop()

        state = orch.get_state() if hasattr(orch, "get_state") else getattr(orch, "state", None)
        state_dict = state.to_dict() if hasattr(state, "to_dict") else dict(state)

        return jsonify({
            "status": "ok",
            "action": action,
            "is_running": state_dict.get("is_running", action == "start"),
            "state": state_dict,
        }), 200
    except Exception as exc:
        LOG.exception("Control operation failed: %s", exc)
        return jsonify({"error": f"Failed to execute control action: {str(exc)}"}), 500


@people_bp.get("/profiles")
def list_profiles() -> Response:
    """GET /api/people/profiles: List anonymous person profiles with pagination and filters."""
    # Validate limit
    limit_param = request.args.get("limit", "50")
    try:
        limit = int(limit_param)
        if limit < 1 or limit > 1000:
            return jsonify({"error": "Query param 'limit' must be an integer between 1 and 1000"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Query param 'limit' must be a valid integer"}), 400

    # Validate offset
    offset_param = request.args.get("offset", "0")
    try:
        offset = int(offset_param)
        if offset < 0:
            return jsonify({"error": "Query param 'offset' must be a non-negative integer"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Query param 'offset' must be a valid integer"}), 400

    # Validate status filter
    status = request.args.get("status")
    if status is not None:
        status = status.strip()
        if status and status not in ("active", "lost", "completed"):
            return jsonify({"error": "Query param 'status' must be one of: 'active', 'lost', 'completed'"}), 400
        if not status:
            status = None

    # Validate min_quality filter
    min_quality_param = request.args.get("min_quality")
    min_quality: Optional[float] = None
    if min_quality_param is not None and min_quality_param != "":
        try:
            min_quality = float(min_quality_param)
            if min_quality < 0.0 or min_quality > 100.0:
                return jsonify({"error": "Query param 'min_quality' must be a float between 0.0 and 100.0"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Query param 'min_quality' must be a valid numeric float"}), 400

    # Validate sort_by
    sort_by = request.args.get("sort_by")
    if sort_by is not None:
        field_name = sort_by.lstrip("-").lstrip("+")
        allowed_sort_fields = {
            "first_seen",
            "last_seen",
            "duration",
            "best_quality_score",
            "quality",
            "total_observations",
            "track_id",
            "recent",
            "created_at",
        }
        if field_name not in allowed_sort_fields:
            return jsonify({
                "error": f"Invalid sort_by field '{sort_by}'. Allowed: {', '.join(sorted(allowed_sort_fields))}"
            }), 400

    try:
        repo = get_repository()
        profiles, total = _fetch_profiles_and_total(
            repo=repo,
            limit=limit,
            offset=offset,
            status=status,
            sort_by=sort_by,
            min_quality=min_quality,
        )

        serialized_profiles = [p.to_dict() if hasattr(p, "to_dict") else p for p in profiles]
        return jsonify({
            "profiles": serialized_profiles,
            "total": total,
            "limit": limit,
            "offset": offset,
        }), 200
    except Exception as exc:
        LOG.exception("Failed to list profiles: %s", exc)
        return jsonify({"error": f"Failed to list profiles: {str(exc)}"}), 500


@people_bp.get("/profiles/<session_id>")
def get_profile(session_id: str) -> Response:
    """GET /api/people/profiles/<session_id>: Detailed profile with all capture angles and history."""
    if not _is_safe_identifier(session_id):
        return jsonify({"error": "Invalid session_id parameter"}), 400

    try:
        repo = get_repository()
        profile = repo.get_profile(session_id)
        if profile is None:
            return jsonify({"error": f"Profile '{session_id}' not found"}), 404

        data = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)
        # Ensure complete capture history records are included
        history = getattr(profile, "capture_history", [])
        data["capture_history"] = [c.to_dict() if hasattr(c, "to_dict") else c for c in history]

        return jsonify(data), 200
    except Exception as exc:
        LOG.exception("Failed to get profile %s: %s", session_id, exc)
        return jsonify({"error": f"Failed to get profile: {str(exc)}"}), 500


@people_bp.delete("/profiles/<session_id>")
def delete_profile(session_id: str) -> Response:
    """DELETE /api/people/profiles/<session_id>: Delete profile and associated captures."""
    if not _is_safe_identifier(session_id):
        return jsonify({"error": "Invalid session_id parameter"}), 400

    try:
        repo = get_repository()
        existing = repo.get_profile(session_id)
        if existing is None:
            return jsonify({"error": f"Profile '{session_id}' not found"}), 404

        deleted = repo.delete_profile(session_id)
        if not deleted:
            return jsonify({"error": f"Profile '{session_id}' could not be deleted"}), 404

        return jsonify({"status": "deleted", "deleted": True, "session_id": session_id}), 200
    except Exception as exc:
        LOG.exception("Failed to delete profile %s: %s", session_id, exc)
        return jsonify({"error": f"Failed to delete profile: {str(exc)}"}), 500


@people_bp.get("/captures/<capture_id>/image")
def get_capture_image(capture_id: str) -> Response:
    """GET /api/people/captures/<capture_id>/image: Serve original or enhanced face crop image."""
    # Path traversal protection on capture_id
    if not _is_safe_identifier(capture_id):
        return jsonify({"error": "Invalid capture_id: path traversal detected"}), 400

    # Type parameter validation
    image_type = request.args.get("type", "original").lower().strip()
    if image_type not in ("original", "enhanced"):
        return jsonify({"error": "Query param 'type' must be either 'original' or 'enhanced'"}), 400

    try:
        repo = get_repository()
        capture = repo.get_capture(capture_id)
        if capture is None:
            return jsonify({"error": f"Capture '{capture_id}' not found"}), 404

        # Select file path based on type
        if image_type == "enhanced":
            raw_path = getattr(capture, "enhanced_path", None)
            if not raw_path:
                return jsonify({"error": f"Enhanced image not available for capture '{capture_id}'"}), 404
        else:
            raw_path = getattr(capture, "original_path", None)
            if not raw_path:
                return jsonify({"error": f"Original image not available for capture '{capture_id}'"}), 404

        # Path traversal protection on the file path itself
        verified_path = _resolve_and_verify_image_path(raw_path)
        if verified_path is None:
            return jsonify({"error": "Access denied: Path traversal detected on capture file"}), 403

        if not verified_path.is_file():
            return jsonify({"error": "Capture image file not found on disk"}), 404

        # Serve file with proper MIME type and caching headers
        mime = "image/png" if verified_path.suffix.lower() == ".png" else "image/jpeg"
        response = make_response(send_file(str(verified_path), mimetype=mime, as_attachment=False))
        response.headers["Cache-Control"] = "public, max-age=86400, immutable"
        response.headers["ETag"] = f'"{capture_id}-{image_type}"'
        return response
    except Exception as exc:
        LOG.exception("Failed to serve capture image for %s: %s", capture_id, exc)
        return jsonify({"error": f"Failed to serve image: {str(exc)}"}), 500


@people_bp.get("/metrics")
def get_metrics() -> Response:
    """GET /api/people/metrics: Detailed latency, FPS, memory, and pipeline statistics."""
    try:
        orch = get_orchestrator()
        metrics = orch.get_metrics() if hasattr(orch, "get_metrics") else getattr(orch, "metrics", None)
        metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else (dict(metrics) if metrics else {})

        # Latency breakdown
        latency_ms = {
            "detection": metrics_dict.get("detection_ms", 0.0),
            "tracking": metrics_dict.get("tracking_ms", 0.0),
            "face_detection": metrics_dict.get("face_detection_ms", 0.0),
            "quality_eval": metrics_dict.get("quality_eval_ms", 0.0),
            "enhancement": metrics_dict.get("enhancement_ms", 0.0),
            "total_pipeline_ms": round(
                metrics_dict.get("detection_ms", 0.0)
                + metrics_dict.get("tracking_ms", 0.0)
                + metrics_dict.get("face_detection_ms", 0.0)
                + metrics_dict.get("quality_eval_ms", 0.0)
                + metrics_dict.get("enhancement_ms", 0.0),
                2,
            ),
        }

        # FPS stats
        acq_fps = metrics_dict.get("acquisition_fps", 0.0)
        tot_ms = latency_ms["total_pipeline_ms"]
        fps_stats = {
            "acquisition_fps": acq_fps,
            "processing_fps": round(1000.0 / max(1.0, tot_ms), 1) if tot_ms > 0 else 0.0,
        }

        # Pipeline stats
        pipeline_stats = {
            "frame_sequence": metrics_dict.get("frame_sequence", 0),
            "active_tracks": metrics_dict.get("active_tracks", 0),
            "queue_depth": metrics_dict.get("queue_depth", 0),
            "dropped_frames": metrics_dict.get("dropped_frames", 0),
        }

        # Hardware memory statistics
        memory_stats: Dict[str, float] = {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0}
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            mem_info = proc.memory_info()
            memory_stats = {
                "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
                "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
                "percent": round(proc.memory_percent(), 2),
            }
        except Exception:
            pass

        return jsonify({
            **metrics_dict,
            "latency_ms": latency_ms,
            "fps": fps_stats,
            "pipeline": pipeline_stats,
            "memory": memory_stats,
        }), 200
    except Exception as exc:
        LOG.exception("Failed to retrieve pipeline metrics: %s", exc)
        return jsonify({"error": f"Failed to retrieve metrics: {str(exc)}"}), 500


@people_bp.post("/config")
def update_config() -> Response:
    """POST /api/people/config: Update runtime settings (thresholds, enhancement toggle)."""
    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a valid JSON object"}), 400
    if len(payload) == 0:
        return jsonify({"error": "Request body cannot be empty"}), 400

    # Validate confidence thresholds (0.0 to 1.0)
    conf_keys = [
        "face_confidence_threshold",
        "confidence_threshold",
        "tracker_confidence_threshold",
        "detection_confidence_threshold",
        "iou_threshold",
    ]
    for key in conf_keys:
        if key in payload:
            val = payload[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return jsonify({"error": f"Field '{key}' must be a numeric float"}), 400
            if val < 0.0 or val > 1.0:
                return jsonify({"error": f"Field '{key}' must be between 0.0 and 1.0"}), 400

    # Validate quality thresholds (0.0 to 100.0)
    qual_keys = ["quality_threshold", "min_quality_score", "min_quality"]
    for key in qual_keys:
        if key in payload:
            val = payload[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return jsonify({"error": f"Field '{key}' must be a numeric float"}), 400
            if val < 0.0 or val > 100.0:
                return jsonify({"error": f"Field '{key}' must be between 0.0 and 100.0"}), 400

    # Validate enhancement toggle (bool)
    enh_keys = ["enhancement_enabled", "enable_enhancement", "enhancement"]
    for key in enh_keys:
        if key in payload:
            val = payload[key]
            if not isinstance(val, bool):
                return jsonify({"error": f"Field '{key}' must be a boolean"}), 400

    # Validate positive integers
    int_keys = ["max_captures_per_person", "max_history_length"]
    for key in int_keys:
        if key in payload:
            val = payload[key]
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                return jsonify({"error": f"Field '{key}' must be a positive integer"}), 400

    try:
        orch = get_orchestrator()
        if hasattr(orch, "configure"):
            updated_config = orch.configure(payload)
        elif hasattr(orch, "update_config"):
            updated_config = orch.update_config(payload)
        else:
            updated_config = payload

        return jsonify({"status": "ok", "config": updated_config}), 200
    except Exception as exc:
        LOG.exception("Config update failed: %s", exc)
        return jsonify({"error": f"Failed to update config: {str(exc)}"}), 500
