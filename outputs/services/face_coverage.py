"""Multi-scale face discovery, independent of body detection and association.

Only spatial tracking is used. IDs describe a short continuous observation,
not a person's identity. All saved crops come from the original source frame.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import uuid
import time
import math
import cv2
import numpy as np

from services.face_contracts import FaceObservation, FramePacket, PersonDetection
from services.face_detector import FaceDetector, box_iou, is_valid_face
from services.person_tracker import PersonTracker


@dataclass
class CoverageConfig:
    detail_size: int = 512
    tile_size: int = 512
    detail_regions_per_frame: int = 4
    tiles_per_frame: int = 2
    min_face_pixels: int = 16


def merge_faces(observations: list[FaceObservation]) -> list[FaceObservation]:
    """NMS across scales; containment also removes duplicate enlarged boxes."""
    kept = []
    for obs in sorted(observations, key=lambda x: x.confidence, reverse=True):
        a = obs.bbox
        # A coarse-scale box enclosing two separate, tighter faces is a merge,
        # not a third person. Prefer the distinct observations at the finer scale.
        contained = [o for o in observations if o is not obs
                     and (o.bbox[2]-o.bbox[0])*(o.bbox[3]-o.bbox[1]) < .55*(a[2]-a[0])*(a[3]-a[1])
                     and a[0] <= (o.bbox[0]+o.bbox[2])/2 <= a[2]
                     and a[1] <= (o.bbox[1]+o.bbox[3])/2 <= a[3]]
        if any(box_iou(x.bbox,y.bbox)<.1 for i,x in enumerate(contained) for y in contained[i+1:]):
            continue
        duplicate = False
        for other in kept:
            b = other.bbox
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            small = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
            if box_iou(a, b) > .35 or (small > 0 and intersection / small > .85):
                duplicate = True
                break
        if not duplicate:
            kept.append(obs)
    return kept


class FaceCoverage:
    def __init__(self, detector: FaceDetector, config: CoverageConfig | None = None):
        self.detector = detector
        # Separate fixed-size detail graph avoids rebuilding the global graph
        # between every ROI and the next full-frame scan.
        self.detail_detector = FaceDetector(replace(detector.config, max_inference_dim=0, fixed_input_size=0))
        self.tile_detector = FaceDetector(replace(detector.config, max_inference_dim=0, fixed_input_size=0))
        self.config = config or CoverageConfig()
        self.tracker = PersonTracker(session_id='faces', lifetime=3.0, confirm_hits=2,
                                     high_score_threshold=.45, low_score_threshold=.25,
                                     match_threshold=.95, match_threshold_low=.95,
                                     unconfirmed_lifetime=2.5, max_area_ratio=2.8)
        self._epoch = None
        self._tile_cursor = 0
        self._last_detail: dict[str, float] = {}
        self.diagnostics = {}
        self._previous_gray = None
        self._previous_thumb = None
        self._previous_faces = []
        self._flow_boxes = {}
        self._camera_shift = (0.0, 0.0)
        self._hann_window = cv2.createHanningWindow((240, 135), cv2.CV_32F)

    def reset(self, epoch: str):
        self._epoch = epoch
        self._tile_cursor = 0
        self._last_detail.clear()
        self.tracker.reset(session_id=f'faces:{epoch}')
        self._previous_gray = None
        self._previous_thumb = None
        self._previous_faces = []
        self._flow_boxes = {}
        self._camera_shift = (0.0, 0.0)

    def _motion_hints(self, packet):
        """Forward/backward checked optical flow assists spatial association (< 3ms on 4K).

        A hint never creates an observation or a saved face. A fresh detector
        result is always required to keep a track visible and to capture it.
        """
        img = packet.image
        h, w = img.shape[:2]

        # Fast strided decimation before grayscale conversion cuts 25MB 4K prep to < 1ms.
        # Row-strided views preserve contiguous scanlines so OpenCV SIMD resize runs copy-free.
        if w >= 3200:
            target_w = min(960, max(1, w // 4))
            target_h = max(1, round(h * (target_w / float(w))))
            sub = cv2.resize(img[::4], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
        elif w > 1280:
            target_w = min(960, max(1, w // 2))
            target_h = max(1, round(h * (target_w / float(w))))
            sub = cv2.resize(img[::2], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
        elif w > 960:
            target_h = max(1, round(h * (960.0 / float(w))))
            sub = cv2.resize(img, (960, target_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        thumb = cv2.resize(gray, (240, 135), interpolation=cv2.INTER_AREA).astype(np.float32)
        thumb -= float(thumb.mean())
        self._flow_boxes = {}
        dx_cam = 0.0
        dy_cam = 0.0
        pc_response = 0.0

        prev_gray = self._previous_gray
        if prev_gray is not None and prev_gray.shape == gray.shape:
            if self._previous_thumb is not None:
                shift_thumb, pc_response = cv2.phaseCorrelate(
                    self._previous_thumb, thumb, self._hann_window
                )
                # Subtract 0.5 on odd height 135 to compensate OpenCV fftShift integer-half offset
                raw_dx = float(shift_thumb[0]) * (w / 240.0)
                raw_dy = (float(shift_thumb[1]) - 0.5) * (h / 135.0)
                if (
                    pc_response > 0.15
                    and math.isfinite(raw_dx + raw_dy)
                    and math.hypot(raw_dx, raw_dy) < w * 0.25
                ):
                    dx_cam = raw_dx
                    dy_cam = raw_dy

            if self._previous_faces:
                gh, gw = gray.shape[:2]
                sx = gw / float(w)
                sy = gh / float(h)
                points = np.asarray(
                    [
                        ((a + (c - a) * x) * sx, (b + (d - b) * y) * sy)
                        for _, (a, b, c, d) in self._previous_faces
                        for y in (0.2, 0.5, 0.8)
                        for x in (0.2, 0.5, 0.8)
                    ],
                    dtype=np.float32,
                ).reshape(-1, 1, 2)

                initial = None
                flags = 0
                if pc_response > 0.20 and (abs(dx_cam) > 0.2 or abs(dy_cam) > 0.2):
                    initial = points + np.array([[[dx_cam * sx, dy_cam * sy]]], dtype=np.float32)
                    flags = cv2.OPTFLOW_USE_INITIAL_FLOW

                # Restrict PyrLK Scharr/Gaussian pyramid build to the active face union ROI
                pad = 96
                if initial is not None:
                    min_x = min(float(points[:, 0, 0].min()), float(initial[:, 0, 0].min()))
                    min_y = min(float(points[:, 0, 1].min()), float(initial[:, 0, 1].min()))
                    max_x = max(float(points[:, 0, 0].max()), float(initial[:, 0, 0].max()))
                    max_y = max(float(points[:, 0, 1].max()), float(initial[:, 0, 1].max()))
                else:
                    min_x = float(points[:, 0, 0].min())
                    min_y = float(points[:, 0, 1].min())
                    max_x = float(points[:, 0, 0].max())
                    max_y = float(points[:, 0, 1].max())

                rx1 = max(0, min(gw, int(math.floor(min_x)) - pad))
                ry1 = max(0, min(gh, int(math.floor(min_y)) - pad))
                rx2 = max(0, min(gw, int(math.ceil(max_x)) + pad))
                ry2 = max(0, min(gh, int(math.ceil(max_y)) + pad))

                if (rx2 - rx1) >= 32 and (ry2 - ry1) >= 32:
                    p_sub = prev_gray[ry1:ry2, rx1:rx2]
                    c_sub = gray[ry1:ry2, rx1:rx2]
                    off = np.array([[[float(rx1), float(ry1)]]], dtype=np.float32)
                    pts_loc = points - off
                    init_loc = (initial - off) if initial is not None else None
                else:
                    p_sub = prev_gray
                    c_sub = gray
                    pts_loc = points
                    init_loc = initial

                lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03)
                forward, ok, _ = cv2.calcOpticalFlowPyrLK(
                    p_sub, c_sub, pts_loc, init_loc,
                    winSize=(21, 21), maxLevel=2, criteria=lk_criteria, flags=flags,
                )
                if forward is not None and ok is not None:
                    backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(
                        c_sub, p_sub, forward, pts_loc.copy(),
                        winSize=(21, 21), maxLevel=2, criteria=lk_criteria,
                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                    )
                    if backward is not None and back_ok is not None:
                        fb_err = np.linalg.norm(backward - pts_loc, axis=2).ravel()
                        valid = (ok.ravel() != 0) & (back_ok.ravel() != 0) & (fb_err < 1.5)
                        inv_sx = 1.0 / sx
                        inv_sy = 1.0 / sy
                        for n, (ident, box) in enumerate(self._previous_faces):
                            mask = valid[n * 9 : (n + 1) * 9]
                            if int(np.count_nonzero(mask)) < 5:
                                continue
                            delta = (
                                forward[n * 9 : (n + 1) * 9] - pts_loc[n * 9 : (n + 1) * 9]
                            ).reshape(-1, 2)[mask]
                            med_dx, med_dy = np.median(delta, axis=0)
                            dx = float(med_dx) * inv_sx
                            dy = float(med_dy) * inv_sy
                            if math.isfinite(dx + dy) and math.hypot(dx, dy) < w * 0.25:
                                self._flow_boxes[ident] = (
                                    box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy
                                )

        self._camera_shift = (round(dx_cam, 2), round(dy_cam, 2))
        self.diagnostics['camera_shift'] = [self._camera_shift[0], self._camera_shift[1]]
        self._previous_gray = gray
        self._previous_thumb = thumb

    def _velocity_predicted_box(
        self, track, timestamp: float
    ) -> tuple[float, float, float, float]:
        """Predict current track bounding box from kinematic velocity when flow is absent."""
        x1, y1, x2, y2 = track.bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        dt = min(1.0, max(0.0, float(timestamp) - float(track.last_seen)))
        vx, vy = getattr(track, 'velocity', (0.0, 0.0))
        max_shift = max(3.5 * max(bw, bh), 280.0)
        dx = max(-max_shift, min(max_shift, float(vx) * dt))
        dy = max(-max_shift, min(max_shift, float(vy) * dt))
        if math.hypot(vx, vy) < 5.0 and (abs(self._camera_shift[0]) > 1.0 or abs(self._camera_shift[1]) > 1.0):
            dx = max(-max_shift, min(max_shift, dx + self._camera_shift[0]))
            dy = max(-max_shift, min(max_shift, dy + self._camera_shift[1]))
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    @staticmethod
    def _expand_and_clamp_region(
        box: tuple[float, float, float, float],
        pad_x: float,
        pad_y: float,
        min_side: float,
        w: int,
        h: int,
    ) -> tuple[float, float, float, float]:
        """Expand box by (pad_x * bw, pad_y * bh) and clamp to at least min_side within (w, h)."""
        x1, y1, x2, y2 = box
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        rx1 = float(x1) - pad_x * bw
        ry1 = float(y1) - pad_y * bh
        rx2 = float(x2) + pad_x * bw
        ry2 = float(y2) + pad_y * bh
        rcx = 0.5 * (rx1 + rx2)
        rcy = 0.5 * (ry1 + ry2)
        rw = min(float(w), max(min_side, rx2 - rx1))
        rh = min(float(h), max(min_side, ry2 - ry1))
        cx1 = max(0.0, min(float(w) - rw, rcx - 0.5 * rw))
        cy1 = max(0.0, min(float(h) - rh, rcy - 0.5 * rh))
        return (cx1, cy1, min(float(w), cx1 + rw), min(float(h), cy1 + rh))

    def _extract_hint_regions(self, extra_hints, w: int, h: int) -> list[tuple[float, float, float, float]]:
        """Normalize heterogeneous extra_hints into valid native-frame regions (>= 96x96 px)."""
        out: list[tuple[float, float, float, float]] = []
        if not extra_hints:
            return out
        for hint in extra_hints:
            if hint is None:
                continue
            raw_box = None
            is_body = False
            if isinstance(hint, (tuple, list)) and len(hint) >= 4:
                raw_box = (float(hint[0]), float(hint[1]), float(hint[2]), float(hint[3]))
            elif isinstance(hint, dict):
                reg = hint.get('region') or hint.get('face_bbox')
                if reg is not None and len(reg) >= 4:
                    raw_box = (float(reg[0]), float(reg[1]), float(reg[2]), float(reg[3]))
                elif hint.get('bbox') is not None and len(hint['bbox']) >= 4:
                    b = hint['bbox']
                    raw_box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                    is_body = True
            elif hasattr(hint, 'bbox'):
                fb = getattr(hint, 'face_bbox', None)
                if fb is not None and len(fb) >= 4:
                    raw_box = (float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3]))
                else:
                    b = hint.bbox
                    raw_box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                    is_body = True
            if raw_box is None or raw_box[2] <= raw_box[0] or raw_box[3] <= raw_box[1]:
                continue
            if is_body:
                bx1, by1, bx2, by2 = raw_box
                bw, bh = bx2 - bx1, by2 - by1
                head_box = (bx1 - 0.12 * bw, by1 - 0.08 * bh, bx2 + 0.12 * bw, by1 + 0.38 * bh)
                out.append(self._expand_and_clamp_region(head_box, 0.0, 0.0, 96.0, w, h))
            else:
                out.append(self._expand_and_clamp_region(raw_box, 0.0, 0.0, 96.0, w, h))
        return out

    def _detail(self, packet: FramePacket, region, *, tile=False) -> list[FaceObservation]:
        h, w = packet.image.shape[:2]
        x1, y1, x2, y2 = region
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2-x1 < 16 or y2-y1 < 16:
            return []
        crop = packet.image[y1:y2, x1:x2]
        side = self.config.tile_size if tile else self.config.detail_size
        ratio = min(1., side / max(crop.shape[:2]))
        nw, nh = max(1, round(crop.shape[1]*ratio)), max(1, round(crop.shape[0]*ratio))
        view = np.zeros((side, side, 3), dtype=np.uint8)
        view[:nh, :nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA) if ratio < 1 else crop
        result = []
        detector = self.tile_detector if tile else self.detail_detector
        threshold = max(0.75, self.detector.config.score_threshold) if tile else self.detector.config.roi_score_threshold
        for raw in detector._detect_raw(view, threshold):
            a, b, c, d = raw['bbox']
            if not (0 <= (a+c)/2 < nw and 0 <= (b+d)/2 < nh):
                continue
            bbox = (x1+max(0,a)/ratio, y1+max(0,b)/ratio,
                    x1+min(nw,c)/ratio, y1+min(nh,d)/ratio)
            if min(bbox[2]-bbox[0], bbox[3]-bbox[1]) < self.config.min_face_pixels:
                continue

            # Extract face crop from view to validate against landmark geometry and contrast
            fa, fb, fc, fd = max(0, int(round(a))), max(0, int(round(b))), min(nw, int(round(c))), min(nh, int(round(d)))
            face_crop = view[fb:fd, fa:fc] if (fd > fb and fc > fa) else None
            if not is_valid_face(crop=face_crop, landmarks=raw.get('landmarks'), bbox=(a, b, c, d)):
                continue

            obs = FaceObservation(uuid.uuid4().hex, -1, packet.timestamp, bbox, raw['confidence'],
                                  [(x1+x/ratio, y1+y/ratio) for x,y in raw['landmarks']])
            result.append(obs)
        return result

    def detect(
        self,
        packet: FramePacket,
        body_tracks=(),
        *,
        fast_only: bool = False,
        extra_hints=(),
    ) -> list[FaceObservation]:
        started = time.perf_counter()
        self._motion_hints(packet)
        motion_done = time.perf_counter()
        h, w = packet.image.shape[:2]

        detail_unpatched = ('_detail' not in self.__dict__) and (type(self)._detail is FaceCoverage._detail)
        use_mosaic = detail_unpatched and hasattr(self.detail_detector, 'detect_regions_mosaic')
        use_tile_mosaic = detail_unpatched and hasattr(self.tile_detector, 'detect_regions_mosaic')

        # Focused fast_only tracking mode: skip heavy full-frame 1280px scan & background tiles
        if fast_only and (self.tracker.active_tracks or self.tracker.lost_tracks or extra_hints):
            tracked_candidates = [
                t for t in (self.tracker.active_tracks + self.tracker.lost_tracks)
                if (packet.timestamp - t.last_seen) <= 1.2
            ]
            tracked_candidates.sort(
                key=lambda t: (0 if t.state == 'active' else 1, 0 if t.confirmed else 1, -t.last_seen)
            )
            regions: list[tuple[float, float, float, float]] = []
            for t in tracked_candidates:
                pred_box = self._flow_boxes.get(
                    t.track_id, self._velocity_predicted_box(t, packet.timestamp)
                )
                regions.append(
                    self._expand_and_clamp_region(pred_box, 0.75, 0.70, 96.0, w, h)
                )
                if len(regions) >= 8:
                    break

            if len(regions) < 8 and extra_hints:
                for hint_reg in self._extract_hint_regions(extra_hints, w, h):
                    regions.append(hint_reg)
                    if len(regions) >= 8:
                        break

            observations: list[FaceObservation] = []
            if regions:
                if use_mosaic:
                    observations.extend(
                        self.detail_detector.detect_regions_mosaic(
                            packet.image,
                            regions,
                            score_threshold=self.detector.config.roi_score_threshold,
                            cell_size=256,
                            min_face_pixels=self.config.min_face_pixels,
                            timestamp=packet.timestamp,
                            frame_sequence=packet.sequence,
                        )
                    )
                else:
                    for reg in regions:
                        observations.extend(self._detail(packet, reg))

            detail_done = time.perf_counter()
            result = merge_faces(observations)
            result = [
                o for o in result
                if abs(o.timestamp - packet.timestamp) < 1e-4
                and (o.frame_sequence is None or o.frame_sequence == packet.sequence)
            ]
            for obs in result:
                obs.frame_sequence = packet.sequence

            self.diagnostics = {
                'mode': 'fast_track',
                'raw_faces': len(observations),
                'detected_faces': len(result),
                'detail_regions': len(regions),
                'tile_cycle_frames': 3,
                'camera_shift': [self._camera_shift[0], self._camera_shift[1]],
                'motion_ms': round((motion_done - started) * 1000, 1),
                'global_ms': 0.0,
                'detail_ms': round((detail_done - motion_done) * 1000, 1),
                'tiles_ms': 0.0,
                'runtime': self.detector.config.runtime,
            }
            return result

        # Full-frame discovery ALWAYS runs in full mode, even when YOLO finds zero bodies.
        observations = self.detector.detect(frame=packet, tracks=None, mode='full_frame',
                                            timestamp=packet.timestamp, frame_sequence=packet.sequence)
        global_done = time.perf_counter()

        proposals = []
        for track in body_tracks:
            x1, y1, x2, x_max = track.bbox[0], track.bbox[1], track.bbox[2], track.bbox[3]
            y2 = x_max
            bw, bh = x2 - x1, y2 - y1
            if any(x1 <= (o.bbox[0] + o.bbox[2]) / 2 <= x2 and y1 - .08 * bh <= (o.bbox[1] + o.bbox[3]) / 2 <= y1 + .38 * bh for o in observations):
                continue
            proposals.append((f'body:{track.track_id}', (x1 - .12 * bw, y1 - .08 * bh, x2 + .12 * bw, y1 + .38 * bh)))

        # Recheck visible faces even if the body detector misses them next time.
        for track in self.tracker.active_tracks + self.tracker.lost_tracks:
            if packet.timestamp - track.last_seen > 2.5:
                continue
            if any(box_iou(track.bbox, o.bbox) > .35 for o in observations):
                continue
            x1, y1, x2, y2 = self._flow_boxes.get(
                track.track_id, self._velocity_predicted_box(track, packet.timestamp)
            )
            bw, bh = x2 - x1, y2 - y1
            proposals.append((f'face:{track.track_id}', (x1 - .65 * bw, y1 - .6 * bh, x2 + .65 * bw, y2 + .6 * bh)))

        if extra_hints:
            for idx, hint_reg in enumerate(self._extract_hint_regions(extra_hints, w, h)):
                proposals.append((f'hint:{idx}', hint_reg))

        # Fair rotation prevents a crowd being permanently limited to the first N.
        proposals.sort(key=lambda row: self._last_detail.get(row[0], -1e20))
        selected = proposals[:self.config.detail_regions_per_frame]
        for key, _ in selected:
            self._last_detail[key] = packet.timestamp
        if selected:
            selected_regions = [region for _, region in selected]
            if use_mosaic:
                observations.extend(
                    self.detail_detector.detect_regions_mosaic(
                        packet.image,
                        selected_regions,
                        score_threshold=self.detector.config.roi_score_threshold,
                        cell_size=320,
                        min_face_pixels=self.config.min_face_pixels,
                        timestamp=packet.timestamp,
                        frame_sequence=packet.sequence,
                    )
                )
            else:
                for region in selected_regions:
                    observations.extend(self._detail(packet, region))
        active_keys = {key for key, _ in proposals}
        self._last_detail = {k: v for k, v in self._last_detail.items() if k in active_keys}
        detail_done = time.perf_counter()

        # Overlapping tiles discover distant/unassociated faces throughout viable vertical bounds,
        # avoiding empty top sky (< 0.08*h) and bottom asphalt/floor (> 0.78*h).
        tiles = [
            (x * w, y * h, min(w, (x + 0.40) * w), min(h, (y + 0.45) * h))
            for y in (0.08, 0.32)
            for x in (0.0, 0.30, 0.60)
        ]
        tile_count = self.config.tiles_per_frame if max(h, w) > 1280 else 0
        selected_tiles = []
        for _ in range(tile_count):
            selected_tiles.append(tiles[self._tile_cursor % len(tiles)])
            self._tile_cursor += 1
        if selected_tiles:
            if use_tile_mosaic:
                observations.extend(
                    self.tile_detector.detect_regions_mosaic(
                        packet.image,
                        selected_tiles,
                        score_threshold=max(0.75, self.detector.config.score_threshold),
                        cell_size=416,
                        min_face_pixels=self.config.min_face_pixels,
                        timestamp=packet.timestamp,
                        frame_sequence=packet.sequence,
                    )
                )
            else:
                for tile_reg in selected_tiles:
                    observations.extend(self._detail(packet, tile_reg, tile=True))

        result = merge_faces(observations)

        # Body positions and optical flow are search/matching hints only.
        # A missing detector result must never create a face observation.
        result = [o for o in result if abs(o.timestamp-packet.timestamp)<1e-4
                  and (o.frame_sequence is None or o.frame_sequence==packet.sequence)]
        for obs in result:obs.frame_sequence=packet.sequence

        self.diagnostics = {'mode': 'full_discovery',
                            'raw_faces': len(observations), 'detected_faces': len(result),
                            'detail_regions': len(selected), 'tile_cycle_frames': 3,
                            'camera_shift': [self._camera_shift[0], self._camera_shift[1]],
                            'motion_ms': round((motion_done - started) * 1000, 1),
                            'global_ms': round((global_done - motion_done) * 1000, 1),
                            'detail_ms': round((detail_done - global_done) * 1000, 1),
                            'tiles_ms': round((time.perf_counter() - detail_done) * 1000, 1),
                            'runtime': self.detector.config.runtime}
        return result

    def update(self, observations: list[FaceObservation], timestamp: float, epoch: str):
        if epoch != self._epoch:
            self.reset(epoch)

        tracks = self.tracker.update([PersonDetection(o.bbox, o.confidence) for o in observations], timestamp,
                                     predicted_boxes=self._flow_boxes)

        # Match face observations to face tracks
        remaining = list(observations)
        matched = []

        # Second pass: remaining observations matched by IoU
        for track in tracks:
            if not remaining:
                break
            if any(m.track_id == track.track_id for m in matched):
                continue
            obs = max(remaining, key=lambda o: box_iou(o.bbox, track.bbox))
            if box_iou(obs.bbox, track.bbox) < .25:
                continue
            remaining.remove(obs)
            obs.track_id = track.track_id
            obs.velocity = track.velocity
            matched.append(obs)

        actual = {o.track_id: o.bbox for o in matched}
        self._previous_faces = [(t.track_id, actual.get(t.track_id, self._flow_boxes.get(t.track_id, t.bbox)))
                                for t in self.tracker.active_tracks + self.tracker.lost_tracks]
        return tracks, matched
