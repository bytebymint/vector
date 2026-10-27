"""ONNX Runtime adapter for OpenCV Zoo's dynamic-shape YuNet weights.

Decoding follows the documented stride-8/16/32 YuNet outputs and OpenCV's
FaceDetectorYN convention (BGR input, no normalization, five landmarks).
Reference: opencv/modules/objdetect/src/face_detect.cpp.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


class YuNetRuntime:
    def __init__(self, model_path: str, score_threshold=.5, nms_threshold=.3, top_k=5000):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        options.intra_op_num_threads = min(4, max(2, (os.cpu_count() or 4) // 2))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.names = [item.name for item in self.session.get_outputs()]
        self.threshold, self.nms, self.top_k = score_threshold, nms_threshold, top_k
        self._grid_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._input_buf: np.ndarray | None = None
        self._split_buf: list[np.ndarray] | None = None

    def setInputSize(self, size):
        self.size = size

    def setScoreThreshold(self, threshold):
        self.threshold = threshold

    def detect(self, image):
        h, w = image.shape[:2]
        padded_h = h + ((-h) % 32)
        padded_w = w + ((-w) % 32)

        buf = getattr(self, '_input_buf', None)
        if buf is None or buf.shape != (1, 3, padded_h, padded_w):
            buf = np.zeros((1, 3, padded_h, padded_w), dtype=np.float32)
            self._input_buf = buf
            self._split_buf = None
        else:
            if padded_h > h:
                buf[0, :, h:, :] = 0.0
            if padded_w > w:
                buf[0, :, :h, w:] = 0.0

        if image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
            split_buf = getattr(self, '_split_buf', None)
            if split_buf is None or split_buf[0].shape != (h, w):
                split_buf = [np.empty((h, w), dtype=np.uint8) for _ in range(3)]
                self._split_buf = split_buf
            cv2.split(image, split_buf)
            buf[0, 0, :h, :w] = split_buf[0]
            buf[0, 1, :h, :w] = split_buf[1]
            buf[0, 2, :h, :w] = split_buf[2]
        else:
            buf[0, :, :h, :w] = np.transpose(image, (2, 0, 1))

        outputs = dict(zip(self.names, self.session.run(self.names, {self.input_name: buf})))

        cache = getattr(self, '_grid_cache', None)
        if cache is None:
            self._grid_cache = cache = {}

        thresh = float(self.threshold)
        use_fast_sq = 0.0 < thresh <= 1.0
        thresh_sq = thresh * thresh

        candidates = []
        for stride in (8, 16, 32):
            cls_s = outputs[f'cls_{stride}'].reshape(-1)
            obj_s = outputs[f'obj_{stride}'].reshape(-1)
            if use_fast_sq:
                pre_idx = np.flatnonzero((cls_s >= thresh_sq) & (obj_s >= thresh_sq))
                if not len(pre_idx):
                    continue
                scores_pre = np.sqrt(
                    np.clip(cls_s[pre_idx], 0.0, 1.0) * np.clip(obj_s[pre_idx], 0.0, 1.0)
                )
                keep_pre = scores_pre >= thresh
                if not np.any(keep_pre):
                    continue
                selected = pre_idx[keep_pre]
                scores_sel = scores_pre[keep_pre]
            else:
                scores = np.sqrt(np.clip(cls_s, 0.0, 1.0) * np.clip(obj_s, 0.0, 1.0))
                selected = np.flatnonzero(scores >= thresh)
                if not len(selected):
                    continue
                scores_sel = scores[selected]

            grid_key = (padded_h, padded_w, stride)
            grid_xy = cache.get(grid_key)
            if grid_xy is None:
                if len(cache) > 64:
                    cache.clear()
                gh, gw = padded_h // stride, padded_w // stride
                grid_xy = (
                    np.tile(np.arange(gw, dtype=np.float32), gh),
                    np.repeat(np.arange(gh, dtype=np.float32), gw),
                )
                cache[grid_key] = grid_xy
            grid_x, grid_y = grid_xy
            grid = np.column_stack((grid_x[selected], grid_y[selected]))

            values = outputs[f'bbox_{stride}'].reshape(-1, 4)[selected]
            centers = (grid + values[:, :2]) * stride
            sizes = np.exp(np.clip(values[:, 2:], -20.0, 20.0)) * stride
            landmarks = outputs[f'kps_{stride}'].reshape(-1, 5, 2)[selected]
            landmarks = ((landmarks + grid[:, None, :]) * stride).reshape(-1, 10)
            candidates.append(np.column_stack((centers - sizes * 0.5, sizes, landmarks, scores_sel)))

        if not candidates:
            return None, None
        rows = np.concatenate(candidates).astype(np.float32)
        keep = cv2.dnn.NMSBoxes(
            rows[:, :4].astype(int).tolist(),
            rows[:, 14].tolist(),
            thresh,
            float(self.nms),
            top_k=int(self.top_k),
        )
        if len(keep) == 0:
            return None, None
        return None, rows[np.asarray(keep, dtype=int).reshape(-1)]
