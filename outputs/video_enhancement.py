"""Night CCTV restoration settings selected for live CPU playback.

Order: gentle noise reduction -> restrained local contrast -> CNN upscale ->
Lanczos resize to the requested output dimensions. No face/plate synthesis.
"""

from pathlib import Path
import os
import time

import cv2
import numpy as np


class VideoEnhancer:
    def __init__(self, scale=5, sr_model="espcn", model_scale=4, compute="opencl"):
        self.scale = scale
        self.sr_model = sr_model
        self.model_scale = model_scale
        self.clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
        self.previous_raw = None
        self.previous_clean = None
        self.previous_time = 0.0
        self.sr = None
        self.target = cv2.dnn.DNN_TARGET_CPU
        self.compute = "CPU"
        self.models = {}
        if compute == "opencl" and cv2.ocl.haveOpenCL():
            cache = Path(__file__).resolve().parents[1] / "work" / "opencl-cache"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("OPENCV_OCL4DNN_CONFIG_PATH", str(cache))
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                # This UHD driver produced periodic subpixel artifacts in FP16.
                # Full precision was visually checked against the CPU result.
                self.target = cv2.dnn.DNN_TARGET_OPENCL
                self.compute = "Intel OpenCL FP32"
        if sr_model != "lanczos":
            if not hasattr(cv2, "dnn_superres"):
                raise RuntimeError(
                    "CNN enhancement requires opencv-contrib-python. "
                    "Use start-camera.ps1 or the work/video-runtime interpreter."
                )
            self.set_model_scale(model_scale)

    def set_model_scale(self, scale):
        if self.sr_model == "lanczos":
            return
        if scale not in self.models:
            model = Path(__file__).resolve().parent / "models" / (
                f"{self.sr_model.upper()}_x{scale}.pb"
            )
            if not model.exists():
                raise FileNotFoundError(f"Missing super-resolution model: {model}")
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(model))
            sr.setModel(self.sr_model, scale)
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            sr.setPreferableTarget(self.target)
            self.models[scale] = sr
        self.sr = self.models[scale]
        self.model_scale = scale

    def toggle_quality(self):
        self.set_model_scale(2 if self.model_scale == 4 else 4)

    @property
    def description(self):
        if self.sr is None:
            return "RESTORED / LANCZOS"
        return f"RESTORED / {self.sr_model.upper()} x{self.model_scale} / {self.compute}"

    def process(self, image):
        y, cr, cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))
        clean = cv2.bilateralFilter(y, 5, 12, 3)
        now = time.monotonic()
        if (
            self.previous_raw is not None
            and self.previous_raw.shape == y.shape
            and now - self.previous_time < 0.4
        ):
            difference = cv2.absdiff(y, self.previous_raw)
            if float(difference.mean()) < 25:
                motion = cv2.dilate(
                    (difference > 7).astype(np.uint8), np.ones((3, 3), np.uint8)
                )
                # Only stable pixels share a little history. Moving vehicles
                # use this frame directly, avoiding ordinary averaging trails.
                stable = motion == 0
                temporal = cv2.addWeighted(clean, 0.75, self.previous_clean, 0.25, 0)
                clean[stable] = temporal[stable]
        self.previous_raw = y.copy()
        self.previous_clean = clean.copy()
        self.previous_time = now

        contrast = cv2.addWeighted(clean, 0.8, self.clahe.apply(clean), 0.2, 0)
        detail = contrast.astype(np.float32) - cv2.GaussianBlur(
            contrast, (0, 0), 0.8
        ).astype(np.float32)
        detail[np.abs(detail) < 2.5] = 0
        corrected = np.clip(
            contrast.astype(np.float32) + 0.25 * np.clip(detail, -8, 8), 0, 255
        ).astype(np.uint8)
        restored = cv2.cvtColor(
            cv2.merge((
                corrected,
                cv2.GaussianBlur(cr, (3, 3), 0.5),
                cv2.GaussianBlur(cb, (3, 3), 0.5),
            )),
            cv2.COLOR_YCrCb2BGR,
        )
        if self.sr is None:
            return self.resize(restored)
        neural = self.sr.upsample(restored)
        # A small conventional component restrains model-specific texture and
        # keeps the presentation close to the incoming camera image.
        conventional = cv2.resize(restored, (neural.shape[1], neural.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
        neural = cv2.addWeighted(neural, 0.85, conventional, 0.15, 0)
        target = (image.shape[1] * self.scale, image.shape[0] * self.scale)
        if (neural.shape[1], neural.shape[0]) == target:
            return neural
        return cv2.resize(neural, target, interpolation=cv2.INTER_LANCZOS4)

    def resize(self, image):
        return cv2.resize(image, None, fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_LANCZOS4)
