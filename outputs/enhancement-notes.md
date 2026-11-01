# RC-293 live enhancement

The source camera supplies 768 x 432 pixels. The presentation output is
3840 x 2160 (5x); it is enhanced output, not native 4K camera detail.

## Selected pipeline

1. Luminance bilateral denoise: diameter 5, colour sigma 12, spatial sigma 3.
2. Motion-gated temporal denoise: 25% previous clean frame only in stationary
   areas; motion threshold 7/255, 3x3 dilation, reset after long frame gaps.
3. Local contrast: CLAHE clip limit 1.3, 8x8 tiles, 20% blend.
4. Restrained sharpening: sigma 0.8, amount 0.25, threshold 2.5, capped detail.
5. ESPCN neural super-resolution: 4x for quality, 2x for fast mode. Intel OpenCL
   FP32 where supported; CPU mode can be selected explicitly. FP16 was rejected
   after a live comparison revealed a repeating checkerboard artifact on this
   driver. FP32 output was compared against CPU output to confirm the correction.
6. Blend 85% neural output with 15% conventional interpolation, then Lanczos4
   resizing to the final 4K dimensions.

The receiver consumes the newest frame rather than building a backlog when
enhancement is slower than the incoming stream. This trades displayed frame
rate for bounded processing backlog. Original BGR frames remain available
before enhancement for downstream detection.

## Controls and restart

- E: original/enhanced display.
- C: original/enhanced split comparison.
- P: quality/fast neural model (4K output in both modes).
- F: fullscreen. Q: quit.

Run `start-camera.ps1` in PowerShell and supply a fresh playUrl. Tokens remain
short-lived and renewal has not been automated. `-Profile fast` starts with the
2x neural model. The launcher uses `../work/video-runtime/Scripts/python.exe`.

To recreate this runtime, create a virtual environment using the existing
Python installation and install `requirements-video.txt` inside it. Install
only opencv-contrib-python in that environment, not opencv-python alongside it.

## Research and model provenance

There is no universal best setting for all low-quality video. ESPCN was chosen
for the current Intel i7-10810U/UHD machine after local timing of ESPCN/FSRCNN
models. Larger EDSR/Real-ESRGAN restorers have different quality/performance
tradeoffs and were not represented as real-time options on this machine.

- [OpenCV super-resolution algorithms and tradeoffs](https://github.com/opencv/opencv_contrib/tree/4.x/modules/dnn_superres)
- [OpenCV CPU benchmarks](https://docs.opencv.org/4.12.0/dc/d69/tutorial_dnn_superres_benchmark.html)
- [OpenCV inference/preprocessing](https://docs.opencv.org/4.13.0/d5/d29/tutorial_dnn_superres_upscale_image_single.html)
- [NVIDIA guidance on denoise/deblur modes and artifact amplification](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html)
- [ESPCN model implementation](https://github.com/fannymonori/TF-ESPCN/tree/master/export)
- [FSRCNN comparison models](https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models)

Weights were downloaded directly from the two model repositories referenced
by OpenCV. Their license files are included in `models/`.
