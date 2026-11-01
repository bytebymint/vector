# Performance verification — 24 October 2026

The optimized app keeps the existing public YouTube acquisition, original H.264
packet relay, YOLO26n model, 640 input setting, ByteTrack and saved geometry.
No dependencies or model weights were added. No accuracy gain is claimed.

## Actual changes

- Apply a four-thread CPU budget after Ultralytics initializes its predictor
  (its device setup previously overwrote the earlier thread limit). OpenCV uses
  one thread. `TRAFFIC_TORCH_THREADS` can override the default.
- After model loading/warmup, select the newest frame again so startup never
  processes a frame retained before the warmup. Adaptive resolution now uses a
  moving average of timing instead of reacting to an isolated slow sample.
- Compact live frame/state metadata: inspection events and detection histories
  stay in the engine and inspection API; live trajectories retain 40 samples.
  Full inspection details are fetched on demand, at most once per second with
  one request in flight. No Socket.IO serialization/query work occurs with zero
  connected dashboard clients.
- Redraw overlays on video, metadata or editor changes, coalescing updates.
  Cache geometry paths, text widths and per-observation speed calculations.
  Retain unchanged dashboard HTML/chart canvases. Native video fragments use
  views instead of unnecessary copies; JPEG fallback holds one bitmap.
- Composite recordings immediately after overlay drawing, capped at 15 fps.
  The native video keeps its independent presentation rate. Recording cleanup
  releases the canvas stream and removes the compositor callback.
- Add covering/partial SQLite indexes for historical counts and use them for
  summaries, avoiding repeated event JSON parsing and full historical row reads.
- Asset URLs automatically change when JavaScript files change, so reloads
  reliably load the current implementation.
- Unavailable calibrated speed remains unavailable instead of converting null
  to a misleading zero. Track inspection, geometry and counting remain available.

## Measurements

Intel i7-10810U, six physical cores, CPU PyTorch 2.14.1, Python 3.12.14.
The system has no CUDA device. Two 20-sample live windows used the same saved
YOLO26n/640/ByteTrack configuration and public camera:

| Observed metric | Before | After |
|---|---:|---:|
| Mean AI throughput | 12.58 fps | 13.93 fps |
| Mean inference latency | 77.26 ms | 72.26 ms |
| Mean host CPU, warmed nonzero samples | 82.69% | 51.46% |
| Mean JSON state size | 72,521 bytes | 40,647 bytes |
| Foreground browser video presentation | about 30 fps | about 30 fps |

These live windows are observations, not a controlled accuracy or performance
trial: traffic, track count and other desktop work varied. Baseline CPU samples
that returned zero on newly created request threads were excluded from its mean.
Payload sizes vary with the number and age of vehicles. Raw samples are saved
in `performance-baseline-live.json` and `performance-optimized-live.json`.

On the same existing historical database, five summary calls averaged roughly
220 ms before indexing and 14–15 ms after indexing (about 15 times faster).
The covering indexes add disk space and modest write overhead.

A controlled 40-frame recorded-video detector/tracker run showed a smaller
model-only gain: 22.28 fps with six threads versus 22.88 fps with four. Both
produced 10.325 detections per frame and 12 track IDs. See
`benchmark-video-before.json`, `benchmark-video-after.json` and
`benchmark-thread-sweep.json`. This excludes video decoding, browser work and
persistence; matching counts do not establish ground-truth tracking accuracy.

## Verification

- Full Python suite: **54 passed in 11.17 seconds**, including actual models,
  tracker recovery, stream recovery, counting, persistence, compact transport,
  indexed aggregate semantics and stale-warmup-frame prevention.
- Both Node checks passed: overlay speed behavior and the real dashboard's
  render scheduling, recording composition and resource cleanup.
- Foreground live browser: 30.00 video fps during recording; overlay rendering
  around 35 updates/s, about 0.13–0.30 ms per draw in observed samples. Both
  boxes and km/h labels remained visible during the record countdown.
- Full saved recording decoded successfully: 1920×1080 VP8, 409 frames over
  29.916 seconds; target capture rate 15 fps, actual average about 13.7 fps.
  `performance-recording-proof.jpg` is a frame from that recording.

Reproduce:

```powershell
& ..\work\video-runtime\Scripts\python.exe -m pytest tests -q
& 'C:\Program Files\nodejs\node.exe' tests/test_dashboard_render.cjs
& 'C:\Program Files\nodejs\node.exe' tests/test_overlay_speed.cjs
& ..\work\video-runtime\Scripts\python.exe benchmark.py --video linkedin-southampton-traffic-fixed.mp4 --stride 2 --threads 4 --iterations 40 --output benchmark-repeat.json
```

Keep the camera tab foreground while recording: browsers can throttle background
tabs. Physical speeds retain the existing map-based calibration and remain
estimates. CPU inference still samples fewer frames than native video playback;
this update does not promise 30 or 60 AI inferences per second on this laptop.
