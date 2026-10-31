# Southampton Junction Cam — map calibration, 23 October 2026

Source: https://www.youtube.com/watch?v=W0hB9CXqPAc (Solent Ships).
Scene: High Street / Town Quay, Southampton, UK. Junction reference is approximately
50.89644 N, 1.40465 W; this is the measured road area, not a surveyed camera mount.
Satellite view: https://www.google.com/maps/@50.8964426,-1.4046471,25m/data=!3m1!1e3

## Measurements and reproducibility

The yellow-box edge was measured with Google Maps' distance tool: **19.62 m**.
Four corresponding paint corners define a ground-plane homography. An independent
pedestrian-island paint edge was measured at **4.27 m**. The homography estimates
that independent edge at approximately **4.046 m**, a **5.25%** difference.
This local reference agreement is not a guarantee of vehicle-speed accuracy.

Camera image corners in the native 1920 × 1080 frame, in correspondence order:
`[[615,539],[755,515],[1095,646],[922,691]]`.
Satellite-screen corners at the saved view:
`[[921,392],[855,527],[479,339],[553,201]]`.
Map scale: `19.62 / hypot(368,191)` metres per screenshot pixel.
Independent image endpoints: `[[515,555],[603,539]]`.

Evidence: `southampton-map-long-edge.png`, `southampton-map-validation.png`.
All numeric configuration is in `southampton-geometry.json`. The setup script
`../work/setup_southampton.py` regenerates and validates this file. Its `--install`
option persists it and the public YouTube watch URL into the local SQLite database.
The previous RC-339 profile is retained.

## Speed and geometry

Vehicle ground contact is the observed bottom centre of the bounding box.
Recent positions are projected onto the calibrated road plane and fitted against
video presentation timestamps over 0.85 seconds. At least four observations and
0.35 seconds are needed. Gaps above 0.5 seconds, poor fit, invalid road locations,
and implausible fits are suppressed. The dashboard shows `Measuring km/h…` while
a stable estimate is unavailable, and `~N km/h*` when ready. The asterisk denotes
a map estimate. Pixels/second are never relabelled as km/h. Display values use a short exponential smoothing window; an isolated missing estimate retains the most recent valid value for at most 0.8 seconds of source video, then returns to measuring.

Lane polygons cover approach lanes. Vehicles turning through the unmarked junction
can have no lane assignment. The geometry is manually traced, editable, and saved
as `Southampton-Junction`; it is not automatic road segmentation. Reference features,
camera framing, road grade, occlusion, and box-anchor jitter limit accuracy. Recalibrate
if the broadcaster moves or crops the camera. A surveyed reference or radar comparison
would be needed to establish speed accuracy. Candidate alerts are for review.

## Video and installation

The external public HLS representation decoded during integration is **1920 × 1080,
30 FPS**. The source player can offer 4K to browsers, but 4K was not exposed by the
public extractor session. No artificial upscaling is applied. The local WebRTC relay
retains native pixels and caps preview cadence at 20 FPS to keep CPU time available
for inference. Acquisition, inference, and preview cadences are distinct.

Install the updated `requirements-traffic.txt` into the existing environment; Node.js
must be on PATH for yt-dlp's official JavaScript challenge support. Run with:

```powershell
cd C:\Users\dhanu\Documents\Codex\2026-10-22\t\outputs
..\work\video-runtime\Scripts\python.exe -m pip install "yt-dlp[default]==2026.10.21"
..\work\video-runtime\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000/. The public watch URL and geometry persist across restarts;
expiring media URLs remain in memory and are resolved again after interruptions.
If the operator ends this broadcast and creates a new video ID, supply its current
public watch URL. Pattaya WebRTC remains available through the source dialog.

## Verified integration

48 Python tests pass, including the preserved Pattaya receiver/relay, YouTube URL selection, source presentation timestamps under bursty delivery, map-plane velocity, and independence from network arrival time. JavaScript speed-label checks and syntax checks pass. A 15.1-second live observation produced 25 telemetry samples: median acquisition 30.0 FPS, median inference 5.4 FPS, median analytics latency 178.1 ms, and no sampled frame age above two seconds. There were 207 actual numeric speed readings in sampled tracks. This is a short host-specific observation, not a sustained-rate or physical-speed-accuracy guarantee.
