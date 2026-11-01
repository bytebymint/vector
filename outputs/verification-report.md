# Implementation and verification report

Verified locally on 22 October 2026. The Flask platform is running at **http://127.0.0.1:5000** with the actual Pattaya **บ้านกระบก33 / RC-293** stream. The final camera session was authorized through the official browser player, and its preview was closed after connecting the shared Python receiver. There is one Pattaya connection and one local browser relay client.

## What changed

The original receiver's SDP negotiation and RSA-compatible DTLS setup were extracted into a shared helper, used by both the retained desktop viewer and the new Flask integration. Native decoded BGR frames feed an independent AI worker through a latest-frame mailbox. Video presentation does not wait for inference. The original digital restoration implementation and weights remain available in the desktop viewer.

Added real official YOLO26 inference; persistent ByteTrack/BoT-SORT instances; class voting and confirmation; normalized road geometry; lane hysteresis; motion; finite-line crossing rules with idempotency and uncertainty; zone transitions; review candidate events; SQLite WAL persistence; measured telemetry; a local WebRTC browser relay; timestamp-aligned Canvas overlays; vehicle inspection; CSV/JSON exports; a live flow chart; and an interactive geometry editor.

The initial manual profile **RC-293 initial manual geometry** contains a road ROI, a broad far-carriageway region, two near-road lane regions and independent IN/OUT counting lines. These are approximate manual regions based on the actual camera view, not automatically established lane truth. They persist in SQLite and are backed up in `rc293-geometry.json`. No physical speed calibration has been invented.

## Automated verification

Final command:

```powershell
& ..\work\video-runtime\Scripts\python.exe -m pytest tests -q --junitxml=test-results.xml
```

**34 passed in 10.31 seconds**, with both official model weights installed. `test-results.xml` contains the machine-readable report. Python source compilation also passed.

Covered: box normalization; letterbox mapping; polygon inclusion; finite-segment crossings and direction; repeated frames/idempotency; line jitter; uncertain gaps; simultaneous crossings; lane hysteresis and transitions; zones; temporal class voting; motion timing; track expiry; session identity isolation; SQLite uniqueness/reopen/config; original SDP headers and audio/video transceivers; simulated upstream disconnection/recovery; bounded mailbox behavior; a real aiortc-to-aiortc local WebRTC relay and its clock; actual ByteTrack and BoT-SORT brief-occlusion identity recovery; skipped-frame tentative confirmation; clock-origin-safe worker scheduling; Flask routes/CSRF/Host guard; profiles and exports; secret non-persistence; both YOLO26n and YOLO26s on a recorded native camera image; and optional ONNX backend fallback.

Tests use isolated temporary databases. Synthetic fixtures are never fed into production analytics.

## Browser and live integration checks

- Actual RC-293 native frames reached the Flask receiver and browser through WebRTC.
- Real model detections generated tracked IDs, bounding boxes, voted classes, lane assignments and trajectories.
- Lane polygons and two counting lines were drawn in the browser, renamed, assigned directions/roles and saved as a profile.
- A line endpoint was dragged, its normalized coordinates were checked, and the saved profile was restored.
- Freeze-for-editing retained the displayed frame while acquisition/inference continued.
- Actual vehicles generated both IN and OUT events, persisted across a server restart.
- A vehicle was selected through the live table. Its inspection panel displayed real confidence, lane, timestamps, trajectory, detection history and crossing history.
- Local browser developer logs showed no dashboard JavaScript errors during the checked run (unrelated extension messages were present).
- CSV/JSON routes and persistence were exercised. No bearer tokens/cookies were found in saved text source/config artifacts.
- A temporary responsive viewport override did not change this Chrome session's effective viewport, so mobile breakpoint behavior was not independently verified. Responsive CSS is implemented; the actual desktop viewport was checked. Fullscreen browser automation was inconclusive; normalized letterbox mapping is covered by offline tests and the Canvas hit-mapping implementation.

## Measured live performance

30 successive one-second samples, **all source-live and AI-running**, with YOLO26n, 640 input, ByteTrack, CPU and one browser client:

| Measurement | Mean | Observed range |
|---|---:|---:|
| Decoded camera arrivals | 16.72 FPS | 16.50–16.91 |
| AI update rate | 10.58 FPS | 9.88–10.98 |
| Inference | 54.09 ms | 45.52–75.24 |
| Tracking | 1.55 ms | 0.88–2.56 |
| Receive-to-analytics age | 86.03 ms | 51.03–153.77 |
| Host CPU | 74.48% | 68.5–94.0% |
| Python process RAM | 452.05 MB | 447.23–454.41 |

Host CPU includes the rest of the user's desktop. Source-to-server capture/network latency is unknown. See `live-verification.json` for all samples; no credential URLs are included.

During that sample the actual persisted role totals advanced from **IN 34 / OUT 64** to **IN 42 / OUT 78**. At sample completion the database contained **121 crossing events**, **42 lane transitions**, **703 confirmed track observations**, and real candidate events. The database had **zero duplicate event keys**. These totals continued changing afterwards and are **not a ground-truth counting accuracy score**.

A later browser WebRTC stats snapshot reported **15 FPS**, **768×432**, **1,454 decoded frames**, **0 dropped frames**, **5 cumulative freezes** and **6 ms jitter**. These are a separate browser snapshot, not the same 30-second average. See `browser-video-stats.json`.

The separate captured-frame benchmark measured YOLO26n mean 47.66 ms (p95 50.42) and YOLO26s mean 85.95 ms (p95 90.93) at 640 input. That excludes relay/network/persistence and repeats one real frame. It does not evaluate tracking accuracy. See both `benchmark-yolo26*.json` files and the reproducible `benchmark.py` command in README.

## Scope and limits

- Supported base classes: car, motorcycle, bicycle, bus and truck. Detection quality varies, particularly for small distant motorcycles at night. COCO labels do not establish sedan/SUV/taxi/etc.; subtype is Unknown without trained optional classifier weights.
- Tracker identities can break or switch; cumulative unique values count confirmed track observations, not globally unique physical vehicles. Brief synthetic occlusion recovery passed; long-term real-street identity accuracy has not been scored against annotations.
- Arrival-time jitter, box jitter and identity mismatches can affect image-space velocity and movement candidates. Candidate alerts can have false positives and require review. Ground-truth precision/recall and IDF1/HOTA were not measured.
- Lane geometry is manual, assisted by unverified OpenCV suggestions. No camera-domain-validated lane segmentation weights are installed.
- Physical calibration is absent; km/h remains disabled. Measured reference coordinates and an independent distance validation are required.
- Vehicle-validated Re-ID/subtype weights and ONNX/TensorRT exports/runtimes are optional and not supplied. Missing optional acceleration falls back to CPU PyTorch. CUDA is unavailable on this Intel machine.
- Token renewal depends on Pattaya's legitimate authorization process. Its API/verification intermittently failed during setup; normal official-player retries succeeded. The application does not make temporary tokens permanent or bypass verification.
- This is a local single-process Flask/Socket.IO deployment. Public hosting/authentication and multi-worker coordination are outside the implemented deployment. Historical video and full expired trajectories are not archived. Export endpoints currently cap each export at 100,000 recent events.

## File inventory

Modified existing file: `camera.py` (shared negotiation helper).

Retained existing files: `video_enhancement.py`, `start-camera.ps1`, `requirements-video.txt`, existing restoration models, restoration notes and comparison image.

Created platform files:

```text
app.py
services/__init__.py
services/stream_manager.py
services/detector.py
services/tracker.py
services/road_geometry.py
services/counting.py
services/traffic_engine.py
services/traffic_service.py
services/storage.py
services/performance.py
templates/index.html
static/css/dashboard.css
static/js/dashboard.js
static/js/overlay.js
static/js/zone_editor.js
static/js/socket.io.min.js
static/js/SOCKETIO-LICENSE.txt
models/yolo26n.pt
models/yolo26s.pt
models/ULTRALYTICS-LICENSE.txt
tests/conftest.py
tests/test_geometry_counting.py
tests/test_traffic.py
tests/test_tracker_storage_stream.py
tests/test_receiver_relay.py
tests/test_worker.py
tests/test_app.py
tests/test_detector.py
tests/fixtures/trajectory.json
tests/fixtures/rc293-native.png
benchmark.py
benchmark-yolo26n.json
benchmark-yolo26s.json
requirements-traffic.txt
install-traffic.ps1
start-traffic.ps1
.gitignore
README.md
rc293-geometry.json
test-results.xml
live-verification.json
browser-video-stats.json
traffic-dashboard.png
verification-report.md
```

Runtime-only database: `data/traffic.sqlite3` (and SQLite WAL/SHM files). Installation/launch commands, settings and operation details are in README.
