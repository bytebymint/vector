# Southampton live source update

The app now supports a public YouTube live watch URL alongside the existing Pattaya receiver. For the current Southampton source, map measurement evidence, km/h calibration, native resolution, and startup commands, see [southampton-calibration.md](southampton-calibration.md).

# VECTOR — Traffic Intelligence AI

Local vehicle detection, persistent multi-object tracking, editable road geometry, counting, review alerts and SQLite analytics, extending the existing working Pattaya RC-293 camera application.

## Launch on this machine

Dependencies and both official YOLO26 weights are already installed. In PowerShell:

```powershell
cd C:\Users\dhanu\Documents\Codex\2026-10-22\t\outputs
.\start-traffic.ps1
```

Exact Python command using the prepared environment:

```powershell
& "C:\Users\dhanu\Documents\Codex\2026-10-22\t\work\video-runtime\Scripts\python.exe" -u "C:\Users\dhanu\Documents\Codex\2026-10-22\t\outputs\app.py"
```

Open **http://127.0.0.1:5000**. Stop with Ctrl+C. Only run one server process; Flask reload is disabled to prevent duplicate camera connections. `TRAFFIC_PORT` can select another localhost port.

### Stream authorization

The existing working negotiation is preserved: video **and audio** receive transceivers, no external ICE server, raw SDP POST with the Pattaya Origin header, SDP answer, and the RSA-compatible DTLS cipher setup that fixed this camera. Certificate fingerprints, DTLS encryption and SRTP remain enabled.

The public-stream API is tried once when authorization is needed. If Pattaya requires its browser verification, open the official camera page using **Stream authorization → Open official camera**, select RC-293, then paste its fresh `playUrl` into the dashboard. In browser developer tools, look for the `public-stream` JSON `details.playUrl`, or the successful `/media/index/api/webrtc?...` request. A previously issued URL may expire quickly. The platform cannot promise permanent tokens or bypass the site's verification. A connected media session can continue after the authorization URL expires.

Tokens stay in process memory. The application never persists play URLs or cookies in SQLite. No browser cookies are copied. Error logs omit upstream URL text. Config and events are excluded from Git by default. The dashboard's own local CSRF token is unrelated to the camera token.

When interrupted, the same authorized receiver retries, times out missing frames, and requests new authorization if necessary. Historical events remain. A reconnect starts a new tracking session because identity across disconnected video cannot be established reliably. The UI reports disconnected/authorization states; stale detections are not counted as visible vehicles.

## Installation on another Windows computer

Python **3.12 recommended** (3.11+ required; pinned wheels must support the chosen Python version):

```powershell
cd C:\path\to\outputs
.\install-traffic.ps1 -Python python
.\start-traffic.ps1
```

Equivalent commands:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-traffic.txt
.\.venv\Scripts\python.exe -m pip install --no-deps ultralytics==8.4.183
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('models/yolo26n.pt'); YOLO('models/yolo26s.pt')"
.\.venv\Scripts\python.exe app.py
```

`opencv-contrib-python` provides `cv2` and the original super-resolution module. Do not install a second OpenCV wheel into the same environment. Ultralytics is deliberately installed with `--no-deps`: the required local inference dependencies are listed explicitly; the unrelated `ultralytics-platform` cloud SDK is omitted. Its optional packaging requirement can therefore appear in `pip check`; actual local predict/tracker APIs are exercised by the tests. There is no cloud AI dependency or LLM.

For an NVIDIA machine, use an official [PyTorch CUDA installation](https://pytorch.org/get-started/locally/) compatible with the driver instead of the CPU wheel. Runtime device detection selects CUDA and FP16 (`quantize=16`) when available, otherwise CPU FP32. This machine has an i7-10810U and Intel UHD, so it runs CPU PyTorch. Intel GPU utilization is unavailable through NVML and displays **—**, not a fabricated value. The original enhancement viewer separately supports Intel OpenCL FP32.

After installation and weight download, inference and the UI work locally. The **live public camera still requires network access and valid authorization**. Tests and recorded-frame benchmarks work offline. All JavaScript and styles are local.

## Camera data flow and preservation

### Add your own stream

Use **Add stream** in the main dashboard. Paste a link to identify its format,
then select **Connect stream**. **Try live demo** prefills the recommended
Southampton YouTube camera; connect to start. Switching cameras restores that
camera's saved geometry or clears it for a new source. New cameras require their
own road geometry and measured calibration before physical speeds are available.

Supported source inputs: HTTPS YouTube watch/live/short links, existing Pattaya
authorization URLs, and direct HTTP(S), HLS (.m3u8), MPEG-DASH (.mpd), HTTP-FLV,
RTSP(S), RTMP(S) media endpoints supported by the bundled FFmpeg/PyAV runtime.
RTSP uses TCP. Direct H.264 sources reuse native MP4 playback; other codecs use
the existing decoded local WebRTC relay. One upstream connection supplies AI
and presentation. Recorded finite HTTP media will reconnect at EOF.

Private server links need network reachability, compatible codecs, and valid
URL credentials/tokens. These URLs are held in memory and are not restored
after an app restart. No browser cookies are copied. Arbitrary web pages, DRM,
custom headers and private WebRTC signaling protocols are not automatically
supported. The localhost-only API accepts operator-specified LAN addresses;
do not expose this service to untrusted networks.

UI verification: 67 Python tests and both Node suites passed; live YouTube demo
played and a temporary local HTTP server supplied 26 decoded frames plus native
MP4 initialization. RTSP/RTMP classification is tested; no private RTSP/RTMP
server was available for an end-to-end check. The visual treatment uses local
CSS transitions with reduced-motion support and limited panel blur; camera
footage is the existing public source, with no decorative stock assets or new
animation dependencies.

Original: Pattaya SDP/DTLS → aiortc video track → decoded BGR → one-frame buffer → `VideoEnhancer` → OpenCV desktop window.

Extended:

```text
one Pattaya WebRTC connection
  └─ decoded native BGR frame + sequence + receive timestamp
       └─ latest-frame mailbox (one frame, independent consumers)
            ├─ AI worker → YOLO26 → persistent tracker → geometry / traffic events
            │                                      ├─ bounded SQLite writer
            │                                      └─ Socket.IO metadata
            └─ local WebRTC relay → browser video → bounded presentation buffer
                                               └─ timestamp-matched Canvas overlays
```

The Pattaya relay encodes video for each local WebRTC peer (maximum four) and creates **no second Pattaya connection**. YouTube H.264 sources use a different presentation branch: the same upstream packets are copied into fragmented MP4 for native browser playback, without re-encoding. The decoder independently supplies the AI mailbox. A bounded reader prefetches HLS packets ahead of the presentation clock to avoid segment-boundary freezes. Browser transport failures fall back to the decoded WebRTC relay; JPEG polling remains an explicit fallback. Tracking and event evaluation run sequentially in the AI worker to preserve ordering.

Both `camera.py` and the Flask service call `services.stream_manager.negotiate()` and the same DTLS setup. Original digital enhancement controls and `start-camera.ps1` remain available. Stop the Flask receiver before launching the old desktop viewer if you want only one upstream session. The browser dashboard displays native video with optional subtle color adjustment; it does not silently run expensive neural 4K restoration in the AI path. Upscaling cannot recover verifiable detail missing from 768×432 video.

## Models, classes and tracking

Official downloaded weights:

- `models/yolo26n.pt`: fast baseline; default.
- `models/yolo26s.pt`: optional higher-accuracy model; selectable live.

Supported COCO classes: **car, motorcycle, bicycle, bus, truck** (IDs 2, 3, 1, 5, 7). Sedan, SUV, pickup, van, jeep, taxi, minibus and three-wheel classifications are **Unknown subtype** unless a suitable trained classifier is supplied. COCO does not reliably provide those labels.

Default settings: 640 input, adaptive 416–640, 12 inference FPS ceiling, class thresholds car .30 / motorcycle .23 / bicycle .25 / bus .35 / truck .35, three observations to confirm, 2.5 seconds lost retention. YOLO's low-confidence output (.08) feeds the tracker's second association; class thresholds gate new displayed tracks. Same-class near-duplicate boxes are suppressed. Road ROI filtering uses observed ground-contact points.

ByteTrack is the default. BoT-SORT is selectable. Both use actual installed Ultralytics Kalman trackers. The worker retains tracker instances across successive decoded camera frames. When AI skips source frames, Kalman prediction and frame age advance without treating those skipped samples as observed empty detections (which would prematurely delete tentative tracks). Motion calculations use actual receive timestamps. BoT-SORT camera-motion compensation is disabled for this fixed camera and variable AI sampling.

Track boxes are smoothed for display; class labels use confidence-weighted temporal voting. Ground contact uses the observed bottom-center, not a lagged display box. Tracks have bounded position/detection/event histories, tentative/confirmed states, lane association, first/last times, motion, dwell and event context. Lost objects are not drawn as if observed. Identity persistence is not guaranteed: heavy overlap, tiny motorcycles, long gaps or changing viewpoints can produce identity breaks or switches. Unique counts are confirmed **track observations**, not a claim of globally unique physical vehicles.

### Optional Re-ID and subtypes

Re-ID requires BoT-SORT, `onnxruntime`, and a compatible trained embedding model placed at `models/yolo26n-reid.onnx`. Validate its suitability for vehicles before enabling it. No appearance weights are installed or claimed to be vehicle-validated. Missing runtime/weights results in a visible warning and disabled Re-ID.

An optional trained YOLO classification model can be supplied via `models/subtypes.json`:

```json
{"weights":"vehicle-subtypes.pt","labels":["sedan","SUV","pickup","van"],"threshold":0.85}
```

Only local models inside `models/` are loaded. Classifier crops are evaluated periodically, not every frame, and fall back to Unknown subtype below threshold. No subtype weights are bundled. Restart after adding a classifier manifest.

Optional ONNX/TensorRT backend selection uses existing `models/yolo26n.onnx` / `yolo26s.onnx` or `.engine` exports and installed compatible runtimes; otherwise it warns and falls back to PyTorch. Export explicitly on the target machine, e.g. `YOLO('models/yolo26n.pt').export(format='onnx', dynamic=True, imgsz=640)`, with Ultralytics exporter dependencies installed. TensorRT is NVIDIA-only; it is not available or benchmarked here. Keep adaptive sizing off for fixed-shape exports. Rebuilding engines and installing optional export tools are operator tasks.

## Road geometry, lanes and counting

### Learn lanes and enter their real distances

Select **Lane speeds** at the top of the dashboard. If this camera has no lanes,
the app starts learning lane corridors from actual tracked vehicle trajectories.
The source continues streaming while a real displayed frame is held for setup.
Learning retains at most 250 track paths, 240 positions each, for ten minutes,
and resets when the source epoch changes. Each proposed straight corridor needs
at least three matching tracks. Repeated observations of one ID cannot satisfy
that requirement. No lane-marking model or guaranteed detection of every lane
is claimed: empty lanes, intersections, turns, perspective and ID breaks can
produce missing, overlapping or incorrect proposals. Review the suggestions.

Each four-corner lane shows **START** on its first edge and **END** on the
opposite edge. Drag the corners to the actual lane segment and enter the real
road distance between these gates in metres. Each lane gets its own input.
No distance is prefilled. Suggestions are drafts; they replace lane shapes only
in the editor until applied. Other configured geometry is retained. Manual
four-corner lane drawing remains available for lanes the learner cannot infer.

**Apply distances & measure speeds** saves the reviewed lanes and lengths for
the camera and selects lane timing instead of the previous ground-plane mode.
The app calculates **average km/h = measured metres / travel seconds × 3.6**.
Crossing times are interpolated from ground-anchor positions and camera source
timestamps. Both gate crossings must be observed for the same tracked ID.
Vehicles first seen midway do not get a fabricated speed. A tracking gap over
0.75 seconds, a departure from the lane or a return across its entry gate cancels
incomplete timing. Results are emitted once per track/lane traversal, persisted
as `lane_speed` events and labeled **km/h avg**. Stops within the segment are
included in the average. This is not instantaneous speed before the END gate.
The displayed value depends on the entered length and reviewed gate positions;
it is not a certified enforcement measurement. Use straight road sections.

The learner uses observed straight paths (PCA and conservative route grouping),
not standard COCO segmentation mislabeled as a lane model. On the live camera
used during verification it initially proposed three supported corridors, then
six after further traffic observations. Every distance remained blank pending
operator input. Synthetic 60 m / 6 s tests produced 36 km/h in both directions,
and simultaneous vehicles retained independent 36 and 72 km/h measurements.
77 Python tests and all three Node checks passed. The new files are
`services/lane_learning.py`, `services/lane_speed.py`, `static/js/lane_setup.js`,
and their regression checks. The CSRF-protected proposal endpoint is
`POST /api/geometry/lanes/suggest`; it never saves proposed lengths or lanes.

1. Select **Road geometry**. The displayed image freezes while acquisition and AI continue.
2. Draw a lane/road/zone polygon with clicks. Press **Enter**, double-click, or use **Finish polygon** to complete. Escape cancels a draft.
3. A counting line takes exactly two clicks. Click a geometry row, then drag its vertices to adjust.
4. Rename, recolor, set lane direction, set line role, hide or delete shapes.
5. **Apply & save geometry** activates it without a server restart. Save/load named profiles separately; loaded profiles must be applied.
6. Resume the picture using **Resume** or return to Live monitor.

All points are normalized to native image width/height. Both canvases share the source aspect ratio, so resizing retains alignment. A manual road polygon filters detections; lane polygons assign bottom-center ground contact. Several consecutive observations are required before lane transitions, with overlap preference for the existing lane. Every lane transition and zone entry/exit has a timestamp. Initial observation inside a zone does not fabricate an entry.

The OpenCV suggestion button uses low-chroma connected regions and Hough edges. It is **unverified**, may select shadows, pavement, vehicles or kerbs, and must be inspected/edited before applying. [Ultra-Fast Lane Detection v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) provides CULane/TuSimple/CurveLanes pretrained weights; those driving-scene datasets do not establish reliability for this elevated oblique night CCTV view. Consequently the application provides precise manual lane polygons rather than claiming that COCO segmentation or an unvalidated driving model understands this scene.

### Crossing convention

Coordinates increase rightward/downward. A line is oriented from first endpoint A to B. Signed side is `cross(B-A, P-A)`. For A on the left and B on the right, below the line is the positive side. A negative→positive crossing takes the configured role (IN or OUT); the reverse direction takes the opposite role. Without a role it is counted directionally only. Lane direction is a separate allowed motion vector.

Crossings require a side change beyond a four-source-pixel deadband and intersection of the finite segment. Remaining visible for 100 frames does not add 100 counts. Each `(tracking session + ID, line ID, direction)` is idempotent in memory and with a UNIQUE SQLite event key. Crossing observations separated by more than .9 seconds are marked uncertain and excluded from confirmed IN/OUT totals. Uncertain events remain exportable for review. Repeated passage by the same ID in the same direction on the same line is intentionally not recounted. Editing geometry resets association baselines; it does not clear historical events.

## Motion, alerts and calibration

Motion uses normalized trajectories scaled back to native pixels and real elapsed receive time. It exposes displacement, smoothed velocity (px/s), compass direction, stationary duration, lane dwell, trajectory curvature and movement consistency. Wrong-way, stopped-vehicle, prolonged obstruction, sudden movement, U-turn-like motion, restricted entry and opposing-direction lane transitions produce **review candidates**, with per-track cooldowns. These are heuristic candidates, never legal violation determinations.

Speed defaults to px/s. The calibration editor accepts four nondegenerate normalized image/ground-plane point pairs in metres, a measured independent two-point distance check and `measured:true`. Homography is accepted only if the independent error is ≤10%. Only then is estimated km/h emitted. All reference points must be on the same plane. A passing check does not remove perspective, box-anchor, timing or road-slope error. No physical calibration is bundled; road dimensions are not guessed.

## Analytics and persistence

SQLite database: `data/traffic.sqlite3`, WAL mode, indexed event time/type/lane and track UID. A bounded single writer decouples commits from acquisition. Persistence errors/queue size are exposed. Config and named geometry profiles persist. Historical analytics survive model switches, reconnects and server restarts.

Visible detections, confirmed active tracks and cumulative track observations are distinct. The dashboard shows visible/unique class counts, IN/OUT crossings, lane activity, recent events, confirmed crossings per minute, observed tracks per minute, stopped vehicles and observed dwell. Box occupancy is the union of detected boxes intersecting the manually configured road area; it is not physical road occupancy. Queue estimate is the number of sufficiently stopped visible tracks when geometry exists; this is a rough proxy, not a perspective-calibrated queue length. Candidate and uncertain events are separated from confirmed crossing totals.

CSV and JSON exports are available from the dashboard (`/api/export/csv`, `/api/export/json`). Exports currently include the most recent 100,000 events; the database retains older events. Back up the SQLite database with SQLite's backup API for long-term deployment. Archived track event details are available after RAM expiration; full trajectories/detection histories are bounded live data, not a video archive.

## Timing and performance

### Source fidelity and Studio recording

YouTube selection has no resolution ceiling and orders compatible H.264 feeds
by resolution, frame rate and bitrate. H.264 is preferred for native browser
packet passthrough; other codecs retain the decoded relay fallback. This means
the selection prioritizes compatible native playback, rather than promising the
highest rendition in every possible codec. Direct sources retain the rendition
specified by their URL. The video footer shows actual dimensions, published
frame rate, advertised source bitrate when available, and passthrough/relay mode.
Bitrate metadata is advertised manifest bandwidth, not a measured network rate.

On 24 October, both the Southampton demo and the newly selected camera
`6dp-bvQ7RWo` offered 1080p30 as their highest published rendition, at about
4.56 Mbps advertised bandwidth. The app copies this H.264 video into MP4
fragments without re-encoding. No upscale, sharpening or color filter is applied
by default: the original source is retained. 4K requires a source offering 4K.

Display & calibration now offers **Studio** recording (default): native pixel
dimensions, up to 30 fps, H.264 MP4 where the browser supports it, and a
size-dependent requested bitrate (about 17.4 Mbps at 1080p30). WebM/VP8 remains
the fallback. Efficient mode retains 15 fps. Higher recording bitrate avoids
unnecessary export loss; it does not add source detail. Playback and AI remain
separate, but browser encoding still competes for CPU. The actual Studio test on
this laptop saved 1080p H.264, 627 frames in 29.94 seconds, averaging 20.95 fps
and about 12.38 Mbps. Playback during recording was around 24 fps. Choose
Efficient when smooth monitoring matters more than recording cadence.

Verification: 68 Python tests and both Node suites passed. Available-format
metadata is saved in `source-quality-formats.json` and
`active-source-quality-formats.json`; the recorded file was fully decoded and
its measured result saved in `studio-quality-verification.json`.

The 24 October optimization keeps native video independent of inference and
coalesces overlay work around decoded frames and analytics changes. Geometry,
text measurements, speed fits and unchanged dashboard sections are cached.
Live metadata is compact; full inspection histories are fetched on demand.
Historical SQLite summaries use covering indexes, measured at about 14–15 ms
instead of 220 ms on this existing database. See [performance-update.md](performance-update.md)
for the measurements, caveats and regression results.

CPU inference defaults to at most four PyTorch threads (reserving capacity for
video and the browser), enforced after predictor initialization. Set
`TRAFFIC_TORCH_THREADS` before launch to override this on different hardware.
The benchmark now accepts successive video frames, e.g.:

```powershell
& ..\work\video-runtime\Scripts\python.exe benchmark.py --video linkedin-southampton-traffic-fixed.mp4 --stride 2 --threads 4 --iterations 40 --output benchmark-repeat.json
```

The expanded-view recorder composites the camera and completed overlay at up
to 15 fps while native playback remains independent. Keep the tab foreground
when recording. Actual recording FPS can be lower under CPU load. Inspect
`#stage.dataset` for measured `playbackFps`, `overlayFps`, and `overlayDrawMs`.

The source mailbox is one frame deep; inference always takes the newest frame after scheduling. Skipped AI samples are measured separately from camera FPS. The camera FPS is measured from decoded arrivals. The dashboard separately reports actual video presentation callbacks as VIDEO PLAYBACK FPS; this is independent of AI FPS. Native video is displayed directly by the browser, with no per-frame bitmap copies or full-image Canvas repaint loop. Only geometry/boxes/speed labels use Canvas. Source presentation timestamps align a bounded metadata history to the video; small ≤300ms extrapolation uses image velocity. Native MP4 uses half-second fragments and approximately one second of playback buffer. Encoded history is bounded to 40 fragments / 24 MiB, prefetch to 180 packets. The view may lag the public live edge because of upstream HLS latency. Source-to-server network latency is unknown. Setup snapshots and JPEG fallback still use the scene Canvas.

Telemetry includes camera FPS, measured inference FPS/latency/p95, tracker/event latency, receive-to-analytics age, skipped frames, host/process CPU, process RAM, optional NVIDIA GPU utilization/memory and connected Socket.IO clients. Unavailable values remain null. Heavy inference never runs in a Flask request handler. Local H.264 re-encoding adds CPU cost per browser client.

Captured-frame benchmark on this i7-10810U CPU at 640 input (30 measured iterations after warmup):

| Model | Mean inference | p95 | Mean tracker | Serial inference + tracker |
|---|---:|---:|---:|---:|
| YOLO26n | 47.7 ms | 50.4 ms | 1.38 ms | 20.4 FPS |
| YOLO26s | 85.9 ms | 90.9 ms | 1.23 ms | 11.5 FPS |

These are repeated-real-frame throughput results, not live transport measurements or a tracking accuracy evaluation. Live results vary with source delivery, scene, desktop load, clients, model and resolution. See `benchmark-yolo26n.json`, `benchmark-yolo26s.json` and the final verification report for live measurements.

Reproduce offline:

```powershell
& ..\work\video-runtime\Scripts\python.exe benchmark.py --image tests\fixtures\rc293-native.png --model yolo26n --resolution 640 --iterations 30 --output benchmark-repeat.json
```

The prepared runtime is outside outputs; a fresh install uses `.\.venv\Scripts\python.exe` instead.

## Tests and verification

```powershell
& ..\work\video-runtime\Scripts\python.exe -m pytest tests -q
```

Tests cover bounding boxes, browser letterboxing, polygon inclusion, finite-segment direction, jitter, idempotency, uncertain gaps, simultaneous crossings, lane hysteresis/transitions, zones, temporal voting, expiry, session recovery, SQLite persistence/reopen, CSRF/host guards, config profiles/exports, secret non-persistence, mailbox bounds, original negotiation headers/transceivers, actual local WebRTC relay/clock, real ByteTrack/BoT-SORT brief occlusion, skipped-frame tentative confirmation, clock-origin-safe scheduling, and both official models on a recorded native frame. Synthetic fixtures are confined to tests and never injected into production analytics. Passing tests do not establish ground-truth counting accuracy on the live street.

## Files and APIs

- Modified `camera.py`: shared, unchanged-protocol SDP/DTLS helper; original desktop viewer retained.
- Existing `video_enhancement.py`, `start-camera.ps1`, restoration weights: retained.
- `app.py`: localhost Flask/Socket.IO routes, CSRF, relay lifecycle, exports.
- `services/stream_manager.py`: shared negotiation, authorization, acquisition, latest-frame mailbox, browser relay/recovery.
- `services/detector.py`, `tracker.py`: real local models, backend fallback, tracker integration and subtype extension.
- `services/road_geometry.py`, `counting.py`, `traffic_engine.py`: geometry, homography, motion, lane/zone/crossing rules.
- `services/storage.py`, `performance.py`, `traffic_service.py`: WAL persistence, telemetry and AI worker.
- `templates/index.html`, `static/css/dashboard.css`, `static/js/{dashboard,overlay,zone_editor}.js`: responsive dashboard, Canvas overlays, charts and geometry editor.
- `static/js/socket.io.min.js`: vendored Socket.IO 4.8.1 client (MIT license header).
- `tests/`, `benchmark.py`, `requirements-traffic.txt`, install/start scripts: reproducible setup and validation.

Main endpoints: GET `/api/state`, `/api/tracks/<uid>`, `/api/geometry/profiles`, `/api/export/{csv,json}`; POST `/api/settings`, `/api/geometry`, `/api/geometry/profiles/<name>`, `/api/stream/authorize`, `/api/stream/restart`, `/api/webrtc/offer`. Mutation requests require the local CSRF header. Socket.IO broadcasts `frame` metadata and measured `state`.

## Limits and operational scope

This is a localhost desktop platform. It is not configured for public hosting, multi-process camera workers, user accounts or internet access. Do not expose it publicly without a separate authenticated deployment design. Flask/Socket.IO's development server is used for this local workflow. CUDA, ONNX, TensorRT, optional Re-ID and subtype weights have not been benchmarked on this Intel machine. Automatic road understanding remains suggestions plus manual editing. Field accuracy still needs annotated video and calibration; no detection model can guarantee every motorcycle or preserve every identity. The saved street geometry, if supplied during setup, is an initial manual approximation and should be adjusted against lane markings. Alerts require review. No license plates, face recognition or personal identities are inferred.

Official technical references: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [tracking](https://docs.ultralytics.com/modes/track/), [ByteTrack implementation](https://docs.ultralytics.com/reference/trackers/byte_tracker/), [BoT-SORT](https://docs.ultralytics.com/reference/trackers/bot_sort/), [aiortc API](https://aiortc.readthedocs.io/en/latest/api.html). Ultralytics code/weights have their upstream AGPL-3.0/enterprise licensing terms; retain upstream model and library licenses when distributing.

### Live lane speed estimates
Four-corner lane polygons with a user-entered start-to-end distance now provide rolling longitudinal km/h before either boundary is crossed. The lane is projected onto a straight planar rectangle; only longitudinal motion is measured. These are approximate estimates dependent on accurate lane corners, parallel road edges, road planarity, and the supplied distance. No DPI or arbitrary pixel multiplier is used. At least five uninterrupted observations spanning 0.6 seconds inside the lane are required; gaps, excessive regression residuals, and implausible speeds suppress the reading. Complete two-gate crossings still provide a separately identified average. Entered distances are not independently verified by the app.


## People and face monitoring

Use the People counter sidebar item with the active streaming link. While face analysis is running, the video overlay, confirmed count, track registry and capture gallery use anonymous **face** tracks. Dashed boxes are tentative observations awaiting confirmation. Counts describe observed tracks and can increase again after an identity break; they are not a guaranteed census of different people.

### Direct face coverage and native captures

Face discovery now runs over the full frame independently of YOLO body boxes. A 1280px global scan, two rotating overlapping detail tiles per pass, and up to four fairly scheduled 512px head/face regions recover smaller and angled faces. Detection is not capped at four faces: the budget applies to additional search regions. Cross-scale NMS suppresses duplicates and coarse boxes covering multiple distinct faces. Anonymous spatial face IDs use Kalman prediction, forward/backward checked optical flow, two-observation confirmation, scale-change gates and short occlusion recovery. No identity recognition or cross-camera face matching is performed.

A dedicated local perception process consumes a single bounded shared-memory copy of the existing decoded native frame. Detection, motion matching, pose and quality scoring run there; capture selection, SQLite persistence and WebSocket events remain in the app. It does not wait for body inference or open a second camera connection. The slot cannot be overwritten until that frame completes, so saved crops and coordinates remain synchronized. Worker errors and a 15-second response timeout surface as errors; the worker can restart, and shutdown releases shared memory. The selected dashboard view gets processing priority. In People view, vehicle context runs at one pass per second; returning to Traffic restores its configured cadence and puts face analysis on standby. Historical data and saved settings are preserved. Default body inference uses two CPU threads to leave capacity for face analysis and video; `TRAFFIC_TORCH_THREADS` can override this. The gallery defers hidden rebuilds during expanded playback and renders at most 60 matching loaded profiles.

Only the latest frame is processed; older shared body results cannot move the face clock backwards. `people:frame` metadata now drives a separate source-timestamp synchronized face overlay, including interpolated face boxes. Counts, the face registry and saved captures use the same face IDs. Dashed faces are tentative; they are not yet counted or saved.

The optional ONNX Runtime backend uses OpenCV Zoo's official dynamic-shape YuNet export (the same detector weights). It avoids OpenCL's expensive first-use compilation and falls back to OpenCV if unavailable. Install with the traffic requirements. To fetch the optional model after a fresh install:

```powershell
Invoke-WebRequest -Uri "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx" -OutFile "models/face_detection_yunet_2026may.onnx"
```

Every face proposal must agree with an independent OpenCV ResNet-10 SSD face detector at confidence ≥0.90, with overlapping boxes and consistent scale/centre. Five-landmark geometry and contrast sanity checks precede verification; skin colour is not a classification rule. Missing verifier weights fail closed. Body locations and optical flow only guide search and association: they never fabricate face observations. The legacy asynchronous-cache path is disabled because applying old coordinates to newer source frames could capture walls or clothing. Captures require the exact original frame sequence and timestamp.

After two-observation track confirmation, a face must be at least 32 native source pixels wide/high and pass all quality gates (default score 55, sharpness 5, motion blur ≤0.80). A failed gate cannot be bypassed by the legacy best-available option. Better observations update selections. High-resolution category requires at least 160 native face pixels and quality 70. These scores and five-landmark pose estimates are heuristic measurements, not forensic ratings.

The independent verifier files are bundled in `models/`. To restore them from the official OpenCV sources:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt" -OutFile "models/face_ssd.prototxt"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel" -OutFile "models/face_ssd_fp16.caffemodel"
```

SHA256: config `dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9`; weights `510ffd2471bd81e3fcc88a5beb4eae4fb445ccf8333ebc54e7302b83f4158a76`. See the [official weights manifest](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/weights.meta4).

The gallery shows verified captures by default with actual native crop dimensions. Earlier unverified captures remain accessible with the explicit checkbox; nothing is deleted by this filter. Original images are the default thumbnails; optional processed versions remain separate in the inspector. The Canvas renderer never guesses a head box or supplies invented quality scores.

Reproduce the full capture-path sample and inspect its generated native crops:

```powershell
python -m pytest tests/test_face_verifier.py -q
python tools/benchmark_face_precision.py --output data/face_precision_report.json
```

The fixture checks reject 24 historical background/clothing crops and retain five selected manually reviewed genuine faces. This small sample is not overall precision/recall. Some negative crops are duplicate category saves from the original bug. The benchmark replays six static 4K source-frame passes through the production worker, quality gates and persistence; its measured latency is analytics latency, not playback FPS.

Validation on this host: all functional regressions and all eight JavaScript test files passed. Two timing checks passed when rerun without simultaneous 4K decoding. The existing `test_pipeline_sustains_30fps_throughput` target remains unmet: its isolated 640×480 synthetic-frame workload measured 23.1 analysis FPS, versus 7.9 while the 4K application was active. These are not real-scene face-detection FPS measurements. Actual 4K processing is slower; the independent verifier trades throughput for fewer false captures. No fixed 30/60 FPS AI guarantee is made.

Original crops are lossless PNGs taken directly from the native frame. The gallery shows real source dimensions. Automatic enhancement is off; enlargement does not recover absent facial detail. Historical captures remain readable. Blurred, occluded, profile and very small faces can still be missed, and anonymous IDs can change after uncertain matches.

Reproducible offline selected-target benchmark (supply your own manually annotated source frames):

```powershell
python tools/benchmark_face_coverage.py path/to/annotations.json --output data/face_coverage_benchmark.json
```

Annotation format: `[ {"image":"frame.jpg", "boxes":[[x1,y1,x2,y2]]} ]`. Coordinates are original source pixels; image paths are relative to the annotation JSON. The report measures selected-target coverage over six **static** search passes. It is not per-frame recall, a complete census, dataset AP, or a tracking accuracy benchmark. The two local Dubai samples matched 4/4 and 8/8 selected visible faces over the six-pass search; this small sample does not establish overall accuracy. All detections outside those selected boxes still require review.

Video remains on the original native MP4 path (H.264, VP9 or AV1 when the browser supports it). Hardware decode is used where available; analytic work is separate from playback. Completed face-analysis FPS and skipped frames are measured independently of source/playback FPS.

### Smooth playback with native capture detail

For YouTube sources above 1080p, the resolver selects a separate rendition from the same video for browser playback. It prefers 1080p60 H.264 and direct DASH media for recorded videos. The original highest-resolution source still feeds the AI and lossless face captures. The browser stream is packet-copied into fragmented MP4: there is no server video decode or re-encode on the preview path. A second connection is used only for this selected YouTube rendition; direct/private streams and Pattaya keep their existing pipeline.

Both renditions share the original media presentation timestamps and one clock. Clock rebases reset the preview at a recent keyframe, and overlays retain their 4K coordinate dimensions when rendered on the 1080p canvas. A failed preview falls back to original-source playback. Expiring media URLs remain in memory. This mode reduces browser decode and transfer load; the server still needs to decode the native master for AI. AI inference and visible playback have independent frame rates.

For the configured Dubai video, the verified formats are 1920×1080 H.264 at 60 fps for preview and 3840×2160 VP9 at 60 fps for the AI source. The dashboard reports both resolutions and measures visible presentation FPS using actual video-frame callbacks. Sources without 60 fps cannot produce real 60 fps in this mode. Actual playback depends on the browser, hardware and network; 60 fps is not fabricated through frame duplication.
