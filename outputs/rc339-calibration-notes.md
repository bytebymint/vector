# RC-339 geometry and live speed

The four right-carriageway polygons in `rc339-geometry.json` follow the three visible dashed dividers and the median/curb edges. They use shared piecewise boundaries and clip at the actual image edges. This is manual image calibration, not lane segmentation. The active geometry and `RC-339-four-lanes` profile are persisted in SQLite.

Live labels and the registry use the same least-squares image velocity over the last 0.85 seconds of capture timestamps. A label needs at least four observations spanning 0.35 seconds. This produces px/s while the vehicle is approaching, without waiting for two crossing events. Perspective means equal physical speeds can produce different px/s at different distances. Physical km/h is shown only when the existing backend ground-plane calibration supplies it.

Official metadata, checked 2026-10-23:
https://livestream.pattaya.go.th/kapi/live/map

RC-339 position: 12.942079, 100.903757.

Satellite imagery inspected:
https://www.google.com/maps/@12.942079,100.903757,19z/data=!3m1!1e3

The original image marks (0.795, 0.106) and (0.314, 0.429) could not be uniquely associated with two surveyed/map ground landmarks. Camera GPS alone does not determine those ground points. No physical distance or homography has been inferred from camera GPS, an assumed lane width, or generic dash lengths. The original checkpoint lines remain stored but hidden to keep the live overlay clean.

Verification: `node tests/test_overlay_speed.cjs` checks irregular timestamps, startup, jitter, calibration units, and independence from checkpoint averages. `python -m pytest -q`: 43 passed, including simultaneous lane association for four vehicles across the saved perspective geometry. Live stream and labels were visually checked; see `rc339-lanes-live-speed.png`.
