"""VECTOR - Real-Time 4K Video Intelligence & Face Recognition Studio

Main application entrypoint.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

if str(OUTPUTS_DIR) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_DIR))

os.chdir(OUTPUTS_DIR)

if __name__ == "__main__":
    from app import app, sio, publish

    sio.start_background_task(publish)
    port = int(os.environ.get("PORT", os.environ.get("TRAFFIC_PORT", 5000)))
    print(f"================================================================")
    print(f"  VECTOR - Real-Time 4K Video Intelligence & Face Recognition")
    print(f"  Server running on http://127.0.0.1:{port}")
    print(f"================================================================")
    sio.run(app, host="127.0.0.1", port=port, debug=False, use_reloader=False)
