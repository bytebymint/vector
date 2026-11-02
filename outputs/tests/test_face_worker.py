"""Real local worker/IPC checks without a public camera or synthetic faces."""
import os
from pathlib import Path
import numpy as np
from services.face_worker import FaceWorkerClient
from services.face_contracts import FramePacket


def test_worker_process_uses_exact_shared_frame_and_recovers_after_restart():
    client=FaceWorkerClient(Path(__file__).resolve().parents[1]/'models')
    config={'face_confidence_threshold':.5,'quality_threshold':20.,'min_face_pixels':24}
    try:
        packet=FramePacket(1,1.,np.zeros((240,320,3),np.uint8),'camera','a',320,240)
        tracks,faces,diagnostics=client.analyze(packet,[],'a',config)
        assert tracks==[] and faces==[]
        assert diagnostics['worker_pid'] != os.getpid()
        first_process=client.process
        first_memory=client.memory.name
        packet=FramePacket(2,1.2,np.zeros((540,960,3),np.uint8),'camera','a',960,540)
        assert client.analyze(packet,[],'a',config)[:2]==([],[])
        assert client.memory.name != first_memory
        client.close()
        assert first_process.poll() is not None
        assert client.memory is None
        assert client.analyze(packet,[],'b',config)[:2]==([],[])
    finally:
        client.close()


def test_analysis_focus_prioritizes_selected_view_without_changing_saved_settings(tmp_path):
    import re
    from app import create_app
    app=create_app(start_services=False,database=tmp_path/'traffic.db')
    browser=app.test_client()
    try:
        page=browser.get('/').get_data(as_text=True)
        token=re.search(r'name="csrf-token" content="([^"]+)"',page).group(1)
        before=dict(app.extensions['traffic'].settings)
        response=browser.post('/api/analysis/focus',json={'mode':'people'},headers={'X-CSRF-Token':token})
        assert response.status_code==200
        assert app.extensions['traffic'].analysis_focus=='people'
        assert app.extensions['people_orchestrator'].analysis_focus=='people'
        assert app.extensions['traffic'].settings==before
        response=browser.post('/api/analysis/focus',json={'mode':'traffic'},headers={'X-CSRF-Token':token})
        assert response.status_code==200
        assert app.extensions['people_orchestrator'].analysis_focus=='traffic'
        assert browser.post('/api/analysis/focus',json={'mode':'invalid'},headers={'X-CSRF-Token':token}).status_code==400
    finally:
        # No stream/inference threads were started in this application fixture.
        app.extensions['people_orchestrator'].face_enhancer.stop()
        app.extensions['people_orchestrator'].pipeline_scheduler.shutdown()
        app.extensions['storage'].close()
