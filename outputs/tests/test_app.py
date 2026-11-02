import re
import pytest
from app import create_app
from services.stream_manager import STREAM_ID

@pytest.fixture
def client(tmp_path):
    app=create_app(False,tmp_path/'test.sqlite3')
    app.testing=True
    client=app.test_client()
    page=client.get('/').text
    token=re.search('name="csrf-token" content="([^"]+)"',page)[1]
    yield app,client,{'X-CSRF-Token':token}
    app.extensions['storage'].close()
    app.extensions['stream'].loop.close()

def test_flask_pages_and_empty_real_state(client):
    app,c,h=client
    assert c.get('/').status_code==200
    state=c.get('/api/state').json
    assert state['analytics']['unique']==0 and state['performance']['inference_fps'] is None
    assert c.get('/api/frame.jpg').status_code==204

def test_mutations_and_origin_guard(client):
    app,c,h=client
    assert c.post('/api/settings',json={}).status_code==403
    assert c.get('/api/state',headers={'Host':'evil.example'}).status_code==403
    assert c.post('/api/settings',json={'model':'invented'},headers=h).status_code==400
    assert c.post('/api/settings',json={'model':'yolo26s'},headers=h).json['model']=='yolo26s'

def test_profiles_geometry_and_exports(client):
    app,c,h=client
    geo={'items':[{'id':'one','kind':'line','points':[[.2,.3],[.8,.6]],'name':'Line A'}]}
    assert c.post('/api/geometry',json=geo,headers=h).status_code==200
    assert c.post('/api/geometry/profiles/Main',json=geo,headers=h).status_code==200
    assert c.get('/api/geometry/profiles').json==['Main']
    assert c.get('/api/geometry/profiles/Main').json['items'][0]['id']=='one'
    assert c.get('/api/export/json').json==[]
    assert 'timestamp,type,uid' in c.get('/api/export/csv').text

def test_secret_is_not_returned_or_persisted(client):
    app,c,h=client
    secret='unit-test-secret-never-persist'
    response=c.post('/api/stream/authorize',json={'url':'https://livestream.pattaya.go.th/media/index/api/webrtc?stream='+STREAM_ID+'&token='+secret+'&type=play'},headers=h)
    assert response.status_code==200
    assert secret not in c.get('/api/state').text
    assert secret not in app.extensions['storage'].path
    assert app.extensions['storage'].load('play_url') is None


def test_lane_learning_requires_live_source_and_keeps_drafts_unsaved(client):
    from types import SimpleNamespace
    import numpy as np
    app,c,h=client
    assert c.post('/api/geometry/lanes/suggest',json={}).status_code==403
    assert c.post('/api/geometry/lanes/suggest',json={},headers=h).status_code==409
    stream=app.extensions['stream'];service=app.extensions['traffic']
    stream.state='live'
    stream.frames.latest=SimpleNamespace(epoch='test-camera',image=np.zeros((1080,1920,3),np.uint8))
    service.lane_learner.observe('test-camera',[
        dict(uid=str(j),confirmed=True,history=[(i*.1,.4+j*.002,.2+i*.015) for i in range(40)])
        for j in range(3)])
    result=c.post('/api/geometry/lanes/suggest',json={},headers=h).json
    assert len(result['items'])==1
    assert result['source_id']==stream.stream_id and result['epoch']=='test-camera'
    assert result['items'][0]['distance_metres'] is None
    assert app.extensions['storage'].load('geometry') is None
