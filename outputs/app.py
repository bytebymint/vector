"""Traffic Intelligence AI — local Flask dashboard around the working camera receiver."""
from __future__ import annotations
import atexit
import asyncio
import logging
import os
from pathlib import Path
import secrets
import time
import cv2
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO
from services.storage import Storage
from services.stream_manager import StreamManager
from services.traffic_service import TrafficService
from services.road_geometry import validate_geometry, suggestions
from services.presentation import compact_frame
from services.people_orchestrator import PeopleSystemOrchestrator
from services.people_routes import init_people_routes, people_bp

ROOT=Path(__file__).resolve().parent


def create_app(start_services=True, database=None):
    # Reserve CPU capacity for PyTorch and video decoding instead of allowing
    # each concurrent OpenCV DNN/resize to start another full-size thread pool.
    cv2.setNumThreads(2)
    app=Flask(__name__)
    app.config.update(SECRET_KEY=secrets.token_hex(32),MAX_CONTENT_LENGTH=256*1024)
    csrf=secrets.token_urlsafe(32)
    sio=SocketIO(app,async_mode='threading',logger=False,engineio_logger=False)
    storage=Storage(database or ROOT/'data'/'traffic.sqlite3')
    stream=StreamManager()
    service=TrafficService(stream,storage,ROOT/'models')
    saved_source=storage.load('camera-source',{})
    if saved_source.get('youtube_url'):
        stream.authorize(saved_source['youtube_url'])
    people_orchestrator = PeopleSystemOrchestrator(
        stream=stream,
        storage=storage,
        models_dir=ROOT / 'models',
        data_dir=(Path(database).resolve().parent / 'people') if database else ROOT / 'data',
        socketio=sio,
        auto_start=False,
        shared_inference=True,
    )
    people_orchestrator.set_focus('traffic')
    service.person_frame_listener = people_orchestrator.submit_shared_frame
    init_people_routes(
        app,
        orchestrator=people_orchestrator,
        repository=people_orchestrator.profile_repository,
        captures_dir=people_orchestrator.captures_dir,
    )
    app.extensions.update(
        traffic=service,
        stream=stream,
        storage=storage,
        socketio=sio,
        people_orchestrator=people_orchestrator,
        people_repository=people_orchestrator.profile_repository,
    )

    @app.before_request
    def protect():
        if request.method in ('POST','PUT','DELETE'):
            if not secrets.compare_digest(request.headers.get('X-CSRF-Token',''),csrf):
                if not request.path.startswith(('/api/people', '/api/video/control')):
                    return jsonify(error='Reload the dashboard to refresh the local request token'),403
        if request.host.split(':')[0] not in ('127.0.0.1','localhost','[::1]'):
            return jsonify(error='This service is bound to localhost'),403

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Cache-Control']='no-store' if request.path.startswith('/api/') else 'no-cache'
        return response

    @app.errorhandler(ValueError)
    def bad_value(exc):return jsonify(error=str(exc)),400

    @app.post('/api/analysis/focus')
    def analysis_focus():
        value=request.get_json(silent=True) or {}
        mode=value.get('mode')
        if mode not in ('traffic','people'):raise ValueError('Choose traffic or people')
        service.analysis_focus=mode
        people_orchestrator.set_focus(mode)
        return jsonify(mode=mode,vehicle_context_fps=1 if mode=='people' else None)

    @app.get('/')
    def index():
        asset_version=max((ROOT/rel).stat().st_mtime_ns
                          for rel in ('static/css/dashboard.css','static/js/dashboard.js','static/js/overlay.js','static/js/people_overlay.js','static/js/face_timeline.js','static/js/zone_editor.js','static/js/lane_setup.js','static/js/people_dashboard.js'))
        return render_template('index.html',csrf=csrf,asset_version=asset_version)

    @app.get('/api/state')
    def state():return jsonify(service.state())

    @app.post('/api/settings')
    def settings():return jsonify(service.configure(request.get_json()))

    @app.post('/api/geometry')
    def geometry():
        value=validate_geometry(request.get_json())
        service.set_geometry(value)
        storage.save('camera-geometry:'+stream.stream_id,value)
        return jsonify(value)

    @app.get('/api/geometry/profiles')
    def profiles():return jsonify(storage.profiles())

    @app.post('/api/geometry/profiles/<name>')
    def save_profile(name):
        if len(name)>60:raise ValueError('Profile name too long')
        value=validate_geometry(request.get_json())
        storage.save('geometry:'+name,value)
        return jsonify(saved=name)

    @app.get('/api/geometry/profiles/<name>')
    def load_profile(name):
        value=storage.load('geometry:'+name)
        return (jsonify(value),200) if value else (jsonify(error='Profile not found'),404)

    @app.get('/api/geometry/suggest')
    def suggest():
        packet=stream.frames.latest
        if packet is None:return jsonify(error='Wait for a live frame'),409
        return jsonify(suggestions(packet.image))

    @app.post('/api/geometry/lanes/suggest')
    def lane_suggestions():
        from services.lane_learning import suggest_lanes
        packet=stream.frames.latest
        if packet is None or stream.state!='live':return jsonify(error='Wait for a live camera and moving vehicles'),409
        source_id=stream.stream_id
        with service.lock:
            paths=service.lane_learner.snapshot() if service.lane_learner.epoch==packet.epoch else []
        result=suggest_lanes(paths,packet.image.shape[1],packet.image.shape[0])
        return jsonify(**result,source_id=source_id,epoch=packet.epoch)

    @app.get('/api/tracks/<uid>')
    def inspect(uid):
        with service.lock:value=service.engine.inspect(uid)
        if value is None:
            return jsonify(uid=uid,state='expired',events=[e for e in storage.events(10000) if e.get('uid')==uid])
        return jsonify(value)

    @app.post('/api/stream/identify')
    def identify_source():
        from services.sources import identify
        return jsonify(identify(request.get_json().get('url','')))

    @app.post('/api/stream/authorize')
    def authorize():
        previous_id=stream.stream_id
        previous_geometry=service.geometry
        stream.authorize(request.get_json().get('url',''))
        if previous_id!=stream.stream_id:
            storage.save('camera-geometry:'+previous_id,previous_geometry)
            service.set_geometry(storage.load('camera-geometry:'+stream.stream_id,{'items':[],'calibration':None}))
        storage.save('camera-source',{'youtube_url':f'https://www.youtube.com/watch?v={stream.stream_id}'} if stream.source_type=='youtube' else {})
        return jsonify(state='connecting')

    @app.post('/api/stream/restart')
    def restart():
        stream.restart()
        return jsonify(state='reconnecting')

    @app.post('/api/webrtc/offer')
    def offer():
        value=request.get_json()
        if value.get('type')!='offer' or not isinstance(value.get('sdp'),str):raise ValueError('Expected SDP offer')
        future=asyncio.run_coroutine_threadsafe(stream.relay_offer(value['sdp'],'offer'),stream.loop)
        try:return jsonify(future.result(15))
        except Exception as exc:
            logging.getLogger(__name__).warning('Local relay negotiation failed: %s',type(exc).__name__)
            return jsonify(error='Local WebRTC negotiation failed; retry or use the JPEG fallback'),503

    @app.get('/api/webrtc/clock/<ident>')
    def clock(ident):return jsonify(origin=stream.clock(ident))

    @app.get('/api/video/info')
    def native_info():
        return jsonify(available=stream.source_type in ('youtube','direct') and stream.native_h264 is not False,
            ready=stream.native_video.init is not None,info=stream.native_video.info)

    @app.get('/api/video/live')
    def native_video():
        if stream.source_type not in ('youtube','direct'):return Response(status=409)
        return Response(stream.native_video.stream(lambda:stream.stopping or stream.source_type not in ('youtube','direct') or stream.native_h264 is False),
            mimetype='application/octet-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

    @app.post('/api/video/control')
    def video_control():
        data = request.get_json(silent=True) or {}
        action = data.get('action')
        if action == 'pause':
            res = stream.pause()
            sio.emit('video_control', res)
            return jsonify(res)
        elif action == 'resume':
            res = stream.resume()
            sio.emit('video_control', res)
            return jsonify(res)
        elif action == 'seek':
            seconds = float(data.get('seconds', data.get('target', 0.0)))
            res = stream.seek(seconds)
            sio.emit('video_control', res)
            return jsonify(res)
        elif action == 'skip':
            delta = float(data.get('delta', 30.0))
            res = stream.skip(delta)
            sio.emit('video_control', res)
            return jsonify(res)
        return jsonify(error='Invalid action. Use pause, resume, seek, or skip'), 400

    @app.get('/api/video/state')
    def video_state():
        return jsonify(
            is_paused=stream.is_paused,
            position=round(stream.playback_position, 2),
            duration=stream.playback_duration,
            source_type=stream.source_type,
            camera=stream.source_title,
        )

    @app.post('/api/webrtc/close/<ident>')
    def close_peer(ident):
        peer=stream.peers.get(ident)
        if peer:asyncio.run_coroutine_threadsafe(peer[0].close(),stream.loop)
        return jsonify(closed=True)

    @app.get('/api/frame.jpg')
    def frame():
        packet=stream.frames.latest
        if packet is None:return Response(status=204)
        ok,data=cv2.imencode('.jpg',packet.image,[cv2.IMWRITE_JPEG_QUALITY,90])
        return Response(data.tobytes() if ok else b'',mimetype='image/jpeg',headers={'X-Frame-Timestamp':str(packet.timestamp),'X-Frame-Sequence':str(packet.sequence),'Cache-Control':'no-store'})

    @app.get('/api/export/<kind>')
    def export(kind):
        if kind=='csv':return Response(storage.export_csv(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=traffic-events.csv'})
        if kind=='json':return jsonify(storage.events(100000))
        raise ValueError('Choose csv or json')

    @sio.on('connect')
    def connected():
        service.clients+=1
        people_orchestrator.people_event_emitter.clients+=1
    @sio.on('disconnect')
    def disconnected(reason=None):
        service.clients=max(0,service.clients-1)
        people_orchestrator.people_event_emitter.clients=max(0,people_orchestrator.people_event_emitter.clients-1)

    def publish():
        last=0;last_state=0
        while not service.stopping.is_set():
            snap=service.snapshot
            if service.clients and snap['sequence']!=last:
                sio.emit('frame',compact_frame(snap))
                last=snap['sequence']
            if service.clients and time.monotonic()-last_state>=1:
                sio.emit('state',service.state());last_state=time.monotonic()
            sio.sleep(.04)

    def shutdown():
        if not service.stopping.is_set():
            service.close();stream.close();storage.close()
        if 'people_orchestrator' in app.extensions and app.extensions['people_orchestrator']:
            try:app.extensions['people_orchestrator'].stop()
            except Exception:pass
    app.extensions['shutdown']=shutdown
    if start_services:
        stream.start();service.start();people_orchestrator.start();sio.start_background_task(publish)
        atexit.register(shutdown)
    return app


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    # Request libraries must never include token-bearing URL details in logs.
    logging.getLogger('aiohttp').setLevel(logging.ERROR)
    application=create_app()
    application.extensions['socketio'].run(application,host='127.0.0.1',port=int(os.getenv('TRAFFIC_PORT','5000')),
        debug=False,use_reloader=False,allow_unsafe_werkzeug=True)
