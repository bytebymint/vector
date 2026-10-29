"""Timestamp-based movement, stable lane assignment, events and bounded track state."""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass, field
import math
import time
import cv2
import numpy as np
from .counting import LineState
from .lane_speed import LaneTimer, lane_projection, live_lane_speed
from .detector import CLASSES
from .road_geometry import anchor, normalize_box, inside, calibration_matrix, ground_speed


@dataclass
class Vehicle:
    uid: str
    ident: int
    first: float
    last: float
    bbox: np.ndarray
    hits: int=0
    confirmed: bool=False
    confidence: float=0
    votes: deque=field(default_factory=lambda:deque(maxlen=20))
    history: deque=field(default_factory=lambda:deque(maxlen=90))
    detection_history: deque=field(default_factory=lambda:deque(maxlen=40))
    events: deque=field(default_factory=lambda:deque(maxlen=50))
    lane: str|None=None
    lane_since: float=0
    candidate: str|None=None
    candidate_hits: int=0
    zones: set|None=None
    lines: dict=field(default_factory=dict)
    velocity: np.ndarray=field(default_factory=lambda:np.zeros(2))
    stopped_since: float|None=None
    cooldown: dict=field(default_factory=dict)
    subtype: str='Unknown subtype'
    subtype_confidence: float|None=None
    speed_marks: dict=field(default_factory=dict)
    marked_speed_px_s: float|None=None
    marked_speed_at: float|None=None
    marked_speed_direction: str|None=None
    lane_timers: dict=field(default_factory=dict)
    lane_speed: dict|None=None
    def category(self):
        totals=Counter()
        for label,weight in self.votes: totals[label]+=weight
        return totals.most_common(1)[0][0] if totals else 'unknown'


class TrafficEngine:
    def __init__(self,storage):
        self.storage=storage
        self.tracks={}
        self.geometry={'items':[],'calibration':None}
        self.matrix=None
        self.lane_matrices={}
        self.epoch=''
        self.current=[]
        self.recent=deque(maxlen=100)
        self.frame_size=(768,432)
    def set_geometry(self,geometry):
        self.geometry=geometry
        self.matrix=calibration_matrix(geometry['calibration']) if geometry.get('calibration') else None
        self.lane_matrices={i['id']:lane_projection(i) for i in geometry['items'] if i['kind']=='lane' and i.get('distance_metres')}
        for track in self.tracks.values():
            track.lines.clear();track.zones=None;track.candidate_hits=0;track.lane=None
            track.lane_timers.clear();track.lane_speed=None
    def reset_session(self,epoch):
        for track in self.tracks.values():
            if track.confirmed:self.persist(track)
        self.tracks.clear();self.current=[];self.epoch=epoch
    def emit(self,track,kind,stamp,**context):
        key=context.pop('key',f'{track.uid}:{kind}:{stamp:.6f}')
        event=dict(key=key,timestamp=stamp,type=kind,uid=track.uid,id=track.ident,
            **{'class':track.category()},lane=track.lane,confidence=track.confidence,confirmed=True)
        event['lane_name']=next((i['name'] for i in self.geometry['items'] if i['id']==track.lane),None)
        event.update(context)
        self.storage.enqueue('event',event)
        self.recent.appendleft(event);track.events.append(event)
        return event
    def persist(self,track):
        self.storage.enqueue('track',dict(uid=track.uid,first_seen=track.first,last_seen=track.last,
            **{'class':track.category()},lane=track.lane,duration=track.last-track.first))
    def _alert(self,track,kind,stamp,**context):
        if stamp-track.cooldown.get(kind,0)>30:
            self.emit(track,kind,stamp,confirmed=False,review_required=True,**context)
            track.cooldown[kind]=stamp
    def update(self,rows,stamp,width,height,settings,motion_stamp=None):
        self.frame_size=(width,height)
        motion_time=stamp if motion_stamp is None else motion_stamp
        seen=set()
        for row in rows:
            box=np.asarray(row[:4],float);ident=int(row[4]);score=float(row[5]);label=CLASSES.get(int(row[6]))
            if label is None:continue
            uid=f'{self.epoch}:{ident}'
            # Suppress near-identical active tracks; never merge identities heuristically.
            duplicate=False
            for previous in seen:
                other=self.tracks[previous]
                if other.category()!=label:continue
                wh=np.maximum(0,np.minimum(box[2:],other.bbox[2:])-np.maximum(box[:2],other.bbox[:2]))
                area=np.prod(wh);union=np.prod(box[2:]-box[:2])+np.prod(other.bbox[2:]-other.bbox[:2])-area
                if area/max(union,1)>.9: duplicate=True;break
            if duplicate:continue
            track=self.tracks.get(uid)
            if track is None:
                if score<settings.get('thresholds',{}).get(label,0.20):continue
                track=Vehicle(uid,ident,stamp,stamp,box)
                self.tracks[uid]=track
            dt=stamp-track.last
            if dt>settings['lifetime']:
                continue
            # Ground-contact geometry uses observed box; render box is smoothed separately.
            point=anchor(box,width,height)
            motion_dt=motion_time-track.history[-1][0] if track.history else 0
            if track.history and motion_dt>0:
                raw=(np.array(point)-np.array(track.history[-1][1:]))*[width,height]/motion_dt
                change=float(np.linalg.norm(raw-track.velocity))
                if track.confirmed and track.hits>=8 and change>max(150,float(np.linalg.norm(track.velocity))*2.5):
                    self._alert(track,'sudden_movement_candidate',stamp,velocity_change_px_s=change)
                track.velocity=.45*raw+.55*track.velocity
            track.bbox=.75*box+.25*track.bbox
            track.last=stamp;track.hits+=1;track.confidence=score
            track.votes.append((label,score))
            track.history.append((motion_time,*point))
            track.detection_history.append(dict(timestamp=stamp,confidence=score,category=label))
            confirm_req=min(2,settings['confirm_hits']) if label in ('motorcycle','bicycle') or track.category() in ('motorcycle','bicycle') else settings['confirm_hits']
            newly=not track.confirmed and track.hits>=confirm_req
            if newly:
                track.confirmed=True
                self.persist(track)
                self.emit(track,'observed',stamp,key=f'{uid}:observed')
            seen.add(uid)
            if not track.confirmed:continue
            self._movement(track,stamp,settings)
            self._geometry(track,point,stamp,width,height,settings)
            for lane in self.geometry['items']:
                if lane['kind'] != 'lane' or not lane.get('distance_metres'): continue
                timer = track.lane_timers.setdefault(lane['id'], LaneTimer())
                reading = timer.update(point,motion_time,lane,width,height)
                if reading:
                    track.lane_speed = dict(reading, lane_id=lane['id'], lane_name=lane['name'])
                    self.emit(track,'lane_speed',stamp,key=f"{uid}:lane-speed:{lane['id']}",
                        speed_kmh=reading['kmh'],average=True,measurement_basis='user measured lane length / source gate time',
                        speed_lane=lane['id'],speed_lane_name=lane['name'],**reading)
            if track.hits%20==0:self.persist(track)
        # Expire from RAM; historical analytics remain in SQLite.
        for uid,track in list(self.tracks.items()):
            if stamp-track.last>settings['lifetime']:
                if track.confirmed:self.persist(track)
                del self.tracks[uid]
        self.current=[self.serialize(self.tracks[uid],stamp) for uid in seen]
        return self.current
    def _movement(self,track,stamp,settings):
        speed=float(np.linalg.norm(track.velocity))
        if len(track.history)>2 and speed<settings['stationary_px_s']:
            if track.stopped_since is None:track.stopped_since=stamp
        elif speed>settings['stationary_px_s']*1.5:
            track.stopped_since=None
        if track.stopped_since and stamp-track.stopped_since>=settings['stopped_seconds']:
            self._alert(track,'stopped_vehicle_candidate',stamp,stopped_seconds=stamp-track.stopped_since)
            if track.lane and stamp-track.stopped_since>=settings['obstruction_seconds']:
                self._alert(track,'lane_obstruction_candidate',stamp)
        if len(track.history)>=8:
            old=np.asarray(track.history[-8][1:]);middle=np.asarray(track.history[-4][1:]);new=np.asarray(track.history[-1][1:])
            a,b=middle-old,new-middle
            cosine=np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9)
            if np.linalg.norm(a)>.025 and np.linalg.norm(b)>.025 and cosine<-.5:
                self._alert(track,'uturn_like_candidate',stamp,turn_degrees=math.degrees(math.acos(max(-1,min(1,cosine)))))
    def _geometry(self,track,point,stamp,width,height,settings):
        items=self.geometry['items']
        lanes=[i for i in items if i['kind']=='lane' and inside(point,i['points'])]
        lane=next((i['id'] for i in lanes if i['id']==track.lane),lanes[0]['id'] if lanes else None)
        if lane!=track.lane:
            if track.candidate==lane:track.candidate_hits+=1
            else:track.candidate=lane;track.candidate_hits=1
            if track.candidate_hits>=settings['lane_hysteresis']:
                previous=track.lane;track.lane=lane;track.lane_since=stamp;track.candidate_hits=0
                if previous and lane:
                    self.emit(track,'lane_transition',stamp,from_lane=previous,to_lane=lane)
                    old=next((i for i in items if i['id']==previous),None)
                    new=next((i for i in items if i['id']==lane),None)
                    if old and new and np.dot(old['direction'],new['direction'])<-.3:
                        self._alert(track,'unusual_lane_transition_candidate',stamp,from_lane=previous,to_lane=lane)
        else:track.candidate_hits=0
        for item in items:
            if item['kind']=='line':
                state=track.lines.setdefault(item['id'],LineState())
                crossing=state.update(point,stamp,item['points'],width,height)
                if crossing:
                    role=item.get('role')
                    if role and crossing['direction']=='positive_to_negative':role='out' if role=='in' else 'in'
                    self.emit(track,'crossing',stamp,key=f"{track.uid}:{item['id']}:{crossing['direction']}",
                        line=item['id'],line_name=item['name'],role=role,**crossing)
                    if item.get('name') in ('Speed checkpoint 1','Speed checkpoint 2') and crossing.get('confirmed'):
                        mark_name=item['name']
                        track.speed_marks[mark_name]=stamp
                        first=next((x for x in items if x.get('name')=='Speed checkpoint 1'),None)
                        second=next((x for x in items if x.get('name')=='Speed checkpoint 2'),None)
                        if first and second and all(k in track.speed_marks for k in ('Speed checkpoint 1','Speed checkpoint 2')):
                            t1=track.speed_marks['Speed checkpoint 1'];t2=track.speed_marks['Speed checkpoint 2']
                            elapsed=abs(t2-t1)
                            a=np.mean(first['points'],axis=0)*[width,height]
                            b=np.mean(second['points'],axis=0)*[width,height]
                            distance=float(np.linalg.norm(b-a))
                            if .5<=elapsed<=60 and distance>1:
                                track.marked_speed_px_s=distance/elapsed
                                track.marked_speed_at=stamp
                                track.marked_speed_direction='1→2' if t2>t1 else '2→1'
                                self.emit(track,'speed_estimate',stamp,
                                    distance_px=distance,elapsed_seconds=elapsed,
                                    speed_px_s=track.marked_speed_px_s,
                                    checkpoint_direction=track.marked_speed_direction,
                                    speed_basis='distance between image checkpoints / crossing time')
        zones={i['id'] for i in items if i['kind'] in ('restricted','entry','exit','zone') and inside(point,i['points'])}
        if track.zones is not None:
            for ident in zones-track.zones:
                item=next(i for i in items if i['id']==ident)
                self.emit(track,'zone_entry',stamp,zone=ident,zone_name=item['name'])
                if item['kind']=='restricted':self._alert(track,'restricted_entry_candidate',stamp,zone=ident)
            for ident in track.zones-zones:self.emit(track,'zone_exit',stamp,zone=ident)
        track.zones=zones
        if track.lane:
            item=next((i for i in items if i['id']==track.lane),None)
            if item and np.linalg.norm(track.velocity)>settings['stationary_px_s']*2:
                direction=np.asarray(item['direction'],float)
                cosine=np.dot(direction,track.velocity)/max(np.linalg.norm(direction)*np.linalg.norm(track.velocity),1e-9)
                if cosine<-.6 and stamp-track.lane_since>2:self._alert(track,'wrong_way_candidate',stamp,lane_direction=item['direction'])
    def serialize(self,track,stamp=None):
        width,height=self.frame_size
        speed=float(np.linalg.norm(track.velocity))
        angle=math.degrees(math.atan2(track.velocity[1],track.velocity[0]))
        directions=['E','SE','S','SW','W','NW','N','NE']
        history=list(track.history)
        consistency=None;curvature=None;kmh=None
        if len(history)>2:
            points=np.asarray([p[1:] for p in history])*[width,height]
            delta=np.diff(points,axis=0)
            length=float(np.linalg.norm(delta,axis=1).sum())
            consistency=float(np.linalg.norm(points[-1]-points[0])/max(length,1e-9))
            angles=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
            curvature=float(np.abs(np.diff(angles)).sum()/max(length,1))
        if self.matrix is not None:
            physical=ground_speed(history,self.matrix,(self.geometry.get('calibration') or {}).get('valid_area'))
            if physical is not None:kmh=physical*3.6
        lane_average = self.matrix is None and track.lane_speed is not None
        if lane_average: kmh = track.lane_speed['kmh']
        distance_lane = next((i for i in self.geometry['items'] if i['kind']=='lane' and i.get('distance_metres') and history and inside(history[-1][1:],i['points'])),None)
        live_speed = None
        if self.matrix is None and not lane_average and distance_lane is not None:
            live_speed = live_lane_speed(history, distance_lane, self.lane_matrices.get(distance_lane['id']))
            if live_speed is not None: kmh = live_speed
        lane=next((i['name'] for i in self.geometry['items'] if i['id']==track.lane),None)
        marked_recent=track.marked_speed_at is not None and stamp is not None and stamp-track.marked_speed_at<=120
        speed_px_s=track.marked_speed_px_s if marked_recent else speed
        speed_basis='two checkpoint distance / crossing time' if marked_recent else 'tracked image displacement / elapsed time'
        return dict(uid=track.uid,id=track.ident,category=track.category(),subtype=track.subtype,subtype_confidence=track.subtype_confidence,
            confidence=track.confidence,bbox=normalize_box(track.bbox,width,height),raw_anchor=list(history[-1][1:]) if history else None,
            lane=track.lane,lane_name=lane,velocity=track.velocity.tolist(),speed_px_s=speed_px_s,speed_kmh=kmh,
            speed_calibrated=self.matrix is not None or lane_average or distance_lane is not None,
            speed_method='lane_average' if lane_average else 'ground_plane' if self.matrix is not None else 'lane_live' if live_speed is not None else 'pending_lane' if distance_lane else 'image',
            lane_speed=track.lane_speed,
            speed_basis='user measured lane length / interpolated gate crossing time (average)' if lane_average else 'map-calibrated ground-plane regression' if self.matrix is not None else 'lane perspective + user distance / source time; straight planar road assumed' if distance_lane else speed_basis,
            displacement_px=(float(np.linalg.norm((np.asarray(history[-1][1:])-np.asarray(history[-2][1:]))*[width,height])) if len(history)>1 else None),
            direction=directions[round(angle/45)%8] if speed>2 else '—',motion='Stopped' if track.stopped_since else 'Moving' if len(history)>2 else 'Assessing',
            stopped_seconds=(track.last-track.stopped_since) if track.stopped_since else 0,
            first_seen=track.first,last_seen=track.last,duration=track.last-track.first,
            lane_seconds=track.last-track.lane_since if track.lane else None,confirmed=track.confirmed,
            state='confirmed' if track.confirmed else 'tentative',history=history,events=list(track.events),
            detection_history=list(track.detection_history),consistency=consistency,curvature_rad_per_pixel=curvature,
            tracking_confidence='observed; identity continuity uncertain across occlusion')
    def inspect(self,uid):
        track=self.tracks.get(uid)
        if track:return self.serialize(track)
        return None
