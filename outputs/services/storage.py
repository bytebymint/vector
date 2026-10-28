"""SQLite WAL persistence with one bounded asynchronous writer."""
from __future__ import annotations
import csv
from contextlib import contextmanager
import io
import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


class Storage:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.pending = queue.Queue(maxsize=4096)
        self.error = None
        self.closed = False
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS config(name TEXT PRIMARY KEY, value TEXT NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS tracks(uid TEXT PRIMARY KEY, first_seen REAL, last_seen REAL, class TEXT, lane TEXT, duration REAL);
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, event_key TEXT UNIQUE NOT NULL, timestamp REAL NOT NULL,
                    type TEXT NOT NULL, uid TEXT, class TEXT, lane TEXT, direction TEXT, confirmed INTEGER NOT NULL, context TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_time ON events(timestamp);
                CREATE INDEX IF NOT EXISTS events_type_lane_time ON events(type,lane,timestamp);
                CREATE INDEX IF NOT EXISTS events_uid ON events(uid);
                -- Dashboard aggregates must read small covering indexes instead
                -- of reparsing every historical event's JSON every two seconds.
                CREATE INDEX IF NOT EXISTS tracks_class_duration ON tracks(class,duration);
                CREATE INDEX IF NOT EXISTS tracks_first_seen ON tracks(first_seen);
                CREATE INDEX IF NOT EXISTS crossing_direction ON events(direction)
                    WHERE type='crossing' AND confirmed=1;
                CREATE INDEX IF NOT EXISTS crossing_lane ON events(COALESCE(lane,'Unassigned'))
                    WHERE type='crossing' AND confirmed=1;
                CREATE INDEX IF NOT EXISTS crossing_role ON events(json_extract(context,'$.role'))
                    WHERE type='crossing' AND confirmed=1;
                CREATE INDEX IF NOT EXISTS crossing_time ON events(timestamp)
                    WHERE type='crossing' AND confirmed=1;
            ''')
        self.thread = threading.Thread(target=self._write, name='sqlite-writer', daemon=True)
        self.thread.start()

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA busy_timeout=5000')
        return db

    @contextmanager
    def connection(self):
        db=self._connect()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _write(self):
        db = self._connect()
        while True:
            job = self.pending.get()
            if job is None:
                self.pending.task_done()
                break
            try:
                kind, data = job
                if kind == 'event':
                    db.execute('INSERT OR IGNORE INTO events(event_key,timestamp,type,uid,class,lane,direction,confirmed,context) VALUES(?,?,?,?,?,?,?,?,?)',
                        (data['key'], data['timestamp'], data['type'], data.get('uid'), data.get('class'), data.get('lane'), data.get('direction'), int(data.get('confirmed', True)), json.dumps(data, ensure_ascii=False)))
                elif kind == 'track':
                    db.execute('INSERT INTO tracks VALUES(?,?,?,?,?,?) ON CONFLICT(uid) DO UPDATE SET last_seen=excluded.last_seen,class=excluded.class,lane=excluded.lane,duration=excluded.duration',
                        (data['uid'],data['first_seen'],data['last_seen'],data['class'],data.get('lane'),data['duration']))
                db.commit()
                self.error = None
            except Exception as exc:
                db.rollback()
                self.error = type(exc).__name__
                LOG.exception('Persistence failed')
            finally:
                self.pending.task_done()
        db.close()

    def enqueue(self, kind, value):
        if self.closed:
            return False
        try:
            self.pending.put_nowait((kind,value))
            return True
        except queue.Full:
            self.error = 'Persistence queue full; event not saved'
            LOG.error(self.error)
            return False

    def flush(self):
        self.pending.join()

    def close(self):
        if not self.closed:
            self.closed = True
            self.pending.put(None)
            self.thread.join(10)

    def load(self, name, default=None):
        with self.connection() as db:
            row = db.execute('SELECT value FROM config WHERE name=?',(name,)).fetchone()
        return json.loads(row[0]) if row else default

    def save(self, name, value):
        with self.connection() as db:
            db.execute('INSERT INTO config VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value,updated=excluded.updated', (name,json.dumps(value),time.time()))

    def profiles(self):
        with self.connection() as db:
            return [r[0][9:] for r in db.execute("SELECT name FROM config WHERE name LIKE 'geometry:%' ORDER BY name")]

    def events(self, limit=100, since=0):
        with self.connection() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT context FROM events WHERE timestamp>=? ORDER BY timestamp DESC LIMIT ?', (since,min(limit,100000)))]

    def summary(self, interval=60):
        now = time.time()
        with self.connection() as db:
            unique = db.execute('SELECT COUNT(*) FROM tracks').fetchone()[0]
            classes = dict(db.execute('SELECT class,COUNT(*) FROM tracks GROUP BY class').fetchall())
            counts = db.execute("SELECT direction,COUNT(*) FROM events INDEXED BY crossing_direction WHERE type='crossing' AND confirmed=1 GROUP BY direction").fetchall()
            crossings = dict(counts)
            lanes = dict(db.execute("SELECT COALESCE(lane,'Unassigned'),COUNT(*) FROM events INDEXED BY crossing_lane WHERE type='crossing' AND confirmed=1 GROUP BY COALESCE(lane,'Unassigned')"))
            roles = {'in':0,'out':0}
            for role,count in db.execute("SELECT json_extract(context,'$.role'),COUNT(*) FROM events INDEXED BY crossing_role WHERE type='crossing' AND confirmed=1 GROUP BY json_extract(context,'$.role')"):
                if role in roles: roles[role]=count
            flow = [dict(bucket=r[0], count=r[1]) for r in db.execute("SELECT CAST(timestamp/? AS INTEGER)*?,COUNT(*) FROM events INDEXED BY crossing_time WHERE type='crossing' AND confirmed=1 AND timestamp>=? GROUP BY 1 ORDER BY 1", (interval,interval,now-3600))]
            recent = db.execute("SELECT COUNT(*) FROM tracks WHERE first_seen>=?",(now-60,)).fetchone()[0]
            dwell=db.execute('SELECT AVG(duration) FROM tracks').fetchone()[0]
        return dict(unique=unique, classes=classes, directions=crossings, lane_crossings=lanes, roles=roles,
                    flow=flow, observed_per_minute=recent, average_observed_dwell_seconds=dwell, interval_seconds=interval)

    def export_csv(self):
        rows = self.events(100000)
        out = io.StringIO(newline='')
        fields=['timestamp','type','uid','class','lane','direction','confirmed','confidence','context']
        writer=csv.DictWriter(out,fields,extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({**row,'context':json.dumps(row,ensure_ascii=False)})
        return out.getvalue()
