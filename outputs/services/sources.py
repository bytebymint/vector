"""Classify operator-supplied sources without fetching pages or exposing secrets."""
import hashlib
from urllib.parse import urlparse
from .youtube_source import youtube_id

DEMO_URL = 'https://www.youtube.com/watch?v=W0hB9CXqPAc'


def identify(value):
    value = (value or '').strip()
    if not value or len(value) > 8192:
        raise ValueError('Paste a streaming URL (maximum 8192 characters)')
    parsed = urlparse(value)
    if parsed.hostname in ('youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com'):
        ident = youtube_id(value)
        if not ident:
            raise ValueError('Use an HTTPS YouTube live link')
        return dict(kind='youtube', label='YouTube Live', id=ident)
    if parsed.hostname == 'livestream.pattaya.go.th' or (value.count('-') == 4 and len(value) == 36):
        return dict(kind='pattaya', label='Pattaya WebRTC')
    if parsed.scheme not in ('https', 'http', 'rtsp', 'rtsps', 'rtmp', 'rtmps') or not parsed.hostname:
        raise ValueError('Use YouTube, Pattaya, HTTP(S), RTSP(S), or RTMP(S). Local files and browser page links are not stream sources.')
    if parsed.fragment:
        raise ValueError('Remove the URL fragment from your stream link')
    extension = parsed.path.lower()
    label = ('HLS' if extension.endswith('.m3u8') else 'MPEG-DASH' if extension.endswith('.mpd')
             else 'HTTP-FLV' if extension.endswith('.flv') else parsed.scheme.upper() + ' stream')
    return dict(kind='direct', label=label, id=hashlib.sha256(value.encode()).hexdigest()[:16], host=parsed.hostname)
