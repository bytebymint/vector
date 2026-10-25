"""Public YouTube video and live source; expiring media URLs remain in memory."""
from __future__ import annotations
import re
from urllib.parse import urlparse, parse_qs
import yt_dlp


def youtube_id(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.scheme != 'https':
        return None
    if parsed.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        ident = parse_qs(parsed.query).get('v', [''])[0] if parsed.path == '/watch' else parsed.path.split('/')[-1] if parsed.path.startswith(('/live/', '/shorts/', '/embed/', '/v/')) else ''
    elif parsed.hostname == 'youtu.be':
        ident = parsed.path.strip('/')
    else:
        return None
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', ident):
        raise ValueError('Use the public YouTube watch or video URL')
    return ident


class QuietLog:
    def debug(self, *args): pass
    def info(self, *args): pass
    def warning(self, *args): pass
    def error(self, *args): pass


def select_preview(formats: list[dict], master: dict, is_live: bool) -> dict | None:
    """Pick a lighter, native browser rendition from the same extraction.

    Prefer 60fps and H.264 hardware decode. This never caps the AI master.
    URLs/headers are temporary credentials and remain in memory.
    """
    if (master.get('height') or 0) <= 1080:
        return None
    candidates=[]
    for value in formats:
        codec=str(value.get('vcodec') or '')
        if (not value.get('url') or not 0 < (value.get('height') or 0) <= 1080
                or not 0 < (value.get('width') or 0) <= 1920
                or not codec.startswith(('avc1','h264','vp9','vp09','av01'))):
            continue
        candidates.append(value)
    if not candidates:
        return None
    def rank(value):
        codec=str(value.get('vcodec') or '')
        # HLS manifests (m3u8_native) stream reliably without 403 Forbidden errors.
        is_hls=value.get('protocol') in ('m3u8_native', 'm3u8')
        return ((value.get('fps') or 0)>=50, value.get('height') or 0,
                codec.startswith(('avc1','h264')), is_hls,
                min(value.get('fps') or 0,60), value.get('tbr') or 0)
    ranked=sorted(candidates,key=rank,reverse=True)
    chosen=ranked[0]
    keys=('url','http_headers','width','height','fps','vcodec','tbr','format_id','protocol')
    result={key:chosen.get(key) for key in keys}
    # YouTube can reject one client/transport while its equivalent HLS rendition
    # remains playable. Keep matching H.264 variants as bounded alternatives.
    result['alternatives']=[{key:value.get(key) for key in keys} for value in ranked[1:]
        if value.get('height')==chosen.get('height') and value.get('fps')==chosen.get('fps')
        and str(value.get('vcodec','')).startswith(('avc1','h264'))][:2]
    return result


def resolve(ident: str) -> dict:
    with yt_dlp.YoutubeDL({
        'quiet': True,
        'no_warnings': True,
        'logger': QuietLog(),
        'js_runtimes': {'node': {}},
        # Select the absolute highest available video quality (8K, 4K, 1440p, 1080p, etc.)
        # If possible with avc1 for native browser direct copy, but bestvideo ensures highest resolution.
        'format': 'bestvideo/best',
        'format_sort': ['res', 'fps', 'br'],
        'format_sort_force': True,
        'socket_timeout': 15,
        'retries': 2,
        'noplaylist': True,
    }) as ydl:
        info = ydl.extract_info(f'https://www.youtube.com/watch?v={ident}', download=False)
    
    is_live = bool(info.get('live_status') == 'is_live' or info.get('is_live'))
    title = info.get('title') or ('YouTube live camera' if is_live else 'YouTube video')
    
    result = {key: info.get(key) for key in ('url', 'http_headers', 'width', 'height', 'fps', 'title', 'vcodec', 'tbr', 'format_id')}
    result['is_live'] = is_live
    result['title'] = title
    result['duration'] = info.get('duration')
    result['preview'] = select_preview(info.get('formats',[]),result,is_live)
    return result
