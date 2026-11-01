import pytest
from services.sources import identify
from services.stream_manager import StreamManager

@pytest.mark.parametrize('url,label',[
 ('https://www.youtube.com/watch?v=W0hB9CXqPAc','YouTube Live'),
 ('https://youtube.com/live/W0hB9CXqPAc','YouTube Live'),
 ('https://youtu.be/W0hB9CXqPAc','YouTube Live'),
 ('https://camera.example/live.m3u8?token=secret','HLS'),
 ('https://camera.example/live.mpd','MPEG-DASH'),
 ('rtsp://user:secret@192.168.1.20:554/live','RTSP stream'),
 ('rtmp://camera.example/live/key','RTMP stream'),
])
def test_source_detection_is_local_and_redacts_credentials(url,label):
 result=identify(url)
 assert result['label']==label
 assert 'secret' not in str(result)

@pytest.mark.parametrize('url',['file:///etc/passwd','concat:http://example/a','javascript:alert(1)','not a url','https://youtube.com/watch?v=bad'])
def test_invalid_sources_rejected(url):
 with pytest.raises(ValueError):identify(url)

def test_direct_source_uses_existing_receiver_without_exposing_url():
 stream=StreamManager()
 try:
  stream.authorize('rtsp://user:secret@192.168.1.20:554/live')
  assert stream.source_type=='direct'
  assert stream.status()['stream_id']
  assert 'secret' not in str(stream.status())
 finally:stream.loop.close()
