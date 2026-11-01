from services import youtube_source


def test_resolver_preserves_uncapped_source_quality(monkeypatch):
    options = {}
    class Resolver:
        def __init__(self, value): options.update(value)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def extract_info(self, *args, **kwargs):
            return dict(live_status='is_live', url='https://media.example/live',
                        width=3840, height=2160, fps=60, vcodec='avc1.640033',
                        tbr=20000, format_id='high')
    monkeypatch.setattr(youtube_source.yt_dlp, 'YoutubeDL', Resolver)
    selected = youtube_source.resolve('W0hB9CXqPAc')
    assert 'height<=' not in options['format']
    assert options['format'] == 'bestvideo/best'
    assert options['format_sort_force'] is True
    assert options['format_sort'] == ['res', 'fps', 'br']
    assert selected['width'] == 3840 and selected['height'] == 2160
    assert selected['fps'] == 60 and selected['tbr'] == 20000


def test_preview_is_1080p60_h264_without_capping_master():
    master=dict(width=3840,height=2160,fps=60)
    def fmt(name,height,fps,codec,protocol='https'):
        return dict(format_id=name,height=height,width=height*16//9,fps=fps,vcodec=codec,
                    url='https://media.example/'+name,protocol=protocol)
    choices=[fmt('4k',2160,60,'vp9'),fmt('slow',1080,30,'avc1.64002A'),
             fmt('vp9',1080,60,'vp9'),fmt('hls',1080,60,'avc1.64002A','m3u8_native'),
             fmt('avc60',1080,60,'avc1.64002A'),fmt('720',720,60,'avc1.640020')]
    preview=youtube_source.select_preview(choices,master,False)
    assert preview['format_id']=='avc60'
    assert master==dict(width=3840,height=2160,fps=60)
    assert youtube_source.select_preview(choices,dict(height=1080),False) is None
    assert youtube_source.select_preview([],master,False) is None


def test_preview_never_invents_60fps_when_unavailable():
    choices=[dict(format_id='30fps',height=1080,width=1920,fps=30,
                  vcodec='avc1',url='https://media.example/30fps')]
    assert youtube_source.select_preview(choices,dict(height=2160),True)['fps']==30
