"""真实本机 HTTP 验收：令牌、跨站请求、媒体拖动与路径边界。"""
import json
import threading
import urllib.request
import urllib.error

import pytest

from gene_agent.workbench.server import make_server


@pytest.fixture
def web(tmp_path):
    folder=tmp_path/'runs/demo';folder.mkdir(parents=True)
    (folder/'video.mp4').write_bytes(b'0123456789')
    server=make_server(tmp_path,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown();server.service.jobs.close();server.server_close();thread.join()


def test_bootstrap_and_video_range(web):
    with urllib.request.urlopen(web+'/api/bootstrap') as r:data=json.load(r)
    assert data['token'] and len(data['tasks'])==5
    request=urllib.request.Request(web+'/file?path=runs/demo/video.mp4',headers={'Range':'bytes=2-5'})
    with urllib.request.urlopen(request) as response:
        assert response.status==206 and response.read()==b'2345'


def test_write_requires_token_and_local_origin(web):
    request=urllib.request.Request(web+'/api/draft',data=b'{}',headers={'Content-Type':'application/json'})
    with pytest.raises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request)
    assert caught.value.code==403
    with urllib.request.urlopen(web+'/api/bootstrap') as r:token=json.load(r)['token']
    request.add_header('X-Workbench-Token',token);request.add_header('Origin','https://unrelated.example')
    with pytest.raises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request)
    assert caught.value.code==403


def test_private_paths_not_served(web):
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(web+'/file?path=../../etc/passwd')
    assert caught.value.code==400


def test_attachment_and_context_preview_use_same_snapshot(web):
    import base64
    with urllib.request.urlopen(web+'/api/bootstrap') as r:token=json.load(r)['token']
    def post(path,data):
        request=urllib.request.Request(web+path,data=json.dumps(data).encode(),
            headers={'Content-Type':'application/json','X-Workbench-Token':token},method='POST')
        with urllib.request.urlopen(request) as r:return json.load(r)
    a=post('/api/assistant/attachment',{'name':'example.txt','data':base64.b64encode(b'x'*70000).decode()})
    preview=post('/api/assistant/context',{'task':'sweep','attachments':[a['id']]})
    assert preview['attachments'][0]['path']==a['path']
    assert preview['attachments'][0]['truncated'] is True
    assert preview['attachments'][0]['preview']=='x'*1000
    assert preview['images']==[]
    with pytest.raises(urllib.error.HTTPError):post('/api/assistant/context',{'attachments':['../escape']})
