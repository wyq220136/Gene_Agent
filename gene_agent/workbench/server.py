"""仅本机的 HTTP 边界；静态页面和结构化 API，不接收 shell 命令。"""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
from urllib.parse import urlparse,parse_qs

from .catalog import safe_path
from .service import Workbench


def make_server(root,port=8070,python=None):
    service=Workbench(root,python);token=secrets.token_urlsafe(32)
    static=Path(__file__).parent/'static'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass

        def json(self,value,status=200):
            data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)

        def local(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.local():return self.json({'error':'仅允许本机访问'},403)
            request=urlparse(self.path);query=parse_qs(request.query)
            try:
                if request.path=='/api/bootstrap':return self.json({**service.bootstrap(),'token':token})
                if request.path=='/api/runs':return self.json(service.runs())
                if request.path=='/api/jobs':return self.json(service.jobs.list())
                if request.path=='/api/assistant/catalog':
                    from .codex_client import catalog
                    return self.json(catalog(service.root,query.get('search',[''])[0],query.get('cursor',[None])[0],
                                             query.get('codex_home',[None])[0]))
                if request.path=='/api/assistant/history':
                    from .codex_client import history
                    return self.json(history(service.root,query['thread_id'][0],query.get('cursor',[None])[0],
                                             query.get('codex_home',[None])[0]))
                if request.path=='/api/job':return self.json(service.job(query['id'][0]))
                if request.path=='/api/scene':return self.json(service.scene(query.get('task',['sweep'])[0],query.get('scene',[None])[0]))
                if request.path=='/api/run':
                    from .records import describe_run
                    return self.json(describe_run(service.root,query['path'][0]))
                if request.path=='/file':
                    p=safe_path(service.root,query['path'][0],('runs','data'),exists=True)
                    if p.suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.mp4','.json','.txt','.log'}:raise ValueError('不支持此文件类型')
                    return self.file(p)
                path=static/({'/':'index.html'}.get(request.path,request.path.lstrip('/')))
                if not path.resolve().is_relative_to(static.resolve()):raise ValueError('无效静态路径')
                return self.file(path)
            except (ValueError,KeyError,TypeError,OSError) as exc:self.json({'error':str(exc)},400)

        def file(self,path):
            if not path.is_file():return self.json({'error':'找不到文件'},404)
            size=path.stat().st_size;start=0;end=size-1;status=200
            value=self.headers.get('Range')
            if value:
                try:
                    unit,span=value.split('=',1);a,b=span.split('-',1)
                    if unit!='bytes' or not a:raise ValueError()
                    start=int(a);end=min(int(b) if b else end,end)
                    if start<0 or start>end:raise ValueError()
                    status=206
                except ValueError:
                    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
            self.send_response(status)
            self.send_header('Content-Type',mimetypes.guess_type(path)[0] or 'application/octet-stream')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Accept-Ranges','bytes')
            self.send_header('Content-Length',str(max(0,end-start+1)))
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining>0:
                    data=stream.read(min(65536,remaining))
                    if not data:break
                    try:self.wfile.write(data)
                    except (BrokenPipeError,ConnectionResetError):break
                    remaining-=len(data)

        def do_POST(self):
            origin=self.headers.get('Origin')
            allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if not self.local() or self.headers.get('X-Workbench-Token')!=token or (origin and origin not in allowed):
                return self.json({'error':'请求来源或令牌无效，请刷新本机页面'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                limit=14*1024*1024 if self.path=='/api/assistant/attachment' else 65536
                if not 0<size<=limit:raise ValueError('请求长度无效')
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):raise ValueError('请求应为 JSON 对象')
                if self.path=='/api/jobs':return self.json(service.submit(data))
                if self.path=='/api/assistant/attachment':
                    from .attachments import add_attachment
                    return self.json(add_attachment(service.root,data))
                if self.path=='/api/assistant/context':
                    from .assistant import prepare_context
                    from .attachments import context_manifest
                    context=prepare_context(service.root,{**data,'question':data.get('question') or '上下文预览'})
                    return self.json(context_manifest(context))
                if self.path=='/api/cancel':service.jobs.cancel(data['id']);return self.json({'ok':True})
                if self.path=='/api/import':return self.json(service.import_run(data))
                if self.path=='/api/freeze':return self.json(service.freeze_sample(data))
                if self.path=='/api/scene':return self.json(service.save_scene(data))
                if self.path=='/api/draft':return self.json(service.draft(data))
                return self.json({'error':'未知操作'},404)
            except (ValueError,KeyError,TypeError,OSError) as exc:self.json({'error':str(exc)},400)
    try:server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    except OSError:
        service.jobs.close();raise
    server.service=service
    service.base_url=f'http://127.0.0.1:{server.server_port}'
    return server
