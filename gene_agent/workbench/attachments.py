"""用户显式提供的附件快照；上传和本地路径共用同一份清单。"""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import uuid

from PIL import Image

MAX_BYTES = 10 * 1024 * 1024
MAX_FILES = 8
TEXT_SUFFIXES = {'.txt','.md','.py','.json','.yaml','.yml','.toml','.ini','.xml',
                 '.urdf','.csv','.log','.sh','.js','.html','.css','.obj'}


def _name(name):
    name=Path(str(name)).name
    return re.sub(r'[^\w.\- ()\u4e00-\u9fff]', '_', name)[:160] or 'attachment'


def add_attachment(root, request):
    root=Path(root).resolve()
    source=None;listing=None
    if request.get('path'):
        source=Path(str(request['path'])).expanduser()
        source=(root/source).resolve() if not source.is_absolute() else source.resolve()
        if any(part in {'.codex','.aws','.ssh','.git'} for part in source.parts) or source.name in {'.env','auth.json','credentials'}:
            raise ValueError('不能附加凭据或内部配置目录中的内容')
        if any(source.is_relative_to(p) for p in map(Path,('/proc','/sys','/dev'))):
            raise ValueError('不能附加设备或进程文件')
        if source.is_dir():
            children=sorted(source.iterdir(),key=lambda p:p.name)
            listing=[{'name':p.name,'type':'directory' if p.is_dir() else 'file'} for p in children[:200]
                     if not p.name.startswith('.')]
            data=json.dumps({'directory':str(source),'entries':listing,'truncated':len(children)>200},ensure_ascii=False,indent=2).encode()
            name=_name(source.name)+'.directory.json'
        else:
            if not source.is_file():raise ValueError('找不到文件或目录')
            if source.stat().st_size>MAX_BYTES:raise ValueError('单个附件不能超过 10 MiB')
            data=source.read_bytes();name=_name(source.name)
    else:
        try:data=base64.b64decode(request.get('data',''),validate=True)
        except (ValueError,TypeError):raise ValueError('附件编码无效') from None
        name=_name(request.get('name','attachment'))
    if not data or len(data)>MAX_BYTES:raise ValueError('附件不能为空，单个附件不能超过 10 MiB')
    kind='file';mime='application/octet-stream'
    if Path(name).suffix.lower() in {'.png','.jpg','.jpeg','.webp'}:
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width*image.height>25_000_000:raise ValueError('图片不能超过 2500 万像素')
                image.verify()
                mime={'PNG':'image/png','JPEG':'image/jpeg','WEBP':'image/webp'}[image.format]
        except (OSError,KeyError):raise ValueError('图片格式或内容无效') from None
        kind='image'
    elif Path(name).suffix.lower() in TEXT_SUFFIXES:
        try:data.decode('utf-8');kind='text'
        except UnicodeDecodeError:pass
    aid=uuid.uuid4().hex
    folder=root/'runs/workbench/attachments'/aid;folder.mkdir(parents=True)
    path=folder/('content_'+name);path.write_bytes(data)
    row={'id':aid,'name':name,'kind':kind,'mime':mime,'size':len(data),
         'path':str(path.relative_to(root)),'sha256':hashlib.sha256(data).hexdigest(),
         'source':str(source) if source else 'upload'}
    if listing is not None:row['note']='只附目录清单，不递归打包；需要具体内容时请另加文件。'
    (folder/'attachment.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
    return row


def attachment_context(root, ids):
    if not isinstance(ids,list) or len(ids)>MAX_FILES:raise ValueError('最多附加 8 个文件')
    root=Path(root).resolve();rows=[]
    for aid in dict.fromkeys(ids):
        if not isinstance(aid,str) or not re.fullmatch('[a-f0-9]{32}',aid):raise ValueError('无效附件 ID')
        folder=root/'runs/workbench/attachments'/aid
        row=json.loads((folder/'attachment.json').read_text())
        path=(root/row['path']).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():raise ValueError('附件路径无效')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('附件已改变，请重新添加')
        if row['kind']=='text':
            with path.open(encoding='utf-8') as stream:text=stream.read(8001)
            row.update(text=text[:8000],truncated=len(text)>8000)
        rows.append(row)
    return rows


def context_manifest(context):
    """与实际发送共用冻结 context；图片清单必须和 answer(images=...) 完全一致。"""
    evidence=context.get('evidence',{})
    return {'task':context.get('task'),'scene':context.get('scene'),'stage':context.get('stage'),
            'run':evidence.get('summary',{}).get('path'),
            'reports':[{'path':r['path'],'truncated':r.get('truncated',False),'text':r['text'][:1000]}
                       for r in evidence.get('reports',[])],
            'attachments':[{**{k:v for k,v in r.items() if k!='text'},'preview':r.get('text','')[:1000]} for r in context.get('attachments',[])],
            'images':context.get('images',[])}
