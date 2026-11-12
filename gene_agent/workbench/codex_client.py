"""本机 Codex App Server 的有限 JSON-RPC 客户端，不读取会话数据库或凭据。"""
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time


class CodexTurnError(ValueError):
    """助手回合失败，不代表被分析的机器人任务失败。保留服务端原始错误。"""

    def __init__(self, thread_id, turn):
        error = turn.get('error')
        message = str(error.get('message', '') if isinstance(error, dict) else error or turn['status'])
        routing = 'remote compact task' in message and 'workspace routing discovery failed' in message
        summary = ('已有对话的远程上下文压缩失败：服务端未能发现 workspace 路由。'
                   if routing else f"Codex 回合未完成（{turn['status']}）：{message}")
        hint = ('可先在当前分支重试；若重复失败，可在原终端检查该对话能否正常继续。'
                '也可明确选择“新建对话”，重新附加所需上下文；新对话不会继承原历史。'
                if routing else '可查看作业日志中的原始错误后，在当前对话重试。')
        self.details = {'scope': 'assistant', 'code': 'remote_compaction_routing' if routing else 'turn_failed',
                        'thread_id': thread_id, 'turn_id': turn['id'], 'status': turn['status'],
                        'message': summary, 'hint': hint, 'error': error}
        super().__init__(summary)


def resolve_codex_home(value=None, *, must_exist=True):
    """只对本次子进程设置 CODEX_HOME，不修改工作台或 shell 的环境。"""
    value = value or os.environ.get('CODEX_HOME') or str(Path.home()/'.codex')
    if not isinstance(value, str) or len(value)>4096 or '\0' in value:
        raise ValueError('CODEX_HOME 路径无效')
    path = Path(value).expanduser()
    if not path.is_absolute() or (must_exist and not path.is_dir()):
        raise ValueError('CODEX_HOME 必须是已存在的绝对目录，可使用 ~/')
    return str(path.resolve())


class CodexClient:
    def __init__(self, root, timeout=30, codex_home=None):
        executable = shutil.which('codex')
        if not executable:
            raise ValueError('未找到本机 Codex CLI')
        self.root = Path(root).resolve()
        self.codex_home = resolve_codex_home(codex_home)
        self.timeout = timeout
        self.serial = 0
        self.events = []
        self.messages = queue.Queue()
        self.active_turn = None
        self.proc = subprocess.Popen([executable, 'app-server', '--stdio'], cwd=self.root,
                                     env={**os.environ, 'CODEX_HOME': self.codex_home},
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        try:
            initialized = self.call('initialize', {'clientInfo': {'name': 'gene_workbench', 'version': '0.2.0'}})
            actual_home = initialized.get('codexHome')
            if actual_home and str(Path(actual_home).resolve()) != self.codex_home:
                raise ValueError('Codex 返回的 CODEX_HOME 与所选目录不一致，已停止接入')
            self.send({'method': 'initialized'})
        except Exception:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.proc.stdout:
                try:
                    self.messages.put(json.loads(line))
                except ValueError:
                    continue
        finally:
            self.messages.put(None)

    def send(self, message):
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + '\n')
        self.proc.stdin.flush()

    def receive(self, deadline):
        try:
            message = self.messages.get(timeout=max(.001, deadline - time.monotonic()))
        except queue.Empty:
            raise ValueError('Codex 响应超时；请查看作业日志') from None
        if message is None:
            raise ValueError('Codex App Server 已断开；请检查本机 CLI 和登录状态')
        # 工作台不授予 shell、文件写入、外部工具等交互授权。
        if 'method' in message and 'id' in message:
            self.send({'id': message['id'], 'error': {'code': -32601,
                       'message': '工作台仅支持只读诊断，不处理工具授权或交互请求'}})
        return message

    def call(self, method, params):
        self.serial += 1
        request_id = self.serial
        self.send({'id': request_id, 'method': method, 'params': params})
        deadline = time.monotonic() + self.timeout
        while True:
            message = self.receive(deadline)
            if message.get('id') == request_id and 'method' not in message:
                if 'error' in message:
                    raise ValueError(f"Codex {method}: {message['error'].get('message', '请求失败')}")
                return message['result']
            if 'method' in message and 'id' not in message:
                self.events.append(message)

    def notification(self, deadline):
        return self.events.pop(0) if self.events else self.receive(deadline)

    def close(self):
        if self.proc.poll() is None:
            if self.active_turn:
                try:
                    self.send({'id': 'cleanup', 'method': 'turn/interrupt', 'params': self.active_turn})
                    # 让服务接收中断，再关闭连接；不触碰其他对话的 turn。
                    self.proc.stdin.close()
                    self.proc.wait(timeout=2)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        for stream in (self.proc.stdin, self.proc.stdout):
            if not stream.closed:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def catalog(root, search='', cursor=None, codex_home=None):
    """只返回当前项目的标题/ID/状态和模型选项；不发送任何模型推理请求。"""
    with CodexClient(root, codex_home=codex_home) as client:
        threads = client.call('thread/list', {'cwd': str(Path(root).resolve()), 'limit': 40,
                              'sortKey': 'updated_at', 'searchTerm': search[:200] or None,
                              'cursor': cursor, 'archived': False})
        models = []
        model_cursor = None
        while True:
            page = client.call('model/list', {'limit': 100, 'cursor': model_cursor})
            models.extend({k: row[k] for k in ('id', 'model', 'displayName', 'supportedReasoningEfforts',
                          'defaultReasoningEffort', 'isDefault') if k in row} for row in page['data'])
            model_cursor = page.get('nextCursor')
            if not model_cursor:
                break
        return {'threads': [{k: row[k] for k in ('id', 'name', 'preview', 'updatedAt', 'status') if k in row}
                            for row in threads['data']], 'next_cursor': threads.get('nextCursor'),
                'models': models, 'codex_home': client.codex_home}


def history(root, thread_id, cursor=None, codex_home=None):
    """分页读取历史；不 resume、不发 turn，不把工具输出和推理内容传到网页。"""
    if not thread_id:
        raise ValueError('请选择对话')
    with CodexClient(root, codex_home=codex_home) as client:
        thread = client.call('thread/read', {'threadId': thread_id, 'includeTurns': False})['thread']
        if Path(thread.get('cwd', '')).resolve() != Path(root).resolve():
            raise ValueError('只能查看当前项目的对话')
        page = client.call('thread/items/list', {'threadId': thread_id, 'limit': 50,
                           'sortDirection': 'desc', 'cursor': cursor})
        rows = []
        for entry in reversed(page['data']):
            item = entry['item'];kind = item['type']
            if kind == 'userMessage':
                text = '\n'.join(c.get('text', '') if c['type']=='text' else f"[{c['type']}]"
                                 for c in item.get('content', []))
                role = 'user'
            elif kind == 'agentMessage':
                text = item.get('text', '');role = 'assistant'
            elif kind in {'reasoning', 'hookPrompt'}:
                continue
            else:
                text = f"工具/事件：{kind}";role = 'event'
            rows.append({'id':item['id'],'turn_id':entry['turnId'],'role':role,
                         'text':text,'time_ms':entry.get('startedAtMs')})
        return {'thread_id':thread_id,'name':thread.get('name') or thread.get('preview') or thread_id,
                'codex_home':client.codex_home,'messages':rows,'next_cursor':page.get('nextCursor')}


def attach_thread(client, root, request):
    thread_id = str(request.get('thread_id') or '').strip()
    mode = request.get('thread_mode', 'resume')
    if mode not in {'resume', 'fork', 'new'}:
        raise ValueError('无效会话接入方式')
    if mode != 'new' and not thread_id:
        raise ValueError('请先选择本地对话，或选择“新建对话”')
    if thread_id and mode != 'new':
        source = client.call('thread/read', {'threadId': thread_id, 'includeTurns': False})['thread']
        if Path(source.get('cwd', '')).resolve() != Path(root).resolve():
            raise ValueError('只能接入当前项目的对话')
        recent = client.call('thread/turns/list', {'threadId': thread_id, 'limit': 1,
                            'sortDirection': 'desc'})['data']
        if source.get('status', {}).get('type') == 'active' or any(
                turn.get('status') == 'inProgress' for turn in recent):
            raise ValueError('原对话正在运行，请等其结束后再接入；本次未发送问题')
    params = {'cwd': str(Path(root).resolve()), 'sandbox': 'read-only', 'approvalPolicy': 'never'}
    if request.get('model'):
        params['model'] = request['model']
    if mode == 'new':
        method = 'thread/start'
    else:
        method = 'thread/fork' if mode == 'fork' else 'thread/resume'
        params.update(threadId=thread_id, excludeTurns=True)
    return client.call(method, params)['thread']['id']


def answer(client, thread_id, prompt, images=(), model=None, effort=None, on_delta=None):
    inputs = [{'type': 'text', 'text': prompt}]
    inputs.extend({'type': 'localImage', 'path': str(path)} for path in images)
    params = {'threadId': thread_id, 'input': inputs, 'approvalPolicy': 'never',
              'sandboxPolicy': {'type': 'readOnly', 'networkAccess': False}}
    if model:
        params['model'] = model
    if effort:
        params['effort'] = effort
    turn = client.call('turn/start', params)['turn']
    client.active_turn = {'threadId': thread_id, 'turnId': turn['id']}
    deadline = time.monotonic() + 240
    texts = {}
    while True:
        event = client.notification(deadline)
        data = event.get('params', {})
        if data.get('threadId') != thread_id or data.get('turnId', turn['id']) != turn['id']:
            continue
        if event.get('method') == 'item/agentMessage/delta':
            item_id = data['itemId']
            texts[item_id] = texts.get(item_id, '') + data['delta']
            if on_delta:
                on_delta(''.join(texts.values()))
        if event.get('method') == 'item/completed' and data.get('item', {}).get('type') == 'agentMessage':
            item = data['item']
            texts[item['id']] = item['text']
        if event.get('method') == 'turn/completed':
            client.active_turn = None
            final = data['turn']
            if final['status'] != 'completed':
                raise CodexTurnError(thread_id, final)
            result = '\n\n'.join(texts.values()).strip()
            if not result:
                raise ValueError('Codex 未返回文字答案')
            return result
