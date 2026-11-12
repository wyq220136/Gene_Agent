"""工作台会话身份与只读续聊契约；不调用外部模型。"""
from unittest.mock import Mock

import pytest

from gene_agent.workbench.codex_client import attach_thread, answer, CodexClient
from gene_agent.workbench.service import Workbench


def source_client(root, status='completed'):
    def call(method, params):
        if method == 'thread/read':
            return {'thread': {'cwd': str(root), 'status': {'type': 'notLoaded'}}}
        if method == 'thread/turns/list':
            return {'data': [{'status': status}]}
        return {'thread': {'id': 'branch' if method == 'thread/fork' else 'original'}}
    return Mock(call=Mock(side_effect=call))


@pytest.mark.parametrize('mode,method,expected', [('resume','thread/resume','original'),
                                               ('fork','thread/fork','branch')])
def test_attach_identity_and_readonly(tmp_path, mode, method, expected):
    client=source_client(tmp_path)
    assert attach_thread(client,tmp_path,{'thread_id':'original','thread_mode':mode}) == expected
    actual,params=client.call.call_args.args
    assert actual == method
    assert params['threadId']=='original'
    assert params['sandbox']=='read-only' and params['approvalPolicy']=='never'
    assert 'developerInstructions' not in params


def test_default_continues_original_and_no_missing_id_fallback(tmp_path):
    client=source_client(tmp_path)
    assert attach_thread(client,tmp_path,{'thread_id':'original'})=='original'
    with pytest.raises(ValueError,match='选择'):
        attach_thread(client,tmp_path,{})
    assert not any(c.args[0]=='thread/start' for c in client.call.call_args_list)


def test_busy_source_is_not_resumed_or_forked(tmp_path):
    for mode in ['resume','fork']:
        client=source_client(tmp_path,'inProgress')
        with pytest.raises(ValueError,match='正在运行'):
            attach_thread(client,tmp_path,{'thread_id':'original','thread_mode':mode})
        assert len(client.call.call_args_list)==2


def test_other_project_is_rejected(tmp_path):
    client=source_client(tmp_path/'other')
    with pytest.raises(ValueError,match='当前项目'):
        attach_thread(client,tmp_path,{'thread_id':'original'})


def test_new_is_explicit_and_persistent(tmp_path):
    client=source_client(tmp_path)
    attach_thread(client,tmp_path,{'thread_mode':'new'})
    method,params=client.call.call_args.args
    assert method=='thread/start' and params.get('ephemeral') is not True


def test_answer_filters_events_and_transmits_model_effort():
    client=Mock()
    client.call.return_value={'turn':{'id':'turn-1'}}
    client.notification.side_effect=[
        {'method':'item/agentMessage/delta','params':{'threadId':'other','turnId':'turn-1','itemId':'m','delta':'wrong'}},
        {'method':'item/agentMessage/delta','params':{'threadId':'original','turnId':'turn-1','itemId':'m','delta':'答案'}},
        {'method':'item/completed','params':{'threadId':'original','turnId':'turn-1','item':{'id':'m','type':'agentMessage','text':'答案。'}}},
        {'method':'turn/completed','params':{'threadId':'original','turn':{'id':'turn-1','status':'completed'}}},
    ]
    updates=[]
    assert answer(client,'original','question',model='example',effort='high',on_delta=updates.append)=='答案。'
    _,params=client.call.call_args.args
    assert params['model']=='example' and params['effort']=='high'
    assert params['sandboxPolicy']=={'type':'readOnly','networkAccess':False}
    assert updates==['答案'] and client.active_turn is None


def test_rpc_errors_do_not_become_answers():
    client=object.__new__(CodexClient)
    client.serial=0;client.timeout=1;client.events=[];client.send=Mock()
    client.receive=Mock(return_value={'id':1,'error':{'message':'model unavailable'}})
    with pytest.raises(ValueError,match='model unavailable'):
        client.call('turn/start',{})


@pytest.mark.parametrize('message,code', [
    ('Error running remote compact task: workspace routing discovery failed', 'remote_compaction_routing'),
    ('model unavailable', 'turn_failed'),
])
def test_failed_turn_preserves_details_and_never_restarts(message, code):
    from gene_agent.workbench.codex_client import CodexTurnError
    client=Mock()
    client.call.return_value={'turn':{'id':'turn-1'}}
    error={'message':message,'codexErrorInfo':'other'}
    client.notification.side_effect=[
        {'method':'item/agentMessage/delta','params':{'threadId':'branch','itemId':'m','delta':'未完成内容'}},
        {'method':'turn/completed','params':{'threadId':'branch','turn':{
            'id':'turn-1','status':'failed','error':error}}},
    ]
    updates=[]
    with pytest.raises(CodexTurnError) as caught:
        answer(client,'branch','why',on_delta=updates.append)
    detail=caught.value.details
    assert detail['code']==code and detail['error']==error
    assert detail['thread_id']=='branch' and detail['turn_id']=='turn-1'
    assert detail['status']=='failed' and detail['scope']=='assistant'
    assert updates==['未完成内容'] and client.active_turn is None
    assert client.call.call_count==1


def test_assistant_writes_failure_record_without_claiming_success(tmp_path, monkeypatch, capsys):
    import json
    from gene_agent.workbench import assistant
    from gene_agent.workbench.codex_client import CodexTurnError
    request=tmp_path/'request.json'
    request.write_text(json.dumps({'question':'why','thread_id':'source','thread_mode':'fork'}))
    client=Mock(codex_home=str(tmp_path))
    client.__enter__=Mock(return_value=client);client.__exit__=Mock(return_value=False)
    monkeypatch.setattr(assistant,'CodexClient',Mock(return_value=client))
    monkeypatch.setattr(assistant,'attach_thread',Mock(return_value='branch'))
    failure=CodexTurnError('branch',{'id':'turn-1','status':'failed','error':{
        'message':'Error running remote compact task: workspace routing discovery failed'}})
    def fail(*args, **kwargs):
        kwargs['on_delta']('不完整回答')
        raise failure
    monkeypatch.setattr(assistant,'answer',fail)
    monkeypatch.setattr('sys.argv',['assistant','--request',str(request),'--root',str(tmp_path)])
    with pytest.raises(SystemExit) as caught:assistant.main()
    assert caught.value.code==1
    detail=json.loads((tmp_path/'assistant_error.json').read_text())
    assert detail['code']=='remote_compaction_routing'
    assert detail['thread_id']=='branch'
    assert json.loads((tmp_path/'session.json').read_text())['source_thread_id']=='source'
    assert (tmp_path/'answer.txt').read_text()=='不完整回答'
    assert 'Traceback' not in capsys.readouterr().out
    service=object.__new__(Workbench)
    service.root=tmp_path
    service.jobs=Mock()
    service.jobs.get.return_value={'state':'failed','inputs':{}}
    service.jobs.log.return_value='助手未完成'
    service.jobs.directory.return_value=tmp_path
    row=service.job('test')
    assert row['state']=='failed' and row['assistant_error']==detail
    assert row['answer']=='不完整回答' and row['session']['thread_id']=='branch'


def test_workbench_requires_explicit_thread_and_preserves_fork_request(tmp_path):
    service=Workbench(tmp_path)
    try:
        with pytest.raises(ValueError,match='选择'):
            service.submit({'task':'sweep','action':'ask','options':{'question':'why'}})
        service.jobs.submit=Mock(return_value={'id':'test'})
        service.submit({'task':'sweep','action':'ask','options':{
            'question':'why','thread_id':'original','thread_mode':'fork','model':'example','effort':'high','codex_home':str(tmp_path)}})
        import json
        request=next((tmp_path/'runs/workbench/jobs').glob('*/request.json'))
        assert json.loads(request.read_text())['thread_mode']=='fork'
        assert json.loads(request.read_text())['codex_home']==str(tmp_path)
    finally:
        service.jobs.close()


def test_codex_home_explicit_and_inherited(tmp_path, monkeypatch):
    from gene_agent.workbench.codex_client import resolve_codex_home
    first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
    monkeypatch.setenv('CODEX_HOME',str(first))
    assert resolve_codex_home()==str(first)
    assert resolve_codex_home(str(second))==str(second)
    import os
    assert os.environ['CODEX_HOME']==str(first)
    for invalid in ['relative','/not/a/real/codex/home',str(first/'file')]:
        with pytest.raises(ValueError,match='CODEX_HOME'):resolve_codex_home(invalid)


def test_child_receives_selected_home_without_changing_parent(tmp_path, monkeypatch):
    import os
    import gene_agent.workbench.codex_client as module
    parent=tmp_path/'parent';selected=tmp_path/'selected';parent.mkdir();selected.mkdir()
    monkeypatch.setenv('CODEX_HOME',str(parent))
    proc=Mock();proc.stdout=iter([])
    popen=Mock(return_value=proc)
    monkeypatch.setattr(module.shutil,'which',lambda _: '/fake/codex')
    monkeypatch.setattr(module.subprocess,'Popen',popen)
    monkeypatch.setattr(CodexClient,'call',lambda *a: {})
    monkeypatch.setattr(CodexClient,'send',lambda *a: None)
    CodexClient(tmp_path,codex_home=str(selected))
    assert popen.call_args.kwargs['env']['CODEX_HOME']==str(selected)
    assert os.environ['CODEX_HOME']==str(parent)


def test_history_is_readonly_paginated_and_preserves_message_order(tmp_path, monkeypatch):
    from gene_agent.workbench import codex_client as module
    client=Mock();client.codex_home=str(tmp_path/'chosen')
    def call(method, params):
        if method=='thread/read':return {'thread':{'id':'original','name':'Example','cwd':str(tmp_path)}}
        assert method=='thread/items/list'
        assert params['cursor']=='older' and params['sortDirection']=='desc'
        return {'data':[
            {'turnId':'t','item':{'id':'a','type':'agentMessage','text':'<script>literal text</script>'}},
            {'turnId':'t','item':{'id':'r','type':'reasoning','text':'not exposed'}},
            {'turnId':'t','item':{'id':'x','type':'commandExecution','aggregatedOutput':'not exposed'}},
            {'turnId':'t','item':{'id':'u','type':'userMessage','content':[{'type':'text','text':'question'}]}},
        ],'nextCursor':'next'}
    client.call.side_effect=call
    factory=Mock();factory.return_value.__enter__=Mock(return_value=client);factory.return_value.__exit__=Mock()
    monkeypatch.setattr(module,'CodexClient',factory)
    page=module.history(tmp_path,'original','older',str(tmp_path/'chosen'))
    assert [x['id'] for x in page['messages']]==['u','x','a']
    assert page['next_cursor']=='next' and page['codex_home']==client.codex_home
    assert page['messages'][-1]['text']=='<script>literal text</script>'
    assert 'not exposed' not in str(page)
    factory.assert_called_once_with(tmp_path,codex_home=str(tmp_path/'chosen'))
    assert not any(c.args[0] in ['thread/resume','turn/start'] for c in client.call.call_args_list)
