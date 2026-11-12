"""显式附件、发送清单和冻结证据的一致性。"""
import base64
import io
import json
from unittest.mock import Mock

from PIL import Image
import pytest

from gene_agent.workbench.attachments import add_attachment, attachment_context, context_manifest
from gene_agent.workbench.assistant import prepare_context
from gene_agent.workbench.service import Workbench


def test_explicit_text_file_without_run_and_frozen_copy(tmp_path):
    source=tmp_path/'notes.md';source.write_text('original')
    a=add_attachment(tmp_path,{'path':str(source)})
    source.write_text('changed later')
    context=prepare_context(tmp_path,{'question':'解释这个文件','attachments':[a['id']]})
    assert context['attachments'][0]['text']=='original'
    assert context['attachments'][0]['source']==str(source)
    assert 'evidence' not in context


def test_image_is_real_model_input_and_invalid_image_rejected(tmp_path):
    data=io.BytesIO();Image.new('RGB',(4,3),'red').save(data,format='PNG')
    a=add_attachment(tmp_path,{'name':'screenshot.png','data':base64.b64encode(data.getvalue()).decode()})
    c=prepare_context(tmp_path,{'question':'看图','attachments':[a['id']]})
    assert c['images']==[a['path']]
    assert context_manifest(c)['images']==c['images']
    with pytest.raises(ValueError,match='图片'):
        add_attachment(tmp_path,{'name':'bad.png','data':base64.b64encode(b'not image').decode()})


def test_text_truncation_and_metadata_filename(tmp_path):
    a=add_attachment(tmp_path,{'name':'attachment.json','data':base64.b64encode(('a'*9000).encode()).decode()})
    rows=attachment_context(tmp_path,[a['id']])
    assert len(rows[0]['text'])==8000 and rows[0]['truncated']
    assert (tmp_path/a['path']).read_text()=='a'*9000
    assert attachment_context(tmp_path,[])==[]
    with pytest.raises(ValueError,match='最多'):attachment_context(tmp_path,[a['id']]*9)
    with pytest.raises(ValueError):attachment_context(tmp_path,['../escape'])


def test_directory_is_listing_not_recursive_upload(tmp_path):
    source=tmp_path/'data';source.mkdir();(source/'a.txt').write_text('file body not automatically sent')
    a=add_attachment(tmp_path,{'path':str(source)})
    row=attachment_context(tmp_path,[a['id']])[0]
    assert 'a.txt' in row['text'] and 'file body not' not in row['text']
    assert '目录清单' in row['note']
    with pytest.raises(ValueError):add_attachment(tmp_path,{'path':'/proc/self/environ'})


def test_evidence_image_removal_and_run_context_optout(tmp_path):
    run=tmp_path/'runs/a';run.mkdir(parents=True)
    (run/'result.json').write_text('{"success":false}')
    for name in ['one.png','two.png','three.png','four.png']:(run/name).write_bytes(b'image placeholder')
    c=prepare_context(tmp_path,{'question':'why','run':'runs/a','allow_evidence':True})
    first=c['images'][0]
    removed=prepare_context(tmp_path,{'question':'why','run':'runs/a','allow_evidence':True,'excluded_images':[first]})
    assert removed['images']==c['images'][1:]
    excluded=prepare_context(tmp_path,{'question':'why','run':'runs/a','include_run':False,'include_task':False,'task':'sweep'})
    assert 'evidence' not in excluded and excluded['task'] is None


def test_submit_freezes_actual_context_and_reports_manifest(tmp_path):
    run=tmp_path/'runs/a';run.mkdir(parents=True);report=run/'result.json';report.write_text('{"success":false}')
    service=Workbench(tmp_path)
    try:
        service.jobs.submit=Mock(return_value={'id':'test'})
        service.submit({'task':'sweep','action':'ask','options':{'question':'why','thread_id':'test','run':'runs/a','allow_evidence':True}})
        report.write_text('{"success":true}')
        frozen=json.loads(next((tmp_path/'runs/workbench/jobs').glob('*/context.json')).read_text())
        assert frozen['evidence']['summary']['success'] is False
        inputs=service.jobs.submit.call_args.args[1]
        assert inputs['context']['reports'][0]['text']=='{"success":false}'
    finally:service.jobs.close()
