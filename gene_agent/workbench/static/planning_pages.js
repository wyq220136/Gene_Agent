'use strict';
function renderPlanningPages(){
 const catalog=state.visualPreviews;if(!catalog)return;
 for(const [id,rows] of [['preview-knowledge',catalog.knowledge[state.task]],['preview-skill',catalog.skills[state.task]]]){
  const old=$('#'+id).value;$('#'+id).innerHTML=Object.entries(rows||{}).map(([k,v])=>`<option value="${esc(k)}">${esc(v)}</option>`).join('');
  if(Object.hasOwn(rows||{},old))$('#'+id).value=old;
 }
 const info=task();const supported=info.visual_planning;
 $('#execution-mode').querySelector('option[value=visual]').disabled=!supported;
 if(!supported&&$('#execution-mode').value==='visual')$('#execution-mode').value='physical';
 $('#execution-mode').dispatchEvent(new Event('change'));
 $('#placement-mode').disabled=!info.randomization;if(!info.randomization)setPlacementMode('fixed');
 $('#open-knowledge-preview').disabled=!Object.keys(catalog.knowledge[state.task]||{}).length;
 $('#plan-skill-preview').disabled=!info.skill_preview;
 $('#plan-skill-preview').textContent=supported?'cuRobo 规划并预览':'运行到所选 Skill 并预览';
 $('#skill-preview-description').textContent=supported?'无 MuJoCo；场景碰撞由独立选项控制。持物初态可作为明确假设。':'MuJoCo 实际执行：从初态运行到所选 Skill（包含前置阶段）。未接入此任务的无 MuJoCo 规划。';
 const source=state.scenePath||task()?.scene||task()?.config;
 $('#knowledge-preview-source').textContent='场景：'+source+($('#open-knowledge-preview').disabled?'；该 benchmark 尚未接入独立 Knowledge 预览。':'');
 $('#skill-preview-source').textContent='场景：'+source;
}
function clearPureViewers(){
 for(const key of ['knowledge','skill','task']){
  const jid=state[key+'ViewJob'];if(jid)api('/api/cancel',{id:jid}).catch(console.warn);
  state[key+'ViewJob']=null;const frame=$('#'+key+'-preview-frame');if(frame){frame.hidden=true;frame.removeAttribute('src');$('#'+key+'-preview-link').hidden=true;}
 }
}
async function attachPureViewer(job,key){
 const old=state[key+'ViewJob'];if(old&&old!==job.id)await api('/api/cancel',{id:old});
 state[key+'ViewJob']=job.id;$('#'+key+'-preview-status').textContent=state[key+'RunStatus']||'正在加载三维场景…';
}
async function startKnowledgePreview(){
 const epoch=state.epoch;
 state.knowledgeRunStatus='';
 const job=await launch('knowledge_preview',{preview_knowledge:$('#preview-knowledge').value});
 if(epoch!==state.epoch){await api('/api/cancel',{id:job.id});return;}
 await attachPureViewer(job,'knowledge');
}
async function beginLiveRun(action,opts={}){
 if(state.visualPending)return toast('已有任务正在运行，请等待或在作业列表停止');
 if(state.editorDraft)throw Error('编辑器有未保存的摆位，请先保存新场景，再预览并运行');
 const epoch=state.epoch,taskId=state.task;
 const job=await launch(action,opts);
 if(epoch!==state.epoch)return;
 const key=opts.preview_skill?'skill':'task';
 state[key+'RunStatus']='';
 state.visualPending={id:job.id,task:taskId,key};$('#log-details').open=true;
 showPage(key==='skill'?'skills':'results');
 if(key==='task')$('#task-preview-panel').hidden=false;
 try{
  const viewer=await launch('live_viewer',{run:job.inputs.output_run},{selectLog:false});
  if(epoch!==state.epoch){await api('/api/cancel',{id:viewer.id});return;}
  await attachPureViewer(viewer,key);
 }finally{
  // Task 与查看器是两个作业；自动打开查看器不能抢走 Task 的失败日志。
  if(epoch===state.epoch){state.job=job.id;await loadJob(job.id);}
 }
}
async function startSkillPreview(){
 return beginLiveRun(task().visual_planning?'visual_plan':'run',{preview_skill:$('#preview-skill').value});
}
async function startTaskRun(withVideo=false){
 return beginLiveRun($('#execution-mode').value==='visual'?'visual_plan':withVideo===true?'run_video':'run');
}
async function openPlannedRecord(run,key='task'){
 const epoch=state.epoch;
 state[key+'RunStatus']='';
 showPage(key==='skill'?'skills':'results');
 if(key==='task')$('#task-preview-panel').hidden=false;
 const summary=await api('/api/run?path='+encodeURIComponent(run));
 if(epoch!==state.epoch)return;
 const job=await launch(summary.live_available?'live_viewer':'visual_viewer',{run});
 if(epoch!==state.epoch){await api('/api/cancel',{id:job.id});return;}
 await attachPureViewer(job,key);
}
async function refreshPlanningPages(){
 if(state.visualPending){
  const pending=state.visualPending;
  const row=await api('/api/job?id='+pending.id);
  if(state.visualPending!==pending||pending.task!==state.task)return;
  if(!['queued','running'].includes(row.state)){
   state.visualPending=null;
   const key=pending.key||'task';
   const failed=row.state!=='completed';
   state[key+'RunStatus']=failed?'任务'+(states[row.state]||row.state)+'：'+(row.error||row.log?.trim()||'请查看任务日志'):
    '运行已结束，在线窗口保留已记录的全部前缀，可继续回放。';
   $('#'+key+'-preview-status').textContent=state[key+'RunStatus'];
   if(failed){state.job=pending.id;$('#log-details').open=true;await loadJob(pending.id);}
   await refreshRuns();

  }
 }
 for(const key of ['knowledge','skill','task']){
  const jid=state[key+'ViewJob'];if(!jid)continue;
  const row=await api('/api/job?id='+jid);if(state[key+'ViewJob']!==jid)continue;
  const frame=$('#'+key+'-preview-frame'),link=$('#'+key+'-preview-link');
  if(row.viewer_ready){if(frame.getAttribute('src')!==row.inputs.viewer_url)frame.src=row.inputs.viewer_url;frame.hidden=false;link.href=row.inputs.viewer_url;link.hidden=false;$('#'+key+'-preview-status').textContent=state[key+'RunStatus']||'三维窗口已就绪，可拖动视角。证据类型和运行状态见窗口；暂停浏览不停止任务。';}
  else if(!['queued','running'].includes(row.state))$('#'+key+'-preview-status').textContent='查看器未启动：'+(row.error||row.log?.slice(-1500)||states[row.state]);
 }
}
function initPlanningPages(){
 $('#open-knowledge-preview').onclick=guard(startKnowledgePreview);$('#plan-skill-preview').onclick=guard(startSkillPreview);
 const updateMode=()=>{const visual=$('#execution-mode').value==='visual',full=$('#collision-mode').value==='full';$('#run-video').disabled=visual;$('#execution-note').textContent=(visual?'纯视觉规划：不使用 MuJoCo。':'MuJoCo 实际执行：物理接触与成功判定保持启用。')+(full?'cuRobo 检查场景碰撞。':'cuRobo 忽略场景碰撞。')+' 保留机器人碰撞与关节限位。';};
 $('#execution-mode').onchange=updateMode;$('#collision-mode').onchange=()=>{$('#skill-collision-mode').value=$('#collision-mode').value;updateMode();};$('#skill-collision-mode').onchange=()=>{$('#collision-mode').value=$('#skill-collision-mode').value;updateMode();};updateMode();
}
