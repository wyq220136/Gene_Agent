'use strict';
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state={token:'',tasks:[],task:'sweep',runs:[],run:null,stage:'',scene:null,scenePath:null,job:null,history:[],answered:new Set(),busy:false,epoch:0};
const names={run_preview:'运行前场景预览',live_viewer:'在线查看',visual_plan:'纯视觉规划',knowledge_preview:'Knowledge 几何预览',visual_viewer:'纯视觉轨迹查看',run:'运行仿真',run_video:'运行并导出视频',video:'导出视频',viewer:'三维回放',skill:'Skill 回放',distribution:'摆位预览',scene_viewer:'场景编辑',sample_viewer:'样本三维查看',concept:'部件编辑',knowledge:'抓取实验台',ask:'助手分析'};
const states={planned:'规划完成',planning_failed:'规划未完成',planning:'规划中',queued:'排队中',running:'运行中',completed:'已完成',failed:'作业有失败',canceled:'已取消',interrupted:'已中断'};
function toast(text){$('#toast').textContent=text;$('#toast').style.display='block';clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('#toast').style.display='none',5500);}
async function api(path,data){const response=await fetch(path,{method:data?'POST':'GET',headers:data?{'Content-Type':'application/json','X-Workbench-Token':state.token}:{},body:data?JSON.stringify(data):undefined});const result=await response.json();if(!response.ok)throw Error(result.error||response.statusText);return result;}
function guard(fn){return async(...args)=>{try{return await fn(...args);}catch(e){toast(e.message);}};}
function task(){return state.tasks.find(x=>x.id===state.task);}
function file(path){return '/file?path='+encodeURIComponent(path);}
function badge(row){if(row.record_type==='pure_visual_plan')return `<span class="badge ${row.planning_success?'good':'bad'}">${row.planning_success?'规划通过（未执行）':'规划未完成'}</span>`;return row.success===true?`<span class="badge good">${row.simulation_assistance?'仿真辅助成功':'任务成功'}</span>`:row.success===false?'<span class="badge bad">未完成</span>':'<span class="badge">尚无成功判定</span>';}
function modal(title,html){$('#modal').classList.remove('wide');$('#modal-title').textContent=title;$('#modal-body').innerHTML=html;$('#modal').showModal();}
function options(){return {execution_mode:$('#execution-mode').value,collision_mode:$('#collision-mode').value,seed:Number($('#seed').value),tilt:Number($('#tilt').value),randomize:$('#randomize').checked,...(state.scenePath?{scene:state.scenePath}:{})};}
function updateContext(){renderAssistantContext();$('#chat-context').textContent=`${task()?.name||''} · ${state.run?.name||'未选择运行记录'}${state.stage?' · '+state.stage:''}`;}
function showPage(page){if(state.page&&state.page!==page)clearPureViewers();state.page=page;for(const key of ['results','scene','knowledge','skills'])$('#'+key+'-page').hidden=page!==key;document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===page));if(page==='scene')guard(loadScene)();if(['knowledge','skills'].includes(page))renderPlanningPages();}
async function changeTask(){clearRunScenePreview();clearPureViewers();state.visualPending=null;state.epoch++;$('#allow-evidence').checked=false;state.task=$('#task').value;state.run=null;state.stage='';state.scenePath=localStorage.getItem('gene-scene-'+state.task)||null;state.scene=null;state.rulesDirty=false;state.presetDraft=false;
 setPlacementMode(localStorage.getItem('gene-mode-'+state.task)||'fixed');
 $('#seed').value=localStorage.getItem('gene-seed-'+state.task)||(state.task==='microwave'?123:36);
 clearSceneEditor();clearSamplePreview();state.history=[];state.job=null;state.chatJob=null;state.savedConcept=null;state.displayedVideo=null;setChatBusy(false);$('#distribution-images').textContent='';$('#messages').textContent='';$('#task-name').textContent=task().name;$('#description').textContent=task().description;document.querySelectorAll('.sweep-field').forEach(el=>el.hidden=state.task!=='sweep');$('#skill').innerHTML=task().skills.map(s=>`<option value="${esc(s.id)}">${esc(s.name)}</option>`).join('');await refreshRuns();renderDetail();updateContext();renderSceneVersions();renderPlanningPages();if(!$('#scene-page').hidden)await loadScene();}
async function refreshRuns(){state.runs=await api('/api/runs');const rows=state.runs.filter(x=>x.task_id===state.task);$('#run-list').innerHTML=rows.length?rows.map(r=>`<div class="run-item ${state.run?.path===r.path?'selected':''}" tabindex="0" data-run="${esc(r.path)}"><strong>${esc(r.name||r.path)}</strong>${badge(r)}<small>${r.updated_at?new Date(r.updated_at*1000).toLocaleString():''}</small><small>${esc(r.failure_stage?'停止于 '+r.failure_stage:(states[r.status]||r.status))}</small></div>`).join(''):'<p class="muted">还没有记录。可以运行一次，也可以导入已有目录。</p>';document.querySelectorAll('[data-run]').forEach(el=>{const select=()=>{state.epoch++;$('#allow-evidence').checked=false;state.chatJob=null;state.history=[];$('#messages').textContent='';setChatBusy(false);state.run=rows.find(r=>r.path===el.dataset.run);state.stage='';renderDetail();refreshRuns().catch(console.error);updateContext();};el.onclick=select;el.onkeydown=e=>{if(e.key==='Enter')select();};});if(state.run){const next=rows.find(r=>r.path===state.run.path)||state.run;const changed=JSON.stringify(next)!==JSON.stringify(state.run);state.run=next;if(changed)renderDetail();}}
function showMedia(path){const isVideo=path.endsWith('.mp4');$('#media').classList.remove('empty');$('#media').innerHTML=isVideo?`<video controls preload="metadata" src="${esc(file(path))}"></video>`:`<a href="${esc(file(path))}" target="_blank" rel="noopener"><img src="${esc(file(path))}" alt="保存的运行画面"></a>`;}
function renderDetail(){const row=state.run;for(const id of ['view-3d','export','view-skill','ask-failure'])$('#'+id).disabled=!row||(id!=='ask-failure'&&!row?.reports?.length);$('#detail-title').textContent=row?row.name:'选择一条记录';$('#result-badge').innerHTML=row?badge(row):'';$('#detail-hint').textContent=row?`${row.path} · ${row.evidence_mode}`:'先运行任务，或导入已有运行目录。';$('#stage-list').innerHTML='';$('#evidence').innerHTML='';$('#media-strip').innerHTML='';if(!row){$('#media').className='media empty';$('#media').innerHTML='<div><span class="empty-symbol">▷</span><h3>在这里查看动作效果</h3><p>运行完成后查看视频、三维回放和失败阶段。</p></div>';return;}
 if(row.record_type==='pure_visual_plan')$('#export').disabled=true;
 const media=row.media||[];if(media.length){showMedia((media.find(x=>x.type==='video')||media[0]).path);$('#media-strip').innerHTML=media.map((m,i)=>`<button data-media="${i}">${m.type==='video'?'▷':'▧'} ${esc(m.name)}</button>`).join('');document.querySelectorAll('[data-media]').forEach(b=>b.onclick=()=>showMedia(media[Number(b.dataset.media)].path));}else{$('#media').className='media empty';$('#media').innerHTML=row.record_type==='pure_visual_plan'?'<div><h3>规划轨迹可三维播放</h3><p>点击下方“打开三维回放”；规划失败时也能查看已生成的前半段。这不是 MuJoCo 实际视频。</p></div>':row.success===false?'<div><h3>任务失败，已记录的前半段仍可查看</h3><p>点击下方“打开三维回放”，或“导出视频”。“运行一次”只保存执行记录，不自动导出视频。</p></div>':'<div><h3>尚未导出画面</h3><p>可打开三维回放，或点击“导出视频”。</p></div>';}
 $('#stage-list').innerHTML=(row.stages||[]).map((s,i)=>`<button class="stage ${s.success===false?'failed':''}" data-stage="${i}">${esc(s.name||s.stage_id||'阶段 '+(i+1))} ${s.success===true?'✓':s.success===false?'!':''}</button>`).join('')||'<p class="muted">此报告未提供可识别的阶段摘要，请查看原始报告。</p>';
 document.querySelectorAll('[data-stage]').forEach(b=>b.onclick=()=>{const s=row.stages[Number(b.dataset.stage)];state.stage=s.name||s.stage_id;document.querySelectorAll('[data-stage]').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');$('#log').textContent=JSON.stringify(s,null,2);$('#log-details').open=true;updateContext();});
 $('#evidence').innerHTML=(row.failure_stage?`<div class="notice">停止阶段：${esc(row.failure_stage)}。直接原因请结合报告和实际画面检查。</div>`:'')+(row.errors||[]).map(e=>`<div class="error">${esc(e)}</div>`).join('')+(row.reports||[]).map(p=>`<a href="${esc(file(p))}" target="_blank" rel="noopener">↗ ${esc(p.split('/').pop())}</a>`).join('');}
async function launch(action,opts={},ui={}){
 const epoch=state.epoch,requestedTask=state.task;
 if(['run','run_video','distribution','visual_plan','run_preview'].includes(action))await ensureRulesSaved();
 if(epoch!==state.epoch||requestedTask!==state.task)throw Error('任务或记录已切换，请重新选择操作');
 const row=await api('/api/jobs',{task:state.task,action,options:{...options(),...opts}});if(epoch===state.epoch){if(ui.selectLog!==false)state.job=row.id;toast('已提交：'+names[action]);await refreshJobs();}return row;}
async function refreshJobs(){const jobs=await api('/api/jobs');const shown=jobs.filter(j=>j.inputs?.task===state.task).slice(0,12);$('#jobs').innerHTML=shown.length?shown.map(j=>`<div class="job"><span class="badge ${j.state==='failed'?'bad':j.state==='completed'?'good':''}">${esc(states[j.state]||j.state)}</span><span class="job-title">${esc(names[j.kind]||j.kind)} <small>${new Date(j.created_at*1000).toLocaleTimeString()}</small></span><button data-log="${j.id}">日志与产物</button>${j.inputs.viewer_url&&j.state==='running'?`<a href="${esc(j.inputs.viewer_url)}" target="_blank" rel="noopener">打开窗口 ↗</a>`:''}${['queued','running'].includes(j.state)?`<button data-cancel="${j.id}">停止</button>`:''}</div>`).join(''):'<p class="muted">尚无作业。需要运行时，在上方选择操作。</p>';
 document.querySelectorAll('[data-log]').forEach(b=>b.onclick=guard(async()=>{state.job=b.dataset.log;$('#log-details').open=true;await loadJob(state.job);}));document.querySelectorAll('[data-cancel]').forEach(b=>b.onclick=guard(async()=>{await api('/api/cancel',{id:b.dataset.cancel});await refreshJobs();}));
 if(state.job)await loadJob(state.job);
 if(state.editorJob&&state.editorJob!==state.job)await loadJob(state.editorJob);
 if(state.previewJob&&state.previewJob!==state.job)await loadJob(state.previewJob);
 if(state.chatJob)await refreshChat();
 await refreshPlanningPages();
 await refreshRunScene();
}
async function loadJob(jid){const row=await api('/api/job?id='+jid);if(row.inputs?.task!==state.task||(state.job!==jid&&state.editorJob!==jid&&state.previewJob!==jid))return;
 if(jid===state.editorJob)await updateSceneEditor(row);
 if(jid===state.previewJob)updateSamplePreview(row);
 if(state.job!==jid)return;$('#log').textContent=row.log||'等待输出…';if(row.error)$('#log').textContent+='\n'+row.error;if(row.kind==='distribution'&&row.samples?.length){
 $('#distribution-images').innerHTML=row.samples.map(sample=>{
 const media=row.media.find(p=>p.endsWith('/seed_'+sample.seed+'.png'));
 return `<div class="sample-card">${media?`<a href="${esc(file(media))}" target="_blank"><img src="${esc(file(media))}" alt="seed ${sample.seed}"></a>`:''}<b>seed ${sample.seed}</b><a href="${esc(file('runs/workbench/jobs/'+row.id+'/run/distribution.json'))}" target="_blank" rel="noopener">初态原始记录 ↗</a><small>${sample.valid?'初态检查通过，尚未验证任务':esc(sample.error||'初态检查未通过，请查看作业记录')}</small><button data-sample-view="${sample.seed}">三维查看 / 调整视角</button><button data-freeze="${sample.seed}" ${sample.valid?'':'disabled'}>以此摆位固定测试</button></div>`;
 }).join('');
 document.querySelectorAll('[data-sample-view]').forEach(button=>button.onclick=guard(()=>openSamplePreview(row.id,Number(button.dataset.sampleView))));
 document.querySelectorAll('[data-freeze]').forEach(button=>button.onclick=guard(async()=>{
 const epoch=state.epoch;
 const result=await api('/api/freeze',{job:row.id,seed:Number(button.dataset.freeze)});
 if(epoch!==state.epoch)return;
 await selectSceneVersion(result,'fixed');toast('已固定这个样本，后续按此布局测试');
 }));
}
 if(row.kind==='video'||row.kind==='run_video'){const video=row.media.find(x=>x.endsWith('.mp4'));const targetRun=row.inputs.output_run||row.inputs.options.run;if(video&&state.run?.path===targetRun&&state.displayedVideo!==video){state.displayedVideo=video;showMedia(video);toast('视频已导出，可直接播放');}}
 if(row.saved_concept&&state.savedConcept!==row.saved_concept){state.savedConcept=row.saved_concept;toast('已保存新 Concept，可在物体列表应用');await loadScene();}}
async function loadScene(){const requestedTask=state.task,requestedScene=state.scenePath;const sceneResult=await api('/api/scene?task='+state.task+(state.scenePath?'&scene='+encodeURIComponent(state.scenePath):''));if(requestedTask!==state.task||requestedScene!==state.scenePath)return;state.scene=sceneResult;const scene=state.scene;$('#scene-select').value=state.scenePath||'';$('#scene-3d').disabled=!(scene.editable||scene.viewable);$('#scene-3d').textContent=scene.editable?'打开场景编辑器':'查看 Benchmark 场景';$('#distribution').disabled=!task().distribution_preview;$('#knowledge').disabled=!scene.editable||state.task!=='sweep';$('#scene-version').textContent='当前场景输入：'+scene.path;$('#scene-edit-title').textContent=scene.editable?'1　编辑整个场景':'1　查看 Benchmark 场景';$('#scene-edit-help').textContent=scene.editable?'单击选中物体后可拖动和旋转，保存为新场景供后续运行。':scene.message;if(!scene.editable){$('#objects').textContent=scene.message+' 物体：'+scene.objects.slice(0,8).map(o=>o.name).join('、')+(scene.objects.length>8?'…（共 '+scene.objects.length+' 个物体，含球体）':'');$('#placement-rules').textContent=scene.message;$('#rules-note').textContent='';for(const id of ['save-rules','suggest-rules'])$('#'+id).disabled=true;return;}for(const id of ['save-rules','suggest-rules'])$('#'+id).disabled=false;
 renderRules(scene);$('#scene-version').textContent='当前场景输入：'+scene.path;$('#objects').innerHTML=scene.objects.map((o,i)=>`<div class="object-row" data-object="${i}"><b>${esc(({brush:'刷子',cube:'绿方块',cube_red:'红方块',cube_blue:'蓝方块',dustpan:'簸箕',microwave:'微波炉',bowl:'碗'})[o.name]||o.name)}</b><small>部件几何与功能标注</small><button data-edit="${i}" ${!o.mesh||!o.concept?'disabled':''}>编辑部件 ↗</button>${state.savedConcept&&scene.concept_editable?`<button data-apply="${i}">应用刚保存的标注</button>`:''}</div>`).join('');
 document.querySelectorAll('[data-edit]').forEach(b=>b.onclick=guard(()=>{const o=scene.objects[Number(b.dataset.edit)];return launch('concept',{mesh:o.mesh,concept:o.concept});}));document.querySelectorAll('[data-apply]').forEach(b=>b.onclick=guard(async()=>{const o=scene.objects[Number(b.dataset.apply)];if(!confirm(`把刚保存的标注应用给 ${o.name}，创建新场景版本？`))return;const result=await api('/api/scene',{scene:scene.path,concepts:{[o.name.startsWith('cube')?'cube':o.name]:state.savedConcept}});state.scenePath=result.path;clearSceneEditor();localStorage.setItem('gene-scene-'+state.task,result.path);await loadBootstrap();await loadScene();toast('已应用到新场景，后续运行使用此版本');}));}
function bubble(role,text){const el=document.createElement('div');el.className='bubble '+(role==='user'?'user-message':'assistant-message');el.textContent=text;$('#messages').append(el);$('#messages').scrollTop=$('#messages').scrollHeight;state.history.push({role,text});}
function openAssistant(){$('#assistant').hidden=false;updateContext();if(!state.assistantSessions){$('#assistant-catalog-status').textContent='当前后台仍是旧版本，请在保存编辑内容后重启工作台。';return;}if(!state.codexCatalogLoaded)guard(refreshCodexCatalog)();}
async function ask(question){
 if($('#send').disabled)return toast('当前问题正在分析，请等待或停止作业');
 openAssistant();
 if(!state.assistantSessions)return toast('请先保存编辑内容并重启工作台，启用会话接入');
 if(state.catalogLoading)return toast('正在加载所选 CODEX_HOME，请稍候');
 if($('#assistant-home').value.trim()!==state.codexHome)return toast('请先点击“应用目录并刷新”');
 const chosen=$('#assistant-thread').value;
 if(!chosen){$('#assistant-connection').open=true;return toast('请先选择本地对话，或选择“新建对话”');}
 if(state.attachmentPending)return toast('附件正在准备，请稍候');
 if(state.run&&$('#context-run').checked&&!$('#allow-evidence').checked){$('#question').value=question;return toast('请先确认允许向 Codex 服务发送当前记录证据');}
 const epoch=state.epoch;
 bubble('user',question);$('#question').value='';setChatBusy(true);
 state.replyElement=document.createElement('div');state.replyElement.className='bubble assistant-message';state.replyElement.textContent='正在接入对话…';$('#messages').append(state.replyElement);
 try{
  const job=await launch('ask',{question,...assistantContextOptions(),
   model:$('#assistant-model').value,effort:$('#assistant-effort').value,codex_home:state.codexHome,
   thread_id:chosen==='__new__'?null:chosen,
   thread_mode:chosen==='__new__'?'new':$('#assistant-mode').value,
   allow_evidence:$('#allow-evidence').checked});
  if(epoch===state.epoch){state.chatJob=job.id;showSentContext(job.inputs?.context);}
 }catch(e){if(epoch===state.epoch){setChatBusy(false);state.replyElement.textContent=e.message;}throw e;}
}
function setChatBusy(busy){$('#send').disabled=busy;$('#assistant-settings').disabled=busy;$('#assistant-context-settings').disabled=busy;}
function showSession(){const id=$('#assistant-thread').value;$('#assistant-mode').disabled=id==='__new__';$('#assistant-session').textContent=id==='__new__'?'首次发送时新建并保存对话':id?'当前对话 ID：'+id:'尚未选择对话';}
function threadStorageKey(){return 'gene-assistant-thread:'+state.codexHome;}
function setSession(id){
 const select=$('#assistant-thread');
 if(![...select.options].some(o=>o.value===id))select.add(new Option('已接入 · '+id,id));
 select.value=id;$('#assistant-mode').value='resume';localStorage.setItem(threadStorageKey(),id);showSession();
}
function updateEfforts(){
 const selected=(state.codexModels||[]).find(m=>(m.model||m.id)===$('#assistant-model').value);
 const old=$('#assistant-effort').value;
 $('#assistant-effort').innerHTML='<option value="">沿用对话 / Codex 默认</option>'+(selected?.supportedReasoningEfforts||[]).map(e=>`<option value="${esc(e.reasoningEffort)}">${esc(e.reasoningEffort)}</option>`).join('');
 if([...$('#assistant-effort').options].some(o=>o.value===old))$('#assistant-effort').value=old;
}
async function refreshCodexCatalog(more=false){
 const generation=(state.catalogGeneration||0)+1;state.catalogGeneration=generation;state.catalogLoading=true;
 const selectedHome=state.codexHome;
 $('#assistant-catalog-status').textContent='正在查询本机 Codex…';
 try{
  const query=new URLSearchParams({search:$('#assistant-search').value,codex_home:selectedHome});if(more&&state.catalogCursor)query.set('cursor',state.catalogCursor);
  const data=await api('/api/assistant/catalog?'+query);if(generation!==state.catalogGeneration||selectedHome!==state.codexHome)return;
  const select=$('#assistant-thread'),selected=select.value||localStorage.getItem('gene-assistant-thread:'+data.codex_home)||'';
  const remembered=[...select.options].find(o=>o.value===selected)?.textContent;
  if(!more)select.innerHTML='<option value="">请选择已有对话</option><option value="__new__">＋ 新建对话</option>';
  for(const t of data.threads){
   if([...select.options].some(o=>o.value===t.id))continue;
   const label=(t.name||t.preview||t.id).replace(/\s+/g,' ').slice(0,90);
   select.add(new Option(label+' · '+new Date(t.updatedAt*1000).toLocaleString(),t.id));
  }
  if(selected){if(![...select.options].some(o=>o.value===selected))select.add(new Option(remembered||('已选对话 · '+selected),selected));select.value=selected;}
  const oldModel=$('#assistant-model').value;state.codexModels=data.models;
  $('#assistant-model').innerHTML='<option value="">沿用对话 / Codex 默认</option>'+data.models.map(m=>`<option value="${esc(m.model||m.id)}">${esc(m.displayName||m.model||m.id)}</option>`).join('');
  if([...$('#assistant-model').options].some(o=>o.value===oldModel))$('#assistant-model').value=oldModel;
  state.codexHome=data.codex_home;$('#assistant-home').value=data.codex_home;localStorage.setItem('gene-assistant-home',data.codex_home);$('#assistant-home-status').textContent='生效目录：'+data.codex_home;
  updateEfforts();showSession();state.catalogCursor=data.next_cursor;$('#assistant-more').hidden=!data.next_cursor;
  state.codexCatalogLoaded=true;$('#assistant-catalog-status').textContent='仅列出当前项目的本地对话；列表来自本机 Codex。';
 }catch(e){if(generation===state.catalogGeneration)$('#assistant-catalog-status').textContent='加载失败：'+e.message;throw e;}finally{if(generation===state.catalogGeneration)state.catalogLoading=false;}
}
async function refreshChat(){
 const jid=state.chatJob;if(!jid)return;
 const row=await api('/api/job?id='+jid);if(jid!==state.chatJob)return;
 if(row.session?.thread_id&&row.session.codex_home===state.codexHome)setSession(row.session.thread_id);
 if(row.answer&&state.replyElement)state.replyElement.textContent=row.answer;
 if(!['queued','running'].includes(row.state)){
  const ok=row.state==='completed';
  if(!ok&&state.replyElement){
   const failure=row.assistant_error;
   const detail=failure?failure.message+'\n'+failure.hint+'\n对话：'+failure.thread_id+'\n这是助手服务错误，不能据此判断仿真任务失败。':(row.error||row.log?.slice(-2000)||states[row.state]);
   state.replyElement.textContent=(row.answer?'以下是未完成的部分回复：\n'+row.answer+'\n\n':'')+'分析未完成：'+detail;
  }
  state.chatJob=null;setChatBusy(false);state.answered.add(jid);
 }
}
$('#assistant-apply-home').onclick=guard(async()=>{
 if(state.chatJob)return toast('请等当前回复结束后切换 CODEX_HOME');
 state.codexHome=$('#assistant-home').value.trim();state.codexCatalogLoaded=false;state.catalogCursor=null;
 state.historyGeneration=(state.historyGeneration||0)+1;
 $('#assistant-thread').innerHTML='<option value="">请选择已有对话</option><option value="__new__">＋ 新建对话</option>';
 $('#assistant-model').innerHTML='<option value="">沿用对话 / Codex 默认</option>';updateEfforts();
 $('#messages').textContent='';state.history=[];$('#assistant-home-status').textContent='尚未确认目录，请等待加载完成';
 await refreshCodexCatalog();
});
$('#assistant-history').onclick=guard(async()=>{
 const thread=$('#assistant-thread').value,home=state.codexHome;
 if(!thread||thread==='__new__')return toast('先选择一个已有对话');
 if(state.catalogLoading||$('#assistant-home').value.trim()!==home)return toast('请先应用 CODEX_HOME 并等待加载完成');
 const generation=(state.historyGeneration||0)+1;state.historyGeneration=generation;
 modal('对话历史','<p id="history-source"></p><button id="history-more" hidden>加载更早记录</button><label class="history-tools"><input type="checkbox" id="history-tools"> 显示工具事件摘要</label><div id="history-status">正在读取…</div><div id="history-entries"></div>');
 $('#modal').classList.add('wide');let cursor=null;const seen=new Set();
 const load=async()=>{
  $('#history-more').disabled=true;
  try{
   const query=new URLSearchParams({codex_home:home,thread_id:thread});if(cursor)query.set('cursor',cursor);
   const data=await api('/api/assistant/history?'+query);
   if(generation!==state.historyGeneration||home!==state.codexHome||thread!==$('#assistant-thread').value||!$('#modal').open||!$('#history-entries'))return;
   $('#modal-title').textContent='对话历史 · '+data.name;
   $('#history-source').textContent='CODEX_HOME：'+data.codex_home+'\n对话 ID：'+data.thread_id;
   const fragment=document.createDocumentFragment();
   for(const message of data.messages){
    if(seen.has(message.id))continue;seen.add(message.id);
    const entry=document.createElement('details');entry.className='history-entry'+(message.role==='event'?' history-event':'');entry.open=message.role!=='event';
    const summary=document.createElement('summary');summary.textContent=({user:'用户',assistant:'Codex',event:'工具 / 事件'})[message.role]+(message.time_ms?' · '+new Date(message.time_ms).toLocaleString():'');
    const content=document.createElement('pre');content.textContent=message.text;entry.append(summary,content);fragment.append(entry);
   }
   $('#history-entries').prepend(fragment);cursor=data.next_cursor;$('#history-more').hidden=!cursor;
   $('#history-status').textContent=cursor?'按时间顺序显示，较早记录可继续加载。':'已到达历史开头。工具仅显示事件类型，不展示工具输出或推理内容。';
  }catch(e){if(generation===state.historyGeneration&&$('#history-status'))$('#history-status').textContent='读取失败：'+e.message;throw e;}
  finally{if(generation===state.historyGeneration&&$('#history-more'))$('#history-more').disabled=false;}
 };
 $('#history-tools').onchange=()=>$('#history-entries').classList.toggle('show-events',$('#history-tools').checked);
 $('#history-more').onclick=guard(load);await load();
});
$('#assistant-refresh').onclick=guard(()=>refreshCodexCatalog());
$('#assistant-more').onclick=guard(()=>refreshCodexCatalog(true));
$('#assistant-thread').onchange=()=>{state.historyGeneration=(state.historyGeneration||0)+1;localStorage.setItem(threadStorageKey(),$('#assistant-thread').value);$('#messages').textContent='';state.history=[];showSession();};
$('#assistant-model').onchange=updateEfforts;

document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>showPage(b.dataset.page));$('#task').onchange=guard(changeTask);$('#run').onclick=guard(startTaskRun);$('#run-video').onclick=guard(()=>startTaskRun(true));$('#export').onclick=guard(()=>launch('video',{run:state.run.path}));$('#view-3d').onclick=guard(()=>(state.run?.record_type==='pure_visual_plan'||state.run?.live_available)?openPlannedRecord(state.run.path):launch('viewer',{run:state.run.path}));$('#view-skill').onclick=guard(()=>(state.run?.record_type==='pure_visual_plan'||state.run?.live_available)?openPlannedRecord(state.run.path,'skill'):launch('skill',{run:state.run.path,skill:$('#skill').value}));$('#distribution').onclick=guard(()=>launch('distribution',{randomize:true}));$('#scene-3d').onclick=guard(openSceneEditor);$('#knowledge').onclick=guard(()=>launch('knowledge'));$('#assistant-toggle').onclick=()=>{if($('#assistant').hidden)openAssistant();else $('#assistant').hidden=true;};$('#chat-close').onclick=()=>$('#assistant').hidden=true;$('#ask-failure').onclick=guard(()=>ask('请分析当前运行：是否成功？如果失败，直接原因是什么，有什么证据，下一步应该检查什么？'));$('#chat-form').onsubmit=guard(async e=>{e.preventDefault();await ask($('#question').value);});document.querySelectorAll('[data-question]').forEach(b=>b.onclick=guard(()=>ask(b.dataset.question)));

$('#import').onclick=()=>{modal('导入已有运行记录','<p>填写项目内的运行目录，例如 runs/sweep_cube/…；只读取历史文件。</p><label>已有示例<select id="sample-run"><option value="">选择示例，或在下方填写目录</option>'+ (state.samples||[]).filter(s=>s.task===state.task).map(s=>'<option value="'+esc(s.path)+'">'+esc(s.path)+'</option>').join('')+'</select></label><label>运行目录<input id="import-path" placeholder="runs/…"></label><button id="import-confirm" class="primary">导入记录</button>');$('#sample-run').onchange=()=>$('#import-path').value=$('#sample-run').value;$('#import-confirm').onclick=guard(async()=>{await api('/api/import',{task:state.task,run:$('#import-path').value});$('#modal').close();await refreshRuns();});};
$('#new-task').onclick=()=>{modal('创建新任务草稿','<p>先记录操作目标并选择参考模板。新动作逻辑需要进一步开发，草稿不会直接执行。</p><label>任务名称<input id="draft-title" maxlength="100" placeholder="例如：把三块积木扫进簸箕"></label><label>希望机器人做什么<textarea id="draft-intent" maxlength="4000"></textarea></label><label>参考模板<select id="draft-template">'+state.tasks.map(t=>`<option value="${t.id}">${esc(t.name)}</option>`).join('')+'</select></label><button id="draft-confirm" class="primary">保存草稿</button>');$('#draft-confirm').onclick=guard(async()=>{await api('/api/draft',{title:$('#draft-title').value,intent:$('#draft-intent').value,template:$('#draft-template').value});$('#modal').close();await loadBootstrap();toast('草稿已保存，可让助手分析开发步骤');});};
$('#guide-link').onclick=e=>{e.preventDefault();modal('从一次运行开始','<p>① 在“场景与物体”生成摆位预览，检查尺度和分布。</p><p>② 在“运行与结果”点击“运行并导出视频”，等待仿真和视频作业完成。</p><p>③ 选择运行记录，查看实际动作、失败阶段和原始报告。</p><p>④ 打开开发助手，询问原因；修改参数后使用相同 seed 对照重跑。</p><p>Concept 描述功能部件，Knowledge 生成操作目标，Skill 组织动作阶段，Task 组合技能。</p><p>编辑器在新窗口打开，保存后通过“应用刚保存的标注”创建新场景版本。</p>');};
async function loadBootstrap(){const result=await api('/api/bootstrap');state.token=result.token;state.assistantSessions=result.assistant_sessions===true;if(!state.codexHome){state.codexHome=localStorage.getItem('gene-assistant-home')||result.codex_home;$('#assistant-home').value=state.codexHome||'';}state.tasks=result.tasks;state.samples=result.samples;state.visualPreviews=result.visual_previews;state.scenes=result.scenes||[];renderSceneVersions();$('#task').innerHTML=state.tasks.map(t=>`<option value="${t.id}">${esc(t.name)}</option>`).join('');$('#task').value=state.task;$('#drafts').innerHTML=result.drafts.map(d=>`<button data-draft="${d.id}">◇ ${esc(d.title)} · 待开发</button>`).join('');document.querySelectorAll('[data-draft]').forEach(b=>b.onclick=guard(async()=>{const d=result.drafts.find(x=>x.id===b.dataset.draft);$('#task').value=d.template;await changeTask();await ask(`我想开发新任务“${d.title}”：${d.intent}。请基于参考任务说明需要开发和验收哪些部分，先只分析。`);}));if(!result.assistant_available)$('#chat-context').textContent='本机未找到 Codex CLI，请检查启动终端 PATH。';}
initAssistantContext();initPlanningPages();initRunScene();
guard(async()=>{await loadBootstrap();await changeTask();await refreshJobs();setInterval(async()=>{if(state.busy)return;state.busy=true;try{await refreshJobs();await refreshRuns();}catch(e){console.warn(e);}finally{state.busy=false;}},2500);})();

$('#scene-select').onchange=guard(async()=>{state.scenePath=$('#scene-select').value||null;clearSceneEditor();localStorage.setItem('gene-scene-'+state.task,state.scenePath||'');$('#distribution-images').textContent='';await loadScene();toast('已切换场景输入，后续运行使用此版本');});
const compareButton=document.createElement('button');compareButton.textContent='对比两次运行';compareButton.id='compare';$('.detail-actions').append(compareButton);
compareButton.onclick=()=>{if(!state.run)return toast('先选择一条运行记录');const candidates=state.runs.filter(r=>r.task_id===state.task&&r.path!==state.run.path);if(!candidates.length)return toast('需要同一任务的另一条记录');modal('对比运行记录','<p>先对照相同 seed 和场景版本，再判断参数改动是否有效。</p><label>对照记录<select id="compare-run">'+candidates.map((r,i)=>`<option value="${i}">${esc(r.name)}</option>`).join('')+'</select></label><div id="comparison"></div>');const show=()=>{const other=candidates[Number($('#compare-run').value)];const rows=[['结果',state.run.success,other.success],['停止阶段',state.run.failure_stage,other.failure_stage],['证据类型',state.run.evidence_mode,other.evidence_mode],['seed',state.run.parameters?.seed,other.parameters?.seed],['抓取倾角（°）',state.run.parameters?.grasp_tilt_deg,other.parameters?.grasp_tilt_deg],['场景版本 hash',state.run.parameters?.scene_sha256,other.parameters?.scene_sha256]];$('#comparison').innerHTML='<table><thead><tr><th>项目</th><th>当前记录</th><th>对照记录</th></tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>`<td>${esc(x===null||x===undefined?'未记录':x)}</td>`).join('')+'</tr>').join('')+'</tbody></table>';};$('#compare-run').onchange=show;show();};

function clearSceneEditor(){
 if(state.editorJob)api('/api/cancel',{id:state.editorJob}).catch(console.warn);
 state.editorJob=null;state.editorSavedPath=null;state.editorDraft=false;
 $('#scene-editor-frame').hidden=true;$('#scene-editor-frame').removeAttribute('src');
 $('#scene-editor-link').hidden=true;$('#scene-editor-placeholder').hidden=false;
 $('#scene-editor-placeholder').textContent='点击“打开场景编辑器”，在桌面和机械臂旁调整物体。';
 $('#scene-editor-status').textContent='';
}
async function openSceneEditor(){
 if(state.rulesDirty)await ensureRulesSaved();
 const old=state.editorJob;
 if(old)await api('/api/cancel',{id:old});
 clearSceneEditor();
 const requestedTask=state.task, requestedScene=state.scenePath;
 const row=await launch('scene_viewer',{randomize:false});
 if(state.task!==requestedTask||state.scenePath!==requestedScene){await api('/api/cancel',{id:row.id});return;}
 state.editorJob=row.id;
 $('#scene-editor-placeholder').textContent=task().editable_scene?'正在加载完整场景和编辑控件…':'正在加载 Benchmark 三维场景（只读）…';
 await loadJob(row.id);
}
async function updateSceneEditor(row){
 if(row.id!==state.editorJob)return;
 state.editorDraft=Boolean(row.scene_draft?.unsaved);renderRunScene();
 if(row.viewer_ready){
   const url=row.inputs.viewer_url;
   if($('#scene-editor-frame').getAttribute('src')!==url)$('#scene-editor-frame').src=url;
   $('#scene-editor-frame').hidden=false;$('#scene-editor-placeholder').hidden=true;
   $('#scene-editor-link').href=url;$('#scene-editor-link').hidden=false;
 }
 if(['failed','canceled','interrupted','completed'].includes(row.state)){
   $('#scene-editor-status').textContent='场景编辑器已停止，可点击上方按钮重新打开。'+(row.error||'');
 }
 if(row.saved_scene&&row.saved_scene.path!==state.editorSavedPath){
   state.editorSavedPath=row.saved_scene.path;state.scenePath=row.saved_scene.path;
   localStorage.setItem('gene-scene-'+state.task,state.scenePath);
   setPlacementMode('fixed');$('#distribution-images').textContent='';
   await loadBootstrap();await loadScene();
   $('#scene-editor-status').textContent='已保存并选用新场景；随机摆位已关闭。后续运行使用此布局。';
   toast('已保存场景，后续运行使用你调整的摆位');
 }
}

function setPlacementMode(mode){
 $('#placement-mode').value=mode;
 $('#randomize').checked=mode==='random';
 localStorage.setItem('gene-mode-'+state.task,mode);
}
$('#placement-mode').onchange=()=>setPlacementMode($('#placement-mode').value);
$('#seed').onchange=()=>localStorage.setItem('gene-seed-'+state.task,$('#seed').value);
function renderSceneVersions(){
 $('#scene-select').innerHTML='<option value="">原始场景</option>'+(state.scenes||[]).filter(s=>s.task===state.task).map(s=>`<option value="${esc(s.path)}">${esc(s.label)}</option>`).join('');
 $('#scene-select').value=state.scenePath||'';renderRunScene();
}
function renderRules(scene){
 state.ruleDraft=structuredClone(scene.placement_rules);state.rulesDirty=false;
 drawRules();
}
function drawRules(){
 const rules=state.ruleDraft;
 $('#placement-rules').innerHTML=Object.entries(rules.objects).map(([name,row])=>{
 const label=({brush:'刷子',cube:'绿方块',cube_red:'红方块',cube_blue:'蓝方块',dustpan:'簸箕',microwave:'微波炉',bowl:'碗'})[name]||name;
 return `<div class="rule-row" data-rule="${esc(name)}"><b>${esc(label)}</b><label>摆位<select data-rule-mode><option value="fixed" ${row.mode==='fixed'?'selected':''}>固定</option><option value="random" ${row.mode==='random'?'selected':''}>随机</option></select></label><label>参考系<select data-rule-frame>${['world',...Object.keys(rules.objects).filter(n=>n!==name)].map(f=>`<option value="${esc(f)}" ${row.frame===f?'selected':''}>${f==='world'?'世界':esc(f)}</option>`).join('')}</select></label><label>X ±m<input type="number" min="0" max="1" step=".005" data-range="0" value="${row.translation_half_range_m[0]}"></label><label>Y ±m<input type="number" min="0" max="1" step=".005" data-range="1" value="${row.translation_half_range_m[1]}"></label><label>朝向 ±°<input type="number" min="0" max="180" step="1" data-yaw value="${row.yaw_half_range_deg}"></label></div>`;
 }).join('');
 $('#placement-rules').oninput=()=>{state.rulesDirty=true;state.presetDraft=false;$('#rules-note').textContent='规则尚未保存；运行或预览前会保存为新版本。';};
 $('#rules-note').textContent=(rules.source==='codex'?'来源：Codex 建议。':'来源：用户修改。')+(rules.note||'')+' 固定物体若选择另一物体为参考系，会随参考物体整体移动。';
}
function collectRules(){
 const rules=structuredClone(state.ruleDraft);
 for(const el of document.querySelectorAll('[data-rule]')){
   const row=rules.objects[el.dataset.rule];
   row.mode=el.querySelector('[data-rule-mode]').value;row.frame=el.querySelector('[data-rule-frame]').value;
   el.querySelectorAll('[data-range]').forEach(i=>row.translation_half_range_m[Number(i.dataset.range)]=Number(i.value));
   row.yaw_half_range_deg=Number(el.querySelector('[data-yaw]').value);
 }
 if(state.rulesDirty&&!state.presetDraft)rules.source='user';
 return rules;
}
async function selectSceneVersion(result,mode=null){
 if(result.task&&result.task!==state.task)return;
 clearSceneEditor();state.scenePath=result.path;state.rulesDirty=false;state.presetDraft=false;
 localStorage.setItem('gene-scene-'+state.task,result.path);
 if(mode)setPlacementMode(mode);
 $('#distribution-images').textContent='';
 await loadBootstrap();await loadScene();
}
async function ensureRulesSaved(){
 if(!task().editable_scene)return;
 const requestedTask=state.task,epoch=state.epoch;
 if(!state.scene||state.scene.task!==state.task||state.scene.path!==(state.scenePath||task().scene||task().config))await loadScene();
 if(requestedTask!==state.task||epoch!==state.epoch)throw Error('任务已切换，未应用旧任务的规则');
 if(!state.rulesDirty&&(!$('#randomize').checked||state.scene.rules_saved))return;
 const result=await api('/api/scene',{task:state.task,scene:state.scene.path,placement_rules:collectRules()});
 if(requestedTask!==state.task||epoch!==state.epoch)throw Error('任务已切换，旧任务规则已另存，未切换当前输入');
 await selectSceneVersion(result);
}
$('#save-rules').onclick=guard(async()=>{await ensureRulesSaved();toast('摆位规则已保存，固定/随机模式由上方选择');});
$('#suggest-rules').onclick=()=>{
 state.ruleDraft=structuredClone(state.scene.suggested_rules);drawRules();
 state.rulesDirty=true;state.presetDraft=true;
 toast('已载入 Codex 编写的建议范围；保存后可预览 6 种摆位');
};

function clearSamplePreview(){
 if(state.previewJob)api('/api/cancel',{id:state.previewJob}).catch(console.warn);
 state.previewJob=null;
 if($('#sample-preview-frame'))$('#sample-preview-frame').removeAttribute('src');
}
$('#modal').addEventListener('close',clearSamplePreview);
async function openSamplePreview(distributionJob,seed){
 clearSamplePreview();
 const epoch=state.epoch;
 const row=await launch('sample_viewer',{distribution_job:distributionJob,seed});
 if(epoch!==state.epoch){await api('/api/cancel',{id:row.id});return;}
 state.previewJob=row.id;
 modal('随机样本 · 三维查看','<p>左键拖动旋转、右键平移、滚轮缩放。可在三维窗口切换 seed，切换时保留视角。</p><p id="sample-preview-status">正在加载已保存的样本快照…</p><iframe id="sample-preview-frame" title="随机样本三维查看" hidden></iframe>');
 $('#modal').classList.add('wide');
 await loadJob(row.id);
}
function updateSamplePreview(row){
 if(row.id!==state.previewJob||!$('#sample-preview-frame'))return;
 if(row.viewer_ready){
   const url=row.inputs.viewer_url;
   if($('#sample-preview-frame').getAttribute('src')!==url)$('#sample-preview-frame').src=url;
   $('#sample-preview-frame').hidden=false;
   $('#sample-preview-status').textContent='当前显示采样时保存的场景快照；旋转视角不会改变物体摆位。';
 }else if(['failed','canceled','interrupted','completed'].includes(row.state)){
   $('#sample-preview-status').textContent='三维查看未运行：'+(row.error||states[row.state]);
 }
}
