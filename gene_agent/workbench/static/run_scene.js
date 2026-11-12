'use strict';
function runSceneOptions(){return {task:state.task,scene:state.scenePath||task()?.scene||task()?.config,execution_mode:$('#execution-mode').value,randomize:$('#randomize').checked,seed:Number($('#seed').value)};}
function runSceneKey(value=runSceneOptions()){return JSON.stringify(value);}
function clearRunScenePreview(){
 const old=state.runScenePreview;state.runScenePreview=null;
 if(old?.id)api('/api/cancel',{id:old.id}).catch(console.warn);
 $('#run-scene-frame').hidden=true;$('#run-scene-frame').removeAttribute('src');$('#run-scene-link').hidden=true;
}
function renderRunScene(){
 if(!task())return;
 const current=runSceneOptions(),versions=(state.scenes||[]).filter(s=>s.task===state.task);
 $('#run-scene-select').innerHTML='<option value="">Benchmark 默认场景</option>'+versions.map(s=>`<option value="${esc(s.path)}">${esc(s.label)}</option>`).join('');
 if(state.scenePath&&!versions.some(s=>s.path===state.scenePath))$('#run-scene-select').insertAdjacentHTML('beforeend',`<option value="${esc(state.scenePath)}">当前指定场景</option>`);
 $('#run-scene-select').value=state.scenePath||'';
 $('#run-scene-path').textContent=current.scene;
 $('#run-scene-setting').textContent=(state.scenePath?'当前网页选用已保存版本':'当前网页选用 Benchmark 默认')+' · '+(current.randomize?'随机摆位 · seed '+current.seed:'固定摆位')+' · '+(current.execution_mode==='visual'?'纯视觉初态':'MuJoCo 初态');
 if(state.runScenePreview&&state.runScenePreview.key!==runSceneKey()){
  clearRunScenePreview();$('#run-scene-status').textContent='设置已改变，旧预览已关闭；请重新预览本次场景。';
 }
 $('#run-scene-draft').hidden=!state.editorDraft;
}
async function previewRunScene(){
 if(state.runSceneRequest)return;state.runSceneRequest=true;$('#preview-run-scene').disabled=true;
 try{
 const epoch=state.epoch;
 await ensureRulesSaved();if(epoch!==state.epoch)return;
 const requested=runSceneOptions(),key=runSceneKey(requested);
 if(state.runScenePreview?.key===key)return toast('当前场景预览已打开');
 clearRunScenePreview();$('#run-scene-status').textContent='正在按本次设置初始化场景；不执行机器人动作…';
 const job=await launch('run_preview',requested);
 if(epoch!==state.epoch||key!==runSceneKey()){await api('/api/cancel',{id:job.id});return;}
 state.runScenePreview={id:job.id,key};await refreshRunScene();
 }finally{state.runSceneRequest=false;$('#preview-run-scene').disabled=false;}
}
async function refreshRunScene(){
 renderRunScene();const pending=state.runScenePreview;if(!pending)return;
 const row=await api('/api/job?id='+pending.id);if(state.runScenePreview!==pending)return;
 if(row.viewer_ready){
  const frame=$('#run-scene-frame');if(frame.getAttribute('src')!==row.inputs.viewer_url)frame.src=row.inputs.viewer_url;frame.hidden=false;
  const link=$('#run-scene-link');link.href=row.inputs.viewer_url;link.hidden=false;
  $('#run-scene-status').textContent='本次初态已就绪，可拖动视角。运行将使用上方同一版本、模式和 seed；此处尚未规划运动。';
 }else if(!['running','queued'].includes(row.state)){clearRunScenePreview();$('#run-scene-status').textContent='预览未就绪：'+(row.error||row.log?.slice(-1200)||row.state);}
}
function initRunScene(){
 $('#preview-run-scene').onclick=guard(previewRunScene);
 $('#edit-run-scene').onclick=()=>showPage('scene');
 $('#run-scene-select').onchange=guard(async()=>{await selectSceneVersion({task:state.task,path:$('#run-scene-select').value});renderRunScene();});
 for(const id of ['execution-mode','placement-mode','seed'])$('#'+id).addEventListener('change',renderRunScene);
 renderRunScene();
}
