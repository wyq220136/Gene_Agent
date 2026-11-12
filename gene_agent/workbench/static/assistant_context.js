'use strict';
function assistantContextOptions(){return {
 task:state.task,scene:state.scenePath||task()?.scene||task()?.config,
 run:$('#context-run').checked?state.run?.path:null,stage:state.stage,
 include_task:$('#context-task').checked,include_scene:$('#context-scene').checked,
 include_stage:$('#context-stage').checked,include_run:$('#context-run').checked,
 attachments:(state.attachments||[]).map(a=>a.id),excluded_images:state.excludedImages||[],
 allow_evidence:$('#allow-evidence').checked,
};}
function renderAssistantContext(){
 if(!$('#assistant-attachments'))return;
 $('#context-task-label').textContent='任务：'+(task()?.name||'未选择');
 $('#context-scene-label').textContent='场景：'+(state.scenePath||task()?.scene||task()?.config||'未选择');
 $('#context-run-label').textContent='运行：'+(state.run?.path||'未选择')+'（报告、日志）';
 $('#context-stage-label').textContent='阶段：'+(state.stage||'未选择');
 const files=state.attachments||[];
 $('#assistant-context-count').textContent=`本次上下文 · ${files.length} 个附件`;
 $('#assistant-attachments').innerHTML=files.map(a=>`<div class="attachment-chip">${a.kind==='image'?`<img src="${esc(file(a.path))}" alt="${esc(a.name)}">`:''}<span>${esc(a.name)}<small>${Math.ceil(a.size/1024)} KiB${a.note?' · '+esc(a.note):''}</small></span><button type="button" data-remove-attachment="${a.id}" aria-label="移除附件">×</button></div>`).join('');
 document.querySelectorAll('[data-remove-attachment]').forEach(b=>b.onclick=()=>{state.attachments=files.filter(a=>a.id!==b.dataset.removeAttachment);renderAssistantContext();});
 const images=state.run&&$('#context-run').checked&&$('#allow-evidence').checked?(state.run.media||[]).filter(m=>m.type==='image').slice(0,3).filter(m=>!(state.excludedImages||[]).includes(m.path)):[];
 $('#assistant-auto-images').innerHTML=images.map((m,i)=>`<div class="attachment-chip"><img src="${esc(file(m.path))}" alt="运行图片"><span>运行图片 ${i+1}<small>${esc(m.path)}</small></span><button type="button" data-remove-image="${i}" aria-label="排除此图片">×</button></div>`).join('');
 document.querySelectorAll('[data-remove-image]').forEach(b=>b.onclick=()=>{(state.excludedImages||=[]).push(images[Number(b.dataset.removeImage)].path);renderAssistantContext();});
}
async function uploadAssistantAttachment(payload){
 if((state.attachments||[]).length+(state.attachmentSlots||0)>=8)throw Error('最多添加 8 个附件，请先移除一些');
 state.attachmentSlots=(state.attachmentSlots||0)+1;state.attachmentPending=(state.attachmentPending||0)+1;
 try{const row=await api('/api/assistant/attachment',payload);(state.attachments||=[]).push(row);renderAssistantContext();return row;}
 finally{state.attachmentPending--;state.attachmentSlots--;}
}
async function attachFiles(files){
 state.attachmentPending=(state.attachmentPending||0)+1;
 try{for(const item of files){
  if(item.size>10*1024*1024)throw Error('单个附件不能超过 10 MiB：'+item.name);
  const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(Error('无法读取文件'));reader.readAsDataURL(item);});
  await uploadAssistantAttachment({name:item.name,data});
 }}finally{state.attachmentPending--;}
}
function manifestText(m){
 return [m.task?'任务：'+m.task:'',m.scene?'场景：'+m.scene:'',m.run?'运行：'+m.run:'',m.stage?'阶段：'+m.stage:'',
  ...(m.reports||[]).map(r=>'报告/日志：'+r.path+(r.truncated?'（已截断）':'')),
  ...(m.attachments||[]).map(a=>'附件：'+a.name+' · '+a.source+(a.truncated?'（文本仅内嵌前 8000 字符，完整快照可读）':'')),
  ...(m.images||[]).map(p=>'实际图片输入：'+p)].filter(Boolean).join('\n')||'仅问题和所选 Codex 对话的历史。';
}
function showSentContext(manifest){
 if(!manifest)return;
 const details=document.createElement('details');details.className='sent-context';
 const title=document.createElement('summary');title.textContent='本条消息实际提供的上下文';
 const text=document.createElement('pre');text.textContent=manifestText(manifest);details.append(title,text);
 if(state.replyElement)state.replyElement.before(details);else $('#messages').append(details);
}
async function previewAssistantContext(){
 const opts=assistantContextOptions();if(!opts.allow_evidence)opts.run=null;
 const m=await api('/api/assistant/context',opts);
 modal('本次上下文预览','<p>下面列出问题之外将提供的内容。所选 Codex 对话已有的历史由 Codex 自己加载。</p><pre id="context-manifest"></pre><div id="context-images"></div><div id="context-reports"></div>');
 $('#modal').classList.add('wide');$('#context-manifest').textContent=manifestText(m);
 $('#context-images').innerHTML=(m.images||[]).map(p=>`<img src="${esc(file(p))}" alt="将附加的图片">`).join('');
 for(const r of [...(m.reports||[]),...(m.attachments||[]).filter(a=>a.preview).map(a=>({path:a.name,text:a.preview}))]){const d=document.createElement('details'),t=document.createElement('summary'),p=document.createElement('pre');t.textContent=r.path+'（预览前 1000 字符）';p.textContent=r.text;d.append(t,p);$('#context-reports').append(d);}
}
async function captureAssistantScreen(){
 if(!navigator.mediaDevices?.getDisplayMedia)throw Error('浏览器不支持屏幕截图，可使用系统截图后 Ctrl+V 粘贴');
 state.attachmentPending=(state.attachmentPending||0)+1;let stream;
 try{
  stream=await navigator.mediaDevices.getDisplayMedia({video:true,audio:false,preferCurrentTab:true});
  const video=document.createElement('video');video.muted=true;video.srcObject=stream;await Promise.race([video.play(),new Promise((_,reject)=>setTimeout(()=>reject(Error('截图画面未就绪，请重试或粘贴系统截图')),5000))]);
  if(!video.videoWidth||!video.videoHeight)throw Error('未获得截图画面');
  const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  canvas.getContext('2d').drawImage(video,0,0);
  const blob=await new Promise(r=>canvas.toBlob(r,'image/png'));
  if(!blob)throw Error('截图失败，请使用系统截图后粘贴');
  await attachFiles([new File([blob],'screenshot-'+Date.now()+'.png',{type:'image/png'})]);
 }finally{stream?.getTracks().forEach(t=>t.stop());state.attachmentPending--;}
}
function initAssistantContext(){
 $('#attach-file').onclick=()=>$('#attachment-files').click();
 $('#attachment-files').onchange=guard(async()=>{await attachFiles($('#attachment-files').files);$('#attachment-files').value='';});
 $('#attach-path').onclick=guard(async()=>{const path=$('#attachment-path').value.trim();if(!path)throw Error('请粘贴文件或目录路径');await uploadAssistantAttachment({path});$('#attachment-path').value='';});
 $('#attach-screen').onclick=guard(captureAssistantScreen);$('#preview-context').onclick=guard(previewAssistantContext);
 for(const id of ['context-task','context-run','context-scene','context-stage','allow-evidence'])$('#'+id).addEventListener('change',renderAssistantContext);
 $('#question').addEventListener('paste',guard(async e=>{if(e.clipboardData.files.length){e.preventDefault();await attachFiles(e.clipboardData.files);}}));
 const panel=$('#assistant');panel.addEventListener('dragover',e=>{if(e.dataTransfer.types.includes('Files'))e.preventDefault();});
 panel.addEventListener('drop',guard(async e=>{if(e.dataTransfer.files.length){e.preventDefault();await attachFiles(e.dataTransfer.files);}}));
}
