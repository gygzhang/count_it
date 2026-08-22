'use strict';
const $ = id => document.getElementById(id);
const pad = i => String(i).padStart(6, '0');

const FIELDS = [
  {k:'method', t:'select', opts:['auto','thresh','otsu','color','bgsub','refbg']},
  {k:'scale', t:'num', step:0.05}, {k:'axis', t:'select', opts:['x','y']},
  {k:'flow', t:'select', opts:['both','pos','neg']},
  {k:'line', t:'num', step:0.01}, {k:'line_band', t:'num', step:0.01},
  {k:'min_hits', t:'num', step:1}, {k:'max_dist', t:'num', step:5},
  {k:'min_area', t:'num', step:10}, {k:'min_area_frac', t:'num', step:0.001},
  {k:'max_aspect', t:'num', step:0.5}, {k:'morph_kernel', t:'num', step:2},
  {k:'morph_iter', t:'num', step:1}, {k:'warmup', t:'num', step:1},
  {k:'thresh_lo', t:'num', step:5}, {k:'thresh_hi', t:'num', step:5},
  {k:'sat_thresh', t:'num', step:5}, {k:'bg_var', t:'num', step:5},
  {k:'roi', t:'text'}, {k:'auto_adapt', t:'check'}, {k:'watershed_split', t:'check'},
];

let defaults={}, videoList=[], anno=null, images=[], masks=[], cur=0, playing=false, lastT=0;
let eventFrames=[], runs=[], activeRunId=null, curMatch=null;
let mode='view', drag=null, trails=null, selTrack=null;
let compareId=null, compareAnno=null;

async function init(){
  const data = await (await fetch('/api/videos')).json();
  defaults = data.defaults; videoList = data.videos;
  fillVideos(videoList); renderParams(); bindEvents(); renderRuns();
}

// ---- inputs / params ----
function fillVideos(videos){
  const sel=$('videoSelect'); sel.innerHTML='';
  videos.forEach(v=>{ const o=document.createElement('option'); o.value=v.path;
    o.textContent=`${v.name}${v.gt!=null?`  (gt ${v.gt})`:''}  [${v.dir}]`; sel.appendChild(o); });
  updVideoHint();
}
function updVideoHint(){ $('videoHint').textContent=$('videoSelect').value||''; }

function renderParams(){
  const box=$('paramForm'); box.innerHTML='';
  FIELDS.forEach(f=>{
    const wrap=document.createElement('div'); wrap.className='pf'+(f.t==='check'?' check':'')+(f.k==='roi'?' full':'');
    if(f.t==='check'){
      const inp=document.createElement('input'); inp.type='checkbox'; inp.id='p_'+f.k; inp.checked=!!defaults[f.k];
      const lab=document.createElement('label'); lab.htmlFor=inp.id; lab.textContent=f.k; wrap.append(inp,lab);
    } else if(f.t==='select'){
      const lab=document.createElement('label'); lab.textContent=f.k;
      const sel=document.createElement('select'); sel.id='p_'+f.k;
      f.opts.forEach(op=>{const o=document.createElement('option');o.value=op;o.textContent=op;
        if(String(defaults[f.k])===op)o.selected=true; sel.appendChild(o);}); wrap.append(lab,sel);
    } else {
      const lab=document.createElement('label'); lab.textContent=f.k;
      const inp=document.createElement('input'); inp.type=f.t==='text'?'text':'number';
      inp.id='p_'+f.k; if(f.step)inp.step=f.step; inp.placeholder=(f.k==='roi'?'x0,y0,x1,y1':defaults[f.k]); wrap.append(lab,inp);
    }
    box.appendChild(wrap);
  });
  const mf=document.createElement('div'); mf.className='pf full';
  mf.innerHTML='<label>max_frames (0=全部, 上限1500)</label><input type="number" id="p_max_frames" step="50" placeholder="600">';
  box.appendChild(mf);
  const sf=document.createElement('div'); sf.className='pf full';
  sf.innerHTML='<label>起始帧 start(长视频分段, 0=从头)</label><input type="number" id="p_start" step="100" placeholder="0">';
  box.appendChild(sf);
}
function collectParams(){
  const p={};
  FIELDS.forEach(f=>{ const el=$('p_'+f.k); if(!el)return;
    if(f.t==='check') p[f.k]=el.checked; else if(el.value!=='') p[f.k]=el.value; });
  return p;
}
function setField(k,v){ const el=$('p_'+k); if(!el)return; if(el.type==='checkbox')el.checked=!!v; else el.value=v; }

function bindEvents(){
  $('videoSelect').onchange=updVideoHint;
  $('resetParams').onclick=()=>{renderParams();};
  $('runBtn').onclick=()=>run();
  $('prevSeg').onclick=()=>gotoSegment(-1);
  $('nextSeg').onclick=()=>gotoSegment(1);
  $('uploadInput').onchange=upload;
  $('preset').onchange=e=>{applyPreset(e.target.value); e.target.value='';};
  $('clearRuns').onclick=()=>{runs=[]; renderRuns();};
  $('clearRoi').onclick=()=>{setField('roi',''); if(anno)run({keepCur:true});};
  $('clearTrack').onclick=()=>{selTrack=null; $('trackSect').classList.add('hidden'); render();};
  $('playBtn').onclick=()=>{playing?stop():play();};
  $('prevBtn').onclick=()=>{stop();step(-1);};
  $('nextBtn').onclick=()=>{stop();step(1);};
  $('prevEvBtn').onclick=()=>{stop();gotoEvent(-1);};
  $('nextEvBtn').onclick=()=>{stop();gotoEvent(1);};
  $('scrub').oninput=e=>{stop();cur=+e.target.value;render();};
  ['lyDet','lyTrack','lyCounted','lyLine','lyTruth','lyMatch','lyMask','lyTrail','lyCompare']
    .forEach(id=>$(id).onchange=render);
  document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
  document.querySelectorAll('.tool').forEach(b=>b.onclick=()=>openTool(b.dataset.tool));
  $('modalClose').onclick=closeModal;
  $('modal').onclick=e=>{if(e.target.id==='modal')closeModal();};
  const cv=$('canvas');
  cv.onmousedown=onDown; cv.onmousemove=onMove; window.addEventListener('mouseup',onUp);
  document.addEventListener('keydown',onKey);
}

async function upload(e){
  const file=e.target.files[0]; if(!file)return;
  $('runStatus').textContent='上传中…';
  const fd=new FormData(); fd.append('file',file);
  const j=await (await fetch('/api/upload',{method:'POST',body:fd})).json();
  if(j.error){$('runStatus').textContent='上传失败: '+j.error;return;}
  const sel=$('videoSelect'); const o=document.createElement('option');
  o.value=j.path; o.textContent=`${j.name}${j.gt!=null?`  (gt ${j.gt})`:''}  [upload]`;
  sel.insertBefore(o,sel.firstChild); sel.value=j.path; updVideoHint();
  videoList.unshift({path:j.path,name:j.name,dir:'upload',gt:j.gt});
  $('runStatus').textContent='已上传，可运行。';
}

// ---- run ----
async function run(opts={}){
  const video=$('videoSelect').value;
  if(!video){$('runStatus').textContent='请先选择视频';return;}
  const body={video, params:collectParams()};
  const mf=$('p_max_frames').value; if(mf!=='')body.max_frames=+mf;
  const st=$('p_start').value; if(st!=='')body.start=+st;
  $('runBtn').disabled=true; $('runStatus').textContent='运行中…';
  try{
    const j=await (await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)})).json();
    if(j.error){$('runStatus').textContent='错误: '+j.error;return;}
    runs.unshift({id:j.run_id, label:paramLabel(body.params), video:video.split('/').pop(), s:j});
    if(runs.length>12)runs.pop();
    await loadRun(j.run_id, opts);
  }catch(err){$('runStatus').textContent='请求失败: '+err;}
  finally{$('runBtn').disabled=false;}
}

async function loadRun(runId, opts={}){
  $('runStatus').textContent='加载标注…';
  const prev=anno;
  anno=await (await fetch(`/api/run/${runId}/annotations.json`)).json();
  const N=anno.frames.length;
  const cv=$('canvas'); cv.width=anno.meta.width; cv.height=anno.meta.height;
  const sameFrames=prev && prev.frames_key===anno.frames_key && images.length===N;
  if(!sameFrames){
    images=new Array(N); let loaded=0;
    await Promise.all(anno.frames.map((_,i)=>new Promise(res=>{
      const im=new Image(); im.onload=im.onerror=()=>{loaded++;
        if(loaded%40===0||loaded===N)$('runStatus').textContent=`加载帧 ${loaded}/${N}`; res();};
      im.src=`/api/frames/${anno.frames_key}/frame_${pad(i)}.jpg`; images[i]=im;
    })));
  }
  masks=new Array(N);   // masks lazy-load per det_key
  activeRunId=runId; eventFrames=anno.frames.filter(f=>f.events.length).map(f=>f.i);
  buildTrails();
  cur = opts.keepCur ? Math.min(cur,N-1) : 0;
  $('emptyState').style.display='none'; cv.style.display='block';
  ['playBtn','prevBtn','nextBtn','scrub','prevEvBtn','nextEvBtn','prevSeg','nextSeg'].forEach(id=>$(id).disabled=false);
  $('scrub').max=N-1; updateSegInfo();
  renderRuns(); renderSummary(); render();
  $('runStatus').textContent=`完成 · ${N} 帧`;
}
function updateSegInfo(){
  if(!anno) return;
  const m=anno.meta, n=m.frames, s=m.start||0, tot=m.total||0;
  $('frameTot').textContent = tot || n;
  $('segInfo').textContent = tot ? `${s}–${s+n}/${tot}` : `${s}–${s+n}`;
  $('prevSeg').disabled = s<=0;
  $('nextSeg').disabled = tot ? (s+n>=tot) : false;
}
function gotoSegment(dir){
  if(!anno) return;
  const n=anno.meta.frames, s=anno.meta.start||0, tot=anno.meta.total||0;
  let ns = s + dir*n;
  if(ns<0) ns=0;
  if(tot && ns>=tot) return;
  $('p_start').value=ns; if($('p_max_frames').value==='') $('p_max_frames').value=n;
  run({keepCur:true});
}

function ensureMask(i){
  if(masks[i])return masks[i];
  const im=new Image(); im.onload=()=>render(); im.src=`/api/masks/${anno.det_key}/frame_${pad(i)}.jpg`;
  masks[i]=im; return im;
}
function buildTrails(){
  trails={};
  anno.frames.forEach(f=>f.tracks.forEach(t=>{(trails[t.id]=trails[t.id]||[]).push([f.i,t.cx,t.cy]);}));
}

// ---- playback ----
function play(){playing=true;$('playBtn').textContent='⏸';lastT=0;requestAnimationFrame(tick);}
function stop(){playing=false;$('playBtn').textContent='▶';}
function step(d){const N=anno.frames.length;cur=(cur+d+N)%N;render();}
function tick(t){if(!playing)return;const fps=Math.max(1,+$('playFps').value||20);
  if(t-lastT>=1000/fps){lastT=t;step(1);}requestAnimationFrame(tick);}

// ---- interaction (F1 ROI/line, F4 track pick) ----
function setMode(m){mode=m;drag=null;
  document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));
  $('canvas').classList.toggle('draw', m!=='view'); render();}
function canvasXY(e){const cv=$('canvas'),r=cv.getBoundingClientRect();
  return [Math.max(0,Math.min(cv.width,(e.clientX-r.left)/r.width*cv.width)),
          Math.max(0,Math.min(cv.height,(e.clientY-r.top)/r.height*cv.height))];}
function onDown(e){ if(!anno)return; const [x,y]=canvasXY(e);
  if(mode==='roi'){drag={x0:x,y0:y,x1:x,y1:y};}
  else if(mode==='line'){drag={line:true}; applyLinePreview(x,y);}
  else { pickTrack(x,y); } }
function onMove(e){ if(!drag||!anno)return; const [x,y]=canvasXY(e);
  if(mode==='roi'){drag.x1=x;drag.y1=y;render();}
  else if(mode==='line'){applyLinePreview(x,y);} }
function onUp(){ if(!drag||!anno){drag=null;return;}
  if(mode==='roi'){ const x0=Math.round(Math.min(drag.x0,drag.x1)),y0=Math.round(Math.min(drag.y0,drag.y1)),
      x1=Math.round(Math.max(drag.x0,drag.x1)),y1=Math.round(Math.max(drag.y0,drag.y1));
    drag=null; if(x1-x0>4&&y1-y0>4){setField('roi',`${x0},${y0},${x1},${y1}`); run({keepCur:true});} else render(); }
  else if(mode==='line'){ drag=null; run({keepCur:true}); } }
function applyLinePreview(x,y){ const frac=(anno.line.axis==='x')?x/anno.meta.width:y/anno.meta.height;
  setField('line', frac.toFixed(3)); anno.line.pos=Math.round((anno.line.axis==='x')?x:y); render(); }
function pickTrack(x,y){ const fr=anno.frames[cur]; let best=1e9,id=null;
  fr.tracks.forEach(t=>{const d=Math.hypot(t.cx-x,t.cy-y); if(d<best&&d<28){best=d;id=t.id;}});
  selTrack=id; renderTrack(); render(); }

// ---- rendering ----
function iou(a,b){const x0=Math.max(a[0],b[0]),y0=Math.max(a[1],b[1]),x1=Math.min(a[0]+a[2],b[0]+b[2]),
  y1=Math.min(a[1]+a[3],b[1]+b[3]),iw=Math.max(0,x1-x0),ih=Math.max(0,y1-y0),I=iw*ih;
  if(I<=0)return 0;const U=a[2]*a[3]+b[2]*b[3]-I;return U>0?I/U:0;}
function matchFrame(dets,truth,thr=0.3){const det=dets.map(()=>'fp'),tr=truth.map(()=>'fn'),used=truth.map(()=>false);
  dets.forEach((d,i)=>{let best=thr,bj=-1;truth.forEach((g,j)=>{if(used[j])return;const v=iou(d,g);if(v>=best){best=v;bj=j;}});
    if(bj>=0){used[bj]=true;det[i]='tp';tr[bj]='tp';}});
  const tp=det.filter(x=>x==='tp').length;return {det,truth:tr,tp,fp:det.length-tp,fn:tr.filter(x=>x==='fn').length};}

function render(){
  if(!anno)return;
  const fr=anno.frames[cur], ctx=$('canvas').getContext('2d'), W=anno.meta.width, H=anno.meta.height, im=images[cur];
  if(im&&im.complete&&im.naturalWidth) ctx.drawImage(im,0,0,W,H); else {ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);}
  if($('lyMask').checked){ const mi=ensureMask(cur);
    if(mi.complete&&mi.naturalWidth){ctx.globalAlpha=0.45;ctx.drawImage(mi,0,0,W,H);ctx.globalAlpha=1;} }
  const L=anno.line;
  if($('lyLine').checked){ctx.strokeStyle='#ff4d4f';ctx.lineWidth=2;ctx.beginPath();
    if(L.axis==='x'){ctx.moveTo(L.pos,0);ctx.lineTo(L.pos,H);}else{ctx.moveTo(0,L.pos);ctx.lineTo(W,L.pos);}ctx.stroke();}
  const hasTruth=anno.truth&&anno.truth.per_frame;
  const tb=hasTruth?(anno.truth.per_frame[cur]||anno.truth.per_frame[String(cur)]||[]):[];
  curMatch=($('lyMatch').checked&&hasTruth)?matchFrame(fr.dets,tb):null;
  if($('lyTruth').checked&&hasTruth){ctx.lineWidth=1.5;
    tb.forEach((b,j)=>{const fn=curMatch&&curMatch.truth[j]==='fn';
      ctx.strokeStyle=fn?'#ff9f40':'#5b8cff';ctx.setLineDash(fn?[5,4]:[]);ctx.strokeRect(b[0],b[1],b[2],b[3]);});ctx.setLineDash([]);}
  if($('lyDet').checked){ctx.lineWidth=2;
    fr.dets.forEach((b,j)=>{ctx.strokeStyle=(curMatch&&curMatch.det[j]==='fp')?'#ff4d4f':'#31d158';ctx.strokeRect(b[0],b[1],b[2],b[3]);});}
  // F3 compare B overlay (orange)
  if($('lyCompare').checked&&compareAnno&&compareAnno.frames[cur]){
    ctx.strokeStyle='#ff9f40';ctx.lineWidth=2;ctx.setLineDash([6,3]);
    compareAnno.frames[cur].dets.forEach(b=>ctx.strokeRect(b[0],b[1],b[2],b[3]));ctx.setLineDash([]);}
  // F4 trails
  if($('lyTrail').checked&&trails){ctx.lineWidth=1;
    Object.values(trails).forEach(pts=>{ctx.strokeStyle='rgba(38,198,218,.35)';ctx.beginPath();
      pts.forEach((p,i)=>i?ctx.lineTo(p[1],p[2]):ctx.moveTo(p[1],p[2]));ctx.stroke();});}
  if(selTrack!=null&&trails[selTrack]){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.beginPath();
    trails[selTrack].forEach((p,i)=>i?ctx.lineTo(p[1],p[2]):ctx.moveTo(p[1],p[2]));ctx.stroke();}
  // tracks
  ctx.font='bold 14px monospace';
  fr.tracks.forEach(t=>{
    if($('lyCounted').checked&&t.counted){ctx.fillStyle='#ffd23f';ctx.beginPath();ctx.arc(t.cx,t.cy,5,0,7);ctx.fill();}
    if(t.id===selTrack){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(t.cx,t.cy,10,0,7);ctx.stroke();}
    if($('lyTrack').checked){ctx.fillStyle=t.id===selTrack?'#fff':'#26c6da';ctx.fillText(t.id+(t.mult>1?`×${t.mult}`:''),t.cx+7,t.cy-7);}
  });
  // ROI rubber-band
  if(drag&&mode==='roi'){ctx.strokeStyle='#fff';ctx.setLineDash([6,4]);ctx.lineWidth=1.5;
    ctx.strokeRect(Math.min(drag.x0,drag.x1),Math.min(drag.y0,drag.y1),Math.abs(drag.x1-drag.x0),Math.abs(drag.y1-drag.y0));ctx.setLineDash([]);}
  ctx.fillStyle='#ffd23f';ctx.font='bold 20px monospace';ctx.fillText('count '+fr.count,12,26);
  if($('lyCompare').checked&&compareAnno) {ctx.fillStyle='#ff9f40';ctx.font='bold 14px monospace';
    ctx.fillText('B '+compareAnno.count+`  (Δ${fr.count-lastCount(compareAnno)})`,12,46);}
  $('frameIdx').textContent=(anno.meta.start||0)+cur; $('scrub').value=cur; renderFrameBox(fr); markTimeline();
}
function lastCount(a){return a.frames[Math.min(cur,a.frames.length-1)].count;}

function kv(box,rows){$(box).innerHTML=rows.map(([k,v,cls])=>`<b>${k}</b><span class="${cls||''}">${v}</span>`).join('');}
function renderSummary(){
  const r=anno.resolved,m=anno.meta,mt=anno.metrics,ti=anno.timing;
  const band=r.thresh_lo!=null?` [${r.thresh_lo},${r.thresh_hi}]`:'';
  $('badgeMethod').textContent='method '+r.method+band; $('badgeCount').textContent='count '+anno.count;
  $('badgeThru').textContent=(ti.throughput_fps||'—')+' fps';
  $('badgeGt').textContent=mt&&mt.gt_total!=null?`gt ${mt.gt_total}`:'gt —';
  const rows=[['method',r.method+band],['source',m.source.split('/').pop()],['dims',`${m.width}×${m.height}`],
    ['frames',m.frames],['src_fps',m.src_fps],['scale',m.scale],['auto_adapt',r.auto_adapt]];
  const d=r.diag||{};
  if(d.motion_px_per_frame!=null)rows.push(['motion px/f',d.motion_px_per_frame]);
  if(d.min_area!=null)rows.push(['auto min_area',d.min_area]);
  if(d.max_dist!=null)rows.push(['auto max_dist',d.max_dist]);
  kv('resolvedBox',rows);
  if(mt){const err=mt.count_error;const mr=[['count',mt.count],['gt',mt.gt_total==null?'—':mt.gt_total],
      ['误差',err==null?'—':(err>0?'+':'')+err,err===0?'good':(err==null?'':'bad')]];
    if(mt.precision!=null)mr.push(['precision',mt.precision],['recall',mt.recall],['tp/fp/fn',`${mt.tp}/${mt.fp}/${mt.fn}`]);
    kv('metricsBox',mr);} else kv('metricsBox',[['真值','无(未找到 *_meta.json / labels)']]);
  kv('timingBox',[['avg',ti.avg_ms+' ms'],['median',ti.median_ms+' ms'],['p95',ti.p95_ms+' ms'],
    ['max',ti.max_ms+' ms'],['吞吐',(ti.throughput_fps||'—')+' fps']]);
}
function renderFrameBox(fr){
  const rows=[['frame',`${cur}/${anno.frames.length-1}`],['检测',fr.dets.length],['轨迹',fr.tracks.length],
    ['累计',fr.count],['越线',fr.events.length?('#'+fr.events.join(' #')):'—',fr.events.length?'warn':'']];
  if(curMatch)rows.push(['本帧 TP/FP/FN',`${curMatch.tp}/${curMatch.fp}/${curMatch.fn}`,curMatch.fp||curMatch.fn?'warn':'good']);
  kv('frameBox',rows);
}
function renderTrack(){
  const sect=$('trackSect'); if(selTrack==null){sect.classList.add('hidden');return;} sect.classList.remove('hidden');
  const t=anno.frames[cur].tracks.find(x=>x.id===selTrack);
  const pts=trails[selTrack]||[]; const cf=anno.frames.find(f=>f.events.includes(selTrack));
  const rows=[['id',selTrack],['出现帧',pts.length?`${pts[0][0]}–${pts[pts.length-1][0]}`:'—'],
    ['越线帧',cf?cf.i:'—',cf?'warn':'']];
  if(t)rows.push(['位置',`${t.cx|0},${t.cy|0}`],['速度',`${t.vx},${t.vy}`],
    ['速率',Math.hypot(t.vx,t.vy).toFixed(1)+' px/f'],['hits/missing',`${t.hits}/${t.missing}`],
    ['已计数',t.counted]); else rows.push(['本帧','不在场']);
  kv('trackBox',rows);
}

// ---- timeline ----
function drawTimeline(){const cv=$('timeline');cv.width=cv.clientWidth||280;
  const ctx=cv.getContext('2d'),N=anno.frames.length,W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);
  const mc=Math.max(1,anno.count);ctx.strokeStyle='#31d158';ctx.lineWidth=1.5;ctx.beginPath();
  anno.frames.forEach((f,i)=>{const x=i/(N-1)*W,y=H-2-(f.count/mc)*(H-6);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  eventFrames.forEach(i=>{const x=i/(N-1)*W;ctx.strokeStyle='rgba(255,210,63,.5)';ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,6);ctx.stroke();});}
function markTimeline(){drawTimeline();const cv=$('timeline'),ctx=cv.getContext('2d'),N=anno.frames.length;
  const x=cur/(N-1)*cv.width;ctx.strokeStyle='#ffd23f';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,cv.height);ctx.stroke();}

// ---- events / history / presets ----
function gotoEvent(dir){if(!eventFrames.length)return;
  let t=dir>0?eventFrames.find(i=>i>cur):[...eventFrames].reverse().find(i=>i<cur);
  if(t==null)t=dir>0?eventFrames[0]:eventFrames[eventFrames.length-1];cur=t;render();}
function onKey(e){if(!anno||/INPUT|SELECT|TEXTAREA/.test(e.target.tagName))return;
  switch(e.key){case ' ':e.preventDefault();playing?stop():play();break;case 'ArrowRight':stop();step(1);break;
    case 'ArrowLeft':stop();step(-1);break;case '.':stop();gotoEvent(1);break;case ',':stop();gotoEvent(-1);break;
    case 'Home':stop();cur=0;render();break;case 'End':stop();cur=anno.frames.length-1;render();break;}}
function paramLabel(p){const parts=Object.keys(p).filter(k=>k!=='axis'&&k!=='flow'&&String(p[k])!==String(defaults[k]))
    .map(k=>`${k}=${p[k]}`);return parts.length?parts.join(' '):'默认参数';}
function renderRuns(){
  const box=$('runsBox');
  if(!runs.length){box.innerHTML='<div class="hint">运行后在此累积，点卡片切换/对比。</div>';return;}
  box.innerHTML='';
  runs.forEach(r=>{
    const s=r.s,m=s.metrics,err=m&&m.count_error;
    const errTxt=m&&m.gt_total!=null?` err ${err>0?'+':''}${err}`:'';
    const pr=m&&m.precision!=null?`<div class="stat"><span>P ${m.precision} · R ${m.recall}</span><span>tp/fp/fn ${m.tp}/${m.fp}/${m.fn}</span></div>`:'';
    const card=document.createElement('div'); card.className='runcard'+(r.id===activeRunId?' active':'');
    card.innerHTML=`<div class="top"><span>${r.video} · ${s.resolved.method}<span class="${err===0?'good':(err==null?'':'bad')}"> count ${s.count}${errTxt}</span></span>`+
      `<span><span class="cmp${r.id===compareId?' on':''}" data-b="${r.id}">B</span> ${s.timing.avg_ms}ms</span></div>`+
      `<div class="lab">${r.label}</div>${pr}`;
    card.onclick=e=>{ if(e.target.classList.contains('cmp')){toggleCompare(r.id);e.stopPropagation();return;} loadRun(r.id,{keepCur:true}); };
    box.appendChild(card);
  });
}
const PRESETS={default:{method:'auto',auto_adapt:false,watershed_split:false},
  autoadapt:{method:'auto',auto_adapt:true,watershed_split:false},thresh:{method:'thresh',auto_adapt:false},
  bgsub:{method:'bgsub',auto_adapt:false},watershed:{method:'auto',auto_adapt:true,watershed_split:true}};
function applyPreset(name){const p=PRESETS[name];if(!p)return;Object.entries(p).forEach(([k,v])=>setField(k,v));}

// ---- F3 compare ----
async function toggleCompare(id){
  if(compareId===id){compareId=null;compareAnno=null;$('lyCompare').checked=false;renderRuns();render();return;}
  const b=await (await fetch(`/api/run/${id}/annotations.json`)).json();
  if(!anno || b.meta.width!==anno.meta.width || b.meta.height!==anno.meta.height
     || b.frames.length!==anno.frames.length){
    $('runStatus').textContent='对比需与当前 run 同分辨率且同帧数(通常同一视频)';
    renderRuns(); return;
  }
  compareId=id; compareAnno=b; $('lyCompare').checked=true; renderRuns(); render();
}

// ---- F5-F8 tools modal ----
function openModal(title){$('modalTitle').textContent=title;$('modal').classList.remove('hidden');}
function closeModal(){$('modal').classList.add('hidden');}
function openTool(name){({autoparams:toolAuto,grid:toolGrid,batch:toolBatch,tests:toolTests})[name]();}

async function toolAuto(){
  const video=$('videoSelect').value; openModal('自动估参 · '+video.split('/').pop());
  $('modalBody').innerHTML='<div class="hint">正在从视频估计初始参数…</div>';
  const j=await (await fetch('/api/autoparams',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video})})).json();
  if(j.error){$('modalBody').innerHTML='<div class="bad">'+j.error+'</div>';return;}
  const s=j.suggested; const rows=Object.entries(s).map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  $('modalBody').innerHTML=`<table class="rep"><tr><th>参数</th><th>建议值</th></tr>${rows}</table>`+
    `<div class="actions"><button class="run" id="applyAuto">应用到参数表并运行</button></div>`+
    `<div class="hint">诊断: ${JSON.stringify(j.diagnostics)}</div>`;
  $('applyAuto').onclick=()=>{Object.entries(s).forEach(([k,v])=>setField(k,v));closeModal();run();};
}
function toolGrid(){
  openModal('网格搜索 · '+$('videoSelect').value.split('/').pop());
  const trk=['line','line_band','min_hits','max_dist','min_speed','track_ttl'];
  const det=['min_area','morph_kernel','morph_iter','thresh_lo','thresh_hi','bg_var','sat_thresh'];
  const opt=[...trk,...det].map(k=>`<option value="${k}">${k}</option>`).join('');
  $('modalBody').innerHTML=`<div class="hint">跟踪类参数(line/min_hits/max_dist…)共享同一次检测，扫描很快。</div>`+
    `<div class="gridspec">`+[1,2].map(i=>`<label>参数${i}</label><div style="display:flex;gap:6px">`+
      `<select id="gk${i}"><option value="">—</option>${opt}</select>`+
      `<input id="gv${i}" placeholder="逗号分隔值, 如 0.3,0.5,0.7" style="flex:1"></div>`).join('')+`</div>`+
    `<div class="actions"><button class="run" id="runGrid">运行网格</button></div><div id="gridOut"></div>`;
  $('runGrid').onclick=runGrid;
}
async function runGrid(){
  const grid={}; [1,2].forEach(i=>{const k=$('gk'+i).value,v=$('gv'+i).value.trim();
    if(k&&v)grid[k]=v.split(',').map(x=>x.trim()).filter(Boolean);});
  if(!Object.keys(grid).length){$('gridOut').innerHTML='<div class="bad">请至少填一个参数+值</div>';return;}
  $('gridOut').innerHTML='<div class="hint">搜索中…</div>';
  const body={video:$('videoSelect').value, base_params:collectParams(), grid};
  const mf=$('p_max_frames').value; if(mf!=='')body.max_frames=+mf;
  const st=$('p_start').value; if(st!=='')body.start=+st;
  const j=await (await fetch('/api/grid',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(j.error){$('gridOut').innerHTML='<div class="bad">'+j.error+'</div>';return;}
  const head=`<tr>${j.keys.map(k=>`<th>${k}</th>`).join('')}<th>count</th><th>gt</th><th>误差</th><th>P</th><th>R</th><th>ms</th></tr>`;
  const body2=j.rows.map((r,idx)=>`<tr class="click${idx===0?' best':''}" data-combo='${JSON.stringify(r.combo)}'>`+
    j.keys.map(k=>`<td>${r.combo[k]}</td>`).join('')+`<td>${r.count}</td><td>${r.gt??'—'}</td>`+
    `<td>${r.error==null?'—':r.error}</td><td>${r.precision??'—'}</td><td>${r.recall??'—'}</td><td>${r.avg_ms}</td></tr>`).join('');
  $('gridOut').innerHTML=`<table class="rep">${head}${body2}</table><div class="hint">点某行=套用该组合并运行</div>`;
  $('gridOut').querySelectorAll('tr.click').forEach(tr=>tr.onclick=()=>{
    const combo=JSON.parse(tr.dataset.combo);Object.entries(combo).forEach(([k,v])=>setField(k,v));closeModal();run();});
}
function toolBatch(){
  openModal('批量跑（当前参数扫多个视频）');
  const list=videoList.map(v=>`<label><input type="checkbox" value="${v.path}" ${v.gt!=null?'':'disabled title="无真值"'}> ${v.name}${v.gt!=null?` (gt ${v.gt})`:' (无真值)'}</label>`).join('');
  $('modalBody').innerHTML=`<div class="hint">勾选视频（建议 max_frames=0 跑全片以匹配真值）：</div>`+
    `<div class="chk-list" id="batchList">${list}</div>`+
    `<div class="actions"><button class="run" id="runBatch">运行批量</button></div><div id="batchOut"></div>`;
  $('runBatch').onclick=runBatch;
}
async function runBatch(){
  const videos=[...document.querySelectorAll('#batchList input:checked')].map(c=>c.value);
  if(!videos.length){$('batchOut').innerHTML='<div class="bad">请勾选至少一个视频</div>';return;}
  $('batchOut').innerHTML='<div class="hint">运行中…</div>';
  const body={videos, params:collectParams()}; const mf=$('p_max_frames').value; if(mf!=='')body.max_frames=+mf;
  const st=$('p_start').value; if(st!=='')body.start=+st;
  const j=await (await fetch('/api/batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(j.error){$('batchOut').innerHTML='<div class="bad">'+j.error+'</div>';return;}
  const rows=j.rows.map(r=>r.ok?`<tr><td>${r.video}</td><td>${r.method}</td><td>${r.count}</td><td>${r.gt??'—'}</td>`+
    `<td class="${r.error===0?'good':'bad'}">${r.error==null?'—':r.error}</td><td>${r.precision??'—'}</td><td>${r.recall??'—'}</td><td>${r.avg_ms}</td></tr>`
    :`<tr><td>${r.video}</td><td colspan="7" class="bad">${r.error}</td></tr>`).join('');
  $('batchOut').innerHTML=`<table class="rep"><tr><th>视频</th><th>方式</th><th>count</th><th>gt</th><th>误差</th><th>P</th><th>R</th><th>ms</th></tr>${rows}</table>`;
}
function toolTests(){
  openModal('测试面板');
  $('modalBody').innerHTML=`<div class="actions"><button class="run" id="runPytest">运行 pytest</button>`+
    `<button class="tool" id="runValidate">读验证报告</button></div><div id="testOut"></div>`;
  $('runPytest').onclick=async()=>{$('testOut').innerHTML='<div class="hint">运行 pytest…</div>';
    const j=await (await fetch('/api/pytest',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).json();
    $('testOut').innerHTML=`<div class="${j.ok?'good':'bad'}">pytest: ${j.passed} passed / ${j.failed} failed</div><pre class="out">${j.output||''}</pre>`;};
  $('runValidate').onclick=async()=>{$('testOut').innerHTML='<div class="hint">读取验证报告…</div>';
    const j=await (await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run:false})})).json();
    if(j.error){$('testOut').innerHTML='<div class="bad">'+j.error+'</div>';return;}
    const rows=(j.results||[]).map(r=>`<tr><td>${r.scenario||r.name||''}</td><td>${r.mode||''}</td><td>${r.detected??''}</td><td>${r.ground_truth??''}</td><td>${r.error??''}</td><td>${r.count_accuracy_pct??''}</td></tr>`).join('');
    $('testOut').innerHTML=`<table class="rep"><tr><th>场景</th><th>模式</th><th>检测</th><th>gt</th><th>误差</th><th>准确率</th></tr>${rows}</table>`;};
}

window.viewer={get anno(){return anno;}, seek(i){stop();cur=Math.max(0,Math.min((anno?anno.frames.length-1:0),i));render();},
  firstDetFrame(){return anno?anno.frames.findIndex(f=>f.dets.length):-1;},
  state(){return {cur,n:anno?anno.frames.length:0,count:anno?anno.count:null,mode,selTrack};},
  setMode, run};
init();
