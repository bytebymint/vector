'use strict';
const $=id=>document.getElementById(id),csrf=document.querySelector('meta[name=csrf-token]').content;
const htmlCache=new Map();
function setHTML(id,html){if(htmlCache.get(id)===html)return false;$(id).innerHTML=html;htmlCache.set(id,html);return true;}
const esc=value=>String(value??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,n=0)=>v==null?'—':Number(v).toFixed(n),timeText=t=>t?new Date(t*1000).toLocaleTimeString([], {hour12:false}):'—';
let toastTimer;
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,7000);}
async function api(path,body){const response=await fetch(path,{method:body===undefined?'GET':'POST',headers:body===undefined?{}:{'Content-Type':'application/json','X-CSRF-Token':csrf},body:body===undefined?undefined:JSON.stringify(body)});const value=await response.json();if(!response.ok)throw Error(value.error||response.statusText);return value;}
const video=$('video'),scene=$('scene'),ctx=scene.getContext('2d'),overlay=new VehicleOverlay($('overlay'));
const faceTimeline=new FaceTimeline();
let state=null,metadata=[],pictures=[],origin=null,peer=null,peerId=null,jpeg=false,frozen=false,renderTime=0,renderMeta=null,inspected=null,configTab='monitor',formLoaded=false,viewMode='traffic';
let nativeAbort=null,nativeUrl=null,renderRequest=0,recordFrame=null;
function requestRender(){if(!renderRequest)renderRequest=requestAnimationFrame(render);}
if(!$('speed-legend')){const legend=document.createElement('span');legend.id='speed-legend';legend.className='speed-legend';legend.hidden=true;legend.textContent='km/h* · map-calibrated estimate';$('stage').appendChild(legend);}
const editor=new ZoneEditor(overlay,updateEditor,value=>freeze(value));
function freeze(value=!frozen){if(value&&!frozen){if(jpeg){const picture=pictures.at(-1);if(!picture){toast('Wait for the first camera frame before setting up lanes.');return false;}resize(picture.bitmap.width,picture.bitmap.height);ctx.drawImage(picture.bitmap,0,0,scene.width,scene.height);}else{if(video.readyState<2||!video.videoWidth){toast('Wait for the first camera frame before setting up lanes.');return false;}resize(video.videoWidth,video.videoHeight);ctx.drawImage(video,0,0,scene.width,scene.height);}}frozen=value;$('stage').classList.toggle('frame-held',frozen);$('freeze').textContent=frozen?'▶ Resume':'Ⅱ Setup';$('freeze').classList.toggle('primary',frozen);if(!frozen){editor.active=false;document.body.classList.remove('editing');}requestRender();return true;}
function resize(w,h){if(scene.width!==w||scene.height!==h){scene.width=w;scene.height=h;overlay.size(w,h);$('stage').style.aspectRatio=`${w}/${h}`;}}
function addMetadata(frame){if(!frame.timestamp)return;metadata.push(frame);if(metadata.length>70)metadata.shift();requestRender();}
async function connectVideo(forceRelay=false){nativeAbort?.abort();nativeAbort=null;if(nativeUrl){URL.revokeObjectURL(nativeUrl);nativeUrl=null;}if(peer){peer.close();if(peerId)api('/api/webrtc/close/'+peerId,{}).catch(()=>{});}peer=null;origin=null;peerId=null;jpeg=false;pictures.forEach(p=>p.bitmap.close());pictures=[];
  $('stage').classList.remove('jpeg-mode');
  if(!forceRelay&&window.MediaSource){try{const source=await api('/api/video/info');if(source.available){const controller=nativeAbort=new AbortController();nativeVideo(controller).catch(e=>{if(!controller.signal.aborted){console.warn('Native playback reconnecting:',e.message);setTimeout(()=>connectVideo(false),800);}});return;}}catch(e){}}
  video.removeAttribute('src');$('stage').dataset.transport='webrtc';
  const pc=peer=new RTCPeerConnection({iceServers:[]});pc.addTransceiver('video',{direction:'recvonly'});
  pc.ontrack=e=>{video.srcObject=new MediaStream([e.track]);video.play().catch(()=>{});};
  pc.onconnectionstatechange=()=>{if(pc!==peer)return;if(pc.connectionState==='failed')toast('Local WebRTC failed. Use Display → JPEG fallback or reconnect.');};
  try{await pc.setLocalDescription(await pc.createOffer());if(pc.iceGatheringState!=='complete')await new Promise(resolve=>{const done=()=>{if(pc.iceGatheringState==='complete'){pc.removeEventListener('icegatheringstatechange',done);resolve();}};pc.addEventListener('icegatheringstatechange',done);setTimeout(resolve,3500);});const answer=await api('/api/webrtc/offer',{sdp:pc.localDescription.sdp,type:'offer'});peerId=answer.peer_id;await pc.setRemoteDescription(answer);pollClock();}catch(e){toast(e.message);}
}
async function nativeVideo(controller){
  const media=new MediaSource();video.srcObject=null;nativeUrl=URL.createObjectURL(media);video.src=nativeUrl;
  await new Promise((resolve,reject)=>{media.addEventListener('sourceopen',resolve,{once:true});controller.signal.addEventListener('abort',()=>reject(Error('Playback stopped')),{once:true});});
  const response=await fetch('/api/video/live',{signal:controller.signal});if(!response.ok)throw Error('Native source unavailable');
  const reader=response.body.getReader();let pending=new Uint8Array(0),sourceBuffer=null,started=false;
  const updated=()=>new Promise((resolve,reject)=>{const clean=()=>{sourceBuffer.removeEventListener('updateend',done);sourceBuffer.removeEventListener('error',failed);};const done=()=>{clean();resolve();};const failed=()=>{clean();reject(Error('Video fragment decode failed'));};sourceBuffer.addEventListener('updateend',done);sourceBuffer.addEventListener('error',failed);});
  try{while(!controller.signal.aborted){
    const {done,value}=await reader.read();if(done)break;
    if(!pending.length)pending=value;else{const merged=new Uint8Array(pending.length+value.length);merged.set(pending);merged.set(value,pending.length);pending=merged;}
    let offset=0;
    while(pending.length-offset>=4){const length=new DataView(pending.buffer,pending.byteOffset+offset,4).getUint32(0);if(length>32*1024*1024)throw Error('Video fragment exceeds buffer limit');if(pending.length-offset<length+4)break;
      const data=pending.subarray(offset+4,offset+4+length);offset+=length+4;
      if(!sourceBuffer){const info=JSON.parse(new TextDecoder().decode(data));if(!MediaSource.isTypeSupported(info.mime))throw Error('Browser does not support this source codec');origin=info.origin;sourceBuffer=media.addSourceBuffer(info.mime);sourceBuffer.mode='segments';$('stage').dataset.transport='native-mp4';continue;}
      if(sourceBuffer.updating)await updated();
      const appended=updated();sourceBuffer.appendBuffer(data);await appended;
      if(video.buffered.length){const start=video.buffered.start(0),end=video.buffered.end(video.buffered.length-1);
        if(!started&&end-start>=.8){video.currentTime=Math.max(start,end-.85);started=true;video.play().catch(e=>{video.muted=true;return video.play();}).catch(()=>{});}
        else if(started){const lag=end-video.currentTime;if(lag>3.5){video.currentTime=Math.max(start,end-1.0);if(video.playbackRate!==1)video.playbackRate=1;}else if(lag>2.4){if(video.playbackRate!==1.04)video.playbackRate=1.04;}else if(lag<1.2&&video.playbackRate!==1){video.playbackRate=1;}}
        if(start<video.currentTime-12&&!sourceBuffer.updating){const removed=updated();sourceBuffer.remove(0,video.currentTime-10);await removed;}
      }
    }pending=pending.subarray(offset);
  }}finally{reader.cancel().catch(()=>{});}
  if(!controller.signal.aborted){setTimeout(()=>connectVideo(false),800);}
}
async function pollClock(){if(!peerId||jpeg)return;try{const value=await api('/api/webrtc/clock/'+peerId);origin=value.origin;}catch(e){}if(!origin)setTimeout(pollClock,500);}
let playbackFPS=null,playbackCount=0,playbackSample=performance.now(),lastPresented=0;
function videoFrame(now,info){
  playbackCount++;lastPresented=now;
  if(now-playbackSample>=2000){playbackFPS=playbackCount*1000/(now-playbackSample);playbackCount=0;playbackSample=now;$('stage').dataset.playbackFps=playbackFPS.toFixed(2);
    const quality=video.getVideoPlaybackQuality?.();
    $('stage').dataset.playbackStats=JSON.stringify({presentedFrames:info.presentedFrames,
      totalFrames:quality?.totalVideoFrames,droppedFrames:quality?.droppedVideoFrames,
      width:video.videoWidth,height:video.videoHeight,mediaTime:info.mediaTime});
  }
  if(!jpeg&&!frozen&&origin!==null&&video.readyState>=2){resize(video.videoWidth,video.videoHeight);renderTime=origin+info.mediaTime;$('empty').hidden=true;requestRender();}
  else if(recordFrame)requestRender();
  video.requestVideoFrameCallback(videoFrame);
}
if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(videoFrame);else{const fallback=()=>{if(!jpeg&&!frozen&&origin!==null&&video.readyState>=2){const stamp=origin+video.currentTime;if(stamp!==renderTime){resize(video.videoWidth,video.videoHeight);renderTime=stamp;$('empty').hidden=true;requestRender();}}requestAnimationFrame(fallback);};requestAnimationFrame(fallback);}
setInterval(async()=>{if(!peer)return;try{const report=await peer.getStats();for(const stat of report.values())if(stat.type==='inbound-rtp'&&stat.kind==='video')$('stage').dataset.rtpStats=JSON.stringify({framesReceived:stat.framesReceived,framesDecoded:stat.framesDecoded,framesDropped:stat.framesDropped,packetsReceived:stat.packetsReceived,packetsLost:stat.packetsLost,keyFramesDecoded:stat.keyFramesDecoded,jitter:stat.jitter,decoder:stat.decoderImplementation});}catch(e){}},2000);
async function jpegLoop(){if(!jpeg)return;try{const response=await fetch('/api/frame.jpg');if(response.status===200){const timestamp=Number(response.headers.get('X-Frame-Timestamp'));const bitmap=await createImageBitmap(await response.blob());pictures.forEach(p=>p.bitmap.close());pictures=[{timestamp,bitmap}];requestRender();}}catch(e){}setTimeout(jpegLoop,100);}
let lastPicture=null,lastFrameInfo='',lastSyncInfo='',overlayFrames=0,overlaySamples=0,overlayDrawTime=0,overlaySample=performance.now();
function render(){renderRequest=0;if(!frozen){
    if(jpeg&&pictures.length){const picture=pictures.at(-1);if(picture!==lastPicture){resize(picture.bitmap.width,picture.bitmap.height);ctx.drawImage(picture.bitmap,0,0,scene.width,scene.height);renderTime=picture.timestamp;lastPicture=picture;$('empty').hidden=true;}}
    let best=null,before=null,after=null;for(const f of metadata){if(f.timestamp<=renderTime&&(!before||f.timestamp>before.timestamp))before=f;if(f.timestamp>=renderTime&&(!after||f.timestamp<after.timestamp))after=f;if(Math.abs(f.timestamp-renderTime)<.4&&(!best||Math.abs(f.timestamp-renderTime)<Math.abs(best.timestamp-renderTime)))best=f;}renderMeta=best;if(viewMode==='people'&&before&&after&&before!==after&&after.timestamp-before.timestamp<=.5){const smoothed=VehicleOverlay.interpolatePeople(before,after,renderTime);if(smoothed)renderMeta=smoothed;}
    const faceSample=(typeof faceTimeline!=='undefined'&&faceTimeline?.sample)?faceTimeline.sample(renderTime,typeof PeopleOverlay!=='undefined'?PeopleOverlay.interpolatePeople:null):null;
    if(faceSample?.people?.length){
      if(!renderMeta)renderMeta={...faceSample,timestamp:renderTime,syncTimestamp:faceSample.syncTimestamp??faceSample.timestamp,people:[...faceSample.people]};
      else{
        renderMeta={...renderMeta,width:faceSample.width??renderMeta.width,height:faceSample.height??renderMeta.height,syncTimestamp:faceSample.syncTimestamp??faceSample.timestamp,tracking_kind:viewMode==='people'?(faceSample.tracking_kind??renderMeta.tracking_kind):renderMeta.tracking_kind,people:renderMeta.people?[...renderMeta.people]:[]};
        for(const fp of faceSample.people){
          const fId=fp.id??fp.track_id??fp.uid;
          const idx=renderMeta.people.findIndex(p=>(p.id??p.track_id??p.uid)===fId);
          if(idx>=0)renderMeta.people[idx]={...renderMeta.people[idx],...fp};
          else renderMeta.people.push(fp);
        }
      }
    }else if(viewMode==='people'&&window.peopleDashboard?.isRunning&&faceSample){
      renderMeta=faceSample;
    }
    if(renderMeta?.tracks?.length){
      const personTracks=renderMeta.tracks.filter(t=>t.category==='person');
      if(personTracks.length){
        if(!renderMeta.people)renderMeta={...renderMeta,people:[]};
        for(const pt of personTracks){
          const pId=pt.id??pt.uid;
          if(!renderMeta.people.some(p=>(p.id??p.track_id??p.uid)===pId)){
            renderMeta.people.push({id:pId,confirmed:pt.confirmed!==false,bbox:pt.bbox,confidence:pt.confidence,velocity:pt.velocity,history:pt.history,hits:pt.hits,duration:pt.duration});
          }
        }
      }
    }
    if(typeof $==='function'&&$('stage'))$('stage').dataset.faceOverlayTracks=String(renderMeta?.people?.length||0);
  }
  const analyticsAvailable=!state||(state.engine.state==='running'&&state.stream.frame_age_seconds<3);
  const drawStarted=performance.now();
  overlay.draw(analyticsAvailable||frozen?renderMeta:null,editor.geometry,editor,renderTime);
  const drawEnded=performance.now();overlayFrames++;overlaySamples++;overlayDrawTime+=drawEnded-drawStarted;
  if(drawEnded-overlaySample>=2000){Object.assign($('stage').dataset,{overlayFrames:String(overlayFrames),overlayDrawMs:(overlayDrawTime/overlaySamples).toFixed(3),overlayFps:(overlaySamples*1000/(drawEnded-overlaySample)).toFixed(2)});overlayDrawTime=0;overlaySamples=0;overlaySample=drawEnded;}
  const frameInfo=frozen?'SETUP FRAME FROZEN · ACQUISITION CONTINUES':`${scene.width} × ${scene.height} · ${Number.isFinite(playbackFPS)?playbackFPS.toFixed(1)+' fps view · ':''}${timeText(renderTime)}`;
  const syncInfo=renderMeta&&analyticsAvailable?`OVERLAY Δ ${Math.abs((renderTime-(renderMeta.syncTimestamp??renderMeta.timestamp))*1000).toFixed(0)} ms`:'NO MATCHED ANALYTICS';
  if(frameInfo!==lastFrameInfo){$('frame-info').textContent=frameInfo;lastFrameInfo=frameInfo;}
  if(syncInfo!==lastSyncInfo){$('sync-info').textContent=syncInfo;lastSyncInfo=syncInfo;}
  // Composite only after the visible overlay is ready for this video timestamp.
  // A separate timer could capture a cleared/previous overlay and did redundant work.
  recordFrame?.(performance.now());
}
requestRender();
function setView(mode){viewMode=mode;api('/api/analysis/focus',{mode}).catch(e=>toast(e.message));try{localStorage.setItem('vector-monitor-view',mode);}catch(e){}document.body.classList.toggle('people-mode',mode==='people');overlay.peopleMode=mode==='people';$('vehicle-metrics').hidden=mode==='people';$('people-metrics').hidden=mode!=='people';$('people-panel').hidden=mode!=='people';$('people-registry').hidden=mode!=='people';$('view-title').firstChild.textContent=mode==='people'?'People monitor':'Street intelligence';$('view-subtitle').textContent=mode==='people'?'Direct face tracking · native source captures.':'Every movement. A clearer picture.';document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',mode==='people'?b.dataset.view==='people':b.dataset.tab==='monitor'));if(mode==='people'&&window.peopleDashboard){if(!window.peopleDashboard.isRunning&&!window.peopleDashboard.userStopped)window.peopleDashboard.startAnalysis?.();else window.peopleDashboard.fetchState?.().catch?.(()=>{});}requestRender();}
function switchTab(tab){setView('traffic');configTab=tab;document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));$('configuration').hidden=tab==='monitor';['geometry','engine','display'].forEach(name=>$(name+'-config').hidden=name!==tab);$('config-title').textContent={geometry:'Road geometry & counting',engine:'AI engine configuration',display:'Display & physical calibration'}[tab]||'';if(tab==='geometry')editor.setActive(true);else{editor.setActive(false);freeze(false);}if(tab!=='monitor')$('configuration').scrollIntoView({behavior:'smooth',block:'nearest'});}
document.querySelectorAll('.nav[data-tab]').forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));document.querySelector('[data-view=people]').onclick=()=>{if(configTab!=='monitor')switchTab('monitor');setView('people');};$('close-config').onclick=()=>switchTab('monitor');$('freeze').onclick=()=>freeze();$('fullscreen').onclick=()=>{document.body.classList.add('stage-expanded');$('record-clip').style.display='block';};$('stage-exit').onclick=()=>{document.body.classList.remove('stage-expanded');window.peopleDashboard?.renderGallery();$('record-clip').style.display='none';};document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.body.classList.remove('stage-expanded');window.peopleDashboard?.renderGallery();$('record-clip').style.display='none';}});$('diagnostic').onclick=()=>{overlay.diagnostic=!overlay.diagnostic;$('diagnostic').classList.toggle('primary',overlay.diagnostic);requestRender();};

// Record the already-playing camera and its synchronized canvas overlay together.
// This does not open another camera connection or include dashboard controls.
const recordButton=$('record-clip')||document.createElement('button');
if(!recordButton.id){recordButton.id='record-clip';recordButton.type='button';recordButton.className='quiet';recordButton.textContent='● Record 30 s';recordButton.setAttribute('aria-label','Record a 30-second clip');Object.assign(recordButton.style,{display:'none',position:'absolute',left:'16px',top:'16px',zIndex:'11',borderColor:'#4c737d',background:'#0c2028e8',color:'#d7f7f4'});$('stage').appendChild(recordButton);}
recordButton.onclick=()=>{
  const existing=window.__trafficRecorder;
  if(existing?.state==='recording'){existing.stop();return;}
  const button=$('record-clip');
  if(!video.videoWidth||video.readyState<2||!window.MediaRecorder||!HTMLCanvasElement.prototype.captureStream){toast('The live camera is not ready for browser recording.');return;}
  const w=video.videoWidth,h=video.videoHeight,canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;
  // Keep recording work below the live overlay's render cadence. 1080p canvas
  // Studio preserves source dimensions; Efficient remains available on busy CPUs.
  const recordFps=Math.min(Number($('record-quality')?.value)||30,state?.stream?.source_fps||30),c=canvas.getContext('2d',{alpha:false,desynchronized:true});
  const mime=[w>1920||h>1080?'video/mp4;codecs=avc1.640033':'video/mp4;codecs=avc1.42E028','video/webm;codecs=vp8','video/webm'].find(t=>MediaRecorder.isTypeSupported(t));
  if(!mime){toast('This browser does not support local video recording.');return;}
  const stream=canvas.captureStream(recordFps);
  const recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:Math.max(14000000,Math.min(60000000,Math.round(w*h*recordFps*.28)))}),parts=[];
  let active=true,lastPaint=-Infinity;
  const paint=now=>{if(!active||video.readyState<2||now-lastPaint<1000/recordFps-2)return;lastPaint=now;try{c.drawImage(jpeg||frozen?scene:video,0,0,w,h);if(overlay.canvas.width&&overlay.canvas.height)c.drawImage(overlay.canvas,0,0,w,h);}catch(e){console.warn('Recording frame skipped:',e);}};
  recorder.ondataavailable=e=>{if(e.data?.size)parts.push(e.data);};
  recorder.onerror=e=>toast('Recording failed: '+(e.error?.message||'media encoder error'));
  recorder.onstop=()=>{
    active=false;recordFrame=null;clearInterval(countdown);stream.getTracks().forEach(t=>t.stop());window.__trafficRecorder=null;button.classList.remove('recording');
    button.textContent='● Record 30 s';button.disabled=false;button.style.display=document.body.classList.contains('stage-expanded')?'block':'none';
    const blob=new Blob(parts,{type:mime}),a=document.createElement('a'),stamp=new Date().toISOString().replace(/[:.]/g,'-');
    a.href=URL.createObjectURL(blob);a.download=`traffic-live-${stamp}.${mime.startsWith('video/mp4')?'mp4':'webm'}`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),60000);
    toast(`${Math.min(30,Math.max(0,(Date.now()-startedAt)/1000)).toFixed(1)}-second ${w}×${h} camera clip saved.${mime.startsWith('video/mp4')?' MP4 ready to share.':' WebM saved; convert to MP4 for sharing.'}`);
  };
  button.disabled=false;button.classList.add('recording');button.style.display='block';window.__trafficRecorder=recorder;
  const startedAt=Date.now(),deadline=startedAt+30000;const countdown=setInterval(()=>{const remaining=Math.max(0,Math.ceil((deadline-Date.now())/1000));button.textContent=remaining?`■ Stop / ${remaining}s`:'■ Saving…';if(!remaining&&recorder.state==='recording')recorder.stop();},250);
  try{recordFrame=paint;requestRender();recorder.start(1000);button.textContent='■ Stop / 30s';}catch(error){clearInterval(countdown);active=false;recordFrame=null;stream.getTracks().forEach(t=>t.stop());window.__trafficRecorder=null;button.classList.remove('recording');button.textContent='● Record 30 s';toast('Browser could not start the recorder: '+error.message);}
};
['labels','confidence','trails','geometry'].forEach(key=>$('show-'+key).onchange=e=>{overlay.options[key]=e.target.checked;requestRender();});$('focus-only').onchange=e=>{overlay.options.focus=e.target.checked;requestRender();};$('opacity').oninput=e=>{overlay.options.opacity=Number(e.target.value);requestRender();};
$('jpeg-fallback').onclick=()=>{nativeAbort?.abort();jpeg=true;$('stage').classList.add('jpeg-mode');peer?.close();if(peerId)api('/api/webrtc/close/'+peerId,{}).catch(()=>{});jpegLoop();$('display-mode').textContent='JPEG fallback · native frames / timestamp-aligned';toast('JPEG fallback enabled. This uses the same upstream camera connection.');};
$('enhancement').onchange=()=>{video.style.filter=$('enhancement').value==='subtle'?'contrast(1.07) brightness(1.03) saturate(1.03)':'none';};
for(const id of ['authorize-open','empty-auth'])$(id).onclick=()=>$('auth-dialog').showModal();
let identifyTimer,identifyVersion=0;
$('add-stream').onclick=()=>$('auth-dialog').showModal();
const demoURL='https://www.youtube.com/watch?v=W0hB9CXqPAc';
function chooseDemo(){ $('play-url').value=demoURL;$('auth-dialog').showModal();identifySource(); }
$('demo-source').onclick=()=>{$('play-url').value=demoURL;identifySource();};$('quick-demo').onclick=chooseDemo;
async function identifySource(){const version=++identifyVersion,url=$('play-url').value.trim();if(!url){$('source-kind').textContent='AUTO DETECT';return;}try{const source=await api('/api/stream/identify',{url});if(version!==identifyVersion)return;$('source-kind').textContent=source.label;$('source-feedback').textContent=source.kind==='youtube'?'YouTube detected. Using a lighter 1080p preview when available, with native-resolution AI captures.':source.kind==='pattaya'?'Pattaya receiver detected. Authorization will be refreshed when available.':'Direct '+source.label+' detected. Your server must allow this computer to receive the media.';}catch(e){if(version===identifyVersion){$('source-kind').textContent='CHECK LINK';$('source-feedback').textContent=e.message;}}}
$('play-url').oninput=()=>{clearTimeout(identifyTimer);identifyVersion++;identifyTimer=setTimeout(identifySource,350);};
$('play-url').onkeydown=e=>{if(e.key==='Enter'){$('connect-source').click();}};
$('connect-source').onclick=async()=>{const button=$('connect-source');button.disabled=true;button.textContent='Connecting…';try{await api('/api/stream/authorize',{url:$('play-url').value.trim()});formLoaded=false;metadata=[];faceTimeline.clear();renderMeta=null;await connectVideo();$('play-url').value='';$('auth-dialog').close();toast('Source accepted. Waiting for live video; connection status appears above the camera.');}catch(e){$('source-feedback').textContent=e.message;toast(e.message);}finally{button.disabled=false;button.textContent='Connect stream →';}};
$('reconnect').onclick=async()=>{await api('/api/stream/restart',{});connectVideo();toast('Reconnecting source and local playback.');};
function loadForm(settings){for(const [key,value] of Object.entries(settings)){const field=$('engine-config').elements.namedItem(key);if(field){if(field.type==='checkbox')field.checked=value;else field.value=value;}}
  $('threshold-fields').innerHTML=Object.entries(settings.thresholds).map(([key,value])=>`<label>${esc(key)}<input type="number" data-threshold="${key}" min=".1" max=".95" step=".01" value="${value}"></label>`).join('');}
$('engine-config').onsubmit=async e=>{e.preventDefault();try{const value={...state.settings};for(const field of e.target.elements){if(!field.name)continue;value[field.name]=field.type==='checkbox'?field.checked:field.type==='number'||field.name==='resolution'?Number(field.value):field.value;}value.thresholds={};document.querySelectorAll('[data-threshold]').forEach(i=>value.thresholds[i.dataset.threshold]=Number(i.value));await api('/api/settings',value);toast('Engine settings saved. Applying to the inference worker.');}catch(error){toast(error.message);}};
function updateEditor(){const selected=editor.item();setHTML('geometry-list',editor.geometry.items.map(i=>`<div class="geometry-row ${editor.selected===i.id?'selected':''}" data-geometry="${esc(i.id)}"><span style="color:${/^#[\da-f]{6}$/i.test(i.color)?i.color:'#36d9d0'}">◇ ${esc(i.name)}</span><small>${esc(i.kind)} · ${i.points.length} vertices</small></div>`).join('')||'<p class="muted">No geometry saved. Draw a polygon or counting line over the image.</p>');
  overlay.invalidateGeometry();requestRender();window.refreshLaneSetup?.();
  if(selected){$('geo-name').value=selected.name;$('geo-color').value=selected.color;$('geo-direction').value=selected.direction.join(',');$('geo-role').value=selected.role||'';$('geo-visible').checked=selected.visible!==false;}
}
document.querySelectorAll('[data-draw]').forEach(b=>b.onclick=()=>editor.start(b.dataset.draw));$('finish-shape').onclick=()=>editor.finish();$('cancel-shape').onclick=()=>editor.cancel();
for(const id of ['geo-name','geo-color','geo-direction','geo-role','geo-visible'])$(id).onchange=()=>{const item=editor.item();if(!item)return;item.name=$('geo-name').value;item.color=$('geo-color').value;item.direction=$('geo-direction').value.split(',').map(Number);item.role=$('geo-role').value||null;item.visible=$('geo-visible').checked;updateEditor();};
$('delete-shape').onclick=()=>{editor.geometry.items=editor.geometry.items.filter(i=>i.id!==editor.selected);editor.selected=null;updateEditor();};
$('reset-geometry').onclick=()=>{editor.geometry.items=[];editor.selected=null;updateEditor();toast('Geometry cleared in editor. Apply & save to change active rules.');};
$('save-geometry').onclick=async()=>{try{editor.geometry=await api('/api/geometry',editor.geometry);toast('Geometry saved. Association and crossing baselines refreshed.');updateEditor();}catch(e){toast(e.message);}};
async function profiles(){const list=await api('/api/geometry/profiles');$('profiles').innerHTML='<option value="">Load saved profile…</option>'+list.map(v=>`<option>${esc(v)}</option>`).join('');}
$('save-profile').onclick=async()=>{try{await api('/api/geometry/profiles/'+encodeURIComponent($('profile-name').value),editor.geometry);profiles();toast('Geometry profile saved.');}catch(e){toast(e.message);}};
$('profiles').onchange=async e=>{if(e.target.value){editor.geometry=await api('/api/geometry/profiles/'+encodeURIComponent(e.target.value));updateEditor();toast('Profile loaded into editor. Apply & save to activate.');}};
$('suggest').onclick=async()=>{try{const suggestion=await api('/api/geometry/suggest');editor.geometry.items.push(...suggestion.items);updateEditor();toast(suggestion.warning);}catch(e){toast(e.message);}};
$('save-calibration').onclick=async()=>{try{const calibration=JSON.parse($('calibration').value);const value=await api('/api/geometry',{...editor.geometry,calibration});editor.geometry=value;toast('Measured calibration accepted (independent distance error ≤10%). Speeds remain estimates.');}catch(e){toast(e.message);}};
$('clear-calibration').onclick=async()=>{editor.geometry=await api('/api/geometry',{...editor.geometry,calibration:null});$('calibration').value='';toast('Physical speed disabled; showing px/s.');};
function eventHtml(event){return `<div class="event"><small>${timeText(event.timestamp)} · #${esc(event.id)} ${event.confirmed?'':'· REVIEW'}</small><p>${esc(event.type.replaceAll('_',' '))} ${event.line_name?'· '+esc(event.line_name):''}</p></div>`;}
function updateState(value){state=value;const a=value.analytics,p=value.performance,s=value.stream;
  $('connection').textContent=s.state==='live'?'SOURCE LIVE':s.state.replaceAll('_',' ').toUpperCase();$('connection').classList.toggle('amber',s.state!=='live');$('stream-status').textContent=s.state.toUpperCase();$('stream-message').textContent=s.message;$('acquisition').textContent=s.state;$('source-size').textContent=s.width?`${s.width} × ${s.height}`:'—';const display=s.playback||s.quality||{};const split=display.separate_rendition;const viewSize=split?`${display.width||'—'} × ${display.height||'—'}`:`${s.width||'—'} × ${s.height||'—'}`;$('stream-details').textContent=`${viewSize} · ${num(display.fps||s.source_fps||s.fps,0)} fps source · ${split?`${s.width||'—'} × ${s.height||'—'} AI / capture · `:''}${s.relay_mode==='native-mp4'?`${(display.codec||display.vcodec||s.quality?.codec||'video').toUpperCase()} packet copy`:'Decoded relay'}`;if($('source-transport'))$('source-transport').textContent=split?'1080p preview / native AI':s.relay_mode==='native-mp4'?`Original ${(s.quality?.codec||s.quality?.vcodec||'video').toUpperCase()} / MP4`:'WebRTC relay';
  if(s.camera){const parts=s.camera.split(' · ');if($('active-camera-name'))$('active-camera-name').textContent=parts[0]||s.camera;if($('active-camera-sub'))$('active-camera-sub').textContent=value.stream.location||'Pattaya, Thailand';if($('top-camera-code'))$('top-camera-code').textContent=parts[1]||s.camera;if($('main-camera-code'))$('main-camera-code').textContent=parts[1]||s.camera;}
  $('m-visible').textContent=num(a.visible);$('m-unique').textContent=num(a.unique);$('m-in').textContent=num(a.roles?.in);$('m-out').textContent=num(a.roles?.out);$('m-fps').textContent=value.engine.state==='running'?num(p.inference_fps,1):'—';$('m-latency').textContent=p.inference_ms==null?'Waiting for inference':`${num(p.inference_ms)} ms inference · ${p.resolution}px input`;
  $('engine-label').textContent=viewMode==='people'?`${value.settings.model.toUpperCase()} · ${value.engine.people_tracker||'Person tracker'}`:`${value.settings.model.toUpperCase()} · ${value.settings.tracker==='bytetrack'?'ByteTrack':'BoT-SORT'}`;
  const names={car:'Cars',motorcycle:'Motorcycles',bicycle:'Bicycles',bus:'Buses',truck:'Trucks'};
  setHTML('class-list',Object.entries(names).map(([key,name])=>`<div class="class-row"><i style="background:${overlay.colors[key]}"></i><span>${name}</span><strong>${a.visible_classes[key]||0} <small>/ ${a.classes?.[key]||0}</small></strong></div>`).join(''));
  const configuredLanes=value.geometry.items.filter(item=>item.kind==='lane');
  const laneCounts={};
  for(const lane of configuredLanes)laneCounts[lane.name]=a.lanes[lane.name]||0;
  for(const [lane,count] of Object.entries(a.lanes))if(!(lane in laneCounts))laneCounts[lane]=count;
  setHTML('lane-list',Object.entries(laneCounts).map(([lane,count])=>`<div><span>${esc(lane)}</span><b>${count}</b></div>`).join('')||'<p class="muted">Draw lanes to begin association.</p>');
  $('speed-legend').hidden=true;$('stopped').textContent=a.stopped;$('flow-minute').textContent=a.observed_per_minute;$('occupancy').textContent=a.box_occupancy_estimate_percent==null?'—':num(a.box_occupancy_estimate_percent)+'%';$('queue').textContent=num(a.queue_estimate);
  setHTML('event-feed',value.events.filter(e=>e.type!=='observed').slice(0,8).map(eventHtml).join('')||'<p class="muted">No crossing or review events recorded yet.</p>');
  const tracks=(value.frame.tracks||[]).filter(t=>t.confirmed&&s.frame_age_seconds<3&&value.engine.state==='running');$('track-count').textContent=tracks.length;$('no-tracks').hidden=!!tracks.length;
  setHTML('track-table',tracks.map(t=>`<tr data-uid="${esc(t.uid)}" class="${t.uid===overlay.selected?'selected':''}"><td>#${t.id}</td><td>${esc(t.category)} <small>${num(t.confidence*100)}%</small></td><td>${esc(t.lane_name||'Unassigned')}</td><td>${esc(t.direction)}</td><td><span class="motion ${t.motion==='Stopped'?'stopped':''}">${esc(t.motion)}</span></td><td>${esc(VehicleOverlay.speedReading(t,s.width||scene.width,s.height||scene.height).text)}</td><td>${num(t.duration,1)}s</td></tr>`).join(''));
  if(!window.peopleDashboard?.isRunning){
  const personRows=(value.frame.people||[]).filter(t=>t.confirmed&&s.frame_age_seconds<3&&value.engine.state==='running');const pc=value.people||{visible:0,unique:0};if(!window.peopleDashboard?.isRunning){$('people-visible').textContent=String(pc.visible||0);$('people-panel-visible').textContent=String(pc.visible||0);}$('people-unique').textContent=String(pc.unique||0);$('people-panel-unique').textContent=String(pc.unique||0);$('people-detections').textContent=String(value.frame.person_detections??(value.frame.people||[]).length);if(!window.peopleDashboard?.isRunning)$('people-status').textContent=value.engine.state==='running'&&s.state==='live'?'Tracking from live frames':'Waiting for live detections';$('people-track-count').textContent=String(personRows.length);$('people-empty').hidden=!!personRows.length;setHTML('people-track-table',personRows.map(t=>`<tr><td>#${esc(t.id)}</td><td>Confirmed</td><td>${num(t.confidence*100)}%</td><td>${num(t.duration,1)}s</td></tr>`).join(''));
  }
  requestRender();
  $('hardware').textContent=(p.device==='cpu'?'CPU · ':'')+(p.cuda?'CUDA available':'Intel CPU · CUDA unavailable');
  const displayedFPS=performance.now()-lastPresented<2000?playbackFPS:0;
  const telem=[['CAMERA RATE',s.fps,'FPS'],['VIDEO PLAYBACK',displayedFPS,'FPS'],['AI INFERENCE',value.engine.state==='running'?p.inference_fps:null,'FPS'],['TRACKER',p.tracking_ms,'ms'],['ANALYTICS AGE',p.analytics_latency_ms,'ms'],['SKIPPED FRAMES',p.skipped_frames,''],['CPU / MEMORY',p.cpu_percent,'%']];
  setHTML('telemetry',telem.map(([label,value,unit])=>`<div><span>${label}</span><strong>${num(value,label==='SKIPPED FRAMES'?0:1)}<small>${unit}</small></strong></div>`).join('')+`<div><span>PROCESS MEMORY</span><strong>${num(p.memory_mb)}<small>MB</small></strong></div><div><span>GPU UTIL / MEMORY</span><strong>${num(p.gpu_percent)}<small>% · ${num(p.gpu_memory_mb)} MB</small></strong></div><div><span>CONNECTED CLIENTS</span><strong>${p.connected_clients}</strong></div><div><span>LOST TRACKS</span><strong>${Math.max(0,a.lost_tracks||0)}</strong></div><div><span>DWELL (OBSERVED)</span><strong>${num(a.average_observed_dwell_seconds,1)}<small>s</small></strong></div><div><span>PERSISTENCE</span><strong style="font-size:12px">${value.persistence.error?esc(value.persistence.error):'SQLite · healthy'}</strong></div>`);
  $('last-update').textContent=`Updated ${timeText(Date.now()/1000)}`;
  if($('video-playback-bar')){
    const canControl=s.can_control!==false&&(s.source_type==='youtube'||s.source_type==='direct');
    $('video-playback-bar').style.display=canControl?'flex':'none';
    if(canControl){
      const pos=s.playback_position||0,dur=s.playback_duration,isPaused=Boolean(s.is_paused);
      if($('playback-pause'))$('playback-pause').classList.toggle('active',isPaused);
      if($('playback-resume'))$('playback-resume').classList.toggle('active',!isPaused);
      if(!isUserScrubbing&&$('playback-seek-slider')){
        if(dur&&dur>0){
          $('playback-seek-slider').max=Math.floor(dur);
          $('playback-seek-slider').value=Math.floor(pos);
          if($('playback-duration'))$('playback-duration').textContent=formatTime(dur);
        }else{
          $('playback-seek-slider').max=Math.max(100,Math.floor(pos)+60);
          $('playback-seek-slider').value=Math.floor(pos);
          if($('playback-duration'))$('playback-duration').textContent='--:--';
        }
        if($('playback-current-time'))$('playback-current-time').textContent=formatTime(pos);
      }
    }
  };
  if(!formLoaded){loadForm(value.settings);editor.geometry=structuredClone(value.geometry);$('calibration').value=value.geometry.calibration?JSON.stringify(value.geometry.calibration,null,2):'';updateEditor();formLoaded=true;}
  if(value.engine.error&&value.engine.error!==updateState.lastError){toast(value.engine.error);updateState.lastError=value.engine.error;}
  if(value.engine.warning&&value.engine.warning!==updateState.lastWarning){toast(value.engine.warning);updateState.lastWarning=value.engine.warning;}
  if(frozen&&scene.width&&scene.height){$('empty').hidden=true;}else if(s.frame_age_seconds!=null&&s.frame_age_seconds>4){$('empty').hidden=false;$('empty').querySelector('h2').textContent='Camera connection interrupted';}
  drawChart(a.flow||[]);
  if(inspected)refreshInspection();
}
let chartKey='';
function drawChart(flow){const canvas=$('flow-chart'),dpr=devicePixelRatio||1,w=canvas.clientWidth||300,h=155;const values=[];const start=Math.floor(Date.now()/60000)*60-59*60;const map=new Map(flow.map(f=>[f.bucket,f.count]));for(let i=0;i<60;i++)values.push(map.get(start+i*60)||0);const key=[w,dpr,start,...values].join(',');if(key===chartKey)return;chartKey=key;const bw=Math.round(w*dpr),bh=Math.round(h*dpr);if(canvas.width!==bw||canvas.height!==bh){canvas.width=bw;canvas.height=bh;}const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);const max=Math.max(1,...values);c.font='8px Segoe UI';for(let i=0;i<4;i++){let y=20+i*33;c.strokeStyle='#233545';c.beginPath();c.moveTo(35,y);c.lineTo(w-15,y);c.stroke();c.fillStyle='#5c7b90';c.fillText(num(max*(1-i/3)),8,y+3);}c.beginPath();values.forEach((v,i)=>{const x=35+i*(w-50)/59,y=119-v/max*99;i?c.lineTo(x,y):c.moveTo(x,y);});c.strokeStyle='#36cfc7';c.lineWidth=1.5;c.stroke();c.lineTo(w-15,119);c.lineTo(35,119);c.closePath();const g=c.createLinearGradient(0,20,0,119);g.addColorStop(0,'#36cfc733');g.addColorStop(1,'#36cfc700');c.fillStyle=g;c.fill();c.fillStyle='#5c7b90';c.fillText('−60 MIN',35,142);c.fillText('NOW',w-35,142);}
if(window.ResizeObserver)new ResizeObserver(()=>{if(state)drawChart(state.analytics.flow||[]);}).observe($('flow-chart'));
$('track-table').addEventListener('click',event=>{const row=event.target.closest('[data-uid]');if(row)inspectTrack(row.dataset.uid);});
$('geometry-list').addEventListener('click',event=>{const row=event.target.closest('[data-geometry]');if(row){editor.selected=row.dataset.geometry;updateEditor();}});
let inspectionPending=false,inspectionUpdated=0;
async function refreshInspection(force=false){if(!inspected||inspectionPending||(!force&&performance.now()-inspectionUpdated<900))return;const uid=inspected;inspectionPending=true;try{const detail=await api('/api/tracks/'+encodeURIComponent(uid));if(inspected===uid)drawInspection(detail);}catch(e){if(force)toast(e.message);}finally{inspectionPending=false;inspectionUpdated=performance.now();}}
async function inspectTrack(uid){inspected=overlay.selected=uid;requestRender();$('inspector').hidden=false;await refreshInspection(true);}
function drawInspection(t){const speed=t.speed_kmh??t.speed_px_s;const pairs={'Speed method':t.speed_basis||'Not calibrated','Lane distance':t.lane_speed?num(t.lane_speed.distance_metres,1)+' m':'—','Lane travel time':t.lane_speed?num(t.lane_speed.elapsed_seconds,2)+' s':'—','Subtype':t.subtype||'Unknown subtype','Detection confidence':num(t.confidence*100)+'%','Lane':t.lane_name||'Unassigned','Track state':t.state,'Direction':t.direction,'Motion':t.motion,'Speed':num(speed,1)+(t.speed_kmh==null?' px/s':' km/h (calibrated estimate)'),'First seen':timeText(t.first_seen),'Last seen':timeText(t.last_seen),'Duration':num(t.duration,1)+' s','Stopped':num(t.stopped_seconds,1)+' s','Lane occupancy':num(t.lane_seconds,1)+' s','Motion consistency':num(t.consistency,2)};
  $('inspection-body').innerHTML=`<div class="eyebrow">PERSISTENT SESSION ID</div><h3 style="color:${overlay.colors[t.category]||'#aaa'}">${esc(t.category||'Expired track')} #${t.id||esc(t.uid)}</h3><dl>${Object.entries(pairs).map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl><h4 class="section-label">RECENT TRAJECTORY</h4><canvas id="trajectory" width="320" height="140"></canvas><h4 class="section-label" style="margin-top:25px">EVENT HISTORY</h4><div class="event-feed">${(t.events||[]).slice(-15).reverse().map(eventHtml).join('')||'<p class="muted">No crossings or zone transitions recorded.</p>'}</div><h4 class="section-label" style="margin-top:25px">DETECTION HISTORY</h4><div class="event-feed">${(t.detection_history||[]).slice(-8).reverse().map(d=>`<div class="event"><small>${timeText(d.timestamp)}</small><p>${esc(d.category)} · ${num(d.confidence*100)}%</p></div>`).join('')}</div>`;
  const c=$('trajectory').getContext('2d');c.strokeStyle=overlay.colors[t.category]||'#36d9d0';c.lineWidth=2;c.beginPath();(t.history||[]).forEach((p,i)=>i?c.lineTo(p[1]*320,p[2]*140):c.moveTo(p[1]*320,p[2]*140));c.stroke();}
$('overlay').addEventListener('click',e=>{if(editor.active)return;const hit=overlay.hit(overlay.point(e));if(viewMode==='people'){overlay.selected=hit?(hit.id??hit.track_id??hit.uid):null;if(hit&&window.peopleDashboard?.profiles){const tid=hit.track_id??hit.id;for(const [sid,prof] of window.peopleDashboard.profiles){if(sid===hit.session_id||prof.track_id===tid){window.peopleDashboard.openProfileModal(sid);break;}}}requestRender();return;}if(hit)inspectTrack(hit.uid);});$('close-inspector').onclick=()=>{inspected=overlay.selected=null;$('inspector').hidden=true;requestRender();};
setInterval(()=>$('clock').textContent=new Date().toLocaleTimeString([], {hour12:false}),1000);
const socket=io({transports:['websocket','polling']});socket.on('frame',addMetadata);socket.on('people:frame',frame=>{faceTimeline.add(frame);requestRender();});socket.on('state',updateState);socket.on('connect',()=>api('/api/state').then(v=>{updateState(v);addMetadata(v.frame);}).catch(e=>toast(e.message)));socket.on('disconnect',()=>{metadata=[];faceTimeline.clear();renderMeta=null;requestRender();$('connection').textContent='DASHBOARD OFFLINE';$('connection').classList.add('amber');});
profiles().catch(()=>{});connectVideo();try{setView(localStorage.getItem('vector-monitor-view')==='people'?'people':'traffic');}catch(e){setView('traffic');}window.addEventListener?.('DOMContentLoaded',()=>{if(viewMode==='people'&&window.peopleDashboard&&!window.peopleDashboard.isRunning&&!window.peopleDashboard.userStopped)window.peopleDashboard.startAnalysis?.();});
window.addEventListener('pagehide',()=>{nativeAbort?.abort();peer?.close();pictures.forEach(p=>p.bitmap.close());});


// Video Playback Controls (Pause, Resume, Skip +30s, Back -30s, Seek Slider)
let isUserScrubbing=false;
function formatTime(seconds){
  if(isNaN(seconds)||seconds==null)return'--:--';
  const s=Math.max(0,Math.floor(seconds)),m=Math.floor(s/60),sec=s%60,h=Math.floor(m/60),min=m%60;
  return h>0?`${h}:${String(min).padStart(2,'0')}:${String(sec).padStart(2,'0')}`:`${String(min).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
}

const pauseBtn=$('playback-pause'),resumeBtn=$('playback-resume'),back30Btn=$('playback-back-30'),skip30Btn=$('playback-skip-30'),seekSlider=$('playback-seek-slider'),currTimeText=$('playback-current-time');

if(pauseBtn){
  pauseBtn.onclick=async()=>{
    try{
      video.pause();
      await api('/api/video/control',{action:'pause'});
      pauseBtn.classList.add('active');resumeBtn.classList.remove('active');
      toast('Video paused.');
    }catch(e){toast('Failed to pause: '+e.message);}
  };
}
if(resumeBtn){
  resumeBtn.onclick=async()=>{
    try{
      video.play().catch(()=>{});
      await api('/api/video/control',{action:'resume'});
      resumeBtn.classList.add('active');pauseBtn.classList.remove('active');
      toast('Video resumed.');
    }catch(e){toast('Failed to resume: '+e.message);}
  };
}
if(back30Btn){
  back30Btn.onclick=async()=>{
    try{
      toast('Rewinding 30 seconds?');
      await api('/api/video/control',{action:'skip',delta:-30});
    }catch(e){toast('Failed to skip back: '+e.message);}
  };
}
if(skip30Btn){
  skip30Btn.onclick=async()=>{
    try{
      toast('Skipping forward 30 seconds?');
      await api('/api/video/control',{action:'skip',delta:30});
    }catch(e){toast('Failed to skip: '+e.message);}
  };
}
if(seekSlider){
  seekSlider.oninput=()=>{
    isUserScrubbing=true;
    if(currTimeText)currTimeText.textContent=formatTime(Number(seekSlider.value));
  };
  seekSlider.onchange=async()=>{
    const target=Number(seekSlider.value);
    isUserScrubbing=false;
    try{
      toast(`Seeking to ${formatTime(target)}?`);
      await api('/api/video/control',{action:'seek',seconds:target});
    }catch(e){toast('Failed to seek: '+e.message);}
  };
}

document.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))return;
  if(e.code==='Space'||e.key==='k'){
    e.preventDefault();
    if(state?.stream?.is_paused)resumeBtn?.click();
    else pauseBtn?.click();
  }else if(e.key==='ArrowLeft'||e.key==='j'){
    e.preventDefault();
    back30Btn?.click();
  }else if(e.key==='ArrowRight'||e.key==='l'){
    e.preventDefault();
    skip30Btn?.click();
  }
});
