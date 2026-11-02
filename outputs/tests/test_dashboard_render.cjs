// Exercise the real dashboard scheduler and recorder without a camera or browser.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
let now=0,nextRaf=1,overlayClears=0,boxes=0,labels=0,recordedImages=0;
const rafs=new Map(),elements=new Map();
function context(recording=false){return new Proxy({}, {get(target,key){
  if(key==='measureText')return text=>({width:text.length*7});
  if(key==='createLinearGradient')return ()=>({addColorStop(){}});
  if(key==='clearRect')return ()=>{if(!recording)overlayClears++;};
  if(key==='strokeRect')return ()=>boxes++;
  if(key==='fillText')return ()=>labels++;
  if(key==='drawImage')return ()=>{if(recording)recordedImages++;};
  return target[key]??(()=>{});
}});}
function element(id='',recording=false){return {id,width:640,height:360,clientWidth:640,clientHeight:360,readyState:4,videoWidth:640,videoHeight:360,
  dataset:{},firstChild:{textContent:''},style:{},classList:{add(){},remove(){},toggle(){},contains(){return false;}},
  getContext:()=>context(recording),addEventListener(){},setAttribute(){},removeAttribute(){},querySelector(){return element();},
  getBoundingClientRect:()=>({width:640,height:360,left:0,top:0}),requestVideoFrameCallback(fn){this.onVideoFrame=fn;},
  captureStream:fps=>({fps,getTracks:()=>[{stop(){}}]}),click(){},play:async()=>{},appendChild(){},
};}
const document={getElementById(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id);},
  querySelector:()=>({content:'fixture'}),querySelectorAll:()=>[],addEventListener(){},body:element(),
  createElement:tag=>element('',tag==='canvas')};
const socket={on(){}};
const sandbox={console,document,performance:{now:()=>now},devicePixelRatio:1,structuredClone,Blob,
  requestAnimationFrame(fn){const id=nextRaf++;rafs.set(id,fn);return id;},
  setInterval(){return 1;},clearInterval(){},setTimeout(){return 1;},clearTimeout(){},
  fetch:()=>new Promise(()=>{}),io:()=>socket,
  URL:{createObjectURL:()=>'',revokeObjectURL(){}},
  Path2D:class{moveTo(){}lineTo(){}},HTMLCanvasElement:{prototype:{captureStream(){}}},
  MediaRecorder:class{static isTypeSupported(){return true;}constructor(stream){this.stream=stream;this.state='inactive';}start(){this.state='recording';}stop(){this.state='inactive';this.onstop();}},
};
sandbox.window=sandbox;sandbox.MediaSource=class{};sandbox.addEventListener=()=>{};
vm.createContext(sandbox);
for(const file of ['people_overlay.js','overlay.js','zone_editor.js','face_timeline.js','dashboard.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/js',file),'utf8'),sandbox,{filename:file});
function flush(){const jobs=[...rafs.values()];rafs.clear();jobs.forEach(fn=>fn(now));}
flush();
assert.equal(rafs.size,0,'an idle dashboard must not schedule continuous redraws');
const fixture={timestamp:100,tracks:[{uid:'car-a',id:1,category:'car',confirmed:true,bbox:[.2,.3,.4,.6],velocity:[20,0],
  speed_kmh:40,speed_calibrated:true,history:[[100,.3,.6]]}]};
sandbox.fixture=fixture;
vm.runInContext('origin=100;addMetadata(fixture);addMetadata(fixture);',sandbox);
elements.get('video').onVideoFrame(0,{mediaTime:0});
assert.equal(rafs.size,1,'metadata and decoded-video updates must coalesce');
flush();
assert.ok(boxes>0&&labels>0,'real vehicle boxes and km/h labels should render');
const before=overlayClears;
for(let i=0;i<144;i++)flush();
assert.equal(overlayClears,before,'monitor refreshes without a decoded frame do no overlay work');
elements.get('record-clip').onclick();
flush();
assert.equal(sandbox.__trafficRecorder.stream.fps,30,'studio recorder retains the native 30fps cadence');
assert.equal(recordedImages,2,'record video and the completed overlay together');
const recordingBoxes=boxes;
for(let i=1;i<=30;i++){
  now=i*1000/30;
  if(i%3===0){sandbox.nextFixture={...fixture,timestamp:100+i/30};vm.runInContext('addMetadata(nextFixture)',sandbox);}
  elements.get('video').onVideoFrame(now,{mediaTime:i/30});flush();
}
assert.ok(boxes>=recordingBoxes+30,'recording must keep the live boxes and speed overlays rendering on every presented frame');
assert.ok(recordedImages>=56&&recordedImages<=64,'recording composites at approximately 30 fps, independently of video at 30 fps');
elements.get('record-clip').onclick();
assert.equal(sandbox.__trafficRecorder,null,'stopping releases the recorder');
const imagesAtStop=recordedImages;
now+=100;elements.get('video').onVideoFrame(now,{mediaTime:1.1});flush();
assert.equal(recordedImages,imagesAtStop,'stopping recording releases composite work');
const boxesAtStop=boxes;
vm.runInContext("state={engine:{state:'running'},stream:{frame_age_seconds:10}};requestRender();",sandbox);flush();
assert.equal(boxes,boxesAtStop,'a disconnected source must not retain apparent live boxes');
vm.runInContext("state={engine:{state:'error'},stream:{frame_age_seconds:0}};requestRender();",sandbox);flush();
assert.equal(boxes,boxesAtStop,'a failed inference worker must not display stale boxes');
vm.runInContext('metadata=[];renderMeta=null;requestRender();',sandbox);flush();
assert.equal(sandbox.VehicleOverlay.prototype.hit.call({frame:null},[.3,.4]),undefined);
console.log('Dashboard idle scheduling, decoded-frame coalescing, overlay recording and cleanup checks passed');

elements.get('video').readyState=0;
assert.equal(vm.runInContext('freeze(true)',sandbox),false,'setup must wait for a real camera image');
assert.equal(vm.runInContext('frozen',sandbox),false,'an unready camera must not leave a blank held frame');
elements.get('video').readyState=4;

// A 4K face observation must retain its coordinate dimensions on a 1080p video.
// This also exercises the branch with no simultaneous vehicle metadata.
vm.runInContext(`
  metadata=[];renderMeta=null;viewMode='people';overlay.peopleMode=true;
  state={engine:{state:'running'},stream:{frame_age_seconds:0}};
  faceTimeline.clear();
  faceTimeline.add({epoch:'dual',tracking_kind:'face',timestamp:102,width:3840,height:2160,
    tracks:[{track_id:1,confirmed:true,bbox:[1920,400,2120,600]}],
    faces:[{track_id:1,bbox:[1920,400,2120,600],confidence:.99}]});
  overlay.peopleOverlay.drawFaceBox=(ctx,face,box)=>{observedFaceRect=box;};
`,sandbox);
elements.get('video').videoWidth=1920;elements.get('video').videoHeight=1080;
now=2000;elements.get('video').onVideoFrame(now,{mediaTime:2});flush();
assert.equal(vm.runInContext('renderMeta.width',sandbox),3840);
assert.equal(elements.get('overlay').width,1920);
assert.deepEqual(Array.from(vm.runInContext('observedFaceRect',sandbox)),[960,200,100,100]);
console.log('4K AI face coordinates align with the independent 1080p video canvas');
