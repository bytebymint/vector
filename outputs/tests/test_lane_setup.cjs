const assert=require('node:assert/strict');
const fs=require('node:fs');const vm=require('node:vm');const path=require('node:path');
const nodes=new Map(),calls=[];
function $(id){if(!nodes.has(id))nodes.set(id,{textContent:'',innerHTML:'',scrollIntoView(){},click(){return this.onclick?.();}});return nodes.get(id);}
const lane={id:'learned-one',kind:'lane',name:'Lane 1',color:'#56ded2',points:[[.2,.2],[.4,.2],[.4,.8],[.2,.8]],distance_metres:null,suggestion:{supporting_tracks:4}};
const sandbox={window:{},editor:{active:false,geometry:{items:[],calibration:null}},state:{stream:{stream_id:'camera'}},$,setHTML:(id,html)=>{$(id).innerHTML=html;},esc:String,updateEditor(){sandbox.window.refreshLaneSetup?.();},setTimeout(){return 1;},clearTimeout(){},toast(text){sandbox.lastToast=text;},switchTab(tab){sandbox.editor.active=tab==='geometry';},api:async(url,body)=>{calls.push({url,body});return url.endsWith('/suggest')?{items:[structuredClone(lane)],source_id:'camera',warning:'Review boundaries'}:structuredClone(body);}};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/js/lane_setup.js'),'utf8'),sandbox);
(async()=>{
  $('open-lane-speed').click();await new Promise(setImmediate);
  assert.equal(sandbox.editor.geometry.items.length,1,'opening an unconfigured camera learns lanes');
  assert.equal(sandbox.editor.geometry.items[0].distance_metres,null,'never invent a real-world road length');
  assert.ok($('lane-distances').innerHTML.includes('Start-to-end road distance'));
  assert.equal(calls.filter(c=>c.url==='/api/geometry').length,0,'suggestions are drafts until applied');
  await $('apply-lane-speeds').click();
  assert.equal(calls.filter(c=>c.url==='/api/geometry').length,0,'missing distance cannot enable physical speed');
  $('lane-distances').onchange({target:{dataset:{laneDistance:lane.id},value:'50'}});
  assert.equal(sandbox.editor.geometry.items[0].distance_metres,50);
  await $('apply-lane-speeds').click();
  const save=calls.find(c=>c.url==='/api/geometry');
  assert.equal(save.body.items[0].distance_metres,50);
  assert.equal(save.body.calibration,null,'lane timing mode is selected explicitly on apply');
  assert.match(sandbox.lastToast,/Live km\/h estimates appear during travel/);
  assert.equal(sandbox.editor.active,false,'return to live video after saving');
  console.log('Lane discovery, distance prompting, no guessed metres and apply flow checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
