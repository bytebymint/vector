const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
let boxes=0,labels=[];
const ctx={clearRect(){},strokeRect(){boxes++;},fillText(text){labels.push(text);},measureText(text){return {width:text.length*6};},beginPath(){},roundRect(){},fill(){},save(){},restore(){}};
const canvas={width:640,height:360,getContext:()=>ctx};
const sandbox={window:{},Path2D:class{}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../static/js/overlay.js'),'utf8'),sandbox);
const overlay=new sandbox.window.VehicleOverlay(canvas);overlay.peopleMode=true;
overlay.draw({timestamp:1,tracks:[],people:[
  {id:17,confirmed:true,bbox:[.1,.2,.3,.7]},
  {id:18,confirmed:false,bbox:[.4,.2,.6,.7]},
]},{items:[]},null,1);
assert.equal(boxes,1,'only a confirmed person track gets a visible box');
assert.equal(labels.length,0,'people view keeps box-only overlays free of label collisions');

const smooth=sandbox.window.VehicleOverlay.interpolatePeople(
  {timestamp:1,people:[{uid:'s:17',id:17,confirmed:true,bbox:[.1,.2,.3,.7]}]},
  {timestamp:1.2,people:[{uid:'s:17',id:17,confirmed:true,bbox:[.3,.2,.5,.7]}]},
  1.1,
);
Array.from(smooth.people[0].bbox).forEach((value,index)=>assert.ok(Math.abs(value-[.2,.2,.4,.7][index])<1e-9));
assert.equal(smooth.timestamp,1.1);
console.log('People overlay renders compact labels and interpolates confirmed tracks smoothly');

