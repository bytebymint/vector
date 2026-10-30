'use strict';
window.ZoneEditor=class {
  constructor(overlay,onChange,onFreeze){this.overlay=overlay;this.onChange=onChange;this.onFreeze=onFreeze;this.geometry={items:[],calibration:null};this.active=false;this.selected=null;this.kind=null;this.draft=[];this.drag=null;
    const canvas=overlay.canvas;
    canvas.addEventListener('pointerdown',e=>{if(!this.active)return;const p=overlay.point(e);if(p.some(v=>v<0||v>1))return;if(this.kind){this.draft.push(p);if(this.kind==='line'&&this.draft.length===2)this.finish();this.onChange();return;}
      let found=null;for(const item of this.geometry.items){item.points.forEach((v,i)=>{if(Math.hypot((v[0]-p[0])*canvas.clientWidth,(v[1]-p[1])*canvas.clientHeight)<14)found={item,index:i};});}
      if(found){this.selected=found.item.id;this.drag=found;canvas.setPointerCapture(e.pointerId);this.onChange();}
    });
    canvas.addEventListener('pointermove',e=>{if(!this.drag)return;const p=overlay.point(e).map(v=>Math.max(0,Math.min(1,v)));this.drag.item.points[this.drag.index]=p;this.onChange();});
    canvas.addEventListener('pointerup',()=>{this.drag=null;});
    canvas.addEventListener('dblclick',()=>{if(this.kind&&this.kind!=='line'){const a=this.draft.at(-1),b=this.draft.at(-2);if(a&&b&&Math.hypot(a[0]-b[0],a[1]-b[1])<.005)this.draft.pop();this.finish();}});
    document.addEventListener('keydown',e=>{if(!this.active||/INPUT|SELECT|TEXTAREA/.test(e.target.tagName))return;if(e.key==='Enter'){e.preventDefault();this.finish();}if(e.key==='Escape')this.cancel();});
  }
  setActive(value){if(value&&this.onFreeze(true)===false)return;this.active=value;document.body.classList.toggle('editing',value);if(!value){this.kind=null;this.draft=[];}this.onChange();}
  start(kind){this.setActive(true);this.kind=kind;this.draft=[];document.getElementById('editor-hint').textContent=kind==='line'?'Click two endpoints. Orient the line to define its signed sides.':'Click vertices. Enter or double-click to finish; Escape to cancel. Drag vertices after drawing.';document.getElementById('editor-hint').hidden=false;document.getElementById('stage').scrollIntoView({behavior:'smooth',block:'center'});}
  finish(){if(this.draft.length<(this.kind==='line'?2:3))return;const item={id:crypto.randomUUID(),kind:this.kind,name:`${this.kind[0].toUpperCase()+this.kind.slice(1)} ${this.geometry.items.filter(x=>x.kind===this.kind).length+1}`,color:this.kind==='restricted'?'#ef9197':'#36d9d0',points:this.draft.map(p=>[...p]),direction:[0,-1],visible:true,role:null};this.geometry.items.push(item);this.selected=item.id;this.kind=null;this.draft=[];document.getElementById('editor-hint').hidden=true;this.onChange();}
  cancel(){this.kind=null;this.draft=[];document.getElementById('editor-hint').hidden=true;this.onChange();}
  item(){return this.geometry.items.find(i=>i.id===this.selected);}
};
