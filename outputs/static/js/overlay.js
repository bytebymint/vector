'use strict';
window.VehicleOverlay = class VehicleOverlay {
  static speedCache=new Map();
  static readingCache=new WeakMap();
  static speedReading(track,width,height){
    const sample=track.history?.at(-1)?.[0],cached=VehicleOverlay.readingCache.get(track);
    // A received track is shared by the registry and many video presentations.
    // Fit each observation once, rather than once per monitor refresh.
    if(cached&&cached.sample===sample&&cached.width===width&&cached.height===height&&cached.kmh===track.speed_kmh&&cached.calibrated===track.speed_calibrated)return cached.reading;
    const reading=VehicleOverlay.measureSpeed(track,width,height);
    VehicleOverlay.readingCache.set(track,{sample,width,height,kmh:track.speed_kmh,calibrated:track.speed_calibrated,reading});
    return reading;
  }
  static measureSpeed(track,width,height){
    if(track.speed_calibrated){
      const key=track.uid,clock=track.history?.at(-1)?.[0],old=key?VehicleOverlay.speedCache.get(key):null;
      let value=Number.isFinite(track.speed_kmh)?Math.abs(track.speed_kmh):NaN;
      if(Number.isFinite(value)){
        if(old&&clock===old.sample)value=old.value;
        else if(old&&clock>old.sample&&clock-old.sample<1){const alpha=1-Math.exp(-(clock-old.sample)/.25);value=old.value+alpha*(value-old.value);}
        if(key&&Number.isFinite(clock)){VehicleOverlay.speedCache.set(key,{value,sample:clock});if(VehicleOverlay.speedCache.size>500)VehicleOverlay.speedCache.delete(VehicleOverlay.speedCache.keys().next().value);}
      }else if(old&&clock>=old.sample&&clock-old.sample<=.8)value=old.value;
      if(Number.isFinite(value))return {ready:true,value,unit:'km/h*',text:`~${value.toFixed(0)} km/h*${track.speed_method==='lane_average'?' avg':''}`};
      return {ready:false,text:track.speed_method==='pending_lane'?'Measuring lane speed…':'Measuring km/h…'};
    }
    const history=track.history||[],last=history.at(-1);
    if(!last)return {ready:false,text:'Measuring speed…'};
    // Fit motion over recent capture timestamps, rather than amplifying box jitter
    // by dividing a single frame displacement by a very short frame interval.
    const samples=history.filter(p=>last[0]-p[0]<=.85);
    const span=samples.length?last[0]-samples[0][0]:0;
    if(samples.length<4||span<.35)return {ready:false,text:'Measuring speed…'};
    let mt=0,mx=0,my=0;
    for(const p of samples){mt+=p[0]-last[0];mx+=p[1]*width;my+=p[2]*height;}
    mt/=samples.length;mx/=samples.length;my/=samples.length;
    let tt=0,tx=0,ty=0;
    for(const p of samples){const dt=p[0]-last[0]-mt;tt+=dt*dt;tx+=dt*(p[1]*width-mx);ty+=dt*(p[2]*height-my);}
    if(tt<=0)return {ready:false,text:'Measuring speed…'};
    const pixels=Math.hypot(tx/tt,ty/tt);
    if(Number.isFinite(track.speed_kmh)){const value=Math.abs(track.speed_kmh);return {ready:true,value,unit:'km/h*',text:`~${value.toFixed(0)} km/h*${track.speed_method==='lane_average'?' avg':''}`};}
    return {ready:true,value:pixels,unit:'px/s',text:`${pixels.toFixed(0)} px/s`};
  }
  constructor(canvas){
    this.canvas=canvas;this.ctx=canvas.getContext('2d');this.selected=null;this.frame=null;this.diagnostic=false;this.peopleMode=false;
    this.paths=new Map();this.labelWidths=new Map();
    this.options={labels:true,confidence:false,trails:false,geometry:true,focus:false,opacity:.9};
    this.colors={car:'#56ded2',motorcycle:'#8eb6ff',bicycle:'#b3a2ff',bus:'#f5c36b',truck:'#ef9197',person:'#79edb6'};
    const PeopleOverlayClass = typeof PeopleOverlay !== 'undefined' ? PeopleOverlay : (typeof window !== 'undefined' ? window.PeopleOverlay : null);
    this.peopleOverlay = PeopleOverlayClass ? new PeopleOverlayClass(canvas, this.options) : null;
  }
  static interpolatePeople(before,after,stamp){
    if(!before||!after||after.timestamp<=before.timestamp||stamp<before.timestamp||stamp>after.timestamp)return null;
    const ratio=(stamp-before.timestamp)/(after.timestamp-before.timestamp),nearest=ratio<.5?before:after,other=ratio<.5?after:before;
    const byId=new Map((other.people||[]).filter(p=>p.confirmed).map(p=>[p.uid??p.id,p]));
    const people=(nearest.people||[]).filter(p=>p.confirmed).map(p=>{
      const q=byId.get(p.uid??p.id);if(!q)return p;
      const a=ratio<.5?p.bbox:q.bbox,b=ratio<.5?q.bbox:p.bbox;
      return {...p,bbox:a.map((v,i)=>v+(b[i]-v)*ratio)};
    });
    return {...nearest,timestamp:stamp,syncTimestamp:nearest.timestamp,people};
  }
  invalidateGeometry(){this.paths.clear();}
  size(w,h){if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;this.paths.clear();this.labelWidths.clear();if(this.peopleOverlay)this.peopleOverlay.size(w,h);}}
  labelWidth(text,u){const key=u+':'+text;if(!this.labelWidths.has(key)){if(this.labelWidths.size>=256)this.labelWidths.clear();this.labelWidths.set(key,this.ctx.measureText(text).width);}return this.labelWidths.get(key);}
  point(event){const r=this.canvas.getBoundingClientRect(),scale=Math.min(r.width/this.canvas.width,r.height/this.canvas.height),w=this.canvas.width*scale,h=this.canvas.height*scale;return [(event.clientX-r.left-(r.width-w)/2)/w,(event.clientY-r.top-(r.height-h)/2)/h];}
  geometry(items,selected,draft,editing=false){const c=this.ctx,w=this.canvas.width,h=this.canvas.height,u=Math.max(1,w/768);
    if(!editing){
      // Display open, traced markings only. Association polygons belong in the editor;
      // closing them across the junction invents road boundaries that aren't there.
      c.save();c.lineCap='round';c.lineJoin='round';
      for(const item of items){if(item.visible===false||!['guide','line'].includes(item.kind))continue;
        let path=this.paths.get(item);if(!path){path=new Path2D();item.points.forEach((p,i)=>i?path.lineTo(p[0]*w,p[1]*h):path.moveTo(p[0]*w,p[1]*h));this.paths.set(item,path);}
        c.strokeStyle=item.color;c.lineWidth=.85*u;c.globalAlpha=this.options.opacity*.58;
        c.setLineDash(item.kind==='line'?[5*u,5*u]:[]);c.stroke(path);
      }c.restore();return;
    }
    for(const item of items){const pts=item.points;if(!pts.length)continue;c.beginPath();pts.forEach((p,i)=>i?c.lineTo(p[0]*w,p[1]*h):c.moveTo(p[0]*w,p[1]*h));if(!['line','guide'].includes(item.kind))c.closePath();c.strokeStyle=item.color;c.lineWidth=(item.id===selected?2.5:1)*u;c.setLineDash(item.kind==='line'?[7,5]:[]);c.stroke();c.setLineDash([]);if(!['line','guide'].includes(item.kind)){c.fillStyle=item.color+'09';c.fill();}c.font=`${10*u}px Segoe UI`;c.fillStyle=item.color;
      if(item.kind==='lane'){
        if(pts.length===4){for(const [text,pair] of [['START',[pts[0],pts[1]]],['END',[pts[3],pts[2]]]]){const gx=(pair[0][0]+pair[1][0])*w/2,gy=(pair[0][1]+pair[1][1])*h/2;c.font=`bold ${11*u}px Segoe UI`;c.fillStyle='#081622ee';c.fillRect(gx-27*u,gy-16*u,54*u,21*u);c.fillStyle=item.color;c.fillText(text,gx-22*u,gy);}}

        // Place names inside their lane, away from the crowded vanishing point.
        const scan=.44,cross=[];
        for(let j=0;j<pts.length;j++){const a=pts[j],b=pts[(j+1)%pts.length];if((a[1]<=scan&&b[1]>scan)||(b[1]<=scan&&a[1]>scan))cross.push(a[0]+(scan-a[1])*(b[0]-a[0])/(b[1]-a[1]));}
        const px=cross.length>=2?(Math.min(...cross)+Math.max(...cross))/2:pts[0][0],py=cross.length>=2?scan:pts[0][1];
        const tw=c.measureText(item.name).width;c.fillStyle='#081622cc';c.fillRect(px*w-tw/2-4*u,py*h-20*u,tw+8*u,15*u);c.fillStyle=item.color;c.fillText(item.name,px*w-tw/2,py*h-9*u);
        const d=item.direction||[0,-1];this.arrow(px*w,py*h+5*u,d[0]*14*u,d[1]*14*u,item.color);
      }else c.fillText(item.name,pts[0][0]*w+5,pts[0][1]*h-7);
      if(item.id===selected){for(const p of pts){c.beginPath();c.arc(p[0]*w,p[1]*h,5,0,Math.PI*2);c.fillStyle='#0a1a26';c.fill();c.strokeStyle=item.color;c.stroke();}}
    }
    if(draft?.length){c.beginPath();draft.forEach((p,i)=>i?c.lineTo(p[0]*w,p[1]*h):c.moveTo(p[0]*w,p[1]*h));c.setLineDash([4,4]);c.strokeStyle='#fff';c.stroke();c.setLineDash([]);draft.forEach(p=>{c.beginPath();c.arc(p[0]*w,p[1]*h,4,0,7);c.fillStyle='#fff';c.fill();});}
  }
  arrow(x,y,dx,dy,color){const c=this.ctx;if(Math.hypot(dx,dy)<2)return;const a=Math.atan2(dy,dx);c.strokeStyle=color;c.beginPath();c.moveTo(x,y);c.lineTo(x+dx,y+dy);c.moveTo(x+dx-6*Math.cos(a-.6),y+dy-6*Math.sin(a-.6));c.lineTo(x+dx,y+dy);c.lineTo(x+dx-6*Math.cos(a+.6),y+dy-6*Math.sin(a+.6));c.stroke();}
  draw(frame,geometry,editor,renderTime){this.frame=frame;const c=this.ctx,w=this.canvas.width,h=this.canvas.height,u=Math.max(1,w/768);c.clearRect(0,0,w,h);c.globalAlpha=this.options.opacity;
    if(!this.peopleMode&&(this.options.geometry||editor?.active))this.geometry(geometry.items,editor?.active?editor.selected:null,editor?.draft,!!editor?.active);
    if(frame&&Math.abs((renderTime||frame.timestamp)-frame.timestamp)<1.25){
      if(this.peopleMode){
        if(this.peopleOverlay){
          this.peopleOverlay.selectedId=this.selected;
          this.peopleOverlay.diagnostic=this.diagnostic;
          this.peopleOverlay.draw(frame,renderTime,this.options);
          c.globalAlpha=1;
          return;
        }
        for(const person of frame.people||[]){if(!person.confirmed)continue;const b=person.bbox,x=b[0]*w,y=b[1]*h,bw=(b[2]-b[0])*w,bh=(b[3]-b[1])*h;c.strokeStyle=this.colors.person;c.lineWidth=1.35*u;c.strokeRect(x,y,bw,bh);}
        c.globalAlpha=1;
        return;
      }
      if(this.diagnostic)for(const t of frame.raw||[]){c.setLineDash([3,3]);c.strokeStyle='#ffbb8888';const b=t.bbox;c.strokeRect(b[0]*w,b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h);c.setLineDash([]);}
      for(const t of frame.tracks||[]){
        if(t.category==='person'&&frame.people?.length&&this.peopleOverlay)continue;if(!t.confirmed&&!this.diagnostic)continue;if(this.options.focus&&t.uid!==this.selected)continue;const col=this.colors[t.category]||'#ccc',b=t.bbox;
        const dt=Math.max(0,Math.min(.3,(renderTime||frame.timestamp)-frame.timestamp));const dx=t.velocity[0]*dt,dy=t.velocity[1]*dt;
        let x=b[0]*w+dx,y=b[1]*h+dy,bw=(b[2]-b[0])*w,bh=(b[3]-b[1])*h;
        if(!this.diagnostic){
          if(editor?.active)continue;
          c.strokeStyle=col;c.lineWidth=(t.uid===this.selected?1.8:1)*u;
          c.strokeRect(x,y,bw,bh);
          if(!this.options.labels)continue;
          const speed=VehicleOverlay.speedReading(t,w,h);
          // No placeholders or pixel-speed labels in the clean live view.
          if(!speed.ready||speed.unit!=='km/h*'||speed.value<2)continue;
          const text=`~${speed.value.toFixed(0)} km/h${t.speed_method==='lane_average'?' avg':''}`;
          c.font=`600 ${11*u}px Segoe UI`;const tw=this.labelWidth(text,u)+12*u;
          const lx=Math.max(3*u,Math.min(w-tw-3*u,x+bw/2-tw/2));
          const ly=Math.max(23*u,Math.min(h-3*u,y-5*u));
          c.fillStyle='#08131cda';c.beginPath();c.roundRect(lx,ly-22*u,tw,24*u,5*u);c.fill();
          c.fillStyle='#f1fcff';c.fillText(text,lx+6*u,ly-6*u);
          continue;
        }
        if(this.options.trails&&t.history.length>1){for(let j=1;j<t.history.length;j++){const a=t.history[j-1],p=t.history[j];c.strokeStyle=col;c.globalAlpha=this.options.opacity*j/t.history.length*.6;c.beginPath();c.moveTo(a[1]*w,a[2]*h);c.lineTo(p[1]*w,p[2]*h);c.stroke();}c.globalAlpha=this.options.opacity;}
        c.lineWidth=(t.uid===this.selected?2.7:1.2)*u;c.strokeStyle=col;c.setLineDash(t.confirmed?[]:[3,3]);c.strokeRect(x,y,bw,bh);c.setLineDash([]);
        c.fillStyle=col+'12';if(t.uid===this.selected)c.fillRect(x,y,bw,bh);
        if(this.options.labels){c.font=`bold ${10*u}px Segoe UI`;let text=`${t.category.toUpperCase()} #${t.id}`;if(t.lane_name)text+=' · '+t.lane_name;if(this.options.confidence)text+=` ${(t.confidence*100).toFixed(0)}%`;const speed=VehicleOverlay.speedReading(t,w,h);let detail=speed.text;if(t.stopped_seconds>3)detail+=' · STOPPED';const tw=Math.max(c.measureText(text).width,c.measureText(detail).width)+10*u;const lx=Math.max(0,Math.min(w-tw,x)),ly=Math.max(30*u,y-3*u);c.fillStyle='#081622ef';c.fillRect(lx,ly-29*u,tw,31*u);c.fillStyle=col;c.fillText(text,lx+5*u,ly-17*u);c.fillStyle='#f1f7fb';c.fillText(detail,lx+5*u,ly-4*u);}
        this.arrow(x+bw/2,y+bh,Math.max(-15,Math.min(15,t.velocity[0]*.15)),Math.max(-15,Math.min(15,t.velocity[1]*.15)),col);
        if(this.diagnostic&&t.raw_anchor){c.beginPath();c.arc(t.raw_anchor[0]*w,t.raw_anchor[1]*h,3,0,7);c.fillStyle='#fff';c.fill();}
      }
      if(frame?.people?.length&&this.peopleOverlay){
        this.peopleOverlay.selectedId=this.selected;
        this.peopleOverlay.diagnostic=this.diagnostic;
        this.peopleOverlay.draw(frame,renderTime,this.options);
      }
    }c.globalAlpha=1;
  }
  hit(point){
    if(this.peopleOverlay){
      const hitPerson=this.peopleOverlay.hit(point);
      if(hitPerson)return hitPerson;
    }
    const tracks=this.peopleMode?(this.frame?.people||[]):(this.frame?.tracks||[]);
    return tracks.filter(t=>{const b=t.bbox;return point[0]>=b[0]&&point[0]<=b[2]&&point[1]>=b[1]&&point[1]<=b[3];}).sort((a,b)=>(a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1])-(b.bbox[2]-b.bbox[0])*(b.bbox[3]-b.bbox[1]))[0];
  }
};
