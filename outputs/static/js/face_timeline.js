'use strict';
(function(root,factory){if(typeof module==='object')module.exports=factory();else root.FaceTimeline=factory();})(globalThis,()=>class FaceTimeline{
  constructor(maxFrames=180){this.maxFrames=maxFrames;this.clear();}
  clear(){this.frames=[];this.epoch=null;this.receivedAt=0;}
  add(frame){
    if(!Number.isFinite(frame?.timestamp))return;
    if(frame.tracking_kind&&frame.tracking_kind!=='face')return;
    if(this.epoch!==frame.epoch){this.clear();this.epoch=frame.epoch;}
    const faces=new Map((frame.faces||[]).map(f=>[f.track_id??f.id,f]));
    const rawList=frame.tracks||frame.people||[];
    const people=rawList.map(t=>{
      const tid=t.track_id??t.id;
      const face=faces.get(tid)||t.face||null;
      return {...t,id:tid,track_id:tid,uid:`${frame.epoch}:${tid}`,face};
    });
    const entry={...frame,people};
    if(this.frames.length&&entry.timestamp<this.frames.at(-1).timestamp){
      const idx=this.frames.findIndex(f=>f.timestamp>entry.timestamp);
      if(idx>=0)this.frames.splice(idx,0,entry);else this.frames.push(entry);
    }else this.frames.push(entry);
    while(this.frames.length>this.maxFrames)this.frames.shift();
    this.receivedAt=Date.now();
  }
  sample(stamp,interpolate){
    if(!this.frames.length||!Number.isFinite(stamp))return null;
    let before=null,after=null,best=null;
    for(const f of this.frames){
      if(f.timestamp<=stamp&&(!before||f.timestamp>before.timestamp))before=f;
      if(f.timestamp>=stamp&&(!after||f.timestamp<after.timestamp))after=f;
      if(Math.abs(f.timestamp-stamp)<=1.2&&(!best||Math.abs(f.timestamp-stamp)<Math.abs(best.timestamp-stamp)))best=f;
    }
    if(!best)return null;
    // Exact 60 FPS sub-frame interpolation between two surrounding frames
    if(before&&after&&before!==after&&after.timestamp-before.timestamp<=1.25){
      const res=interpolate?interpolate(before,after,stamp):null;
      if(res)return res;
    }
    // Damped kinematic forward projection when live video slightly leads newest AI frame
    const last=this.frames.at(-1);
    const dt=stamp-last.timestamp;
    if(dt>0&&dt<=.40){
      const effDt=.28*(1-Math.exp(-dt/.28));
      const prev=this.frames.length>=2?this.frames.at(-2):null;
      const prevSpan=(prev&&last.timestamp>prev.timestamp&&last.timestamp-prev.timestamp<=1.0)?(last.timestamp-prev.timestamp):0;
      const prevById=prevSpan>0?new Map((prev.people||[]).map(q=>[q.uid??q.id??q.track_id,q])):null;
      const fw=last.width||0,fh=last.height||0;
      const shiftBox=(box,dxPx,dyPx,dxNorm,dyNorm)=>{
        if(!Array.isArray(box)||box.length<4)return [box,0,0];
        const isNorm=Math.max(Math.abs(box[0]),Math.abs(box[1]),Math.abs(box[2]),Math.abs(box[3]))<=1.05;
        const dx=isNorm?dxNorm:dxPx,dy=isNorm?dyNorm:dyPx;
        const bw=box[2]-box[0],bh=box[3]-box[1];
        let nx1=box[0]+dx,ny1=box[1]+dy;
        if(isNorm){
          nx1=Math.max(0,Math.min(Math.max(0,1-bw),nx1));
          ny1=Math.max(0,Math.min(Math.max(0,1-bh),ny1));
        }else if(fw>0&&fh>0){
          nx1=Math.max(0,Math.min(Math.max(0,fw-bw),nx1));
          ny1=Math.max(0,Math.min(Math.max(0,fh-bh),ny1));
        }
        return [[nx1,ny1,nx1+bw,ny1+bh],nx1-box[0],ny1-box[1]];
      };
      const projectedPeople=(last.people||[]).map(p=>{
        if(!p||p.confirmed===false)return p;
        const rawVel=(Array.isArray(p.face?.velocity)&&p.face.velocity.length>=2)?p.face.velocity:
          ((Array.isArray(p.velocity)&&p.velocity.length>=2)?p.velocity:null);
        let vxPx=0,vyPx=0,vxNorm=0,vyNorm=0,hasVel=false;
        if(rawVel&&Number.isFinite(rawVel[0])&&Number.isFinite(rawVel[1])&&(rawVel[0]!==0||rawVel[1]!==0)){
          const refBox=p.face?.bbox||p.bbox;
          const refIsNorm=Array.isArray(refBox)&&refBox.length>=4&&Math.max(Math.abs(refBox[0]),Math.abs(refBox[1]),Math.abs(refBox[2]),Math.abs(refBox[3]))<=1.05;
          if(refIsNorm&&Math.max(Math.abs(rawVel[0]),Math.abs(rawVel[1]))<=2.0){
            vxNorm=rawVel[0];vyNorm=rawVel[1];
            vxPx=fw>0?vxNorm*fw:vxNorm;vyPx=fh>0?vyNorm*fh:vyNorm;
          }else{
            vxPx=rawVel[0];vyPx=rawVel[1];
            vxNorm=fw>0?vxPx/fw:0;vyNorm=fh>0?vyPx/fh:0;
          }
          hasVel=true;
        }else if(prevById){
          const q=prevById.get(p.uid??p.id??p.track_id);
          const aBox=q?.face?.bbox||q?.bbox,bBox=p.face?.bbox||p.bbox;
          if(Array.isArray(aBox)&&Array.isArray(bBox)&&aBox.length>=4&&bBox.length>=4){
            const isNorm=Math.max(Math.abs(bBox[0]),Math.abs(bBox[1]),Math.abs(bBox[2]),Math.abs(bBox[3]))<=1.05;
            const dvx=((bBox[0]+bBox[2])-(aBox[0]+aBox[2]))*.5/prevSpan;
            const dvy=((bBox[1]+bBox[3])-(aBox[1]+aBox[3]))*.5/prevSpan;
            if(isNorm){
              vxNorm=dvx;vyNorm=dvy;
              vxPx=fw>0?dvx*fw:0;vyPx=fh>0?dvy*fh:0;
            }else{
              vxPx=dvx;vyPx=dvy;
              vxNorm=fw>0?dvx/fw:0;vyNorm=fh>0?dvy/fh:0;
            }
            hasVel=vxPx!==0||vyPx!==0||vxNorm!==0||vyNorm!==0;
          }
        }
        if(!hasVel)return p;
        const maxPxX=fw>0?fw*1.5:4000,maxPxY=fh>0?fh*1.5:4000;
        vxPx=Math.max(-maxPxX,Math.min(maxPxX,vxPx));
        vyPx=Math.max(-maxPxY,Math.min(maxPxY,vyPx));
        vxNorm=Math.max(-1.5,Math.min(1.5,vxNorm));
        vyNorm=Math.max(-1.5,Math.min(1.5,vyNorm));
        const dxPx=vxPx*effDt,dyPx=vyPx*effDt,dxNorm=vxNorm*effDt,dyNorm=vyNorm*effDt;
        const [bbox]=p.bbox?shiftBox(p.bbox,dxPx,dyPx,dxNorm,dyNorm):[p.bbox,0,0];
        let face=p.face||null;
        if(face?.bbox&&!face.is_locked&&!face.is_head_anchor){
          if(face.coordinate_space==='person'){
            face={...face};
          }else{
            const [fBox,adx,ady]=shiftBox(face.bbox,dxPx,dyPx,dxNorm,dyNorm);
            face={...face,bbox:fBox};
            if(Array.isArray(face.landmarks)&&face.landmarks.length){
              const fIsNorm=Math.max(Math.abs(face.bbox[0]),Math.abs(face.bbox[1]),Math.abs(face.bbox[2]),Math.abs(face.bbox[3]))<=1.05;
              face.landmarks=face.landmarks.map(pt=>{
                if(!pt)return pt;
                const lx=Array.isArray(pt)?pt[0]:pt.x,ly=Array.isArray(pt)?pt[1]:pt.y;
                const ptIsNorm=Math.max(Math.abs(lx),Math.abs(ly))<=1.05;
                const sdx=(ptIsNorm===fIsNorm)?adx:(ptIsNorm?dxNorm:dxPx);
                const sdy=(ptIsNorm===fIsNorm)?ady:(ptIsNorm?dyNorm:dyPx);
                return Array.isArray(pt)?[lx+sdx,ly+sdy]:{...pt,x:lx+sdx,y:ly+sdy};
              });
            }
          }
        }
        return {...p,bbox,face};
      });
      return {...last,timestamp:stamp,syncTimestamp:last.timestamp,people:projectedPeople};
    }
    return best;
  }
});
