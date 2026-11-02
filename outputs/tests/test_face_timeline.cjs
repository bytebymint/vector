const assert=require('node:assert/strict');
const FaceTimeline=require('../static/js/face_timeline.js');
const PeopleOverlay=require('../static/js/people_overlay.js');
const timeline=new FaceTimeline();
function frame(t,x,epoch='a'){return {tracking_kind:'face',epoch,timestamp:t,width:3840,height:2160,
  tracks:[{track_id:1,confirmed:true,tracking_kind:'face',bbox:[x,100,x+50,160]}],
  faces:[{track_id:1,bbox:[x,100,x+50,160],confidence:.9}]};}
timeline.add(frame(1,100));timeline.add(frame(1.4,140));
const sample=timeline.sample(1.2,PeopleOverlay.interpolatePeople);
assert.equal(sample.people.length,1);
assert.equal(sample.people[0].bbox[0],120);
assert.equal(sample.people[0].face.bbox[0],120);
assert.equal(sample.people[0].uid,'a:1');
assert.equal(timeline.sample(3,PeopleOverlay.interpolatePeople),null);
timeline.add(frame(4,300,'b'));
assert.equal(timeline.frames.length,1);
assert.equal(timeline.sample(1.2,PeopleOverlay.interpolatePeople),null);
for(let i=0;i<220;i++)timeline.add(frame(4+i*.1,300,'b'));
assert.equal(timeline.frames.length,180);
timeline.clear();assert.equal(timeline.frames.length,0);
const fs=require('node:fs');
assert.match(fs.readFileSync(require.resolve('../static/js/dashboard.js'),'utf8'),/socket.on\('people:frame'/);
console.log('Face metadata synchronization, interpolation, expiry and epoch reset passed');
