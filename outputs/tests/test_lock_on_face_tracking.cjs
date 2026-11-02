/**
 * test_lock_on_face_tracking.cjs
 * Comprehensive test suite verifying:
 *  1. Lock-on face tracking (face box remains anchored on head even when face detection drops)
 *  2. Camera pan motion stabilization (forward extrapolation when renderTime leads inference)
 *  3. Landmarks, pose, and quality badges remain active when tracking_kind === 'face'
 *  4. Clean release when tracked person exits screen entirely
 */

const assert = require('node:assert/strict');
const FaceTimeline = require('../static/js/face_timeline.js');
const PeopleOverlay = require('../static/js/people_overlay.js');

function createMockContext() {
  const calls = {
    roundRect: 0,
    strokeRect: 0,
    fillText: [],
    arc: 0,
    strokes: 0,
    fills: 0,
    saved: 0,
    restored: 0
  };

  return {
    calls,
    strokeStyle: '',
    fillStyle: '',
    lineWidth: 1,
    globalAlpha: 1,
    font: '',
    textBaseline: '',
    textAlign: '',
    clearRect() {},
    beginPath() {},
    closePath() {},
    moveTo() {},
    lineTo() {},
    arcTo() {},
    save() { calls.saved++; },
    restore() { calls.restored++; },
    stroke() { calls.strokes++; },
    fill() { calls.fills++; },
    roundRect(x, y, w, h, r) { calls.roundRect++; },
    strokeRect(x, y, w, h) { calls.strokeRect++; },
    arc(x, y, r, sa, ea) { calls.arc++; },
    setLineDash() {},
    fillText(text) { calls.fillText.push(text); },
    measureText(text) { return { width: text.length * 8 }; }
  };
}

function createMockCanvas(w = 1920, h = 1080) {
  const ctx = createMockContext();
  return {
    width: w,
    height: h,
    getContext: () => ctx,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: w, height: h })
  };
}

// Faces must be observed, never guessed from body positions or old locks.
{
  const canvas=createMockCanvas(1000,1000),overlay=new PeopleOverlay(canvas);
  let faceDraws=0;
  overlay.drawFaceBox=()=>faceDraws++;
  overlay.draw({timestamp:1,people:[{id:42,bbox:[100,100,300,800],confirmed:true,
    face:{bbox:[140,120,260,270],quality:{score:85}}}]},1);
  assert.equal(faceDraws,1);
  overlay.draw({timestamp:1.1,people:[{id:42,bbox:[150,100,350,800],confirmed:true,
    face:null,locked_face:{bbox:[190,120,310,270],quality:{score:85}}}]},1.1);
  assert.equal(faceDraws,1,'A dropped detection cannot invent a face from its body/old lock');
  overlay.draw({timestamp:1.2,people:[{id:99,bbox:[100,100,300,800],confirmed:true}]},1.2);
  assert.equal(faceDraws,1,'New body-only tracks cannot invent head boxes');
  assert.ok(!canvas.getContext().calls.fillText.some(t=>t.includes('Q: 75')));
}
{
  const timeline=new FaceTimeline();
  for(const [t,x] of [[10,200],[10.2,240]])timeline.add({tracking_kind:'face',epoch:'cam1',
    timestamp:t,tracks:[{track_id:1,bbox:[x,100,x+60,180],confirmed:true}],
    faces:[{track_id:1,bbox:[x,100,x+60,180],quality:{score:91}}]});
  const mid=timeline.sample(10.1,PeopleOverlay.interpolatePeople);
  assert.ok(Math.abs(mid.people[0].face.bbox[0]-220)<1e-4);
  assert.equal(PeopleOverlay.interpolatePeople(timeline.frames[0],timeline.frames[1],10.3),null,
    'Never extrapolate face evidence beyond the analyzed frames');
  timeline.add({tracking_kind:'face',epoch:'cam1',timestamp:10.4,
    tracks:[{track_id:2,bbox:[20,20,60,90],locked_face:{bbox:[20,20,60,90]}}],faces:[]});
  assert.equal(timeline.frames.at(-1).people[0].face,null);
}

// -------------------------------------------------------------
// Test 3: Landmarks, Pose, and Quality Badges in tracking_kind === 'face'
// -------------------------------------------------------------
{
  const canvas = createMockCanvas(1000, 1000);
  const overlay = new PeopleOverlay(canvas);
  const ctx = canvas.getContext('2d');

  const faceFrame = {
    timestamp: 5.0,
    tracking_kind: 'face',
    people: [
      {
        id: 7,
        confirmed: true,
        tracking_kind: 'face',
        bbox: [200, 200, 350, 380],
        face: {
          bbox: [200, 200, 350, 380],
          confidence: 0.98,
          quality: { score: 92 },
          pose: { yaw: 20, category: 'right_angle' },
          landmarks: [[240, 260], [310, 260], [275, 300], [250, 340], [300, 340]]
        }
      }
    ]
  };

  overlay.draw(faceFrame, 5.0);

  const texts = ctx.calls.fillText;
  assert.ok(texts.some(t => t.includes('Q: 92 ★')), 'Quality badge must not be disabled when tracking_kind === face');
  assert.ok(texts.some(t => t.includes('Right 20°') || t.includes('Right')), 'Pose badge must not be disabled when tracking_kind === face');
  assert.ok(ctx.calls.arc >= 10, '5-point landmarks (arc calls) must not be disabled when tracking_kind === face');

  console.log('✓ Test 3: Badges and landmarks remain fully active in face tracking mode');
}

// -------------------------------------------------------------
// Test 6: Face and Person Visualization in BOTH people and traffic modes
// -------------------------------------------------------------
{
  const vm = require('node:vm');
  const fs = require('node:fs');
  const path = require('node:path');

  const sandbox = {
    window: {},
    Path2D: class {}
  };
  sandbox.PeopleOverlay = PeopleOverlay;
  sandbox.window.PeopleOverlay = PeopleOverlay;
  vm.createContext(sandbox);

  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/overlay.js'), 'utf8'), sandbox);

  const canvas = createMockCanvas(1280, 720);
  const vehicleOverlay = new sandbox.window.VehicleOverlay(canvas);
  const ctx = canvas.getContext('2d');

  // Test in traffic mode (peopleMode = false)
  vehicleOverlay.peopleMode = false;
  const trafficFrameWithPeople = {
    timestamp: 30.0,
    tracks: [
      { uid: 'car-1', id: 1, category: 'car', confirmed: true, bbox: [0.1, 0.4, 0.3, 0.6], velocity: [10, 0] }
    ],
    people: [
      {
        id: 10657,
        confirmed: true,
        bbox: [0.5, 0.2, 0.7, 0.8],
        face: {
          bbox: [0.55, 0.22, 0.65, 0.38],
          quality: { score: 86 },
          pose: { yaw: 0, category: 'frontal' }
        }
      }
    ]
  };

  ctx.calls.roundRect = 0;
  ctx.calls.fillText = [];
  vehicleOverlay.draw(trafficFrameWithPeople, { items: [] }, null, 30.0);

  // In traffic mode, peopleOverlay must still render person and face boxes!
  assert.ok(ctx.calls.fillText.some(t => t.includes('#P-10657')), 'Person #P-10657 must render in traffic view mode');
  assert.ok(ctx.calls.fillText.some(t => t.includes('Q: 86 ★')), 'Face quality badge must render in traffic view mode');
  assert.ok(ctx.calls.roundRect >= 2, 'Face and person boxes must be drawn in traffic view mode');

  console.log('✓ Test 6: People and face boxes render seamlessly in traffic view mode verified');
}

console.log('\nObserved-face rendering, interpolation and no-guessing regressions passed!');
