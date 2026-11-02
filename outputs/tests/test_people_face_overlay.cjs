/**
 * Comprehensive verification of PeopleOverlay and facial capture visualization.
 * Exercises:
 *  - Emerald / teal rounded bounding boxes
 *  - Anonymous tracking ID pills (#P-XX, duration, hits)
 *  - Facial bounding boxes with cyan vs gold thresholds
 *  - Pose badges ("Frontal", "Left 25°", "Right 30°")
 *  - Face quality tags ("Q: 85 ★" with green/yellow/red color coding)
 *  - Pulsating / flashing "Captured!" indicator badges
 *  - 5-point facial landmark dots (eyes, nose, mouth)
 *  - Active vs temporarily lost tracking states
 *  - Coordinate normalization & aspect ratio preservation
 *  - Text width caching & GC optimization
 *  - Seamless integration with VehicleOverlay
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const PeopleOverlay = require('../static/js/people_overlay.js');

// Mock Canvas 2D Context tracking rendering calls
function createMockContext() {
  const calls = {
    roundRect: 0,
    strokeRect: 0,
    fillRect: 0,
    fillText: [],
    arc: 0,
    setLineDash: [],
    strokes: 0,
    fills: 0,
    saved: 0,
    restored: 0
  };

  const ctx = {
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
    fillText(text, x, y) { calls.fillText.push(text); },
    save() { calls.saved++; },
    restore() { calls.restored++; },
    stroke() { calls.strokes++; },
    fill() { calls.fills++; },
    roundRect(x, y, w, h, r) { calls.roundRect++; },
    strokeRect(x, y, w, h) { calls.strokeRect++; },
    fillRect(x, y, w, h) { calls.fillRect++; },
    arc(x, y, r, sa, ea) { calls.arc++; },
    setLineDash(dash) { calls.setLineDash.push([...dash]); },
    measureText(text) {
      return { width: text.length * 7.5 };
    }
  };

  return ctx;
}

function createMockCanvas(width = 1280, height = 720) {
  const ctx = createMockContext();
  return {
    width,
    height,
    getContext: (type) => (type === '2d' ? ctx : null),
    getBoundingClientRect: () => ({
      left: 0,
      top: 0,
      width: 1280,
      height: 720
    })
  };
}

// -------------------------------------------------------------
// Test 1: Instantiation, Defaults, and Resize Cache Invalidation
// -------------------------------------------------------------
{
  const canvas = createMockCanvas(1280, 720);
  const overlay = new PeopleOverlay(canvas);

  assert.equal(overlay.canvas.width, 1280);
  assert.equal(overlay.canvas.height, 720);
  assert.equal(overlay.options.showPersonBbox, true);
  assert.equal(overlay.options.showFaceBbox, true);
  assert.equal(overlay.options.showPose, true);
  assert.equal(overlay.options.showQuality, true);
  assert.equal(overlay.options.showLandmarks, true);
  assert.equal(overlay.options.highQualityThreshold, 70);

  // Measure text and check caching
  const w1 = overlay.measureTextWidth('Hello World', '12px sans-serif', canvas.getContext('2d'));
  assert.ok(w1 > 0);
  assert.equal(overlay.textWidthCache.size, 1);

  // Measure again - should be cache hit
  const w2 = overlay.measureTextWidth('Hello World', '12px sans-serif', null);
  assert.equal(w1, w2);

  // Resize canvas - should invalidate cache
  overlay.size(1920, 1080);
  assert.equal(overlay.canvas.width, 1920);
  assert.equal(overlay.canvas.height, 1080);
  assert.equal(overlay.textWidthCache.size, 0);

  console.log('✓ Test 1: Instantiation, options, and resize cache passed');
}

// -------------------------------------------------------------
// Test 2: Coordinate Normalization and Dynamic Video Scaling
// -------------------------------------------------------------
{
  const canvas = createMockCanvas(1920, 1080);
  const overlay = new PeopleOverlay(canvas);

  // Test normalized coordinates [0, 1]
  const normBox = [0.1, 0.2, 0.3, 0.8]; // x1, y1, x2, y2
  const pixelBox = overlay.normalizeBbox(normBox, 1920, 1080, null, null);
  assert.ok(Math.abs(pixelBox[0] - 192) < 1e-6);       // x
  assert.ok(Math.abs(pixelBox[1] - 216) < 1e-6);       // y
  assert.ok(Math.abs(pixelBox[2] - 384) < 1e-6);       // w = (0.3 - 0.1) * 1920
  assert.ok(Math.abs(pixelBox[3] - 648) < 1e-6);       // h = (0.8 - 0.2) * 1080

  // Test pixel coordinates with frame scaling (e.g. source 1280x720 displayed on 1920x1080)
  const rawBox = [100, 100, 300, 500]; // 1280x720 frame coordinates
  const scaledBox = overlay.normalizeBbox(rawBox, 1920, 1080, 1280, 720);
  assert.equal(scaledBox[0], 100 * 1.5);
  assert.equal(scaledBox[1], 100 * 1.5);
  assert.equal(scaledBox[2], 200 * 1.5);
  assert.equal(scaledBox[3], 400 * 1.5);

  // Test parent-relative face box (e.g. normalized crop relative to person bounding box)
  const faceRelBox = [0.2, 0.1, 0.8, 0.4];
  const absFaceBox = overlay.normalizeBbox(faceRelBox, 1920, 1080, null, null, pixelBox);
  assert.ok(Math.abs(absFaceBox[0] - (pixelBox[0] + 0.2 * pixelBox[2])) < 1e-6);
  assert.ok(Math.abs(absFaceBox[1] - (pixelBox[1] + 0.1 * pixelBox[3])) < 1e-6);
  assert.ok(Math.abs(absFaceBox[2] - (0.6 * pixelBox[2])) < 1e-6);
  assert.ok(Math.abs(absFaceBox[3] - (0.3 * pixelBox[3])) < 1e-6);

  // Test landmark normalization
  const normPt = [0.25, 0.4];
  const pixelPt = overlay.normalizePoint(normPt, 1920, 1080, null, null);
  assert.equal(pixelPt[0], 480);
  assert.equal(pixelPt[1], 432);

  console.log('✓ Test 2: Coordinate normalization and aspect scaling passed');
}

// -------------------------------------------------------------
// Test 3: Rendering Complete Person + Face + Pose + Quality + Landmarks
// -------------------------------------------------------------
{
  const canvas = createMockCanvas(1000, 1000);
  const overlay = new PeopleOverlay(canvas);
  const ctx = canvas.getContext('2d');

  const frame = {
    timestamp: 100.0,
    width: 1000,
    height: 1000,
    people: [
      {
        id: 14,
        confirmed: true,
        bbox: [0.1, 0.1, 0.4, 0.8],
        hits: 24,
        duration: 3.5,
        state: 'active',
        face: {
          bbox: [0.18, 0.12, 0.32, 0.32],
          confidence: 0.94,
          pose: { yaw: 24.5, pitch: 3.0, roll: 0.5, category: 'right_angle' },
          quality: { score: 85.0, sharpness: 140, is_acceptable: true },
          landmarks: [
            [0.22, 0.20], // left eye
            [0.28, 0.20], // right eye
            [0.25, 0.25], // nose
            [0.23, 0.28], // left mouth
            [0.27, 0.28]  // right mouth
          ]
        }
      }
    ]
  };

  overlay.draw(frame, 100.0);

  // Check roundRect calls (person box, ID pill, face box, pose badge, quality badge)
  assert.ok(ctx.calls.roundRect >= 4, `expected at least 4 roundRect calls, got ${ctx.calls.roundRect}`);

  // Check landmark arc calls (5 landmarks x 2 for halo + dot = 10 arc calls)
  assert.ok(ctx.calls.arc >= 10, `expected 10 landmark arcs, got ${ctx.calls.arc}`);

  // Check labels rendered
  const texts = ctx.calls.fillText;
  // Must include ID pill
  assert.ok(texts.some(t => typeof t === 'string' && t.includes('#P-14')), 'Should render #P-14 pill');
  // Must include duration or hits
  assert.ok(texts.some(t => typeof t === 'string' && (t.includes('3.5s') || t.includes('24h'))), 'Should render duration/hits');
  // Must include Pose badge
  assert.ok(texts.some(t => typeof t === 'string' && t.includes('Right 25°')), 'Should render Right 25° pose badge');
  // Must include Quality badge with ★ for score 85
  assert.ok(texts.some(t => typeof t === 'string' && t.includes('Q: 85 ★')), 'Should render Q: 85 ★ high quality badge');

  console.log('✓ Test 3: Person, Face, Pose, Quality, Landmarks rendering verified');
}

// -------------------------------------------------------------
// Test 4: Quality Score Color Grading & Thresholds
// -------------------------------------------------------------
{
  const canvas = createMockCanvas(800, 600);
  const overlay = new PeopleOverlay(canvas);
  const ctx = canvas.getContext('2d');

  // Test three individuals: High Quality (88 >= 70), Medium Quality (55 >= 40), Low Quality (32 < 40)
  const frame = {
    timestamp: 10,
    people: [
      { id: 1, confirmed: true, bbox: [0.1, 0.1, 0.3, 0.5], face: { bbox:[.14,.12,.26,.25], quality: { score: 88 } } },
      { id: 2, confirmed: true, bbox: [0.4, 0.1, 0.6, 0.5], face: { bbox:[.44,.12,.56,.25], quality: { score: 55 } } },
      { id: 3, confirmed: true, bbox: [0.7, 0.1, 0.9, 0.5], face: { bbox:[.74,.12,.86,.25], quality: { score: 32 } } }
    ]
  };

  overlay.draw(frame, 10);
  const texts = ctx.calls.fillText;

  assert.ok(texts.some(t => t.includes('Q: 88 ★')), 'High score must include star');
  assert.ok(texts.some(t => t.includes('Q: 55') && !t.includes('★')), 'Medium score must not include star');
  assert.ok(texts.some(t => t.includes('Q: 32') && !t.includes('★')), 'Low score must not include star');

  console.log('✓ Test 4: Quality score grading and badges verified');
}

// -------------------------------------------------------------
// Test 5: Active vs Temporarily Lost Tracking State
// -------------------------------------------------------------
{
  const canvas = createMockCanvas(800, 600);
  const overlay = new PeopleOverlay(canvas);
  const ctx = canvas.getContext('2d');

  const frame = {
    timestamp: 15,
    people: [
      { id: 10, confirmed: true, state: 'lost', duration: 4.2, bbox: [0.2, 0.2, 0.4, 0.7] }
    ]
  };

  overlay.draw(frame, 15);

  // For lost state: line dash should have been set with dashed pattern
  const dashes = ctx.calls.setLineDash;
  assert.ok(dashes.some(d => d.length > 0), 'Lost track must render dashed border');

  // Pill must show LOST tag
  const texts = ctx.calls.fillText;
  assert.ok(texts.some(t => t.includes('LOST')), 'Pill must indicate LOST status');

  console.log('✓ Test 5: Temporarily lost tracking state styling verified');
}

// -------------------------------------------------------------
// Test 6: Pulsating / Flashing "Captured!" Badge Animation
// -------------------------------------------------------------
{
  const canvas = createMockCanvas(800, 600);
  const overlay = new PeopleOverlay(canvas);
  const ctx = canvas.getContext('2d');

  // Explicitly trigger a capture event
  overlay.triggerCapture(42, {
    quality_score: 91,
    category: 'frontal',
    capture_id: 'cap_42_01'
  });

  const frame = {
    timestamp: 20,
    people: [
      { id: 42, confirmed: true, bbox: [0.3, 0.3, 0.6, 0.8], face: { bbox: [0.4, 0.35, 0.5, 0.5] } }
    ]
  };

  overlay.draw(frame, 20);

  // Captured badge should be rendered
  const texts = ctx.calls.fillText;
  assert.ok(texts.some(t => t.includes('CAPTURED! 91★')), 'Must render pulsating 📸 CAPTURED! 91★ badge');

  console.log('✓ Test 6: Capture pulse animation trigger and rendering verified');
}

// -------------------------------------------------------------
// Test 7: Hit Testing & Point Translation
// -------------------------------------------------------------
{
  const canvas = createMockCanvas(800, 600);
  const overlay = new PeopleOverlay(canvas);

  const frame = {
    people: [
      { id: 7, bbox: [0.1, 0.2, 0.3, 0.7] },
      { id: 8, bbox: [0.5, 0.2, 0.8, 0.7] }
    ]
  };
  overlay.draw(frame, 1);

  // Hit inside person #7
  const hit1 = overlay.hit([0.2, 0.4]);
  assert.ok(hit1 !== null);
  assert.equal(hit1.id, 7);

  // Hit inside person #8
  const hit2 = overlay.hit([0.6, 0.5]);
  assert.ok(hit2 !== null);
  assert.equal(hit2.id, 8);

  // Hit outside
  const hit3 = overlay.hit([0.05, 0.05]);
  assert.equal(hit3, null);

  console.log('✓ Test 7: Normalized hit testing verified');
}

// -------------------------------------------------------------
// Test 8: Integration with VehicleOverlay
// -------------------------------------------------------------
{
  const sandbox = {
    window: {},
    Path2D: class {}
  };
  sandbox.PeopleOverlay = PeopleOverlay;
  sandbox.window.PeopleOverlay = PeopleOverlay;
  vm.createContext(sandbox);

  // Load overlay.js in the sandbox where PeopleOverlay is available
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/overlay.js'), 'utf8'), sandbox);

  const mockCanvas = createMockCanvas(1000, 1000);
  const vehicleOverlay = new sandbox.window.VehicleOverlay(mockCanvas);

  assert.ok(vehicleOverlay.peopleOverlay !== null, 'VehicleOverlay must instantiate peopleOverlay when available');

  vehicleOverlay.peopleMode = true;
  const frame = {
    timestamp: 50,
    people: [
      {
        id: 99,
        confirmed: true,
        bbox: [0.2, 0.2, 0.5, 0.7],
        hits: 15,
        duration: 2.1,
        face: {
          bbox: [0.3, 0.25, 0.4, 0.4],
          quality: { score: 78 },
          pose: { yaw: 0, category: 'frontal' }
        }
      }
    ]
  };

  vehicleOverlay.draw(frame, { items: [] }, null, 50);

  const ctx = mockCanvas.getContext('2d');
  const texts = ctx.calls.fillText;

  assert.ok(texts.some(t => t.includes('#P-99')), 'VehicleOverlay delegation should render #P-99');
  assert.ok(texts.some(t => t.includes('Frontal')), 'VehicleOverlay delegation should render Frontal pose');
  assert.ok(texts.some(t => t.includes('Q: 78 ★')), 'VehicleOverlay delegation should render Q: 78 ★');

  console.log('✓ Test 8: VehicleOverlay seamless delegation verified');
}

// -------------------------------------------------------------
// Test 9: Interpolation of People Tracks with Facial Data
// -------------------------------------------------------------
{
  const before = {
    timestamp: 10.0,
    people: [
      {
        id: 20,
        confirmed: true,
        bbox: [0.1, 0.2, 0.3, 0.8],
        face: { bbox: [0.15, 0.25, 0.25, 0.45] }
      }
    ]
  };

  const after = {
    timestamp: 10.4,
    people: [
      {
        id: 20,
        confirmed: true,
        bbox: [0.2, 0.2, 0.4, 0.8],
        face: { bbox: [0.25, 0.25, 0.35, 0.45] }
      }
    ]
  };

  const interpolated = PeopleOverlay.interpolatePeople(before, after, 10.2);
  assert.ok(interpolated !== null);
  assert.equal(interpolated.timestamp, 10.2);

  const p = interpolated.people[0];
  assert.ok(Math.abs(p.bbox[0] - 0.15) < 1e-6);
  assert.ok(Math.abs(p.bbox[2] - 0.35) < 1e-6);
  assert.ok(Math.abs(p.face.bbox[0] - 0.20) < 1e-6);

  console.log('✓ Test 9: People and face temporal interpolation verified');
}

console.log('\nAll PeopleOverlay comprehensive tests passed successfully!');
