/**
 * Standalone Unit Tests for People Dashboard Frontend Engine
 * 
 * Verifies DOM structure, state management, Socket.IO event processing,
 * search/filter toolbar behavior, comparison view modes, and modal interactions.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

console.log('--- Running People Dashboard Unit Tests ---');

// 1. Verify index.html template integrity
const htmlPath = path.join(__dirname, '../templates/index.html');
const indexHtml = fs.readFileSync(htmlPath, 'utf8');

const requiredIds = [
  'people-control-bar',
  'people-live-indicator',
  'people-status-badge',
  'people-pipeline-info',
  'people-toggle-btn',

  'people-metrics',
  'people-visible',
  'people-active-tracks',
  'people-captures-total',
  'people-enhancements-total',
  'people-fps',

  'people-gallery-section',
  'people-profile-count',
  'gallery-sync-status',
  'people-toolbar',
  'people-search-input',
  'people-filter-status',
  'people-filter-quality',
  'people-filter-quality-val',
  'people-sort-by',

  'people-loading',
  'people-error-alert',
  'people-grid',
  'people-gallery-empty',

  'people-detail-modal',
  'modal-close-btn',
  'modal-profile-title',
  'modal-status-badge',
  'modal-quality-badge',
  'modal-duration-badge',
  'modal-session-id',

  'modal-view-side-by-side',
  'modal-view-toggle',
  'modal-view-slider',
  'modal-img-original',
  'modal-img-enhanced',
  'modal-img-toggle',
  'modal-toggle-view-btn',
  'modal-split-container',
  'modal-slider-orig',
  'modal-slider-enh',
  'modal-slider-handle',

  'modal-quality-metrics',
  'modal-angles-list',
  'modal-history-count',
  'modal-history-list',
  'modal-delete-profile-btn',
];

for (const id of requiredIds) {
  assert.ok(
    indexHtml.includes(`id="${id}"`),
    `index.html must include element with id="${id}"`
  );
}
console.log('✓ index.html contains all 38 required People Dashboard DOM element IDs');

// Check script tag inclusion
assert.ok(
  indexHtml.includes('src="/static/js/people_dashboard.js'),
  'index.html must include script tag for people_dashboard.js'
);
console.log('✓ index.html references people_dashboard.js script');

// Verify dashboard.css styling integrity
const cssPath = path.join(__dirname, '../static/css/dashboard.css');
const cssContent = fs.readFileSync(cssPath, 'utf8');
assert.ok(cssContent.includes('.people-card .card-thumb'), 'CSS must include .people-card .card-thumb selector');
assert.ok(cssContent.includes('.person-card .card-thumb'), 'CSS must include .person-card .card-thumb selector');
assert.ok(cssContent.includes('object-fit: cover'), 'CSS must include object-fit: cover for thumbnails');
assert.ok(cssContent.includes('image-rendering: crisp-edges'), 'CSS must include high-DPI crisp-edges rendering');
assert.ok(cssContent.includes('#modal-img-original'), 'CSS must include modal comparison portrait zoom rules');
console.log('✓ dashboard.css verified for anti-letterboxing, cover-fit, and crisp high-DPI thumbnails');

// 2. Mock DOM and browser environment
const elements = new Map();

function createMockElement(id = '', tag = 'div') {
  return {
    id,
    tagName: tag.toUpperCase(),
    value: '',
    textContent: '',
    innerHTML: '',
    hidden: false,
    className: '',
    style: {},
    dataset: {},
    attributes: {},
    children: [],
    classList: {
      _classes: new Set(),
      add(c) { this._classes.add(c); },
      remove(c) { this._classes.delete(c); },
      toggle(c, val) {
        if (val === undefined) {
          if (this._classes.has(c)) this._classes.delete(c);
          else this._classes.add(c);
        } else if (val) {
          this._classes.add(c);
        } else {
          this._classes.delete(c);
        }
      },
      contains(c) { return this._classes.has(c); },
    },
    getAttribute(name) { return this.attributes[name] ?? null; },
    setAttribute(name, val) { this.attributes[name] = String(val); },
    removeAttribute(name) { delete this.attributes[name]; },
    addEventListener(evt, fn) { this[`on_${evt}`] = fn; },
    removeEventListener() {},
    querySelector(selector) {
      if (selector === '.btn-icon') return createMockElement('', 'span');
      if (selector === '.btn-text') return createMockElement('', 'span');
      return null;
    },
    querySelectorAll(selector) {
      if (selector === '[data-cmp-mode]') {
        return [
          { dataset: { cmpMode: 'side-by-side' }, classList: { toggle() {} } },
          { dataset: { cmpMode: 'toggle' }, classList: { toggle() {} } },
          { dataset: { cmpMode: 'slider' }, classList: { toggle() {} } },
        ];
      }
      return [];
    },
    getBoundingClientRect() {
      return { top: 10, left: 10, width: 600, height: 400 };
    },
    showModal() { this.hidden = false; this.open = true; },
    close() { this.hidden = true; this.open = false; },
  };
}

function getElement(id) {
  if (!elements.has(id)) {
    elements.set(id, createMockElement(id));
  }
  return elements.get(id);
}

const mockDocument = {
  getElementById: id => getElement(id),
  querySelector: sel => {
    if (sel === 'meta[name="csrf-token"]') {
      return { getAttribute: () => 'mock-csrf-token-xyz' };
    }
    if (sel === '[data-view="people"]') return createMockElement('', 'button');
    return null;
  },
  querySelectorAll: () => [],
  body: createMockElement('body', 'body'),
};

const socketHandlers = new Map();
const mockSocket = {
  on(event, handler) {
    socketHandlers.set(event, handler);
  },
  emit(event, data) {
    this.lastEmitted = { event, data };
  },
};

const sandbox = {
  console,
  document: mockDocument,
  window: {
    addEventListener: () => {},
    MutationObserver: class {
      observe() {}
      disconnect() {}
    },
  },
  fetch: async () => ({
    ok: true,
    json: async () => ({}),
  }),
  socket: mockSocket,
  io: () => mockSocket,
};
sandbox.globalThis = sandbox;
sandbox.window.socket = mockSocket;

vm.createContext(sandbox);

// 3. Load people_dashboard.js into VM context
const jsPath = path.join(__dirname, '../static/js/people_dashboard.js');
const jsCode = fs.readFileSync(jsPath, 'utf8');
vm.runInContext(jsCode, sandbox, { filename: 'people_dashboard.js' });

const PeopleDashboardClass = sandbox.PeopleDashboard;
assert.ok(PeopleDashboardClass, 'PeopleDashboard class must be exported');
console.log('✓ people_dashboard.js loaded and PeopleDashboard class available');

// 4. Test instantiation and DOM binding
const dashboard = new PeopleDashboardClass();
dashboard.init();

assert.ok(dashboard.initialized, 'Dashboard must be initialized');
assert.ok(dashboard.dom.controlBar, 'Control bar element must be bound');
assert.ok(dashboard.dom.metricFps, 'Metric FPS must be bound');
assert.ok(dashboard.dom.grid, 'Grid element must be bound');
assert.ok(dashboard.dom.modal, 'Modal element must be bound');
console.log('✓ Dashboard initialized and elements successfully bound');

// 5. Test Start / Stop People Analysis
assert.equal(dashboard.isRunning, false, 'Initial running state should be false');

dashboard.setAnalysisRunning(true);
assert.equal(dashboard.isRunning, true, 'Running state should be true');
assert.equal(getElement('people-status-badge').textContent, 'ACTIVE');
assert.ok(getElement('people-live-indicator').classList.contains('active'), 'Indicator must have active class');

dashboard.setAnalysisRunning(false);
assert.equal(dashboard.isRunning, false, 'Running state should be false');
assert.equal(getElement('people-status-badge').textContent, 'STOPPED');
console.log('✓ Start / Stop Analysis toggles state, badge, and live indicator');

// 6. Test Socket.IO Event: people:state
assert.ok(socketHandlers.has('people:state'), 'Must register people:state socket event');
const stateHandler = socketHandlers.get('people:state');

stateHandler({
  is_running: true,
  fps: 18.5,
  active_counts: {
    active_tracks: 3,
    total_profiles: 12,
    total_captures: 45,
  },
  enhancements_done: 8,
  status_message: 'Real-time multi-angle tracking',
});

assert.equal(dashboard.isRunning, true);
assert.equal(dashboard.metrics.fps, 18.5);
assert.equal(dashboard.metrics.active_tracks, 3);
assert.equal(dashboard.metrics.total_captures, 45);
assert.equal(dashboard.metrics.enhancements_done, 8);
assert.equal(getElement('people-fps').textContent, '18.5');
assert.equal(getElement('people-active-tracks').textContent, '3');
assert.equal(getElement('people-captures-total').textContent, '45');
assert.equal(getElement('people-enhancements-total').textContent, '8');
console.log('✓ people:state event updates metrics and control indicators');
socketHandlers.get('people:frame')({tracks:[{confirmed:true,state:'active'},{confirmed:false,state:'active'},{confirmed:true,state:'lost'}]});
assert.equal(dashboard.metrics.visible,1,'Actual backend tracks update the visible count');

// 7. Test Socket.IO Event: people:profile_update
assert.ok(socketHandlers.has('people:profile_update'), 'Must register people:profile_update socket event');
const profileHandler = socketHandlers.get('people:profile_update');

profileHandler({
  session_id: 'session-alpha',
  metadata: {face_validation_version:2},
  track_id: 101,
  camera_id: 'RC-264',
  first_seen: 100.0,
  last_seen: 108.4,
  duration: 8.4,
  status: 'active',
  best_quality_score: 92.4,
  best_capture: {
    capture_id: 'cap-01',
    original_url: '/api/people/captures/cap-01/image?type=original',
    enhanced_url: null,
    is_enhanced: false,
    quality_score: 92.4,
    crop_width: 160,
    crop_height: 160,
    category: 'frontal',
  },
  captures_by_category: {
    frontal: {
      capture_id: 'cap-01',
      original_url: '/api/people/captures/cap-01/image?type=original',
      quality_score: 92.4,
      crop_width: 160,
      crop_height: 160,
    },
    left_angle: {
      capture_id: 'cap-02',
      original_url: '/api/people/captures/cap-02/image?type=original',
      quality_score: 84.0,
      crop_width: 140,
      crop_height: 140,
    },
  },
  capture_history: [
    {
      capture_id: 'cap-01',
      timestamp: 108.0,
      category: 'frontal',
      quality_score: 92.4,
      original_url: '/api/people/captures/cap-01/image?type=original',
    },
  ],
});

assert.equal(dashboard.profiles.size, 1);
const p = dashboard.profiles.get('session-alpha');
assert.ok(p);
assert.equal(p.track_id, 101);
assert.equal(p.best_quality_score, 92.4);
assert.ok(getElement('people-grid').innerHTML.includes('session-alpha'));
assert.ok(getElement('people-grid').innerHTML.includes('Frontal'));
assert.ok(getElement('people-grid').innerHTML.includes('Q: 92 ★'), 'Quality score badge must display Q: 92 ★ format');
assert.ok(getElement('people-grid').innerHTML.includes('card-angle-badge'), 'Card must include card-angle-badge');

// Verify renderCard alias
assert.equal(typeof dashboard.renderCard, 'function', 'renderCard alias must exist');
const renderedCardHtml = dashboard.renderCard(p);
assert.equal(renderedCardHtml, dashboard.renderProfileCard(p), 'renderCard output must equal renderProfileCard');

// Test sub-second duration formatting (<1s instead of 0.0s)
const subSecProfile = {
  session_id: 'session-sub-sec',
  metadata: {face_validation_version:2},
  track_id: 105,
  duration: 0.04,
  status: 'active',
  best_quality_score: 88,
  best_capture: {
    capture_id: 'cap-sub',
    yaw: -24.6,
    quality_score: 88,
    original_url: '/api/people/captures/cap-sub/image',
  },
};
const subSecHtml = dashboard.renderCard(subSecProfile);
assert.ok(subSecHtml.includes('⏱ &lt;1s') || subSecHtml.includes('⏱ <1s'), 'Duration < 0.1s must format as <1s');
assert.ok(subSecHtml.includes('Left 25°'), 'Pose angle yaw -24.6 must format as Left 25°');
assert.ok(subSecHtml.includes('Q: 88 ★'), 'Quality rating must display Q: 88 ★');
console.log('✓ people:profile_update renders gallery card with <1s duration, yaw pose angle, Q: 88 ★ quality score');

// 8. Test Socket.IO Event: people:capture
assert.ok(socketHandlers.has('people:capture'), 'Must register people:capture socket event');
const captureHandler = socketHandlers.get('people:capture');

captureHandler({
  session_id: 'session-alpha',
  metadata: {face_validation_version:2},
  capture: {
    capture_id: 'cap-03',
    session_id: 'session-alpha',
  metadata: {face_validation_version:2},
    category: 'right_angle',
    quality_score: 88.5,
    timestamp: 110.0,
    original_url: '/api/people/captures/cap-03/image?type=original',
  },
});

assert.equal(p.capture_history.length, 2);
assert.ok(p.captures_by_category.right_angle);
assert.equal(dashboard.metrics.total_captures, 46);
console.log('✓ people:capture appends multi-angle capture to profile and increments capture counter');

// 9. Test Socket.IO Event: people:enhancement_ready
assert.ok(socketHandlers.has('people:enhancement_ready'), 'Must register people:enhancement_ready socket event');
const enhancementHandler = socketHandlers.get('people:enhancement_ready');

enhancementHandler({
  session_id: 'session-alpha',
  metadata: {face_validation_version:2},
  capture_id: 'cap-01',
  enhanced_url: '/api/people/captures/cap-01/image?type=enhanced',
  enhancement_method: 'GFPGAN',
  is_enhanced: true,
});

assert.equal(p.best_capture.is_enhanced, true);
assert.equal(p.best_capture.enhanced_url, '/api/people/captures/cap-01/image?type=enhanced');
assert.equal(dashboard.metrics.enhancements_done, 9);
console.log('✓ people:enhancement_ready updates enhanced image URL and method');

// 10. Test Filtering & Search
// Add a second profile
profileHandler({
  session_id: 'session-beta',
  metadata: {face_validation_version:2},
  track_id: 102,
  duration: 2.1,
  status: 'completed',
  best_quality_score: 45.0,
});

assert.equal(dashboard.profiles.size, 2);

// Filter by status: active
dashboard.filters.status = 'active';
let filtered = dashboard.getFilteredProfiles();
assert.equal(filtered.length, 1);
assert.equal(filtered[0].session_id, 'session-alpha');

// Filter by status: completed
dashboard.filters.status = 'completed';
filtered = dashboard.getFilteredProfiles();
assert.equal(filtered.length, 1);
assert.equal(filtered[0].session_id, 'session-beta');

// Filter by minQuality: 80
dashboard.filters.status = 'all';
dashboard.filters.minQuality = 80;
filtered = dashboard.getFilteredProfiles();
assert.equal(filtered.length, 1);
assert.equal(filtered[0].session_id, 'session-alpha');

// Search by query
dashboard.filters.minQuality = 0;
dashboard.filters.search = 'beta';
filtered = dashboard.getFilteredProfiles();
assert.equal(filtered.length, 1);
assert.equal(filtered[0].session_id, 'session-beta');

// Reset filters
dashboard.filters.search = '';
console.log('✓ Filter & Search toolbar handles status, min-quality, and search queries');

// 11. Test Modal Opening and Comparison Modes
dashboard.openProfileModal('session-alpha');
assert.equal(dashboard.selectedProfileId, 'session-alpha');
assert.equal(getElement('modal-session-id').textContent, 'session-alpha');
assert.equal(getElement('modal-profile-title').textContent, 'Profile #101');
assert.ok(getElement('modal-img-original').src.includes('cap-01'));
assert.ok(getElement('modal-img-enhanced').src.includes('type=enhanced'));

// Test comparison mode switcher
dashboard.setComparisonMode('toggle');
assert.equal(dashboard.comparisonMode, 'toggle');
assert.equal(getElement('modal-view-toggle').hidden, false);
assert.equal(getElement('modal-view-side-by-side').hidden, true);

dashboard.setComparisonMode('slider');
assert.equal(dashboard.comparisonMode, 'slider');
assert.equal(getElement('modal-view-slider').hidden, false);

// Test split slider positioning
dashboard.setSplitSliderPosition(70);
assert.equal(dashboard.splitPosition, 70);
assert.equal(getElement('modal-slider-enh-wrapper').style.width, '70%');
assert.equal(getElement('modal-slider-handle').style.left, '70%');

// Test close modal
dashboard.closeProfileModal();
assert.equal(dashboard.selectedProfileId, null);
console.log('✓ Profile Details Modal handles side-by-side, toggle, and split slider comparison');

// 12. Test Profile Deletion
async function testDeletion() {
  let deleteConfirmed = false;
  sandbox.confirm = () => {
    deleteConfirmed = true;
    return true;
  };
  sandbox.fetch = async (url, opts) => {
    if (opts && opts.method === 'DELETE') {
      return { ok: true, json: async () => ({ deleted: true }) };
    }
    return { ok: true, json: async () => ({}) };
  };

  dashboard.openProfileModal('session-beta');
  await dashboard.deleteProfile('session-beta');
  assert.ok(deleteConfirmed, 'Confirmation prompt must be shown');
  assert.equal(dashboard.profiles.has('session-beta'), false, 'Deleted profile must be removed');
  console.log('✓ Profile deletion removes profile and triggers gallery re-render');

  console.log('\n======================================================');
  console.log('ALL PEOPLE DASHBOARD TESTS PASSED SUCCESSFULLY! (12/12)');
  console.log('======================================================\n');
}

testDeletion();

// Historical false positives remain accessible, but are excluded by default.
dashboard.profiles.set('legacy',{session_id:'legacy',metadata:{},track_id:99});
dashboard.filters.search='';dashboard.filters.status='all';dashboard.filters.minQuality=0;
assert.ok(!dashboard.getFilteredProfiles().some(p=>p.session_id==='legacy'));
dashboard.filters.includeLegacy=true;
assert.ok(dashboard.getFilteredProfiles().some(p=>p.session_id==='legacy'));
console.log('Verified-face gallery default and legacy opt-in passed');
