/**
 * VECTOR · Traffic Intelligence & Facial Capture System
 * People Dashboard Frontend Engine
 * 
 * Manages real-time people detection, facial angle captures,
 * quality assessment scores, AI face enhancement previews,
 * and biometric profile gallery management.
 */
'use strict';

(function(root, factory) {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory();
  } else {
    root.PeopleDashboard = factory();
    if (typeof window !== 'undefined') {
      window.addEventListener('DOMContentLoaded', () => {
        window.peopleDashboard = new root.PeopleDashboard();
        window.peopleDashboard.init();
      });
    }
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {

  const DEFAULT_AVATAR = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="%23223848"><rect width="100" height="100" fill="%230b1722"/><circle cx="50" cy="38" r="20" fill="%2338576e"/><path d="M22 88 c0-18 14-28 28-28 s28 10 28 28 Z" fill="%2338576e"/></svg>';

  const esc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));

  const num = (v, n = 0) => v == null ? '—' : Number(v).toFixed(n);

  const formatDuration = sec => {
    if (sec == null || isNaN(sec)) return '<1s';
    const s = Number(sec);
    if (s < 0.1) return '<1s';
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = Math.floor(s / 60);
    const rem = (s % 60).toFixed(0);
    return `${m}m ${rem}s`;
  };

  const formatTime = ts => {
    if (!ts) return '—';
    try {
      const d = new Date(ts > 1e11 ? ts : ts * 1000);
      return d.toLocaleTimeString([], { hour12: false });
    } catch (e) {
      return '—';
    }
  };

  class PeopleDashboard {
    constructor(options = {}) {
      this.options = options;
      this.initialized = false;
      this.isRunning = false;
      this.userStopped = false;
      this.profiles = new Map(); // sessionId -> PersonProfile
      this.metrics = {
        visible: 0,
        active_tracks: 0,
        total_captures: 0,
        enhancements_done: 0,
        fps: 0.0,
        latency_ms: null,
      };
      this.filters = {
        includeLegacy: false,
        search: '',
        status: 'all',
        minQuality: 0,
        sortBy: 'newest',
      };
      this.selectedProfileId = null;
      this.activeCaptureId = null;
      this.comparisonMode = 'side-by-side';
      this.toggleShowingEnhanced = false;
      this.splitPosition = 50;
      this.socket = null;
      this.pollTimer = null;
      this.isDraggingSlider = false;
    }

    getCsrfToken() {
      if (typeof document === 'undefined') return '';
      const meta = document.querySelector('meta[name="csrf-token"]');
      return meta ? meta.getAttribute('content') : '';
    }

    async api(path, method = 'GET', body = null) {
      if (typeof fetch === 'undefined') return null;
      const headers = {
        'Accept': 'application/json',
      };
      if (body !== null) {
        headers['Content-Type'] = 'application/json';
        headers['X-CSRF-Token'] = this.getCsrfToken();
      }
      const response = await fetch(path, {
        method,
        headers,
        body: body !== null ? JSON.stringify(body) : undefined,
      });
      if (!response.ok) {
        let errMessage = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const data = await response.json();
          if (data && data.error) errMessage = data.error;
        } catch (e) {}
        const error = new Error(errMessage);
        error.status = response.status;
        throw error;
      }
      return await response.json();
    }

    init() {
      if (typeof document === 'undefined') return;
      if (this.initialized) return;
      this.initialized = true;

      this.bindElements();
      this.setupSocket();
      this.setupEventListeners();
      this.setupTabObserver();

      // Initial check & load
      this.checkTabState();
      this.fetchState().catch(() => {});
      this.fetchProfiles().catch(() => {});
    }

    bindElements() {
      const get = id => (typeof document !== 'undefined' ? document.getElementById(id) : null);
      this.dom = {
        controlBar: get('people-control-bar'),
        liveIndicator: get('people-live-indicator'),
        statusBadge: get('people-status-badge'),
        pipelineInfo: get('people-pipeline-info'),
        toggleBtn: get('people-toggle-btn'),

        metricVisible: get('people-visible'),
        metricActive: get('people-active-tracks'),
        metricCaptures: get('people-captures-total'),
        metricEnhancements: get('people-enhancements-total'),
        metricFps: get('people-fps'),
        metricLatency: get('people-latency'),

        gallerySection: get('people-gallery-section'),
        profileCount: get('people-profile-count'),
        syncStatus: get('gallery-sync-status'),
        searchInput: get('people-search-input'),
        searchClear: get('people-search-clear'),
        filterStatus: get('people-filter-status'),
        filterQuality: get('people-filter-quality'),
        includeLegacy: get('people-include-legacy'),
        filterQualityVal: get('people-filter-quality-val'),
        sortBy: get('people-sort-by'),

        loading: get('people-loading'),
        errorAlert: get('people-error-alert'),
        errorText: get('people-error-text'),
        errorDismiss: get('people-error-dismiss'),

        grid: get('people-grid'),
        emptyState: get('people-gallery-empty'),

        modal: get('people-detail-modal'),
        modalCloseBtn: get('modal-close-btn'),
        modalTitle: get('modal-profile-title'),
        modalStatusBadge: get('modal-status-badge'),
        modalQualityBadge: get('modal-quality-badge'),
        modalDurationBadge: get('modal-duration-badge'),
        modalSessionId: get('modal-session-id'),

        viewSideBySide: get('modal-view-side-by-side'),
        viewToggle: get('modal-view-toggle'),
        viewSlider: get('modal-view-slider'),
        imgOriginal: get('modal-img-original'),
        imgEnhanced: get('modal-img-enhanced'),
        imgOrigMeta: get('modal-img-orig-meta'),
        imgEnhMeta: get('modal-img-enh-meta'),
        imgToggle: get('modal-img-toggle'),
        toggleViewBtn: get('modal-toggle-view-btn'),
        toggleStateLabel: get('modal-toggle-state-label'),

        splitContainer: get('modal-split-container'),
        sliderOrig: get('modal-slider-orig'),
        sliderEnh: get('modal-slider-enh'),
        sliderEnhWrapper: get('modal-slider-enh-wrapper'),
        sliderHandle: get('modal-slider-handle'),

        qualityMetricsGrid: get('modal-quality-metrics'),
        anglesList: get('modal-angles-list'),
        historyCount: get('modal-history-count'),
        historyList: get('modal-history-list'),
        deleteProfileBtn: get('modal-delete-profile-btn'),
      };
    }

    setupSocket() {
      if (typeof window === 'undefined') return;
      let socketInstance = null;
      try {
        if (typeof socket !== 'undefined' && socket && typeof socket.on === 'function') {
          socketInstance = socket;
        }
      } catch (e) {}
      if (!socketInstance && window.__peopleSocket) {
        socketInstance = window.__peopleSocket;
      }
      if (!socketInstance && typeof window.io === 'function') {
        socketInstance = window.io({ transports: ['websocket', 'polling'] });
        window.__peopleSocket = socketInstance;
      }
      this.socket = socketInstance;

      if (!this.socket) return;

      this.socket.on('people:state', data => this.handlePeopleState(data));
      this.socket.on('people:capture', data => this.handlePeopleCapture(data));
      this.socket.on('people:profile_update', data => this.handlePeopleProfileUpdate(data));
      this.socket.on('people:enhancement_ready', data => this.handlePeopleEnhancementReady(data));
      this.socket.on('people:frame', data => {
        const rows = data?.tracks || data?.persons;
        if (Array.isArray(rows)) {
          this.lastPeopleFrameAt = Date.now();
          if(data.tracking_kind==='face'){
            if(this.faceEpoch!==data.epoch){this.faceEpoch=data.epoch;this.faceSeen=new Set();}
            const confirmed=rows.filter(t=>t.confirmed!==false&&t.state!=='lost');
            confirmed.forEach(t=>this.faceSeen.add(t.track_id));
            for(const id of ['people-unique','people-panel-unique']){const el=document.getElementById(id);if(el)el.textContent=String(this.faceSeen.size);}
            const count=document.getElementById('people-track-count');if(count)count.textContent=String(confirmed.length);
            const empty=document.getElementById('people-empty');if(empty)empty.hidden=!!confirmed.length;
            const table=document.getElementById('people-track-table');if(table)table.innerHTML=confirmed.map(t=>`<tr><td>#F-${Number(t.track_id)}</td><td>Confirmed face</td><td>${Math.round(Number(t.confidence)*100)}%</td><td>${Number(t.duration||0).toFixed(1)}s</td></tr>`).join('');
            const active=document.getElementById('people-active-tracks');if(active)active.textContent=String(confirmed.length);
          }
          const status=document.getElementById('people-status');
          if(status)status.textContent=`${data.coverage?.detected_faces||0} detected · ${data.coverage?.pending_confirmation||0} verifying`;
          if(this.dom.pipelineInfo)this.dom.pipelineInfo.textContent=data.coverage?.error ? `Face detector error: ${data.coverage.error}` : 'Multi-scale face detection · native captures · best-frame upgrades';
          this.metrics.visible = rows.filter(t => t.confirmed !== false && t.state !== 'lost').length;
          if (this.dom.metricVisible) {
            this.dom.metricVisible.textContent = String(this.metrics.visible);
          }
          const sideCount = document.getElementById('people-panel-visible');
          if (sideCount) sideCount.textContent = String(this.metrics.visible);
          if (data.fps !== undefined && this.dom.metricFps) {
            this.metrics.fps = Number(data.fps) || 0.0;
            this.dom.metricFps.textContent = num(this.metrics.fps, 1);
          }
          const latMs = data.total_latency_ms ?? data.latency_ms ?? data.metrics?.total_latency_ms;
          if (latMs !== undefined && latMs !== null && Number.isFinite(Number(latMs))) {
            this.metrics.latency_ms = Number(latMs);
          }
          if (this.dom.metricLatency) {
            if (this.metrics.latency_ms != null && Number.isFinite(this.metrics.latency_ms)) {
              this.dom.metricLatency.textContent = this.metrics.fps > 0
                ? `${Math.round(this.metrics.latency_ms)} ms · ${num(this.metrics.fps, 1)} FPS`
                : `${Math.round(this.metrics.latency_ms)} ms`;
            } else if (this.metrics.fps > 0) {
              this.dom.metricLatency.textContent = `${Math.round(1000 / this.metrics.fps)} ms · ${num(this.metrics.fps, 1)} FPS`;
            }
          }
        }
      });

      // Also listen to general state to sync visible people if provided
      this.socket.on('state', data => {
        if (data && data.people && !this.isRunning) {
          if (data.people.visible !== undefined && this.dom.metricVisible) {
            this.metrics.visible = Number(data.people.visible) || 0;
            this.dom.metricVisible.textContent = String(this.metrics.visible);
          }
        }
      });
    }

    setupEventListeners() {
      const d = this.dom;
      if (!d) return;

      // Start / Stop toggle
      if (d.toggleBtn) {
        d.toggleBtn.onclick = () => this.toggleAnalysis();
      }

      if(d.includeLegacy)d.includeLegacy.onchange=()=>{this.filters.includeLegacy=d.includeLegacy.checked;this.renderGallery();};

      // Search input
      if (d.searchInput) {
        d.searchInput.oninput = () => {
          this.filters.search = d.searchInput.value.trim().toLowerCase();
          if (d.searchClear) {
            d.searchClear.hidden = !this.filters.search;
          }
          this.renderGallery();
        };
      }
      if (d.searchClear) {
        d.searchClear.onclick = () => {
          d.searchInput.value = '';
          this.filters.search = '';
          d.searchClear.hidden = true;
          this.renderGallery();
        };
      }

      // Filter by status
      if (d.filterStatus) {
        d.filterStatus.onchange = () => {
          this.filters.status = d.filterStatus.value;
          this.renderGallery();
        };
      }

      // Quality filter slider
      if (d.filterQuality) {
        if (d.filterQuality.value !== undefined) {
          const initVal = Number(d.filterQuality.value) || 0;
          this.filters.minQuality = initVal;
          if (d.filterQualityVal) {
            d.filterQualityVal.textContent = initVal > 0 ? `${initVal}+` : '0';
          }
        }
        d.filterQuality.oninput = () => {
          const val = Number(d.filterQuality.value) || 0;
          this.filters.minQuality = val;
          if (d.filterQualityVal) {
            d.filterQualityVal.textContent = val > 0 ? `${val}+` : '0';
          }
          this.renderGallery();
        };
      }

      // Sort by
      if (d.sortBy) {
        d.sortBy.onchange = () => {
          this.filters.sortBy = d.sortBy.value;
          this.renderGallery();
        };
      }

      // Error alert dismiss
      if (d.errorDismiss) {
        d.errorDismiss.onclick = () => this.hideError();
      }

      // Modal close
      if (d.modalCloseBtn) {
        d.modalCloseBtn.onclick = () => this.closeProfileModal();
      }
      if (d.modal) {
        d.modal.addEventListener('click', e => {
          if (typeof d.modal.getBoundingClientRect !== 'function') return;
          const rect = d.modal.getBoundingClientRect();
          const isInDialog = (
            rect.top <= e.clientY && e.clientY <= rect.top + rect.height &&
            rect.left <= e.clientX && e.clientX <= rect.left + rect.width
          );
          if (!isInDialog) {
            this.closeProfileModal();
          }
        });
      }

      // Comparison mode switcher buttons
      if (d.modal && typeof d.modal.querySelectorAll === 'function') {
        const modeButtons = d.modal.querySelectorAll('[data-cmp-mode]');
        modeButtons.forEach(btn => {
          btn.onclick = () => {
            const mode = btn.dataset.cmpMode;
            this.setComparisonMode(mode);
          };
        });
      }

      // Toggle comparison view button
      if (d.toggleViewBtn) {
        d.toggleViewBtn.onclick = () => this.toggleComparisonView();
      }

      // Interactive split slider drag
      if (d.splitContainer && typeof window !== 'undefined') {
        const onMove = e => {
          if (!this.isDraggingSlider) return;
          const rect = d.splitContainer.getBoundingClientRect();
          const clientX = e.touches ? e.touches[0].clientX : e.clientX;
          const pos = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
          this.setSplitSliderPosition(pos);
        };
        const onStart = e => {
          this.isDraggingSlider = true;
          onMove(e);
        };
        const onEnd = () => {
          this.isDraggingSlider = false;
        };

        d.splitContainer.addEventListener('mousedown', onStart);
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onEnd);

        d.splitContainer.addEventListener('touchstart', onStart, { passive: true });
        window.addEventListener('touchmove', onMove, { passive: true });
        window.addEventListener('touchend', onEnd);
      }

      // Delete profile button
      if (d.deleteProfileBtn) {
        d.deleteProfileBtn.onclick = () => {
          if (this.selectedProfileId) {
            this.deleteProfile(this.selectedProfileId);
          }
        };
      }

      // Gallery grid delegated clicks
      if (d.grid) {
        d.grid.addEventListener('click', e => {
          const card = e.target.closest ? e.target.closest('[data-profile-session]') : null;
          if (card && card.dataset) {
            const sessionId = card.dataset.profileSession;
            this.openProfileModal(sessionId);
          }
        });
      }
    }

    setupTabObserver() {
      if (typeof document === 'undefined') return;

      const peopleNav = document.querySelector('[data-view="people"]');
      if (peopleNav) {
        peopleNav.addEventListener('click', () => {
          setTimeout(() => this.checkTabState(), 20);
        });
      }
      document.querySelectorAll('.nav[data-tab]').forEach(btn => {
        btn.addEventListener('click', () => {
          setTimeout(() => this.checkTabState(), 20);
        });
      });

      if (typeof MutationObserver !== 'undefined' && document.body) {
        const observer = new MutationObserver(() => this.checkTabState());
        observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
      }
    }

    checkTabState() {
      if (typeof document === 'undefined' || !document.body) return;
      const isPeople = document.body.classList.contains('people-mode');
      if (this.dom.controlBar) {
        this.dom.controlBar.hidden = !isPeople;
      }
      if (this.dom.gallerySection) {
        this.dom.gallerySection.hidden = !isPeople;
      }
      if (isPeople && this.profiles.size === 0) {
        this.fetchProfiles().catch(() => {});
      }
    }

    async fetchState() {
      try {
        const data = await this.api('/api/people/state');
        if (data) this.handlePeopleState(data);
      } catch (e) {
        // Fall back gracefully if endpoint not yet mounted
      }
    }

    async fetchProfiles() {
      this.setLoading(true);
      try {
        const res = await this.api('/api/people/profiles');
        const list = Array.isArray(res) ? res : (res && Array.isArray(res.profiles) ? res.profiles : []);
        if (Array.isArray(list)) {
          this.profiles.clear();
          for (const item of list) {
            if (item && item.session_id) {
              this.profiles.set(item.session_id, item);
            }
          }
          this.renderGallery();
        }
      } catch (e) {
        // Fallback: keep existing in-memory profiles
      } finally {
        this.setLoading(false);
      }
    }

    async startAnalysis() {
      if (this.isRunning) return;
      this.userStopped = false;
      return this.toggleAnalysis(true);
    }

    async toggleAnalysis(forceRun = null) {
      const willRun = typeof forceRun === 'boolean' ? forceRun : !this.isRunning;
      this.userStopped = !willRun;
      this.setAnalysisRunning(willRun);

      const action = willRun ? 'start' : 'stop';
      try {
        await this.api('/api/people/control', 'POST', { action });
      } catch (err) {
        try {
          await this.api(`/api/people/analysis/${action}`, 'POST', {});
        } catch (err2) {
          let handled = false;
          if (this.socket) {
            this.socket.emit(willRun ? 'people:start' : 'people:stop', {});
            handled = true;
          }
          if (!handled) {
            this.setAnalysisRunning(!willRun);
            this.showError(`Failed to ${willRun ? 'start' : 'stop'} analysis: ${err.message}`);
          }
        }
      }
    }

    setAnalysisRunning(running, statusMsg = null) {
      this.isRunning = !!running;
      const d = this.dom;
      if (!d) return;

      if (d.liveIndicator) {
        d.liveIndicator.classList.toggle('active', this.isRunning);
      }

      if (d.statusBadge) {
        d.statusBadge.textContent = this.isRunning ? 'ACTIVE' : 'STOPPED';
        d.statusBadge.className = `people-status-badge ${this.isRunning ? 'active' : 'stopped'}`;
      }

      if (d.pipelineInfo) {
        d.pipelineInfo.textContent = statusMsg || (
          this.isRunning ? 'Real-time multi-angle facial capture active' : 'Engine idle · Awaiting activation'
        );
      }

      if (d.toggleBtn) {
        const icon = d.toggleBtn.querySelector ? d.toggleBtn.querySelector('.btn-icon') : null;
        const text = d.toggleBtn.querySelector ? d.toggleBtn.querySelector('.btn-text') : null;
        if (icon) icon.textContent = this.isRunning ? '■' : '▶';
        if (text) text.textContent = this.isRunning ? 'Stop People Analysis' : 'Start People Analysis';
        d.toggleBtn.classList.toggle('running', this.isRunning);
      }

      if (d.syncStatus) {
        d.syncStatus.textContent = this.isRunning ? 'LIVE' : 'IDLE';
        d.syncStatus.classList.toggle('active', this.isRunning);
      }
    }

    handlePeopleState(data) {
      if (!data) return;
      if(this.isRunning && this.lastPeopleFrameAt && Date.now()-this.lastPeopleFrameAt>3000){
        this.metrics.visible=0;
        if(this.dom.metricVisible)this.dom.metricVisible.textContent='0';
        const side=document.getElementById('people-panel-visible');if(side)side.textContent='0';
      }

      const isRunning = data.is_running !== undefined ? data.is_running : (data.state && data.state.is_running);
      if (typeof isRunning === 'boolean') {
        this.setAnalysisRunning(isRunning, data.status_message || (data.state && data.state.status_message));
      }

      const fps = data.fps !== undefined ? data.fps : (data.state && data.state.fps);
      if (fps !== undefined) {
        this.metrics.fps = Number(fps) || 0.0;
      }

      const activeTracks = data.active_tracks_count !== undefined
        ? data.active_tracks_count
        : (data.active_counts ? data.active_counts.active_tracks : (data.state && data.state.active_tracks_count));
      if (activeTracks !== undefined) {
        this.metrics.active_tracks = Number(activeTracks) || 0;
      }

      const totalCaptures = data.total_captures_count !== undefined
        ? data.total_captures_count
        : (data.active_counts ? data.active_counts.total_captures : (data.state && data.state.total_captures_count));
      if (totalCaptures !== undefined) {
        this.metrics.total_captures = Number(totalCaptures) || 0;
      }

      const totalProfiles = data.total_profiles_count !== undefined
        ? data.total_profiles_count
        : (data.active_counts ? data.active_counts.total_profiles : (data.state && data.state.total_profiles_count));
      if (totalProfiles !== undefined && this.dom.profileCount) {
        this.dom.profileCount.textContent = String(totalProfiles);
      }

      if (data.enhancements_done !== undefined) {
        this.metrics.enhancements_done = Number(data.enhancements_done) || 0;
      }

      const latMs = data.total_latency_ms ?? data.latency_ms
        ?? (data.state && (data.state.total_latency_ms ?? data.state.latency_ms))
        ?? (data.metrics && (data.metrics.total_latency_ms ?? (
          Number.isFinite(data.metrics.face_detection_ms)
            ? (Number(data.metrics.detection_ms || 0) + Number(data.metrics.tracking_ms || 0) + Number(data.metrics.face_detection_ms || 0) + Number(data.metrics.quality_eval_ms || 0))
            : undefined
        )));
      if (latMs !== undefined && latMs !== null && Number.isFinite(Number(latMs)) && Number(latMs) > 0) {
        this.metrics.latency_ms = Number(latMs);
      }

      if (data.error) {
        this.showError(data.error);
      } else {
        this.hideError();
      }

      this.updateMetricsUI();
    }

    handlePeopleCapture(data) {
      if (!data) return;
      this.metrics.total_captures++;
      if (this.dom.metricCaptures) {
        this.dom.metricCaptures.textContent = String(this.metrics.total_captures);
      }

      const cap = data.capture || data;
      const sessionId = cap.session_id || data.session_id || (data.profile && data.profile.session_id);
      if (sessionId) {
        let profile = this.profiles.get(sessionId);
        if (!profile && data.profile) {
          profile = data.profile;
          this.profiles.set(sessionId, profile);
        } else if (profile) {
          if (cap && cap.capture_id) {
            if (!profile.capture_history) profile.capture_history = [];
            profile.capture_history.unshift(cap);
            if (!profile.captures_by_category) profile.captures_by_category = {};
            if (cap.category) {
              profile.captures_by_category[cap.category] = cap;
            }
            if (!profile.best_capture || (cap.quality_score || 0) > (profile.best_quality_score || 0)) {
              profile.best_capture = cap;
              profile.best_quality_score = cap.quality_score;
            }
          }
        }
        this.renderGallery();
        if (this.selectedProfileId === sessionId) {
          this.refreshModalContent();
        }
      }
    }

    handlePeopleProfileUpdate(data) {
      if (!data) return;
      const list = Array.isArray(data) ? data : [data];
      for (const item of list) {
        if (!item || !item.session_id) continue;
        const existing = this.profiles.get(item.session_id) || {};
        const merged = { ...existing, ...item };
        this.profiles.set(item.session_id, merged);

        if (merged.status === 'active') {
          this.metrics.active_tracks = Math.max(1, this.metrics.active_tracks);
        }
      }

      this.renderGallery();

      if (this.selectedProfileId && this.profiles.has(this.selectedProfileId)) {
        this.refreshModalContent();
      }
    }

    handlePeopleEnhancementReady(data) {
      if (!data) return;
      this.metrics.enhancements_done++;
      if (this.dom.metricEnhancements) {
        this.dom.metricEnhancements.textContent = String(this.metrics.enhancements_done);
      }

      const sessionId = data.session_id;
      const captureId = data.capture_id;
      if (sessionId && this.profiles.has(sessionId)) {
        const profile = this.profiles.get(sessionId);
        if (profile.best_capture && (profile.best_capture.capture_id === captureId || !captureId)) {
          profile.best_capture.is_enhanced = true;
          profile.best_capture.enhanced_url = data.enhanced_url || profile.best_capture.enhanced_url;
          profile.best_capture.enhancement_method = data.enhancement_method || data.method || 'ESPCN-x2';
        }
        if (profile.capture_history) {
          for (const c of profile.capture_history) {
            if (c.capture_id === captureId) {
              c.is_enhanced = true;
              c.enhanced_url = data.enhanced_url || c.enhanced_url;
              c.enhancement_method = data.enhancement_method || data.method || 'ESPCN-x2';
            }
          }
        }
        if (profile.captures_by_category) {
          for (const key of Object.keys(profile.captures_by_category)) {
            const c = profile.captures_by_category[key];
            if (c.capture_id === captureId) {
              c.is_enhanced = true;
              c.enhanced_url = data.enhanced_url || c.enhanced_url;
              c.enhancement_method = data.enhancement_method || data.method || 'ESPCN-x2';
            }
          }
        }

        this.renderGallery();
        if (this.selectedProfileId === sessionId) {
          this.refreshModalContent();
        }
      }
    }

    updateMetricsUI() {
      const d = this.dom;
      if (!d) return;

      if (d.metricActive) d.metricActive.textContent = String(this.metrics.active_tracks);
      if (d.metricCaptures) d.metricCaptures.textContent = String(this.metrics.total_captures);
      if (d.metricEnhancements) d.metricEnhancements.textContent = String(this.metrics.enhancements_done);
      if (d.metricFps) d.metricFps.textContent = this.metrics.fps > 0 ? num(this.metrics.fps, 1) : '—';
      if (d.metricLatency) {
        if (this.metrics.latency_ms != null && Number.isFinite(this.metrics.latency_ms)) {
          d.metricLatency.textContent = this.metrics.fps > 0
            ? `${Math.round(this.metrics.latency_ms)} ms · ${num(this.metrics.fps, 1)} FPS`
            : `${Math.round(this.metrics.latency_ms)} ms`;
        } else if (this.metrics.fps > 0) {
          d.metricLatency.textContent = `${Math.round(1000 / this.metrics.fps)} ms · ${num(this.metrics.fps, 1)} FPS`;
        }
      }
      if (d.profileCount) d.profileCount.textContent = String(this.profiles.size);
    }

    getFilteredProfiles() {
      let items = Array.from(this.profiles.values());
      if(!this.filters.includeLegacy)items=items.filter(p=>Number(p.metadata?.face_validation_version)>=2);

      // Search filter
      if (this.filters.search) {
        const q = this.filters.search;
        items = items.filter(p => {
          const sid = String(p.session_id || '').toLowerCase();
          const tid = String(p.track_id || '').toLowerCase();
          const cid = String(p.camera_id || '').toLowerCase();
          return sid.includes(q) || tid.includes(q) || cid.includes(q);
        });
      }

      // Status filter
      if (this.filters.status !== 'all') {
        items = items.filter(p => {
          if (this.filters.status === 'active') return p.status === 'active';
          return p.status !== 'active';
        });
      }

      // Quality score filter
      if (this.filters.minQuality > 0) {
        items = items.filter(p => {
          const score = Number(p.best_quality_score || (p.best_capture && p.best_capture.quality_score) || 0);
          return score >= this.filters.minQuality;
        });
      }

      // Sorting
      items.sort((a, b) => {
        if (this.filters.sortBy === 'highest_quality') {
          const qa = Number(a.best_quality_score || (a.best_capture && a.best_capture.quality_score) || 0);
          const qb = Number(b.best_quality_score || (b.best_capture && b.best_capture.quality_score) || 0);
          return qb - qa;
        }
        if (this.filters.sortBy === 'duration') {
          return (Number(b.duration) || 0) - (Number(a.duration) || 0);
        }
        if (this.filters.sortBy === 'captures') {
          const ca = a.capture_history ? a.capture_history.length : (a.capture_history_count || 0);
          const cb = b.capture_history ? b.capture_history.length : (b.capture_history_count || 0);
          return cb - ca;
        }
        // default: newest
        const ta = Number(a.last_seen || a.first_seen || 0);
        const tb = Number(b.last_seen || b.first_seen || 0);
        return tb - ta;
      });

      return items;
    }

    renderGallery() {
      const d = this.dom;
      if (!d || !d.grid) return;
      // Keep the data current while expanded video owns the screen; rebuilding
      // hundreds of hidden image cards on every capture competes with decoding.
      if(document.body?.classList?.contains('stage-expanded'))return;

      const filtered = this.getFilteredProfiles();
      const profiles = filtered.slice(0,60);

      if (d.profileCount) {
        d.profileCount.textContent = String(this.profiles.size);
        d.profileCount.title = `Showing ${profiles.length} of ${filtered.length} matching loaded profiles`;
      }

      if (profiles.length === 0) {
        d.grid.innerHTML = '';
        if (d.emptyState) d.emptyState.hidden = false;
        return;
      }

      if (d.emptyState) d.emptyState.hidden = true;

      d.grid.innerHTML = profiles.map(p => this.renderProfileCard(p)).join('');
    }

    renderCard(profile) {
      return this.renderProfileCard(profile);
    }

    renderProfileCard(profile) {
      const sid = esc(profile.session_id);
      const tid = profile.track_id != null ? `#${profile.track_id}` : sid;
      const status = profile.status || 'active';
      const isActive = status === 'active';
      const duration = formatDuration(profile.duration);
      const score = Math.round(Number(profile.best_quality_score || (profile.best_capture && profile.best_capture.quality_score) || 0));

      let scoreClass = 'score-high';
      if (score < 50) scoreClass = 'score-low';
      else if (score < 75) scoreClass = 'score-med';

      const bestCap = profile.best_capture || {};
      const hasEnhanced = !!(bestCap.is_enhanced || bestCap.enhanced_url);
      const origUrl = bestCap.original_url || DEFAULT_AVATAR;
      const imgUrl = bestCap.enhanced_url || origUrl;
      const imgOnError = (bestCap.enhanced_url && bestCap.original_url && bestCap.enhanced_url !== bestCap.original_url)
        ? `if(this.dataset.fb){this.onerror=null;this.src='${esc(DEFAULT_AVATAR)}';}else{this.dataset.fb='1';this.src='${esc(origUrl)}';}`
        : `this.onerror=null;this.src='${esc(DEFAULT_AVATAR)}';`;
      const nativeLabel = bestCap.crop_width && bestCap.crop_height ? `${bestCap.crop_width} × ${bestCap.crop_height} px` : "Native crop";

      // Compute best pose angle label
      let angleLabel = 'Frontal';
      if (bestCap.yaw != null && !isNaN(bestCap.yaw)) {
        const yawVal = Number(bestCap.yaw);
        if (Math.abs(yawVal) < 15) {
          angleLabel = 'Frontal';
        } else if (yawVal < -15) {
          angleLabel = `Left ${Math.abs(Math.round(yawVal))}°`;
        } else {
          angleLabel = `Right ${Math.round(yawVal)}°`;
        }
      } else if (bestCap.category || bestCap.pose_category) {
        const cat = (bestCap.category || bestCap.pose_category).toLowerCase();
        if (cat.includes('left')) angleLabel = 'Left 25°';
        else if (cat.includes('right')) angleLabel = 'Right 25°';
        else if (cat.includes('profile')) angleLabel = 'Profile';
        else angleLabel = 'Frontal';
      }

      // Detected angles
      const angles = profile.captures_by_category || {};
      const hasFrontal = !!angles.frontal;
      const hasLeft = !!angles.left_angle;
      const hasRight = !!angles.right_angle;
      const hasHighRes = !!angles.high_res;

      const leftLabel = (angles.left_angle && angles.left_angle.yaw != null && !isNaN(angles.left_angle.yaw))
        ? `Left ${Math.abs(Math.round(angles.left_angle.yaw))}°`
        : 'Left ◖';
      const rightLabel = (angles.right_angle && angles.right_angle.yaw != null && !isNaN(angles.right_angle.yaw))
        ? `Right ${Math.round(angles.right_angle.yaw)}°`
        : 'Right ◗';

      const capCount = profile.capture_history ? profile.capture_history.length : (profile.capture_history_count || (profile.best_capture ? 1 : 0));

      return `
        <article class="person-card ${isActive ? 'card-active' : 'card-inactive'}" data-profile-session="${sid}" tabindex="0" role="button" aria-label="Inspect anonymous capture for ${sid}">
          <div class="card-thumb-container">
            <div class="card-thumb-backdrop" style="background-image: url('${esc(imgUrl)}')"></div>
            <img class="card-thumb" src="${esc(imgUrl)}" alt="Face Portrait for ${sid}" loading="lazy" decoding="async" onerror="${imgOnError}">
            <div class="card-thumb-overlay">
              <div class="card-thumb-top-left">
                <span class="card-status-pill ${isActive ? 'active' : 'inactive'}">${isActive ? 'ACTIVE' : 'COMPLETED'}</span>
                <span class="card-angle-badge" title="Capture Pose: ${esc(angleLabel)}">${esc(angleLabel)}</span>
              </div>
              <div class="card-thumb-top-right">
                <span class="card-score-badge ${scoreClass}" title="Quality Score: ${score}/100">Q: ${score}${score >= 75 ? " ★" : ""}</span>
              </div>
              <span class="card-thumb-badge ${hasEnhanced ? 'card-enhanced-badge' : 'card-hd-badge'}" title="${hasEnhanced ? 'Enhanced studio portrait · ' : 'Actual source crop dimensions: '}${esc(nativeLabel)}">${hasEnhanced ? `AI HD · ${esc(nativeLabel)}` : esc(nativeLabel)}</span>
            </div>
          </div>
          <div class="card-content">
            <div class="card-header">
              <div class="card-title-group">
                <span class="card-session-id" title="${sid}">${sid}</span>
                <span class="card-track-num">${tid}</span>
              </div>
              <span class="card-duration" title="Observation duration">⏱ ${duration}</span>
            </div>
            
            <div class="card-angles-strip">
              <span class="angle-tag ${hasFrontal ? 'present' : 'missing'}" title="${hasFrontal ? 'Frontal Angle Captured' : 'Frontal angle not yet captured'}">Frontal</span>
              <span class="angle-tag ${hasLeft ? 'present' : 'missing'}" title="${hasLeft ? 'Left Angle Captured' : 'Left angle not yet captured'}">${esc(leftLabel)}</span>
              <span class="angle-tag ${hasRight ? 'present' : 'missing'}" title="${hasRight ? 'Right Angle Captured' : 'Right angle not yet captured'}">${esc(rightLabel)}</span>
              <span class="angle-tag ${hasHighRes ? 'present' : 'missing'}" title="${hasHighRes ? 'High-Res Frame Extracted' : 'Standard resolution'}">Hi-Res ★</span>
            </div>

            <div class="card-footer">
              <span class="card-cap-count">📷 ${capCount} capture${capCount === 1 ? '' : 's'}</span>
              <button type="button" class="card-inspect-btn">Inspect →</button>
            </div>
          </div>
        </article>
      `;
    }

    openProfileModal(sessionId) {
      const profile = this.profiles.get(sessionId);
      if (!profile) return;

      this.selectedProfileId = sessionId;
      this.activeCaptureId = profile.best_capture ? profile.best_capture.capture_id : null;
      this.comparisonMode = 'side-by-side';
      this.toggleShowingEnhanced = false;
      this.splitPosition = 50;

      this.refreshModalContent();

      const modal = this.dom.modal;
      if (modal) {
        if (typeof modal.showModal === 'function') {
          modal.showModal();
        } else {
          modal.hidden = false;
        }
      }
    }

    closeProfileModal() {
      const modal = this.dom.modal;
      if (modal) {
        if (typeof modal.close === 'function') {
          modal.close();
        } else {
          modal.hidden = true;
        }
      }
      this.selectedProfileId = null;
    }

    refreshModalContent() {
      const d = this.dom;
      if (!d || !this.selectedProfileId) return;

      const profile = this.profiles.get(this.selectedProfileId);
      if (!profile) return;

      const sid = profile.session_id;
      const tid = profile.track_id != null ? `#${profile.track_id}` : sid;
      const status = profile.status || 'active';
      const score = Math.round(Number(profile.best_quality_score || (profile.best_capture && profile.best_capture.quality_score) || 0));

      if (d.modalTitle) d.modalTitle.textContent = `Profile ${tid}`;
      if (d.modalSessionId) d.modalSessionId.textContent = sid;
      if (d.modalStatusBadge) {
        d.modalStatusBadge.textContent = status.toUpperCase();
        d.modalStatusBadge.className = `badge ${status === 'active' ? 'active' : 'inactive'}`;
      }
      if (d.modalQualityBadge) {
        d.modalQualityBadge.textContent = `Quality: ${score}/100`;
        d.modalQualityBadge.className = `badge quality ${score >= 75 ? 'score-high' : score >= 50 ? 'score-med' : 'score-low'}`;
      }
      if (d.modalDurationBadge) {
        d.modalDurationBadge.textContent = `Observed: ${formatDuration(profile.duration)}`;
      }

      // Determine active capture record
      let activeCap = profile.best_capture || {};
      if (this.activeCaptureId && profile.capture_history) {
        const found = profile.capture_history.find(c => c.capture_id === this.activeCaptureId);
        if (found) activeCap = found;
      }

      this.updateModalComparison(activeCap);
      this.renderQualityMetricsBreakdown(activeCap);
      this.renderMultiAnglesList(profile);
      this.renderCaptureHistoryList(profile);
    }

    updateModalComparison(capture) {
      const d = this.dom;
      if (!d) return;

      const origUrl = capture.original_url || DEFAULT_AVATAR;
      const enhUrl = capture.enhanced_url || origUrl;
      const hasEnhanced = !!capture.is_enhanced && capture.enhanced_url;
      const method = capture.enhancement_method || (hasEnhanced ? 'ESPCN-x2' : 'Standard Source');

      const width = capture.crop_width || '—';
      const height = capture.crop_height || '—';
      const sizeMeta = `${width} × ${height} px · Quality: ${Math.round(capture.quality_score || 0)}`;

      // Side-by-side view
      if (d.imgOriginal) d.imgOriginal.src = origUrl;
      if (d.imgEnhanced) {
        d.imgEnhanced.src = hasEnhanced ? enhUrl : '';
        d.imgEnhanced.classList.toggle('placeholder', !hasEnhanced);
        const box = d.imgEnhanced.closest?.('.img-box');
        if (box) box.hidden = !hasEnhanced;
      }
      d.viewSideBySide?.classList.toggle('single-source', !hasEnhanced);
      if (d.imgOrigMeta) d.imgOrigMeta.textContent = `Original Crop · ${sizeMeta}`;
      if (d.imgEnhMeta) {
        d.imgEnhMeta.textContent = hasEnhanced ? `${method} · Processed preview` : 'Original preserved · processing optional';
      }

      // Toggle view
      if (d.imgToggle) {
        d.imgToggle.src = (this.toggleShowingEnhanced && hasEnhanced) ? enhUrl : origUrl;
      }
      if (d.toggleStateLabel) {
        d.toggleStateLabel.textContent = (this.toggleShowingEnhanced && hasEnhanced)
          ? `Showing: ${method} Enhanced` : 'Showing: Original Source';
      }
      if (d.toggleViewBtn) {
        d.toggleViewBtn.textContent = (this.toggleShowingEnhanced && hasEnhanced)
          ? 'View Original' : 'View Enhanced';
        d.toggleViewBtn.disabled = !hasEnhanced;
      }

      // Split slider view
      if (d.sliderOrig) d.sliderOrig.src = origUrl;
      if (d.sliderEnh) d.sliderEnh.src = enhUrl;
      this.setSplitSliderPosition(this.splitPosition);
    }

    setComparisonMode(mode) {
      this.comparisonMode = mode;
      const d = this.dom;
      if (!d) return;

      if (d.viewSideBySide) d.viewSideBySide.hidden = mode !== 'side-by-side';
      if (d.viewToggle) d.viewToggle.hidden = mode !== 'toggle';
      if (d.viewSlider) d.viewSlider.hidden = mode !== 'slider';

      if (d.modal && typeof d.modal.querySelectorAll === 'function') {
        const buttons = d.modal.querySelectorAll('[data-cmp-mode]');
        buttons.forEach(btn => {
          btn.classList.toggle('active', btn.dataset.cmpMode === mode);
        });
      }
    }

    toggleComparisonView() {
      this.toggleShowingEnhanced = !this.toggleShowingEnhanced;
      const profile = this.profiles.get(this.selectedProfileId);
      if (!profile) return;
      let activeCap = profile.best_capture || {};
      if (this.activeCaptureId && profile.capture_history) {
        const found = profile.capture_history.find(c => c.capture_id === this.activeCaptureId);
        if (found) activeCap = found;
      }
      this.updateModalComparison(activeCap);
    }

    setSplitSliderPosition(percentage) {
      this.splitPosition = Math.max(0, Math.min(100, percentage));
      const d = this.dom;
      if (!d) return;

      if (d.sliderEnhWrapper) {
        d.sliderEnhWrapper.style.width = `${this.splitPosition}%`;
      }
      if (d.sliderHandle) {
        d.sliderHandle.style.left = `${this.splitPosition}%`;
      }
    }

    renderQualityMetricsBreakdown(capture) {
      const d = this.dom;
      if (!d || !d.qualityMetricsGrid) return;

      const score = Math.round(capture.quality_score || 0);
      const yaw = capture.yaw != null ? `${num(capture.yaw, 1)}°` : '0.0°';
      const pitch = capture.pitch != null ? `${num(capture.pitch, 1)}°` : '0.0°';

      const metrics = [
        { label: 'OVERALL SCORE', value: `${score} / 100`, percent: score, good: score >= 70 },
        { label: 'HEAD ORIENTATION', value: `Yaw ${yaw} · Pitch ${pitch}`, percent: Math.max(10, 100 - Math.abs(Number(capture.yaw || 0)) * 2), good: Math.abs(Number(capture.yaw || 0)) < 25 },
        { label: 'POSE CATEGORY', value: (capture.pose_category || capture.category || 'Frontal').toUpperCase(), percent: 90, good: true },
        { label: 'RESOLUTION CROP', value: `${capture.crop_width || 128} × ${capture.crop_height || 128} px`, percent: Math.min(100, Math.round(((capture.crop_width || 128) / 256) * 100)), good: (capture.crop_width || 0) >= 100 },
      ];

      d.qualityMetricsGrid.innerHTML = metrics.map(m => `
        <div class="metric-meter-card">
          <div class="meter-head">
            <span class="meter-label">${esc(m.label)}</span>
            <strong class="meter-value">${esc(m.value)}</strong>
          </div>
          <div class="meter-track">
            <div class="meter-fill ${m.good ? 'good' : 'warning'}" style="width: ${m.percent}%"></div>
          </div>
        </div>
      `).join('');
    }

    renderMultiAnglesList(profile) {
      const d = this.dom;
      if (!d || !d.anglesList) return;

      const angles = profile.captures_by_category || {};
      const angleKeys = [
        { key: 'frontal', label: 'Frontal View', desc: 'Direct face pose' },
        { key: 'left_angle', label: 'Left Profile / Angle', desc: 'Left cheek & ear visible' },
        { key: 'right_angle', label: 'Right Profile / Angle', desc: 'Right cheek & ear visible' },
        { key: 'high_res', label: 'High Resolution', desc: 'Maximum pixel count' },
      ];

      d.anglesList.innerHTML = angleKeys.map(a => {
        const cap = angles[a.key];
        const isSelected = cap && cap.capture_id === this.activeCaptureId;
        if (cap) {
          const origAngleUrl = cap.original_url || DEFAULT_AVATAR;
          const imgUrl = cap.enhanced_url || origAngleUrl;
          const angleOnError = (cap.enhanced_url && cap.original_url && cap.enhanced_url !== cap.original_url)
            ? `if(this.dataset.fb){this.onerror=null;this.src='${esc(DEFAULT_AVATAR)}';}else{this.dataset.fb='1';this.src='${esc(origAngleUrl)}';}`
            : `this.onerror=null;this.src='${esc(DEFAULT_AVATAR)}';`;
          const score = Math.round(cap.quality_score || 0);
          return `
            <div class="angle-tile present ${isSelected ? 'selected' : ''}" data-angle-cap="${esc(cap.capture_id)}" role="button" tabindex="0">
              <img src="${esc(imgUrl)}" alt="${esc(a.label)}" class="angle-thumb" onerror="${angleOnError}">
              <div class="angle-meta">
                <span class="angle-name">${esc(a.label)}</span>
                <span class="angle-score">Score: ${score} · ${esc(cap.crop_width || '—')}px</span>
              </div>
            </div>
          `;
        }
        return `
          <div class="angle-tile missing">
            <div class="angle-thumb placeholder">∅</div>
            <div class="angle-meta">
              <span class="angle-name">${esc(a.label)}</span>
              <span class="angle-desc">Awaiting detection</span>
            </div>
          </div>
        `;
      }).join('');

      d.anglesList.querySelectorAll('[data-angle-cap]').forEach(tile => {
        tile.onclick = () => {
          const capId = tile.dataset.angleCap;
          this.activeCaptureId = capId;
          this.refreshModalContent();
        };
      });
    }

    renderCaptureHistoryList(profile) {
      const d = this.dom;
      if (!d || !d.historyList) return;

      const history = profile.capture_history || (profile.best_capture ? [profile.best_capture] : []);
      if (d.historyCount) d.historyCount.textContent = String(history.length);

      if (history.length === 0) {
        d.historyList.innerHTML = '<p class="muted">No historical captures recorded.</p>';
        return;
      }

      d.historyList.innerHTML = history.map(c => {
        const isSelected = c.capture_id === this.activeCaptureId;
        const timeStr = formatTime(c.timestamp);
        const score = Math.round(c.quality_score || 0);
        const origHistUrl = c.original_url || DEFAULT_AVATAR;
        const imgUrl = c.enhanced_url || origHistUrl;
        const histOnError = (c.enhanced_url && c.original_url && c.enhanced_url !== c.original_url)
          ? `if(this.dataset.fb){this.onerror=null;this.src='${esc(DEFAULT_AVATAR)}';}else{this.dataset.fb='1';this.src='${esc(origHistUrl)}';}`
          : `this.onerror=null;this.src='${esc(DEFAULT_AVATAR)}';`;
        const cat = (c.category || c.pose_category || 'Face').toUpperCase();
        return `
          <div class="history-item ${isSelected ? 'selected' : ''}" data-hist-cap="${esc(c.capture_id)}" role="button" tabindex="0">
            <img src="${esc(imgUrl)}" alt="${esc(cat)}" class="hist-thumb" onerror="${histOnError}">
            <div class="hist-details">
              <div class="hist-top">
                <span class="hist-cat">${esc(cat)}</span>
                <span class="hist-time">${timeStr}</span>
              </div>
              <div class="hist-bottom">
                <span class="hist-score">Quality: ${score}/100</span>
                <span class="hist-status">Original · ${c.crop_width || '—'} × ${c.crop_height || '—'} px</span>
              </div>
            </div>
          </div>
        `;
      }).join('');

      d.historyList.querySelectorAll('[data-hist-cap]').forEach(item => {
        item.onclick = () => {
          this.activeCaptureId = item.dataset.histCap;
          this.refreshModalContent();
        };
      });
    }

    async deleteProfile(sessionId) {
      let confirmed = true;
      if (typeof confirm === 'function') {
        confirmed = confirm(`Are you sure you want to permanently delete profile ${sessionId} and all its captured face assets?`);
      }
      if (!confirmed) return;

      this.setLoading(true);
      try {
        await this.api(`/api/people/profiles/${encodeURIComponent(sessionId)}`, 'DELETE');
      } catch (e) {
        // Fallback: still delete locally
      } finally {
        this.profiles.delete(sessionId);
        this.closeProfileModal();
        this.renderGallery();
        this.setLoading(false);
      }
    }

    showError(message) {
      const d = this.dom;
      if (!d) return;
      if (d.errorAlert && d.errorText) {
        d.errorText.textContent = message;
        d.errorAlert.hidden = false;
      }
    }

    hideError() {
      const d = this.dom;
      if (d && d.errorAlert) {
        d.errorAlert.hidden = true;
      }
    }

    setLoading(loading) {
      const d = this.dom;
      if (d && d.loading) {
        d.loading.hidden = !loading;
      }
    }
  }


  PeopleDashboard.prototype.renderCard = PeopleDashboard.prototype.renderProfileCard;
  return PeopleDashboard;
});
