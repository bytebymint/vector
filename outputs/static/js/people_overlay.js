/**
 * people_overlay.js - High-Performance Real-Time Visual Overlays for People and Faces.
 *
 * Provides canvas-based rendering for:
 *  - Emerald / teal rounded bounding boxes for observed persons
 *  - Anonymous tracking ID pills (#P-XX / #F-XX, duration, hits)
 *  - Cyan / Gold facial bounding boxes with quality thresholds
 *  - Head pose orientation badges ("Frontal", "Left 25°", "Right 30°")
 *  - Intelligent face quality score tags (Q: 85 ★ with color grading)
 *  - Pulsating / flashing "Captured!" indicator badges
 *  - 5-point facial landmark dots with subtle glow
 *  - Active vs temporarily lost tracking states
 *  - Adaptive 60 FPS One-Euro Filter for zero-jitter, zero-lag face & landmark tracking
 *  - Motion velocity compensation and forward extrapolation for camera pan stabilization
 *  - Coordinate normalization with dynamic scaling, aspect ratio preservation, and resize events
 *  - LRU-cached text measurements and minimal GC overhead
 */

'use strict';

(function (root, factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory();
  } else {
    root.PeopleOverlay = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : this, function () {

  const DEFAULT_OPTIONS = {
    showPersonBbox: true,
    showFaceBbox: true,
    showLabels: true,
    showPose: true,
    showQuality: true,
    showLandmarks: true,
    showCapturePulse: true,
    activeTrackingFilter: true, // Only render confirmed/active unless diagnostic
    highQualityThreshold: 70,   // Quality >= 70 is gold / high-quality capture
    mediumQualityThreshold: 55, // Quality >= 55 gets subtle cyan tactical corner accents
    acceptableQualityThreshold: 40,
    capturePulseDuration: 1.8,  // seconds of pulsating capture badge
    oneEuroMinCutoff: 2.2,      // Hz: eliminates 1-2px stationary micro-jitter
    oneEuroBeta: 0.06,          // boosts cutoff immediately on rapid movement for zero lag
    oneEuroDCutoff: 1.0,        // Hz: derivative low-pass cutoff
    cornerRadius: 6,
    opacity: 0.95,
    colors: {
      person: '#10b981',             // emerald / teal
      personFill: 'rgba(16, 185, 129, 0.07)',
      personLost: '#94a3b8',         // muted slate
      personLostFill: 'rgba(148, 163, 184, 0.04)',
      faceCyan: '#06b6d4',           // standard face cyan
      faceCyanFill: 'rgba(6, 182, 212, 0.06)',
      faceCyanAccent: 'rgba(34, 211, 238, 0.88)',
      faceGold: '#f59e0b',           // high quality gold
      faceGoldFill: 'rgba(245, 158, 11, 0.09)',
      landmarkHalo: 'rgba(6, 182, 212, 0.4)',
      landmarkDot: '#ffffff',
      captureGold: '#fbbf24',
      pillBg: 'rgba(9, 24, 36, 0.92)',
      pillBorder: 'rgba(16, 185, 129, 0.45)',
      qualityHigh: '#10b981',
      qualityMed: '#f59e0b',
      qualityLow: '#ef4444'
    }
  };

  class PeopleOverlay {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas?.getContext ? canvas.getContext('2d') : null;
      this.options = { ...DEFAULT_OPTIONS, ...options, colors: { ...DEFAULT_OPTIONS.colors, ...(options.colors || {}) } };
      this.selectedId = null;
      this.lastRenderTime = 0;
      this.diagnostic = false;

      // Map of trackId -> { startTime, timestamp, score, category, id }
      this.capturePulses = new Map();

      // Track the last seen capture ID per track to auto-trigger capture pulses
      this.seenCaptures = new Map();

      // Cache for text measurements: key: `${font}:${text}` -> width
      this.textWidthCache = new Map();
      this.maxCacheSize = 512;

      // Lock-on face tracking memory: trackKey -> { rel: [rx1, ry1, rw, rh], face, lastSeen }
      this.lockedFaces = new Map();

      // Motion tracking memory for camera pan stabilization: trackKey -> { lastTime, lastBox, vx, vy }
      this.trackMotion = new Map();

      // Per-track adaptive One-Euro Filter state for fBox and 5-point landmarks
      this.oneEuroState = new Map();
    }

    static _oneEuroAlpha(cutoff, dt) {
      const tau = 1.0 / (2.0 * Math.PI * cutoff);
      return 1.0 / (1.0 + tau / dt);
    }

    _filterScalar(prevVal, prevDx, nextVal, dt, minCutoff, beta, dCutoff) {
      const rawDx = (nextVal - prevVal) / dt;
      const aD = PeopleOverlay._oneEuroAlpha(dCutoff, dt);
      const edx = aD * rawDx + (1.0 - aD) * prevDx;
      const cutoff = minCutoff + beta * Math.abs(edx);
      const aX = PeopleOverlay._oneEuroAlpha(cutoff, dt);
      return [aX * nextVal + (1.0 - aX) * prevVal, edx];
    }

    /**
     * Smooth face bounding box and 5-point landmarks using an adaptive One-Euro Filter
     * during continuous high-rate rendering (0.004 <= dt <= 0.12), while snapping
     * immediately on first sight, zero-dt redraws, or large time steps (dt > 0.12).
     */
    smoothFaceTrack(trackKey, renderTime, fBox, rawLandmarks, canvasW, canvasH, frameW, frameH, u, opts, nowMs) {
      let pts = null;
      if (rawLandmarks && rawLandmarks.length > 0) {
        pts = [];
        for (let i = 0; i < Math.min(5, rawLandmarks.length); i++) {
          pts.push(this.normalizePoint(rawLandmarks[i], canvasW, canvasH, frameW, frameH));
        }
      }

      const t = Number.isFinite(renderTime) && renderTime > 0 ? renderTime : (this.lastFrame?.timestamp || 0);
      const prev = this.oneEuroState.get(trackKey);
      const dt = prev ? (t - prev.lastRenderTime) : 0;
      const maxJump = Math.max(fBox[2], fBox[3], 48 * u) * 1.6;
      const jumped = prev ? (Math.hypot(fBox[0] - prev.box[0], fBox[1] - prev.box[1]) > maxJump) : false;

      if (prev && !jumped && dt >= 0.004 && dt <= 0.12) {
        const minCutoff = opts.oneEuroMinCutoff ?? 2.2;
        const beta = (opts.oneEuroBeta ?? 0.06) / Math.max(1, u);
        const dCutoff = opts.oneEuroDCutoff ?? 1.0;

        const nextBox = [0, 0, 0, 0];
        const nextBoxD = [0, 0, 0, 0];
        for (let k = 0; k < 4; k++) {
          const [sv, sdv] = this._filterScalar(prev.box[k], prev.boxD[k], fBox[k], dt, minCutoff, beta, dCutoff);
          nextBox[k] = k >= 2 ? Math.max(4, sv) : sv;
          nextBoxD[k] = sdv;
        }

        let nextLm = null;
        let nextLmD = null;
        if (pts) {
          if (prev.lm && prev.lm.length === pts.length) {
            nextLm = [];
            nextLmD = [];
            for (let j = 0; j < pts.length; j++) {
              const [sx, sdx] = this._filterScalar(prev.lm[j][0], prev.lmD[j][0], pts[j][0], dt, minCutoff, beta, dCutoff);
              const [sy, sdy] = this._filterScalar(prev.lm[j][1], prev.lmD[j][1], pts[j][1], dt, minCutoff, beta, dCutoff);
              nextLm.push([sx, sy]);
              nextLmD.push([sdx, sdy]);
            }
          } else {
            nextLm = pts.map(p => [p[0], p[1]]);
            nextLmD = pts.map(() => [0, 0]);
          }
        }

        this.oneEuroState.set(trackKey, {
          lastRenderTime: t,
          lastSeen: nowMs,
          box: nextBox,
          boxD: nextBoxD,
          lm: nextLm,
          lmD: nextLmD
        });
        return { fBox: nextBox, pixelLandmarks: nextLm };
      }

      this.oneEuroState.set(trackKey, {
        lastRenderTime: t,
        lastSeen: nowMs,
        box: [fBox[0], fBox[1], fBox[2], fBox[3]],
        boxD: [0, 0, 0, 0],
        lm: pts ? pts.map(p => [p[0], p[1]]) : null,
        lmD: pts ? pts.map(() => [0, 0]) : null
      });
      return { fBox, pixelLandmarks: pts };
    }

    /**
     * Handle canvas resize smoothly and invalidate cached metrics.
     */
    size(width, height) {
      if (!this.canvas) return;
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.invalidateCache();
      }
    }

    /**
     * Clear caches when dimensions or fonts change.
     */
    invalidateCache() {
      this.textWidthCache.clear();
      this.oneEuroState.clear();
    }

    /**
     * Update runtime options.
     */
    setOptions(newOptions = {}) {
      this.options = {
        ...this.options,
        ...newOptions,
        colors: { ...this.options.colors, ...(newOptions.colors || {}) }
      };
    }

    /**
     * Programmatically trigger a capture pulse animation for a tracked person.
     */
    triggerCapture(trackId, captureData = {}) {
      if (!trackId && trackId !== 0) return;
      const key = String(trackId);
      const score = Number.isFinite(captureData.quality_score) ? captureData.quality_score :
                    (Number.isFinite(captureData.score) ? captureData.score : 85);
      const category = captureData.pose_category || captureData.category || 'Frontal';
      this.capturePulses.set(key, {
        startTime: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
        score: Math.round(score),
        category: category,
        captureId: captureData.capture_id || `cap-${Date.now()}`
      });
      // Limit memory retention of pulse history
      if (this.capturePulses.size > 100) {
        const firstKey = this.capturePulses.keys().next().value;
        this.capturePulses.delete(firstKey);
      }
    }

    /**
     * High performance text measurement with LRU-style bounded Map caching.
     */
    measureTextWidth(text, font, ctx) {
      const key = font + '|' + text;
      let width = this.textWidthCache.get(key);
      if (width !== undefined) return width;

      if (ctx) {
        ctx.font = font;
        width = ctx.measureText(text).width;
      } else {
        width = text.length * 7;
      }

      if (this.textWidthCache.size >= this.maxCacheSize) {
        this.textWidthCache.clear();
      }
      this.textWidthCache.set(key, width);
      return width;
    }

    /**
     * Cross-platform rounded rectangle drawing helper.
     */
    drawRoundRect(ctx, x, y, w, h, radius) {
      if (typeof ctx.roundRect === 'function') {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, radius);
      } else {
        const r = Math.min(radius, Math.abs(w) / 2, Math.abs(h) / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.arcTo(x + w, y, x + w, y + r, r);
        ctx.lineTo(x + w, y + h - r);
        ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h);
        ctx.arcTo(x, y + h, x, y + h - r, r);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
      }
    }

    /**
     * Coordinate normalization:
     * Converts normalized [0, 1] or pixel coordinates to current canvas pixel space.
     * Preserves aspect ratio and handles dynamic scaling.
     */
    normalizeBbox(bbox, canvasW, canvasH, frameW, frameH, parentBox = null) {
      if (!bbox || bbox.length < 4) return [0, 0, 0, 0];
      let [x1, y1, x2, y2] = bbox;

      // Check if coordinates are normalized [0.0 - 1.0]
      const isNormalized = Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) <= 1.05;

      if (isNormalized) {
        if (parentBox) {
          // If relative to a parent bounding box (person crop)
          const pw = parentBox[2], ph = parentBox[3];
          const px = parentBox[0], py = parentBox[1];
          return [
            px + x1 * pw,
            py + y1 * ph,
            Math.max(4, (x2 - x1) * pw),
            Math.max(4, (y2 - y1) * ph)
          ];
        }
        return [
          x1 * canvasW,
          y1 * canvasH,
          Math.max(4, (x2 - x1) * canvasW),
          Math.max(4, (y2 - y1) * canvasH)
        ];
      }

      // Coordinates are already in pixel space
      let scaleX = 1, scaleY = 1;
      if (frameW && frameH && (frameW !== canvasW || frameH !== canvasH)) {
        scaleX = canvasW / frameW;
        scaleY = canvasH / frameH;
      }

      return [
        x1 * scaleX,
        y1 * scaleY,
        Math.max(4, (x2 - x1) * scaleX),
        Math.max(4, (y2 - y1) * scaleY)
      ];
    }

    /**
     * Coordinate normalization for facial landmarks.
     */
    normalizePoint(pt, canvasW, canvasH, frameW, frameH) {
      if (!pt) return [0, 0];
      const x = Array.isArray(pt) ? pt[0] : (pt.x ?? 0);
      const y = Array.isArray(pt) ? pt[1] : (pt.y ?? 0);

      const isNormalized = Math.max(Math.abs(x), Math.abs(y)) <= 1.05;
      if (isNormalized) {
        return [x * canvasW, y * canvasH];
      }

      let scaleX = 1, scaleY = 1;
      if (frameW && frameH && (frameW !== canvasW || frameH !== canvasH)) {
        scaleX = canvasW / frameW;
        scaleY = canvasH / frameH;
      }
      return [x * scaleX, y * scaleY];
    }

    /**
     * Map a DOM Pointer / Mouse event to normalized [0, 1] stage coordinates,
     * taking container aspect ratio, letterboxing, and resize into account.
     */
    point(event) {
      if (!this.canvas) return [0, 0];
      const r = this.canvas.getBoundingClientRect();
      const scale = Math.min(r.width / this.canvas.width, r.height / this.canvas.height);
      const w = this.canvas.width * scale;
      const h = this.canvas.height * scale;
      const offsetX = (r.width - w) / 2;
      const offsetY = (r.height - h) / 2;

      const clientX = event.clientX !== undefined ? event.clientX : (event.x || 0);
      const clientY = event.clientY !== undefined ? event.clientY : (event.y || 0);

      const nx = (clientX - r.left - offsetX) / (w || 1);
      const ny = (clientY - r.top - offsetY) / (h || 1);
      return [Math.max(0, Math.min(1, nx)), Math.max(0, Math.min(1, ny))];
    }

    /**
     * Hit test: Find the person track or face under the given normalized point [0..1].
     * Multi-resolution safe (e.g. 1080p video stage overlaying 4K 3840x2160 AI coordinates).
     */
    hit(normPt) {
      if (!normPt || !this.lastFrame) return null;
      const people = this.lastFrame.people || [];
      const [px, py] = normPt;
      const refW = this.lastFrame?.width || this.canvas?.width || 1;
      const refH = this.lastFrame?.height || this.canvas?.height || 1;

      for (const p of people) {
        const b = p.bbox || p.face?.bbox;
        if (!b || b.length < 4) continue;
        let x1 = b[0], y1 = b[1], x2 = b[2], y2 = b[3];
        // If bbox is in pixel coordinates, normalize against source AI frame dimensions
        if (Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) > 1.05) {
          x1 /= refW;
          y1 /= refH;
          x2 /= refW;
          y2 /= refH;
        }
        if (px >= x1 && px <= x2 && py >= y1 && py <= y2) {
          return p;
        }
      }
      return null;
    }

    /**
     * Select a specific tracking ID.
     */
    select(trackId) {
      this.selectedId = trackId;
    }

    /**
     * Master Draw Loop:
     * High performance canvas rendering routine executed on requestAnimationFrame.
     * Incorporates adaptive 60 FPS One-Euro filtering and camera pan motion compensation.
     */
    draw(frame, renderTime = 0, runtimeOptions = {}) {
      if (!this.canvas || !this.ctx) return;
      const c = this.ctx;
      const w = this.canvas.width;
      const h = this.canvas.height;
      if (!w || !h) return;

      this.lastFrame = frame;
      const effectiveRenderTime = renderTime || (frame?.timestamp || 0);
      this.lastRenderTime = effectiveRenderTime;
      const u = Math.max(1, w / (frame?.tracking_kind === 'face' ? 1280 : 768));
      const opts = { ...this.options, ...runtimeOptions };

      // NOTE: Do NOT turn off landmarks, pose, or quality badges when tracking_kind === 'face'!
      c.save();
      c.lineCap = 'round';
      c.lineJoin = 'round';
      c.globalAlpha = opts.opacity;

      const people = frame?.people || [];
      const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();

      // Periodic cleanup of stale track state (tracks gone for > 3.0s)
      if (this.lockedFaces.size > 200) {
        for (const [key, val] of this.lockedFaces.entries()) {
          if (nowMs - val.lastSeen > 3000) this.lockedFaces.delete(key);
        }
      }
      if (this.oneEuroState.size > 200) {
        for (const [key, val] of this.oneEuroState.entries()) {
          if (nowMs - val.lastSeen > 3000) this.oneEuroState.delete(key);
        }
      }

      // Render each tracked person and their facial features
      for (let i = 0; i < people.length; i++) {
        const person = people[i];
        if (!person) continue;

        const isFaceTrack = person.tracking_kind === 'face' || frame?.tracking_kind === 'face';

        // In active filter mode, skip unconfirmed people unless diagnostic mode is on
        const isConfirmed = person.confirmed !== false;
        if (opts.activeTrackingFilter && !isConfirmed && !isFaceTrack && !this.diagnostic) {
          continue;
        }

        const isSelected = (this.selectedId !== null) &&
          (person.id === this.selectedId || person.uid === this.selectedId || person.track_id === this.selectedId);

        // State: check if track is temporarily lost or inactive
        const isLost = !isConfirmed || person.state === 'lost' || person.status === 'lost' || person.lost === true;

        // Check for new high-quality capture events on this track
        const trackKey = String(person.uid ?? person.id ?? person.track_id ?? i);
        const captureKey = String(person.id ?? person.track_id ?? i);
        const bestCapture = person.best_capture || person.capture;
        if (bestCapture && bestCapture.capture_id) {
          if (this.seenCaptures.get(captureKey) !== bestCapture.capture_id) {
            this.seenCaptures.set(captureKey, bestCapture.capture_id);
            this.triggerCapture(captureKey, bestCapture);
          }
        } else if (person.just_captured || person.new_capture) {
          this.triggerCapture(captureKey, person);
        }

        // Compute pixel bounding box for person / track
        const rawPBox = this.normalizeBbox(person.bbox, w, h, frame?.width, frame?.height);
        const pBox = [...rawPBox];

        // Camera Pan Motion Compensation:
        // When renderTime leads frame.timestamp on un-interpolated frames, advance position by track velocity
        const prevMotion = this.trackMotion.get(trackKey);
        let vx = prevMotion?.vx || 0, vy = prevMotion?.vy || 0;
        const obsStamp = frame?.syncTimestamp ?? frame?.timestamp ?? 0;
        if (prevMotion && obsStamp && prevMotion.lastTime < obsStamp) {
          const dt = obsStamp - prevMotion.lastTime;
          if (dt > 0.01 && dt < 0.8) {
            vx = (rawPBox[0] - prevMotion.lastBox[0]) / dt;
            vy = (rawPBox[1] - prevMotion.lastBox[1]) / dt;
          } else {
            vx = 0;
            vy = 0;
          }
          this.trackMotion.set(trackKey, { lastTime: obsStamp, lastBox: rawPBox, vx, vy });
        } else if (!prevMotion) {
          this.trackMotion.set(trackKey, { lastTime: obsStamp || effectiveRenderTime, lastBox: rawPBox, vx: 0, vy: 0 });
        }
        let compDx = 0, compDy = 0;
        if (!frame?.syncTimestamp && (vx !== 0 || vy !== 0) && effectiveRenderTime > (frame?.timestamp || 0)) {
          const dtComp = Math.min(0.4, effectiveRenderTime - (frame.timestamp || 0));
          const effComp = 0.28 * (1 - Math.exp(-dtComp / 0.28));
          compDx = vx * effComp;
          compDy = vy * effComp;
          pBox[0] += compDx;
          pBox[1] += compDy;
        }

        // 1. Draw Person Bounding Box (for body tracks; avoid duplicate box on direct face tracks)
        if (opts.showPersonBbox) {
          if (person.body_bbox) {
            const bBox = this.normalizeBbox(person.body_bbox, w, h, frame?.width, frame?.height);
            bBox[0] += compDx;
            bBox[1] += compDy;
            this.drawPersonBox(c, person, bBox, u, isSelected, isLost, opts);
          } else if (!isFaceTrack) {
            this.drawPersonBox(c, person, pBox, u, isSelected, isLost, opts);
          }
        }

        // Only draw an observed face. Body anchors, guessed head positions and
        // remembered rectangles are motion hints, not face detector results.
        const face = person.face || (Array.isArray(person.faces) ? person.faces[0] : null) ||
          (frame?.faces && frame.faces.find(f => f.track_id === (person.track_id ?? person.id)));
        let fBox = null;
        let smoothedLandmarks = null;
        const rawLandmarks = face ? (face.landmarks || person.landmarks) : null;
        if (face?.bbox && !face.is_head_anchor && !face.is_locked) {
          const relative = face.coordinate_space === 'person';
          fBox = this.normalizeBbox(face.bbox, w, h, frame?.width, frame?.height, relative ? pBox : null);
          if (!relative && (compDx !== 0 || compDy !== 0)) {
            fBox[0] += compDx;
            fBox[1] += compDy;
          }
          const smoothed = this.smoothFaceTrack(
            trackKey, effectiveRenderTime, fBox, rawLandmarks,
            w, h, frame?.width, frame?.height, u, opts, nowMs
          );
          fBox = smoothed.fBox;
          smoothedLandmarks = smoothed.pixelLandmarks;
        }

        // 3. Draw Anonymous Tracking ID Pill (#P-14 or #F-14)
        if (opts.showLabels) {
          this.drawPersonPill(c, person, isFaceTrack ? (fBox || pBox) : pBox, u, isLost, opts);
        }

        // 4. Draw Face Visualization: Bounding Box, Landmarks, Pose, Quality, Pulsing Capture
        if (face && fBox) {
          const quality = face.quality || person.quality;
          const qualityScore = typeof quality === 'object' ? (quality?.score ?? 0) : (Number(quality) || 0);
          const isHighQuality = qualityScore >= opts.highQualityThreshold;

          // Draw Facial Bounding Box (locked on head)
          if (opts.showFaceBbox) {
            this.drawFaceBox(c, face, fBox, u, qualityScore, isHighQuality, isLost, opts);
          }

          // Draw 5-Point Facial Landmark Dots (cyan/white dots for eyes, nose, mouth corners)
          if (opts.showLandmarks && smoothedLandmarks && smoothedLandmarks.length > 0) {
            this.drawLandmarks(c, smoothedLandmarks, u, w, h, frame?.width, frame?.height, opts, true);
          } else if (opts.showLandmarks && rawLandmarks && rawLandmarks.length > 0) {
            this.drawLandmarks(c, rawLandmarks, u, w, h, frame?.width, frame?.height, opts, false);
          }

          // Draw Pose Indicator Badge ("Frontal", "Left 25°", "Right 30°")
          const pose = face.pose || person.pose;
          if (opts.showPose && pose) {
            this.drawPoseBadge(c, pose, fBox, u, opts);
          }

          // Draw Face Quality Score Tag ("Q: 85 ★")
          if (opts.showQuality && (quality || qualityScore > 0)) {
            this.drawQualityBadge(c, quality, qualityScore, fBox, u, opts);
          }

          // Draw Flashing / Pulsating "Captured!" Badge if active
          if (opts.showCapturePulse) {
            const activePulse = this.capturePulses.get(captureKey) || this.capturePulses.get(trackKey);
            if (activePulse) {
              const elapsedSec = (nowMs - activePulse.startTime) / 1000;
              if (elapsedSec <= opts.capturePulseDuration) {
                this.drawCapturePulse(c, activePulse, fBox || pBox, u, elapsedSec, opts);
              } else {
                this.capturePulses.delete(captureKey);
                this.capturePulses.delete(trackKey);
              }
            }
          }
        }
      }

      c.restore();
    }

    /**
     * Draw person bounding box with rounded corners and subtle fill.
     */
    drawPersonBox(ctx, person, box, u, isSelected, isLost, opts) {
      const [x, y, w, h] = box;
      const radius = opts.cornerRadius * u;
      const colors = opts.colors;

      ctx.save();
      if (isLost) {
        ctx.strokeStyle = colors.personLost;
        ctx.fillStyle = colors.personLostFill;
        ctx.setLineDash([5 * u, 4 * u]);
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1.3 * u;
      } else {
        ctx.strokeStyle = isSelected ? '#34d399' : colors.person;
        ctx.fillStyle = colors.personFill;
        ctx.setLineDash([]);
        ctx.lineWidth = (isSelected ? 2.4 : 1.6) * u;
      }

      this.drawRoundRect(ctx, x, y, w, h, radius);
      ctx.fill();
      ctx.stroke();

      // Selected highlight brackets
      if (isSelected && !isLost) {
        ctx.strokeStyle = '#6ee7b7';
        ctx.lineWidth = 2.8 * u;
        const cornerLen = Math.min(16 * u, w * 0.25, h * 0.25);
        // Top-left
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLen, y);
        // Top-right
        ctx.moveTo(x + w - cornerLen, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + cornerLen);
        // Bottom-left
        ctx.moveTo(x, y + h - cornerLen);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + cornerLen, y + h);
        // Bottom-right
        ctx.moveTo(x + w - cornerLen, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - cornerLen);
        ctx.stroke();
      }

      ctx.restore();
    }

    /**
     * Draw anonymous tracking ID pill (#P-14 / #F-14, duration, hits).
     */
    drawPersonPill(ctx, person, box, u, isLost, opts) {
      const [x, y, w, h] = box;
      const id = person.id ?? person.track_id ?? '?';
      const duration = Number.isFinite(person.duration) ? person.duration :
                       (person.last_seen && person.first_seen ? Math.max(0, person.last_seen - person.first_seen) : 0);
      const hits = person.hits || 1;

      const isFaceTrack = person.tracking_kind === 'face';
      const idPrefix = isFaceTrack ? '#F' : '#P';
      const idText = `${idPrefix}-${id}${person.confirmed === false ? ' ?' : ''}`;
      const metaText = isLost ? `LOST · ${duration.toFixed(1)}s` : `${duration.toFixed(1)}s · ${hits}h`;

      const fontId = `bold ${10.5 * u}px "Segoe UI", system-ui, sans-serif`;
      const fontMeta = `500 ${9.5 * u}px "Segoe UI", system-ui, sans-serif`;

      const idW = this.measureTextWidth(idText, fontId, ctx);
      const metaW = this.measureTextWidth(metaText, fontMeta, ctx);

      const pillPad = 6 * u;
      const pillGap = 6 * u;
      const pillW = idW + metaW + pillPad * 2 + pillGap;
      const pillH = 19 * u;

      // Position pill above box, clamping to canvas top
      let pillX = Math.max(4 * u, Math.min(this.canvas.width - pillW - 4 * u, x));
      let pillY = y - pillH - 3 * u;
      if (pillY < 4 * u) {
        pillY = y + 4 * u; // Move inside box if clamped at top
      }

      ctx.save();
      // Background pill
      this.drawRoundRect(ctx, pillX, pillY, pillW, pillH, 4 * u);
      ctx.fillStyle = opts.colors.pillBg;
      ctx.fill();
      ctx.strokeStyle = isLost ? 'rgba(148, 163, 184, 0.4)' : (isFaceTrack ? 'rgba(6, 182, 212, 0.5)' : opts.colors.pillBorder);
      ctx.lineWidth = 1 * u;
      ctx.setLineDash([]);
      ctx.stroke();

      // Render Text
      ctx.textBaseline = 'middle';
      const textY = pillY + pillH / 2;

      // ID in vibrant emerald/cyan or muted slate
      ctx.font = fontId;
      ctx.fillStyle = isLost ? '#cbd5e1' : (isFaceTrack ? opts.colors.faceCyan : opts.colors.person);
      ctx.fillText(idText, pillX + pillPad, textY);

      // Meta duration/hits in subtle cool white
      ctx.font = fontMeta;
      ctx.fillStyle = isLost ? '#f59e0b' : '#94a3b8';
      ctx.fillText(metaText, pillX + pillPad + idW + pillGap, textY);

      ctx.restore();
    }

    /**
     * Draw facial bounding box (cyan or gold border for high quality, with crisp tactical corner brackets).
     */
    drawFaceBox(ctx, face, fBox, u, qualityScore, isHighQuality, isLost, opts) {
      const [x, y, w, h] = fBox;
      const colors = opts.colors;
      const isMedQuality = !isHighQuality && qualityScore >= (opts.mediumQualityThreshold ?? 55);

      ctx.save();
      const faceColor = isLost ? colors.personLost : (isHighQuality ? colors.faceGold : colors.faceCyan);
      const faceFill = isHighQuality ? colors.faceGoldFill : colors.faceCyanFill;

      ctx.strokeStyle = faceColor;
      ctx.fillStyle = faceFill;
      ctx.lineWidth = (isHighQuality ? 1.8 : 1.35) * u;
      ctx.setLineDash(isLost ? [3 * u, 3 * u] : []);

      const r = Math.min(4 * u, w * 0.2, h * 0.2);
      this.drawRoundRect(ctx, x, y, w, h, r);
      ctx.fill();
      ctx.stroke();

      // Tactical corner brackets: gold for high quality (>= 70), subtle cyan accents for medium quality (>= 55)
      if (!isLost && (isHighQuality || isMedQuality)) {
        ctx.strokeStyle = isHighQuality ? colors.captureGold : (colors.faceCyanAccent || 'rgba(34, 211, 238, 0.88)');
        ctx.lineWidth = (isHighQuality ? 2.0 : 1.55) * u;
        ctx.setLineDash([]);
        const cl = Math.min((isHighQuality ? 7.5 : 6.0) * u, w * (isHighQuality ? 0.3 : 0.26), h * (isHighQuality ? 0.3 : 0.26));

        ctx.beginPath();
        // Top-left
        ctx.moveTo(x, y + cl); ctx.lineTo(x, y); ctx.lineTo(x + cl, y);
        // Top-right
        ctx.moveTo(x + w - cl, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cl);
        // Bottom-left
        ctx.moveTo(x, y + h - cl); ctx.lineTo(x, y + h); ctx.lineTo(x + cl, y + h);
        // Bottom-right
        ctx.moveTo(x + w - cl, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cl);
        ctx.stroke();
      }

      ctx.restore();
    }

    /**
     * Derive human-readable pose string from pose data.
     */
    getPoseText(pose) {
      if (!pose) return 'Frontal';
      if (typeof pose === 'string') return pose;

      const yaw = pose.yaw ?? 0;
      const category = (pose.category || '').toLowerCase();

      if (category === 'frontal' || Math.abs(yaw) < 12) {
        return 'Frontal';
      }
      if (category === 'profile' || Math.abs(yaw) >= 45) {
        return yaw < 0 ? 'Left Profile' : 'Right Profile';
      }
      if (yaw < -12) {
        return `Left ${Math.abs(Math.round(yaw))}°`;
      }
      if (yaw > 12) {
        return `Right ${Math.round(yaw)}°`;
      }
      return 'Frontal';
    }

    /**
     * Draw Pose indicator badge (e.g. "Frontal", "Left 25°", "Right 30°").
     */
    drawPoseBadge(ctx, pose, fBox, u, opts) {
      const [x, y, w] = fBox;
      const text = this.getPoseText(pose);
      const font = `600 ${8.5 * u}px "Segoe UI", sans-serif`;
      const textW = this.measureTextWidth(text, font, ctx);

      const badgeW = textW + 8 * u;
      const badgeH = 14 * u;
      const badgeX = x + w - badgeW;
      const badgeY = y - badgeH - 2 * u;

      ctx.save();
      this.drawRoundRect(ctx, badgeX, badgeY, badgeW, badgeH, 3 * u);
      ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
      ctx.lineWidth = 0.8 * u;
      ctx.setLineDash([]);
      ctx.stroke();

      ctx.font = font;
      ctx.fillStyle = '#38bdf8'; // sky blue
      ctx.textBaseline = 'middle';
      ctx.fillText(text, badgeX + 4 * u, badgeY + badgeH / 2);
      ctx.restore();
    }

    /**
     * Draw Face Quality score tag (e.g. "Q: 85 ★" or colored badge green >=70, yellow >=40, red <40).
     */
    drawQualityBadge(ctx, quality, score, fBox, u, opts) {
      const [x, y] = fBox;
      const roundScore = Math.max(0, Math.min(100, Math.round(score)));

      let badgeColor = opts.colors.qualityHigh;
      let text = `Q: ${roundScore} ★`;
      if (roundScore >= opts.highQualityThreshold) {
        badgeColor = opts.colors.qualityHigh;
        text = `Q: ${roundScore} ★`;
      } else if (roundScore >= opts.acceptableQualityThreshold) {
        badgeColor = opts.colors.qualityMed;
        text = `Q: ${roundScore}`;
      } else {
        badgeColor = opts.colors.qualityLow;
        text = `Q: ${roundScore}`;
      }

      const font = `bold ${8.5 * u}px "Segoe UI", sans-serif`;
      const textW = this.measureTextWidth(text, font, ctx);

      const badgeW = textW + 8 * u;
      const badgeH = 14 * u;
      const badgeX = x;
      const badgeY = y - badgeH - 2 * u;

      ctx.save();
      this.drawRoundRect(ctx, badgeX, badgeY, badgeW, badgeH, 3 * u);
      ctx.fillStyle = 'rgba(10, 20, 30, 0.9)';
      ctx.fill();
      ctx.strokeStyle = badgeColor;
      ctx.lineWidth = 1 * u;
      ctx.setLineDash([]);
      ctx.stroke();

      ctx.font = font;
      ctx.fillStyle = badgeColor;
      ctx.textBaseline = 'middle';
      ctx.fillText(text, badgeX + 4 * u, badgeY + badgeH / 2);
      ctx.restore();
    }

    /**
     * Draw 5-Point facial landmark dots (cyan/white dots for eyes, nose, mouth corners).
     */
    drawLandmarks(ctx, landmarks, u, canvasW, canvasH, frameW, frameH, opts, preNormalized = false) {
      if (!landmarks || landmarks.length === 0) return;
      ctx.save();

      const dots = [];
      for (let i = 0; i < Math.min(5, landmarks.length); i++) {
        dots.push(preNormalized ? landmarks[i] : this.normalizePoint(landmarks[i], canvasW, canvasH, frameW, frameH));
      }

      // Subtle connector between eyes and mouth
      if (dots.length >= 2) {
        ctx.strokeStyle = 'rgba(34, 211, 238, 0.2)';
        ctx.lineWidth = 0.8 * u;
        ctx.beginPath();
        ctx.moveTo(dots[0][0], dots[0][1]);
        ctx.lineTo(dots[1][0], dots[1][1]);
        ctx.stroke();
      }

      for (let i = 0; i < dots.length; i++) {
        const [px, py] = dots[i];
        const isEye = i < 2;

        // Outer glowing halo
        ctx.beginPath();
        ctx.arc(px, py, (isEye ? 3.5 : 2.8) * u, 0, Math.PI * 2);
        ctx.fillStyle = opts.colors.landmarkHalo;
        ctx.fill();

        // Inner solid core dot
        ctx.beginPath();
        ctx.arc(px, py, (isEye ? 1.8 : 1.4) * u, 0, Math.PI * 2);
        ctx.fillStyle = isEye ? opts.colors.landmarkDot : '#22d3ee';
        ctx.fill();
      }

      ctx.restore();
    }

    /**
     * Draw flashing or pulsating "Captured!" indicator badge when a new high-quality capture is triggered.
     */
    drawCapturePulse(ctx, pulse, fBox, u, elapsedSec, opts) {
      const [x, y, w, h] = fBox;
      const duration = opts.capturePulseDuration || 1.8;
      const progress = Math.max(0, Math.min(1, elapsedSec / duration));

      ctx.save();

      // 1. Expanding outer pulsating ripple
      const expand = progress * 14 * u;
      const rippleAlpha = Math.max(0, (1 - progress) * 0.85 * Math.abs(Math.sin(elapsedSec * 9)));
      ctx.strokeStyle = opts.colors.captureGold;
      ctx.lineWidth = 1.8 * u;
      ctx.globalAlpha = rippleAlpha;
      ctx.setLineDash([]);
      this.drawRoundRect(ctx, x - expand, y - expand, w + expand * 2, h + expand * 2, (opts.cornerRadius + 2) * u);
      ctx.stroke();

      // 2. Pulsating "CAPTURED!" Badge
      const pulseScale = 1 + 0.1 * Math.sin(progress * Math.PI * 6) * (1 - progress);
      const text = `📸 CAPTURED! ${pulse.score}★`;
      const font = `bold ${10 * u * pulseScale}px "Segoe UI", sans-serif`;
      const textW = this.measureTextWidth(text, font, ctx);

      const badgeW = textW + 12 * u;
      const badgeH = 20 * u;
      const badgeX = x + (w - badgeW) / 2;
      const badgeY = y + h * 0.35; // Center over the face

      ctx.globalAlpha = Math.max(0, 1 - progress * 0.7);

      // Badge background pill
      this.drawRoundRect(ctx, badgeX, badgeY, badgeW, badgeH, 5 * u);
      ctx.fillStyle = 'rgba(6, 78, 59, 0.94)'; // deep emerald
      ctx.fill();
      ctx.strokeStyle = opts.colors.captureGold;
      ctx.lineWidth = 1.5 * u;
      ctx.stroke();

      // Badge text
      ctx.font = font;
      ctx.fillStyle = '#fef08a'; // bright gold/yellow
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';
      ctx.fillText(text, badgeX + badgeW / 2, badgeY + badgeH / 2);

      ctx.restore();
    }

    /**
     * Linearly interpolate people tracks and observed faces between surrounding metadata timestamps.
     * Smoothly aligns normalized vs pixel coordinate spaces and interpolates velocity when available.
     * Never extrapolates beyond [before.timestamp, after.timestamp].
     */
    static interpolatePeople(before, after, stamp) {
      if (!before || !after || after.timestamp <= before.timestamp || stamp < before.timestamp || stamp > after.timestamp) return null;
      const ratio = (stamp - before.timestamp) / (after.timestamp - before.timestamp);
      const nearest = ratio < 0.5 ? before : after;
      const other = ratio < 0.5 ? after : before;
      const dstW = nearest.width || other.width || 0;
      const dstH = nearest.height || other.height || 0;

      const alignBox = (box, srcFrame, targetNorm) => {
        if (!Array.isArray(box) || box.length < 4) return box;
        const srcNorm = Math.max(Math.abs(box[0]), Math.abs(box[1]), Math.abs(box[2]), Math.abs(box[3])) <= 1.05;
        const srcW = srcFrame.width || dstW;
        const srcH = srcFrame.height || dstH;
        if (srcNorm === targetNorm) {
          if (!targetNorm && srcW && srcH && dstW && dstH && (srcW !== dstW || srcH !== dstH)) {
            return [box[0] * dstW / srcW, box[1] * dstH / srcH, box[2] * dstW / srcW, box[3] * dstH / srcH];
          }
          return box;
        }
        if (srcNorm && !targetNorm && dstW && dstH) {
          return [box[0] * dstW, box[1] * dstH, box[2] * dstW, box[3] * dstH];
        }
        if (!srcNorm && targetNorm && srcW && srcH) {
          return [box[0] / srcW, box[1] / srcH, box[2] / srcW, box[3] / srcH];
        }
        return box;
      };

      const alignPt = (pt, srcFrame, targetNorm) => {
        if (!pt) return pt;
        const x = Array.isArray(pt) ? pt[0] : (pt.x ?? 0);
        const y = Array.isArray(pt) ? pt[1] : (pt.y ?? 0);
        const srcNorm = Math.max(Math.abs(x), Math.abs(y)) <= 1.05;
        const srcW = srcFrame.width || dstW;
        const srcH = srcFrame.height || dstH;
        if (srcNorm === targetNorm) {
          if (!targetNorm && srcW && srcH && dstW && dstH && (srcW !== dstW || srcH !== dstH)) {
            return [x * dstW / srcW, y * dstH / srcH];
          }
          return [x, y];
        }
        if (srcNorm && !targetNorm && dstW && dstH) return [x * dstW, y * dstH];
        if (!srcNorm && targetNorm && srcW && srcH) return [x / srcW, y / srcH];
        return [x, y];
      };

      const lerpVec = (va, vb) => {
        if (Array.isArray(va) && Array.isArray(vb) && va.length >= 2 && vb.length >= 2) {
          return [va[0] + (vb[0] - va[0]) * ratio, va[1] + (vb[1] - va[1]) * ratio];
        }
        return va || vb;
      };

      const byId = new Map((other.people || []).map(p => [p.uid ?? p.id ?? p.track_id, p]));
      const people = (nearest.people || []).map(p => {
        const q = byId.get(p.uid ?? p.id ?? p.track_id);
        if (!q?.bbox || !p.bbox) return p;
        const a = nearest === before ? p : q;
        const b = nearest === after ? p : q;
        const pNorm = Math.max(Math.abs(p.bbox[0]), Math.abs(p.bbox[1]), Math.abs(p.bbox[2]), Math.abs(p.bbox[3])) <= 1.05;
        const aBox = alignBox(a.bbox, before, pNorm);
        const bBox = alignBox(b.bbox, after, pNorm);
        const bbox = aBox.map((v, i) => v + (bBox[i] - v) * ratio);

        let body_bbox = p.body_bbox;
        if (Array.isArray(a.body_bbox) && Array.isArray(b.body_bbox) && a.body_bbox.length >= 4 && b.body_bbox.length >= 4) {
          const bNorm = Math.max(Math.abs(p.body_bbox[0]), Math.abs(p.body_bbox[1]), Math.abs(p.body_bbox[2]), Math.abs(p.body_bbox[3])) <= 1.05;
          const ab = alignBox(a.body_bbox, before, bNorm);
          const bb = alignBox(b.body_bbox, after, bNorm);
          body_bbox = ab.map((v, i) => v + (bb[i] - v) * ratio);
        }

        const velocity = (a.velocity || b.velocity) ? lerpVec(a.velocity, b.velocity) : undefined;

        let face = p.face || null;
        if (face && a.face?.bbox && b.face?.bbox) {
          const fNorm = Math.max(Math.abs(face.bbox[0]), Math.abs(face.bbox[1]), Math.abs(face.bbox[2]), Math.abs(face.bbox[3])) <= 1.05;
          const afBox = alignBox(a.face.bbox, before, fNorm);
          const bfBox = alignBox(b.face.bbox, after, fNorm);
          face = { ...face, bbox: afBox.map((v, i) => v + (bfBox[i] - v) * ratio) };
          if (Array.isArray(a.face.landmarks) && a.face.landmarks.length === b.face.landmarks?.length) {
            face.landmarks = a.face.landmarks.map((ptA, j) => {
              const refPt = face.landmarks?.[j] || ptA;
              const rx = Array.isArray(refPt) ? refPt[0] : (refPt.x ?? 0);
              const ry = Array.isArray(refPt) ? refPt[1] : (refPt.y ?? 0);
              const lmNorm = Math.max(Math.abs(rx), Math.abs(ry)) <= 1.05;
              const pa = alignPt(ptA, before, lmNorm);
              const pb = alignPt(b.face.landmarks[j], after, lmNorm);
              return [pa[0] + (pb[0] - pa[0]) * ratio, pa[1] + (pb[1] - pa[1]) * ratio];
            });
          }
          if (a.face.velocity || b.face.velocity) {
            face.velocity = lerpVec(a.face.velocity, b.face.velocity);
          }
        }

        const merged = { ...p, bbox, face };
        if (body_bbox !== undefined) merged.body_bbox = body_bbox;
        if (velocity !== undefined) merged.velocity = velocity;
        return merged;
      });
      return { ...nearest, width: dstW || nearest.width, height: dstH || nearest.height, timestamp: stamp, syncTimestamp: nearest.timestamp, people };
    }
  }

  return PeopleOverlay;
});
