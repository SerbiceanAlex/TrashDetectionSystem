/* ── Alpine.js video module for Trash Detection System ─────────────────── */

function videoApp() {
  return {
    // Mode: "webcam" or "upload"
    videoMode: 'webcam',

    // Webcam state
    webcamActive: false,
    webcamStream: null,
    webcamVideo: null,       // <video> element reference
    captureCanvas: null,     // offscreen canvas for JPEG capture
    ws: null,
    wsConnected: false,

    // Live stats (updated per frame from WS)
    liveFps: 0,
    liveMs: 0,
    liveObjects: 0,
    liveMaterials: {},
    showBboxes: true,

    // Confidence
    detConf: 0.35,

    // Upload state
    uploadFile: null,
    videoProcessing: false,
    uploadSessionId: null,
    videoProcessingMsg: '',
    uploadPollTimer: null,
    videoProgress: 0,       // 0–100 percent for progress bar

    // Video sessions list
    videoSessions: [],
    videoSessionsTotal: 0,
    videoSessionsPage: 0,
    isLoadingVideoSessions: false,
    selectedVideoSession: null,

    // ── Init ──────────────────────────────────────────────────────────────
    async initVideo() {
      await this.loadVideoSessions();
    },

    // ── Webcam ────────────────────────────────────────────────────────────
    facingMode: 'environment',   // 'environment' = back cam, 'user' = front cam
    _animFrameId: null,          // requestAnimationFrame handle for drawing loop

    async startWebcam() {
      try {
        // Stop existing stream first (needed when flipping camera)
        if (this.webcamStream) {
          this.webcamStream.getTracks().forEach(t => t.stop());
          this.webcamStream = null;
        }

        this.webcamStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: this.facingMode,
          },
          audio: false,
        });

        const videoEl = this.$refs.webcamVideo;
        videoEl.srcObject = this.webcamStream;
        await videoEl.play();

        this.webcamActive = true;
        this.captureCanvas = document.createElement('canvas');

        // Open WebSocket
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${proto}://${location.host}/ws/video/live?det_conf=${this.detConf}`;
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          this.wsConnected = true;
          this._startSendLoop();   // fire-and-forget send loop
        };

        this.ws.onmessage = (evt) => {
          const data = JSON.parse(evt.data);
          this.liveFps   = data.fps            || 0;
          this.liveMs    = data.elapsed_ms     || 0;
          this.liveObjects = data.total_objects || 0;
          this.liveMaterials = data.material_counts || {};
          // Store detections for overlay drawing
          this._lastDetections = data.detections || [];
        };

        this.ws.onclose = () => { this.wsConnected = false; };
        this.ws.onerror = () => { this.wsConnected = false; };

        // Start smooth 30fps render loop (draws live video + bbox overlay)
        this._startRenderLoop();

      } catch (err) {
        this.webcamActive = false;
        showToast('Cannot access camera: ' + err.message, 'error');
      }
    },

    // ── Smooth 30fps render loop (client-side only, always fast) ──────────
    _startRenderLoop() {
      const draw = () => {
        if (!this.webcamActive) return;

        const videoEl  = this.$refs.webcamVideo;
        const overlayEl = this.$refs.displayCanvas;
        if (!videoEl || !overlayEl) { this._animFrameId = requestAnimationFrame(draw); return; }

        const vw = videoEl.videoWidth;
        const vh = videoEl.videoHeight;
        if (!vw || !vh) { this._animFrameId = requestAnimationFrame(draw); return; }

        // Match canvas size to video
        if (overlayEl.width !== vw || overlayEl.height !== vh) {
          overlayEl.width  = vw;
          overlayEl.height = vh;
        }

        const ctx = overlayEl.getContext('2d');

        // Draw the live video frame
        ctx.drawImage(videoEl, 0, 0, vw, vh);

        // Draw detection bounding boxes on top
        if (this._lastDetections && this._lastDetections.length > 0) {
          this._drawBoxes(ctx, this._lastDetections, vw, vh);
        }

        this._animFrameId = requestAnimationFrame(draw);
      };
      this._animFrameId = requestAnimationFrame(draw);
    },

    // ── Draw bboxes client-side (scales from model coords → display) ──────
    _drawBoxes(ctx, detections, canvasW, canvasH) {
      const COLORS = {
        plastic:    '#3b82f6',
        metal:      '#f59e0b',
        glass:      '#10b981',
        paper:      '#8b5cf6',
        cardboard:  '#f97316',
        biological: '#84cc16',
        trash:      '#ef4444',
        clothes:    '#ec4899',
        shoes:      '#14b8a6',
      };

      for (const det of detections) {
        const fw = det.frame_w || canvasW;
        const fh = det.frame_h || canvasH;
        const scaleX = canvasW / fw;
        const scaleY = canvasH / fh;

        let [x1, y1, x2, y2] = det.box;
        x1 *= scaleX; y1 *= scaleY;
        x2 *= scaleX; y2 *= scaleY;

        const color = COLORS[det.material?.toLowerCase()] || '#ef4444';

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2.5;
        ctx.shadowColor = color;
        ctx.shadowBlur  = 6;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.shadowBlur  = 0;

        // Label background
        const label = `${det.material} ${Math.round((det.score || 0) * 100)}%`;
        ctx.font = 'bold 13px Inter, system-ui, sans-serif';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        const lx = Math.max(0, x1);
        const ly = Math.max(18, y1);
        ctx.fillRect(lx - 2, ly - 16, tw + 10, 18);

        // Label text
        ctx.fillStyle = '#fff';
        ctx.fillText(label, lx + 3, ly - 2);
      }
    },

    // ── Async fire-and-forget frame send (doesn't block render loop) ──────
    _startSendLoop() {
      const send = () => {
        if (!this.webcamActive || !this.wsConnected) return;

        const videoEl = this.$refs.webcamVideo;
        const canvas  = this.captureCanvas;
        if (!videoEl || videoEl.readyState < 2) {
          setTimeout(send, 100);
          return;
        }

        const vw = videoEl.videoWidth;
        const vh = videoEl.videoHeight;
        if (!vw || !vh) { setTimeout(send, 100); return; }

        // Downscale to max 640px
        const maxDim = 640;
        let w = vw, h = vh;
        if (w > maxDim || h > maxDim) {
          if (w > h) { h = Math.round(h * maxDim / w); w = maxDim; }
          else       { w = Math.round(w * maxDim / h); h = maxDim; }
        }
        canvas.width = w;
        canvas.height = h;
        canvas.getContext('2d').drawImage(videoEl, 0, 0, w, h);

        canvas.toBlob((blob) => {
          if (blob && this.ws && this.ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then(buf => {
              this.ws.send(new Uint8Array(buf));
            });
          }
          // Send at ~20fps — GPU can handle this rate
          setTimeout(send, 50);
        }, 'image/jpeg', 0.65);
      };
      send();
    },

    // ── Flip camera (front ↔ back) ────────────────────────────────────────
    async flipCamera() {
      this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';
      if (this.webcamActive) {
        // Stop old stream, reconnect WS, restart
        if (this.ws) { this.ws.close(); this.ws = null; }
        if (this._animFrameId) { cancelAnimationFrame(this._animFrameId); this._animFrameId = null; }
        this._lastDetections = [];
        await this.startWebcam();
      }
    },


    stopWebcam() {
      this.webcamActive = false;
      if (this._animFrameId) { cancelAnimationFrame(this._animFrameId); this._animFrameId = null; }
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
      if (this.webcamStream) {
        this.webcamStream.getTracks().forEach(t => t.stop());
        this.webcamStream = null;
      }
      this.wsConnected = false;
      this.liveFps = 0;
      this.liveMs = 0;
      this.liveObjects = 0;
      this.liveMaterials = {};
      this._lastDetections = [];
      this.loadVideoSessions();
    },

    // ── 📸 Save current live frame as a detection session ─────────────────
    snapshotLoading: false,
    snapshotFlash: false,

    async saveSnapshot() {
      const canvas = this.$refs.displayCanvas;
      if (!canvas || !this.webcamActive) {
        showToast('Camera is not active', 'error');
        return;
      }

      this.snapshotLoading = true;

      // Flash effect
      this.snapshotFlash = true;
      setTimeout(() => { this.snapshotFlash = false; }, 300);

      // Silently refresh GPS before saving
      try {
        if (this.geoStatus !== 'ok') {
          const pos = await requestGPS();
          this.geoLat = pos.lat;
          this.geoLng = pos.lng;
          this.geoAccuracy = pos.accuracy;
          this.geoStatus = 'ok';
        }
      } catch (_) {}

      canvas.toBlob(async (blob) => {
        if (!blob) { this.snapshotLoading = false; return; }

        const fd = new FormData();
        fd.append('file', blob, `live_${Date.now()}.jpg`);

        let url = `/api/detect?det_conf=${this.detConf}`;
        if (this.geoLat != null && this.geoLng != null) {
          url += `&latitude=${this.geoLat}&longitude=${this.geoLng}`;
        }

        try {
          const data = await fetchAPI(url, { method: 'POST', body: fd });
          showToast(`Saved! ${data.total_objects} objects detected`);
          // Notify map + history to refresh
          window.dispatchEvent(new CustomEvent('eco:newReport'));
        } catch (e) {
          showToast('Save error: ' + e.message, 'error');
        } finally {
          this.snapshotLoading = false;
        }
      }, 'image/jpeg', 0.92);
    },


    handleVideoFileSelect(ev) {
      const f = ev.target.files[0];
      if (f) {
        this.uploadFile = f;
        this.videoProcessingMsg = '';
        this.uploadSessionId = null;
        this.uploadVideo();
      }
    },

    async uploadVideo() {
      if (!this.uploadFile) return;
      this.videoProcessing = true;
      this.videoProcessingMsg = 'Uploading...';
      this.videoProgress = 0;

      try {
        const fd = new FormData();
        fd.append('file', this.uploadFile);
        const resp = await fetch(`/api/video/upload?det_conf=${this.detConf}`, {
          method: 'POST', body: fd,
        });
        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || 'Upload error');
        }
        const data = await resp.json();
        this.uploadSessionId = data.session_id;
        this.videoProcessingMsg = 'Processing...';

        // Poll for completion
        this._pollUploadStatus(data.session_id);
      } catch (e) {
        this.videoProcessingMsg = 'Error: ' + e.message;
        this.videoProcessing = false;
      }
    },

    _pollUploadStatus(sessionId) {
      this.uploadPollTimer = setInterval(async () => {
        try {
          const resp = await fetch(`/api/video/sessions/${sessionId}`);
          if (!resp.ok) return;
          const vs = await resp.json();

          // Update progress bar
          if (vs.total_frames_expected > 0) {
            this.videoProgress = Math.round((vs.frames_processed / vs.total_frames_expected) * 100);
            this.videoProcessingMsg = `Processing... ${this.videoProgress}% (${vs.frames_processed}/${vs.total_frames_expected} frames)`;
          }

          if (vs.status === 'completed') {
            clearInterval(this.uploadPollTimer);
            this.videoProgress = 100;
            const evCount = vs.littering_count || 0;
            this.videoProcessingMsg = `Gata! ${vs.total_frames} frames · ${vs.total_objects} obiecte detectate${evCount > 0 ? ` · ${evCount} incident${evCount > 1 ? 'e' : ''} salvat${evCount > 1 ? 'e' : ''}` : ''}`;
            this.videoProcessing = false;
            this.selectedVideoSession = vs;
            this.loadVideoSessions();
            // Auto-refresh incidents tab
            if (typeof this.loadIncidents === 'function') this.loadIncidents();
            if (evCount > 0) showToast(`${evCount} incident${evCount > 1 ? 'e' : ''} detectat${evCount > 1 ? 'e' : ''} și salvat${evCount > 1 ? 'e' : ''}!`, 'success');
          } else if (vs.status === 'failed') {
            clearInterval(this.uploadPollTimer);
            this.videoProgress = 0;
            this.videoProcessingMsg = 'Processing failed.';
            this.videoProcessing = false;
          }
        } catch (_) {}
      }, 2000);
    },

    // ── Video sessions list ───────────────────────────────────────────────
    async loadVideoSessions() {
      this.isLoadingVideoSessions = true;
      try {
        const skip = this.videoSessionsPage * 10;
        const resp = await fetch(`/api/video/sessions?skip=${skip}&limit=10`);
        if (resp.ok) {
          const data = await resp.json();
          this.videoSessions = data.items;
          this.videoSessionsTotal = data.total;
        }
      } finally {
        this.isLoadingVideoSessions = false;
      }
    },

    async viewVideoSession(id) {
      const resp = await fetch(`/api/video/sessions/${id}`);
      if (resp.ok) this.selectedVideoSession = await resp.json();
    },

    async deleteVideoSession(id) {
      if (!confirm(`Delete video session #${id}?`)) return;
      const resp = await fetch(`/api/video/sessions/${id}`, { method: 'DELETE' });
      if (resp.ok) {
        this.selectedVideoSession = null;
        await this.loadVideoSessions();
      }
    },

    getVideoDownloadUrl(session) {
      return `/api/video/sessions/${session.id}/download`;
    },

    parseMaterials(jsonStr) {
      try { return JSON.parse(jsonStr || '{}'); } catch (_) { return {}; }
    },

    // ── Helpers ───────────────────────────────────────────────────────────
    formatDuration(sec) {
      if (!sec) return '0s';
      const m = Math.floor(sec / 60);
      const s = Math.round(sec % 60);
      return m > 0 ? `${m}m ${s}s` : `${s}s`;
    },

    // ── Upload video state ────────────────────────────────────────────────
    videoProcessing: false,
    videoProgress: 0,
    videoProcessingMsg: '',
    uploadLoading: false,
    uploadStatus: '',
    uploadSessionId: null,

    // ── Monitor Mode (Littering Event Detection) ───────────────────────────
    monitorSource: 'camera',  // 'camera' | 'video'
    monitorActive: false,
    monitorStream: null,
    monitorWs: null,
    monitorState: 'CLEAR',
    monitorProgress: 0,
    monitorFps: 0,
    monitorPersons: 0,
    monitorTrash: 0,
    monitorAlerts: [],
    monitorPersonConf: 0.35,
    monitorSendFps: 10,
    monitorFacingMode: 'environment',   // 'environment' = spate, 'user' = față
    _monitorAnimFrame: null,
    _monitorCaptureCanvas: null,
    _lastMonitorMsg: null,
    _monitorLastSendAt: 0,
    _monitorSending: false,

    async startMonitor() {
      try {
        this.monitorStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: this.monitorFacingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      } catch (e) {
        // Fallback: try without facingMode constraint if it fails
        try {
          this.monitorStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } catch (e2) {
          showToast('Camera access error: ' + e2.message, 'error');
          return;
        }
      }

      const video = this.$refs.monitorVideo;
      const canvas = this.$refs.monitorCanvas;
      if (!video || !canvas) {
        showToast('Camera elements missing — reload the page.', 'error');
        return;
      }

      video.srcObject = this.monitorStream;
      await video.play();

      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      let wsUrl = `${proto}://${location.host}/ws/video/monitor?det_conf=${this.detConf}&person_conf=${this.monitorPersonConf}`;
      if (this.geoLat != null && this.geoLng != null) {
        wsUrl += `&lat=${this.geoLat}&lng=${this.geoLng}`;
      }

      this.monitorWs = new WebSocket(wsUrl);
      this.monitorWs.binaryType = 'arraybuffer';

      this.monitorWs.onopen = () => {
        this.monitorActive = true;
        this._monitorCaptureCanvas = document.createElement('canvas');
        this._monitorLastSendAt = 0;
        this._monitorSending = false;
        this._startMonitorCapture(video, canvas);
      };

      this.monitorWs.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'alert') {
            this.monitorAlerts.push(msg);
            const material = msg.material || 'unknown';
            showToast(`⚠ Incident detectat! Material: ${material}`, 'error');
            // Vibrate on mobile
            if (navigator.vibrate) navigator.vibrate([200, 100, 200]);
            // Browser push notification
            if ('Notification' in window) {
              if (Notification.permission === 'granted') {
                new Notification('TrashDet — Incident detectat!', {
                  body: `Material: ${material} • ${new Date().toLocaleTimeString('ro-RO')}`,
                  icon: '/static/icons/icon-192.png',
                  tag: 'littering-alert',
                  requireInteraction: true,
                });
              } else if (Notification.permission === 'default') {
                Notification.requestPermission().then(p => {
                  if (p === 'granted') {
                    new Notification('TrashDet — Incident detectat!', {
                      body: `Material: ${material}`,
                      icon: '/static/icons/icon-192.png',
                      tag: 'littering-alert',
                    });
                  }
                });
              }
            }
            // Refresh incidents tab if open
            if (typeof this.loadIncidents === 'function') this.loadIncidents();
            // Notify admin panel
            window.dispatchEvent(new CustomEvent('eco:litteringAlert', { detail: msg }));
          } else {
            this.monitorState = msg.state || 'CLEAR';
            this.monitorProgress = msg.monitor_progress || 0;
            this.monitorFps = msg.fps || 0;
            this.monitorPersons = msg.persons || 0;
            this.monitorTrash = msg.trash || 0;
            // Store msg — overlay is drawn in the RAF loop to avoid accumulation
            this._lastMonitorMsg = msg;
          }
        } catch (_) {}
      };

      this.monitorWs.onerror = () => {
        showToast('Monitor WebSocket connection error', 'error');
        this.stopMonitor();
      };

      this.monitorWs.onclose = (ev) => {
        // Only show toast for unexpected closes (not user-initiated)
        if (this.monitorActive && ev.code !== 1000) {
          showToast('Monitor connection interrupted', 'warning');
        }
        this.stopMonitor();
      };
    },

    _startMonitorCapture(video, displayCanvas) {
      const cc = this._monitorCaptureCanvas;

      const loop = () => {
        if (!this.monitorActive) return;
        this._monitorAnimFrame = requestAnimationFrame(loop);

        if (video.readyState < 2) return;

        const vw = video.videoWidth, vh = video.videoHeight;
        if (!vw) return;

        // Draw video to display canvas — this implicitly clears previous overlays
        const dw = displayCanvas.offsetWidth || vw;
        const dh = displayCanvas.offsetHeight || vh;
        if (displayCanvas.width !== dw) displayCanvas.width = dw;
        if (displayCanvas.height !== dh) displayCanvas.height = dh;
        const ctx = displayCanvas.getContext('2d');
        ctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
        ctx.drawImage(video, 0, 0, displayCanvas.width, displayCanvas.height);
        // Draw bounding boxes fresh on each frame from last WS message
        if (this._lastMonitorMsg) {
          this._drawMonitorOverlay(displayCanvas, this._lastMonitorMsg);
        }

        // Capture a bit more detail for small/soft objects (e.g. wrappers on bed)
        const scale = 768 / Math.max(vw, vh);
        cc.width = Math.round(vw * scale);
        cc.height = Math.round(vh * scale);
        const cctx = cc.getContext('2d');
        cctx.drawImage(video, 0, 0, cc.width, cc.height);

        if (this.monitorWs && this.monitorWs.readyState === WebSocket.OPEN) {
          const now = performance.now();
          const sendIntervalMs = 1000 / Math.max(this.monitorSendFps || 12, 1);

          // Backpressure guard: do not flood WS with more frames than backend can consume.
          if (!this._monitorSending && (now - this._monitorLastSendAt) >= sendIntervalMs) {
            if (this.monitorWs.bufferedAmount < 500000) {
              this._monitorSending = true;
              this._monitorLastSendAt = now;
              cc.toBlob((blob) => {
                if (!blob) {
                  this._monitorSending = false;
                  return;
                }
                blob.arrayBuffer()
                  .then((buf) => {
                    if (this.monitorWs && this.monitorWs.readyState === WebSocket.OPEN) {
                      this.monitorWs.send(buf);
                    }
                  })
                  .finally(() => {
                    this._monitorSending = false;
                  });
              }, 'image/jpeg', 0.78);
            }
          }
        }
      };
      this._monitorAnimFrame = requestAnimationFrame(loop);
    },

    _drawMonitorOverlay(canvas, msg) {
      if (!msg.person_boxes && !msg.trash_boxes && !msg.last_person_zones) return;
      const ctx = canvas.getContext('2d');
      const scaleX = canvas.width / (msg.frame_w || 640);
      const scaleY = canvas.height / (msg.frame_h || 480);

      // Draw monitored zone (dashed orange) when in MONITORING state — shows WHERE to drop object
      if (msg.state === 'MONITORING' && msg.last_person_zones && msg.last_person_zones.length > 0) {
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 300);  // pulsating opacity
        ctx.strokeStyle = `rgba(251,146,60,${0.6 + 0.4 * pulse})`;
        ctx.lineWidth = 3;
        ctx.setLineDash([12, 6]);
        ctx.font = 'bold 13px sans-serif';
        ctx.fillStyle = `rgba(251,146,60,${0.7 + 0.3 * pulse})`;
        for (const z of msg.last_person_zones) {
          const [x1, y1, x2, y2] = z;
          // Draw expanded zone (35% expansion matches zone_expand=0.35)
          const expand = 0.35;
          const dw = (x2 - x1) * expand, dh = (y2 - y1) * expand;
          const zx1 = (x1 - dw) * scaleX, zy1 = (y1 - dh) * scaleY;
          const zw  = (x2 - x1 + 2 * dw) * scaleX, zh = (y2 - y1 + 2 * dh) * scaleY;
          ctx.strokeRect(zx1, zy1, zw, zh);
          ctx.fillText('⬇ Aruncă aici', zx1 + 4, zy1 + 16);
        }
        ctx.setLineDash([]);
      }

      // Draw person boxes (orange solid)
      if (msg.person_boxes) {
        ctx.strokeStyle = 'rgba(251,191,36,0.9)';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.font = '11px sans-serif';
        ctx.fillStyle = 'rgba(251,191,36,0.9)';
        for (const b of msg.person_boxes) {
          const [x1, y1, x2, y2] = b;
          ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
          ctx.fillText('person', x1 * scaleX + 2, y1 * scaleY - 3);
        }
      }

      // Draw trash boxes (red)
      if (msg.trash_boxes) {
        ctx.strokeStyle = 'rgba(239,68,68,0.9)';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(239,68,68,0.9)';
        ctx.font = '11px sans-serif';
        for (const d of msg.trash_boxes) {
          const [x1, y1, x2, y2] = d.box;
          ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
          ctx.fillText('#' + d.track_id, x1 * scaleX + 2, y1 * scaleY - 3);
        }
      }
    },

    // ── Flip camera front ↔ back (fără a reporni WebSocket-ul) ─────────────
    async flipMonitorCamera() {
      this.monitorFacingMode = this.monitorFacingMode === 'environment' ? 'user' : 'environment';
      if (!this.monitorActive) return;

      // Opreste stream-ul curent
      if (this.monitorStream) {
        this.monitorStream.getTracks().forEach(t => t.stop());
        this.monitorStream = null;
      }

      // Porneste stream nou cu camera opusă
      try {
        this.monitorStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: this.monitorFacingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        const video = this.$refs.monitorVideo;
        video.srcObject = this.monitorStream;
        await video.play();
        showToast(this.monitorFacingMode === 'user' ? 'Camera față activată' : 'Camera spate activată', 'success');
      } catch (e) {
        showToast('Eroare la schimbarea camerei: ' + e.message, 'error');
        // Revert if failed
        this.monitorFacingMode = this.monitorFacingMode === 'environment' ? 'user' : 'environment';
      }
    },

    stopMonitor() {
      this.monitorActive = false;
      if (this._monitorAnimFrame) { cancelAnimationFrame(this._monitorAnimFrame); this._monitorAnimFrame = null; }
      // Null WS before closing to prevent onclose → stopMonitor recursion
      const ws = this.monitorWs; this.monitorWs = null;
      if (ws && ws.readyState !== WebSocket.CLOSED) ws.close(1000, 'User stopped');
      if (this.monitorStream) { this.monitorStream.getTracks().forEach(t => t.stop()); this.monitorStream = null; }
      this.monitorState = 'CLEAR';
      this.monitorFps = 0;
      this.monitorPersons = 0;
      this.monitorTrash = 0;
      this.monitorProgress = 0;
      this._lastMonitorMsg = null;
      this._monitorLastSendAt = 0;
      this._monitorSending = false;
    },
  };
}
