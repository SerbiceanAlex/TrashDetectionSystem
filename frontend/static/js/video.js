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
    detConf: 0.50,

    // Upload state
    uploadFile: null,
    uploadLoading: false,
    uploadSessionId: null,
    uploadStatus: '',
    uploadPollTimer: null,
    uploadProgress: 0,       // 0–100 percent for progress bar

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
        showToast('Nu se poate accesa camera: ' + err.message, 'error');
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
        showToast('Camera nu este activă', 'error');
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
          showToast(`Salvat! ${data.total_objects} obiecte detectate`);
          // Notify map + history to refresh
          window.dispatchEvent(new CustomEvent('eco:newReport'));
        } catch (e) {
          showToast('Eroare la salvare: ' + e.message, 'error');
        } finally {
          this.snapshotLoading = false;
        }
      }, 'image/jpeg', 0.92);
    },


    handleVideoFileSelect(ev) {
      const f = ev.target.files[0];
      if (f) {
        this.uploadFile = f;
        this.uploadStatus = '';
        this.uploadSessionId = null;
      }
    },

    async uploadVideo() {
      if (!this.uploadFile) return;
      this.uploadLoading = true;
      this.uploadStatus = 'Se incarca...';
      this.uploadProgress = 0;

      try {
        const fd = new FormData();
        fd.append('file', this.uploadFile);
        const resp = await fetch(`/api/video/upload?det_conf=${this.detConf}`, {
          method: 'POST', body: fd,
        });
        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || 'Eroare upload');
        }
        const data = await resp.json();
        this.uploadSessionId = data.session_id;
        this.uploadStatus = 'Se proceseaza...';

        // Poll for completion
        this._pollUploadStatus(data.session_id);
      } catch (e) {
        this.uploadStatus = 'Eroare: ' + e.message;
        this.uploadLoading = false;
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
            this.uploadProgress = Math.round((vs.frames_processed / vs.total_frames_expected) * 100);
            this.uploadStatus = `Se proceseaza... ${this.uploadProgress}% (${vs.frames_processed}/${vs.total_frames_expected} frame-uri)`;
          }

          if (vs.status === 'completed') {
            clearInterval(this.uploadPollTimer);
            this.uploadProgress = 100;
            this.uploadStatus = `Complet! ${vs.total_frames} frame-uri, ${vs.total_objects} obiecte, ${vs.avg_fps.toFixed(1)} FPS`;
            this.uploadLoading = false;
            this.selectedVideoSession = vs;
            this.loadVideoSessions();
          } else if (vs.status === 'failed') {
            clearInterval(this.uploadPollTimer);
            this.uploadProgress = 0;
            this.uploadStatus = 'Procesarea a esuat.';
            this.uploadLoading = false;
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
      if (!confirm(`Stergi sesiunea video #${id}?`)) return;
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
  };
}
