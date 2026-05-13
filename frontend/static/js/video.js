/* ── Alpine.js video module for Trash Detection System ─────────────────── */

function videoApp() {
  return {
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
            this.videoProcessingMsg = `Gata! ${vs.total_frames} cadre analizate · ${vs.total_objects} obiecte detectate`;
            this.selectedVideoSession = vs;
            this.loadVideoSessions();
            
            // Auto-refresh incidents tab
            if (typeof this.loadIncidents === 'function') this.loadIncidents();
            
            if (evCount > 0) {
              showToast(`Analiză finalizată: ${evCount} incident(e) salvat(e)!`, 'error');
            } else {
              showToast('Analiză finalizată: Niciun incident de aruncare ilegală detectat.', 'success');
            }

            // Lasă mesajul de succes pe ecran 4 secunde înainte să revină la butonul de upload
            setTimeout(() => {
              this.videoProcessing = false;
              this.uploadFile = null;
            }, 4000);

          } else if (vs.status === 'failed') {
            clearInterval(this.uploadPollTimer);
            this.videoProgress = 0;
            this.videoProcessingMsg = 'Eroare la procesarea fișierului.';
            showToast('Eroare la procesarea videoclipului.', 'error');
            
            setTimeout(() => {
              this.videoProcessing = false;
              this.uploadFile = null;
            }, 3000);
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
      const ok = await this.showConfirm(
        'Șterge sesiune video',
        `Sesiunea #${id} și datele asociate vor fi șterse permanent.`,
        { confirmText: 'Șterge', icon: 'trash-2' }
      );
      if (!ok) return;
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
