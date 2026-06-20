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
      if (this.token || getAuthToken()) {
        await this.loadVideoSessions();
      }
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
      if (!this.token && !getAuthToken()) {
        this.openAuth?.('login');
        showToast('Autentifică-te pentru a analiza videoclipuri.', 'error');
        return;
      }
      this.videoProcessing = true;
      this.videoProcessingMsg = 'Uploading...';
      this.videoProgress = 0;

      try {
        const fd = new FormData();
        fd.append('file', this.uploadFile);
        const data = await fetchAPI(`/api/video/upload?det_conf=${this.detConf}`, {
          method: 'POST',
          body: fd,
        });
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
          const vs = await fetchAPI(`/api/video/sessions/${sessionId}`);

          // Update progress bar
          if (vs.total_frames_expected > 0) {
            this.videoProgress = Math.round((vs.frames_processed / vs.total_frames_expected) * 100);
            this.videoProcessingMsg = `Processing... ${this.videoProgress}% (${vs.frames_processed}/${vs.total_frames_expected} frames)`;
          }

          if (vs.status === 'completed') {
            clearInterval(this.uploadPollTimer);
            this.videoProgress = 100;
            const evCount = vs.littering_count || 0;
            this.videoProcessingMsg = evCount > 0
              ? `Gata! ${vs.total_frames} cadre analizate · ${evCount} incident(e) generate`
              : `Gata! ${vs.total_frames} cadre analizate · niciun incident confirmat`;
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
      if (!this.token && !getAuthToken()) {
        this.videoSessions = [];
        this.videoSessionsTotal = 0;
        this.isLoadingVideoSessions = false;
        return;
      }
      this.isLoadingVideoSessions = true;
      try {
        const skip = this.videoSessionsPage * 10;
        const data = await fetchAPI(`/api/video/sessions?skip=${skip}&limit=10`);
        this.videoSessions = data.items;
        this.videoSessionsTotal = data.total;
      } catch (e) {
        if (!String(e.message || '').includes('401')) {
          console.warn('loadVideoSessions', e);
        }
      } finally {
        this.isLoadingVideoSessions = false;
      }
    },

    async viewVideoSession(id) {
      try {
        this.selectedVideoSession = await fetchAPI(`/api/video/sessions/${id}`);
      } catch (e) {
        showToast('Nu pot încărca sesiunea video: ' + e.message, 'error');
      }
    },

    async deleteVideoSession(id) {
      const ok = await this.showConfirm(
        'Șterge sesiune video',
        `Sesiunea #${id} și datele asociate vor fi șterse permanent.`,
        { confirmText: 'Șterge', icon: 'trash-2' }
      );
      if (!ok) return;
      await fetchAPI(`/api/video/sessions/${id}`, {
        method: 'DELETE',
      });
      this.selectedVideoSession = null;
      await this.loadVideoSessions();
    },

    getVideoDownloadUrl(session) {
      const token = getAuthToken();
      const qs = token ? `?token=${encodeURIComponent(token)}` : '';
      return `/api/video/sessions/${session.id}/download${qs}`;
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
    monitorFpsDisplay: 0,
    monitorFpsRaw: 0,
    monitorPersons: 0,
    monitorTrash: 0,
    monitorAlerts: [],
    monitorPersonConf: 0.35,
    monitorSendFps: 24,
    monitorCameraWidth: 1280,
    monitorCameraHeight: 720,
    monitorCaptureMaxDim: 640,
    monitorJpegQuality: 0.75,
    monitorFacingMode: 'environment',   // 'environment' = spate, 'user' = față
    _monitorAnimFrame: null,
    _monitorCaptureCanvas: null,
    _lastMonitorMsg: null,
    _monitorLastSendAt: 0,
    _monitorSending: false,
    _monitorFrameInFlight: false,
    _monitorFramesInFlight: 0,
    _monitorMaxInFlight: 3,
    _monitorFpsLastDisplayAt: 0,

    async startMonitor() {
      try {
        const runtime = this.systemInfo?.runtime || {};
        const configuredTargetFps = Number(runtime.monitor_target_fps || 24);
        if (this.monitorSendFps === 24 && configuredTargetFps !== 24) {
          this.monitorSendFps = configuredTargetFps;
        }
        this.monitorSendFps = Math.max(10, Math.min(Number(this.monitorSendFps || configuredTargetFps || 24), 30));
        this.monitorCameraWidth = Math.max(640, Math.min(Number(runtime.monitor_camera_width || this.monitorCameraWidth || 1280), 1920));
        this.monitorCameraHeight = Math.max(360, Math.min(Number(runtime.monitor_camera_height || this.monitorCameraHeight || 720), 1080));
        this.monitorCaptureMaxDim = Math.max(416, Math.min(Number(runtime.monitor_capture_max_dim || this.monitorCaptureMaxDim || 640), 768));
        this.monitorJpegQuality = Math.max(0.60, Math.min(Number(runtime.monitor_jpeg_quality || this.monitorJpegQuality || 0.75), 0.90));

        this.monitorStream = await navigator.mediaDevices.getUserMedia({
          video: this._monitorVideoConstraints(),
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
      let wsUrl = `${proto}://${location.host}/ws/video/monitor?det_conf=${this.detConf}&person_conf=${this.monitorPersonConf}&analysis_fps=${this.monitorSendFps}`;
      const authToken = getAuthToken();
      if (authToken) {
        wsUrl += `&token=${encodeURIComponent(authToken)}`;
      }
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
        this._monitorFrameInFlight = false;
        this._monitorFramesInFlight = 0;
        this._startMonitorCapture(video, canvas);
      };

      this.monitorWs.onmessage = (ev) => {
        try {
          this._releaseMonitorFrame();
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
            const targetFps = Number(msg.analysis_fps_target || this.monitorSendFps || 25);
            const nextFps = Math.min(Math.max(msg.fps || 0, 0), targetFps);
            this._updateMonitorFps(nextFps, targetFps);
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

    _monitorVideoConstraints() {
      // Cere o rezoluție bună fără plafoane dure (`max`) — pe mobil un cap dur
      // forțează camera pe un mod de captură mic și moale (sursa blur-ului).
      // Camera rulează la 30fps nativ; trimiterea către AI e limitată separat.
      const width = Math.max(640, Math.min(Number(this.monitorCameraWidth || 1280), 1920));
      const height = Math.max(360, Math.min(Number(this.monitorCameraHeight || 720), 1080));
      return {
        facingMode: this.monitorFacingMode,
        width: { ideal: width },
        height: { ideal: height },
        frameRate: { ideal: 30 },
      };
    },

    _startMonitorCapture(video, overlayCanvas) {
      const cc = this._monitorCaptureCanvas;

      const loop = () => {
        if (!this.monitorActive) return;
        this._monitorAnimFrame = requestAnimationFrame(loop);

        if (video.readyState < 2) return;

        const vw = video.videoWidth, vh = video.videoHeight;
        if (!vw) return;

        // Preview-ul este elementul <video> nativ (clar, fără re-desenare).
        // Canvas-ul transparent are exact dimensiunile cadrului video, astfel
        // încât object-fit:cover taie video-ul și overlay-ul identic.
        if (overlayCanvas.width !== vw) overlayCanvas.width = vw;
        if (overlayCanvas.height !== vh) overlayCanvas.height = vh;
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        if (this._lastMonitorMsg) {
          this._drawMonitorOverlay(overlayCanvas, this._lastMonitorMsg);
        }

        if (this.monitorWs && this.monitorWs.readyState === WebSocket.OPEN) {
          const now = performance.now();
          const sendIntervalMs = 1000 / Math.max(this.monitorSendFps || 12, 1);

          // Keep a tiny pipeline of frames in flight. Waiting for a full
          // browser-network-server round trip caps phones around 10-12 FPS,
          // even when the GPU can process faster.
          const maxInFlight = Math.max(1, Math.min(Number(this._monitorMaxInFlight || 2), 3));
          if (!this._monitorSending && this._monitorFramesInFlight < maxInFlight && (now - this._monitorLastSendAt) >= sendIntervalMs) {
            if (this.monitorWs.bufferedAmount < 500000) {
              // Build the JPEG only when a frame is actually due. The preview
              // canvas still renders every animation frame, but encoding is
              // throttled to the AI rhythm.
              const maxDim = Math.max(416, Math.min(Number(this.monitorCaptureMaxDim || 768), 896));
              const scale = Math.min(1, maxDim / Math.max(vw, vh));
              cc.width = Math.round(vw * scale);
              cc.height = Math.round(vh * scale);
              const cctx = cc.getContext('2d');
              cctx.drawImage(video, 0, 0, cc.width, cc.height);

              this._monitorSending = true;
              this._monitorFramesInFlight += 1;
              this._monitorFrameInFlight = this._monitorFramesInFlight > 0;
              this._monitorLastSendAt = now;
              cc.toBlob((blob) => {
                if (!blob) {
                  this._monitorSending = false;
                  this._releaseMonitorFrame();
                  return;
                }
                blob.arrayBuffer()
                  .then((buf) => {
                    if (this.monitorWs && this.monitorWs.readyState === WebSocket.OPEN) {
                      this.monitorWs.send(buf);
                    } else {
                      this._releaseMonitorFrame();
                    }
                  })
                  .catch(() => {
                    this._releaseMonitorFrame();
                  })
                  .finally(() => {
                    this._monitorSending = false;
                  });
              }, 'image/jpeg', this.monitorJpegQuality || 0.72);
            }
          }
        }
      };
      this._monitorAnimFrame = requestAnimationFrame(loop);
    },

    _releaseMonitorFrame() {
      this._monitorFramesInFlight = Math.max(0, Number(this._monitorFramesInFlight || 0) - 1);
      this._monitorFrameInFlight = this._monitorFramesInFlight > 0;
    },

    _updateMonitorFps(rawFps, targetFps) {
      const target = Math.max(1, Number(targetFps || this.monitorSendFps || 15));
      const raw = Math.max(0, Math.min(Number(rawFps || 0), target));
      this.monitorFpsRaw = raw;

      // Netezire ușoară contra jitter-ului, dar valoarea afișată este cea REALĂ
      // (măsurată de server) — nu se rotunjește artificial la țintă.
      this.monitorFps = this.monitorFps > 0
        ? (this.monitorFps * 0.8 + raw * 0.2)
        : raw;

      const now = performance.now();
      if (this._monitorFpsLastDisplayAt === 0 || (now - this._monitorFpsLastDisplayAt) > 500) {
        this.monitorFpsDisplay = Math.round(this.monitorFps);
        this._monitorFpsLastDisplayAt = now;
      }
    },

    _drawMonitorOverlay(canvas, msg) {
      if (!msg.person_boxes && !msg.trash_boxes && !msg.last_person_zones) return;
      const ctx = canvas.getContext('2d');
      const scaleX = canvas.width / (msg.frame_w || 640);
      const scaleY = canvas.height / (msg.frame_h || 480);
      // Canvas-ul are acum rezoluția nativă a camerei (~1280px), nu ~800px —
      // grosimile și fonturile se scalează ca să rămână lizibile.
      const ui = Math.max(1, canvas.width / 640);

      // Zona monitorizată NU se mai desenează: la distanțe mai mari de 2-3 m
      // poziția ei devine imprecisă și induce utilizatorul în eroare.

      // Draw person boxes (orange solid)
      if (msg.person_boxes) {
        ctx.strokeStyle = 'rgba(251,191,36,0.9)';
        ctx.lineWidth = 2 * ui;
        ctx.setLineDash([]);
        ctx.font = `${Math.round(11 * ui)}px sans-serif`;
        ctx.fillStyle = 'rgba(251,191,36,0.9)';
        for (const b of msg.person_boxes) {
          const [x1, y1, x2, y2] = b;
          ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
          ctx.fillText('person', x1 * scaleX + 2 * ui, y1 * scaleY - 3 * ui);
        }
      }

      // Draw trash boxes (red)
      if (msg.trash_boxes) {
        ctx.strokeStyle = 'rgba(239,68,68,0.9)';
        ctx.lineWidth = 2 * ui;
        ctx.fillStyle = 'rgba(239,68,68,0.9)';
        ctx.font = `${Math.round(11 * ui)}px sans-serif`;
        for (const d of msg.trash_boxes) {
          const [x1, y1, x2, y2] = d.box;
          ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
          ctx.fillText('#' + d.track_id, x1 * scaleX + 2 * ui, y1 * scaleY - 3 * ui);
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
          video: this._monitorVideoConstraints(),
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
      this.monitorFpsDisplay = 0;
      this.monitorFpsRaw = 0;
      this.monitorPersons = 0;
      this.monitorTrash = 0;
      this.monitorProgress = 0;
      this._lastMonitorMsg = null;
      this._monitorLastSendAt = 0;
      this._monitorSending = false;
      this._monitorFrameInFlight = false;
      this._monitorFramesInFlight = 0;
      this._monitorFpsLastDisplayAt = 0;
    },
  };
}
