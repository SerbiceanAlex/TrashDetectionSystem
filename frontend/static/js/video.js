/* ── Alpine.js video module for Trash Detection System ─────────────────── */

function videoApp() {
  return {
    // Confidence — prag mai jos pentru a prinde obiecte mai mici/la distanță
    // (>2.5 m). Falsele sunt ținute sub control de suprimarea pe corp, filtrul
    // geometric și logica temporală (un fals tranzitoriu nu creează incident).
    detConf: 0.25,

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
    videoClearingHistory: false,
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

    async clearVideoHistory() {
      if (this.videoClearingHistory || !(this.videoSessionsTotal || this.videoSessions.length)) return;
      const ok = await this.showConfirm(
        'Curăță istoricul video',
        'Vor fi șterse sesiunile de upload finalizate și fișierele video asociate. Incidentele existente se șterg separat din lista de incidente.',
        { confirmText: 'Curăță istoricul', confirmColor: '#dc2626', iconColor: '#dc2626', icon: 'trash-2' }
      );
      if (!ok) return;
      this.videoClearingHistory = true;
      try {
        const res = await fetchAPI('/api/video/sessions', { method: 'DELETE' });
        this.videoSessions = [];
        this.videoSessionsTotal = 0;
        this.videoSessionsPage = 0;
        this.selectedVideoSession = null;
        showToast(res.detail || 'Istoricul video a fost curățat.');
        await this.loadVideoSessions();
        if (typeof this.loadAdminStats === 'function') this.loadAdminStats();
      } catch (e) {
        showToast('Eroare la curățarea istoricului: ' + e.message, 'error');
      } finally {
        this.videoClearingHistory = false;
      }
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
    monitorPersonConf: 0.25,
    monitorSendFps: 24,
    monitorCameraWidth: 1280,
    monitorCameraHeight: 720,
    monitorCaptureMaxDim: 896,
    monitorJpegQuality: 0.75,
    monitorFacingMode: 'environment',   // 'environment' = spate, 'user' = față
    monitorSourceMode: 'local',         // 'local' = cameră browser, 'ip' = cameră IP/RTSP
    monitorIpUrl: '',                   // URL rtsp://… sau http://… al camerei IP
    _monitorIpMode: false,
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
      // Cameră IP/RTSP: flux separat — serverul deschide stream-ul, nu browserul.
      if (this.monitorSourceMode === 'ip') {
        return this._startMonitorIp();
      }
      try {
        const runtime = this.systemInfo?.runtime || {};
        this.detConf = Number(runtime.monitor_min_det_conf || 0.25);
        this.monitorPersonConf = Number(runtime.monitor_person_conf || 0.25);
        const configuredTargetFps = Number(runtime.monitor_target_fps || 24);
        // Adoptă mereu ținta din config (test de ceiling): clientul împinge cât
        // poate, iar contorul onest arată rata reală susținută.
        this.monitorSendFps = configuredTargetFps;
        this.monitorSendFps = Math.max(10, Math.min(Number(this.monitorSendFps || configuredTargetFps || 24), 120));
        this.monitorCameraWidth = Math.max(640, Math.min(Number(runtime.monitor_camera_width || this.monitorCameraWidth || 1280), 1920));
        this.monitorCameraHeight = Math.max(360, Math.min(Number(runtime.monitor_camera_height || this.monitorCameraHeight || 720), 1080));
        this.monitorCaptureMaxDim = Math.max(416, Math.min(Number(runtime.monitor_capture_max_dim || this.monitorCaptureMaxDim || 896), 896));
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

      this._monitorVideo = video;
      this._monitorCanvas = canvas;
      this._monitorWsUrl = wsUrl;
      this._monitorUserStopped = false;
      this._monitorReconnects = 0;
      this._createMonitorWs();
    },

    // ── Cameră IP/RTSP: serverul deschide stream-ul și trimite cadrele adnotate ──
    async _startMonitorIp() {
      const url = (this.monitorIpUrl || '').trim();
      if (!url) { showToast('Introdu URL-ul camerei IP (rtsp://… sau http://…).', 'error'); return; }
      const canvas = this.$refs.monitorCanvas;
      if (!canvas) { showToast('Element canvas lipsă — reîncarcă pagina.', 'error'); return; }

      const runtime = this.systemInfo?.runtime || {};
      this.detConf = Number(runtime.monitor_min_det_conf || 0.25);
      this.monitorPersonConf = Number(runtime.monitor_person_conf || 0.25);
      this.monitorSendFps = 20;  // rată pentru stream IP (sursele bune: telefon/HLS)

      this._monitorCanvas = canvas;
      this._monitorVideo = this.$refs.monitorVideo;
      this._monitorIpMode = true;
      this._monitorUserStopped = false;
      this._monitorReconnects = 0;

      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      let wsUrl = `${proto}://${location.host}/ws/video/monitor?det_conf=${this.detConf}&person_conf=${this.monitorPersonConf}&analysis_fps=${this.monitorSendFps}&source_url=${encodeURIComponent(url)}`;
      const authToken = getAuthToken();
      if (authToken) wsUrl += `&token=${encodeURIComponent(authToken)}`;
      this._monitorWsUrl = wsUrl;
      this._createMonitorWs();
    },

    _createMonitorWs() {
      const video = this._monitorVideo;
      const canvas = this._monitorCanvas;
      this.monitorWs = new WebSocket(this._monitorWsUrl);
      this.monitorWs.binaryType = 'arraybuffer';

      this.monitorWs.onopen = () => {
        this.monitorActive = true;
        this._monitorReconnects = 0;
        // Cameră IP: serverul trimite cadrele; nu capturăm/trimitem din browser.
        if (this._monitorIpMode) return;
        this._monitorCaptureCanvas = document.createElement('canvas');
        this._monitorLastSendAt = 0;
        this._monitorSending = false;
        this._monitorFrameInFlight = false;
        this._monitorFramesInFlight = 0;
        this._startMonitorCapture(video, canvas);
      };

      this.monitorWs.onmessage = (ev) => {
        try {
          // Cameră IP: cadru binar adnotat de server → îl desenăm pe canvas.
          if (typeof ev.data !== 'string') {
            const cnv = this._monitorCanvas;
            if (cnv) {
              const blob = new Blob([ev.data], { type: 'image/jpeg' });
              createImageBitmap(blob).then((bmp) => {
                if (cnv.width !== bmp.width) cnv.width = bmp.width;
                if (cnv.height !== bmp.height) cnv.height = bmp.height;
                cnv.getContext('2d').drawImage(bmp, 0, 0, cnv.width, cnv.height);
                bmp.close();
              }).catch(() => {});
            }
            return;
          }
          this._releaseMonitorFrame();
          const msg = JSON.parse(ev.data);
          if (msg.type === 'error' || msg.type === 'stream_end') {
            showToast(msg.message || 'Stream cameră încheiat.', msg.type === 'error' ? 'error' : 'warning');
            this.stopMonitor();
            return;
          }
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
            msg._receivedAt = performance.now();
            this._lastMonitorMsg = msg;
          }
        } catch (_) {}
      };

      this.monitorWs.onerror = () => {
        // Lăsăm onclose să decidă reconectarea (onerror precede onclose).
      };

      this.monitorWs.onclose = (ev) => {
        // Oprește bucla de captură curentă (nu mai trimite spre un WS închis).
        if (this._monitorAnimFrame) { cancelAnimationFrame(this._monitorAnimFrame); this._monitorAnimFrame = null; }
        // Închidere intenționată (user a apăsat stop) — nu reconecta.
        if (this._monitorUserStopped || ev.code === 1000) return;
        // Deconectare neașteptată (WiFi/timeout) — reconectează automat, păstrând camera.
        if (this._monitorReconnects < 5 && (this.monitorStream || this._monitorIpMode)) {
          this._monitorReconnects++;
          showToast(`Reconectare monitor… (${this._monitorReconnects}/5)`, 'warning');
          setTimeout(() => { if (!this._monitorUserStopped) this._createMonitorWs(); }, 700);
        } else {
          showToast('Monitor deconectat — pornește din nou.', 'error');
          this.stopMonitor();
        }
      };
    },

    _monitorVideoConstraints() {
      // Cere o rezoluție bună fără plafoane dure (`max`) — pe mobil un cap dur
      // forțează camera pe un mod de captură mic și moale (sursa blur-ului).
      // Cerem 60 FPS: dacă hardware-ul camerei suportă, preview-ul e mai neted;
      // dacă nu, browserul revine automat la 30. Analiza e limitată separat.
      const width = Math.max(640, Math.min(Number(this.monitorCameraWidth || 1280), 1920));
      const height = Math.max(360, Math.min(Number(this.monitorCameraHeight || 720), 1080));
      return {
        facingMode: this.monitorFacingMode,
        width: { ideal: width },
        height: { ideal: height },
        frameRate: { ideal: 120 },
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
        if (this._lastMonitorMsg && (performance.now() - (this._lastMonitorMsg._receivedAt || 0)) < 900) {
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
              const maxDim = Math.max(416, Math.min(Number(this.monitorCaptureMaxDim || 896), 896));
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

      // Netezire mai puternică contra jitter-ului. Valoarea rămâne cea REALĂ
      // (măsurată de server), doar mai stabilă vizual.
      this.monitorFps = this.monitorFps > 0
        ? (this.monitorFps * 0.88 + raw * 0.12)
        : raw;

      const now = performance.now();
      if (this._monitorFpsLastDisplayAt !== 0 && (now - this._monitorFpsLastDisplayAt) < 700) return;
      this._monitorFpsLastDisplayAt = now;

      // Bandă moartă: schimbă numărul afișat doar dacă media netezită s-a
      // depărtat cu cel puțin 1.5 față de ce e pe ecran — altfel rămâne fix
      // (nu mai pâlpâie 26↔27 la valori de graniță).
      const current = this.monitorFpsDisplay || 0;
      const smoothed = this.monitorFps;
      if (current === 0 || Math.abs(smoothed - current) >= 1.5) {
        this.monitorFpsDisplay = Math.round(smoothed);
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
        for (const d of msg.trash_boxes) {
          const [x1, y1, x2, y2] = d.box;
          ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
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
      this._monitorUserStopped = true;   // oprire intenționată — nu reconecta
      if (this._monitorAnimFrame) { cancelAnimationFrame(this._monitorAnimFrame); this._monitorAnimFrame = null; }
      // Null WS before closing to prevent onclose → stopMonitor recursion
      const ws = this.monitorWs; this.monitorWs = null;
      if (ws && ws.readyState !== WebSocket.CLOSED) ws.close(1000, 'User stopped');
      if (this.monitorStream) { this.monitorStream.getTracks().forEach(t => t.stop()); this.monitorStream = null; }
      // Cameră IP: curăță canvas-ul cu ultimul cadru și resetează modul.
      if (this._monitorIpMode && this._monitorCanvas) {
        const c = this._monitorCanvas;
        try { c.getContext('2d').clearRect(0, 0, c.width, c.height); } catch (_) {}
      }
      this._monitorIpMode = false;
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
