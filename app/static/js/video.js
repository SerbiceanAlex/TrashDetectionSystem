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
    detConf: 0.25,

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
    async startWebcam() {
      try {
        this.webcamStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
          audio: false,
        });

        const videoEl = this.$refs.webcamVideo;
        videoEl.srcObject = this.webcamStream;
        await videoEl.play();

        // Also wire the raw display video so "no bboxes" mode works
        const displayVideoEl = this.$refs.webcamVideoDisplay;
        if (displayVideoEl) {
          displayVideoEl.srcObject = this.webcamStream;
          await displayVideoEl.play();
        }

        this.webcamActive = true;

        // Create offscreen canvas for JPEG capture
        this.captureCanvas = document.createElement('canvas');

        // Open WebSocket
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${proto}://${location.host}/ws/video/live?det_conf=${this.detConf}`;
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          this.wsConnected = true;
          this._sendFrameLoop();
        };

        this.ws.onmessage = (evt) => {
          const data = JSON.parse(evt.data);
          this.liveFps = data.fps || 0;
          this.liveMs = data.elapsed_ms || 0;
          this.liveObjects = data.total_objects || 0;
          this.liveMaterials = data.material_counts || {};

          // Draw annotated frame on display canvas
          if (data.frame && this.showBboxes) {
            const displayCanvas = this.$refs.displayCanvas;
            if (displayCanvas) {
              const img = new Image();
              img.onload = () => {
                displayCanvas.width = img.width;
                displayCanvas.height = img.height;
                displayCanvas.getContext('2d').drawImage(img, 0, 0);
              };
              img.src = 'data:image/jpeg;base64,' + data.frame;
            }
          }
        };

        this.ws.onclose = () => { this.wsConnected = false; };
        this.ws.onerror = () => { this.wsConnected = false; };

      } catch (err) {
        alert('Nu se poate accesa camera: ' + err.message);
      }
    },

    _sendFrameLoop() {
      if (!this.webcamActive || !this.wsConnected) return;

      const videoEl = this.$refs.webcamVideo;
      const canvas = this.captureCanvas;
      if (!videoEl || videoEl.readyState < 2) {
        requestAnimationFrame(() => this._sendFrameLoop());
        return;
      }

      canvas.width = videoEl.videoWidth;
      canvas.height = videoEl.videoHeight;
      canvas.getContext('2d').drawImage(videoEl, 0, 0);

      canvas.toBlob((blob) => {
        if (blob && this.ws && this.ws.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then(buf => {
            this.ws.send(new Uint8Array(buf));
          });
        }
        // Next frame — throttle to ~15 fps to avoid overwhelming the server
        setTimeout(() => {
          if (this.webcamActive) this._sendFrameLoop();
        }, 66);
      }, 'image/jpeg', 0.75);
    },

    stopWebcam() {
      this.webcamActive = false;
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
      this.loadVideoSessions();
    },

    // ── Upload video ──────────────────────────────────────────────────────
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
