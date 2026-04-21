/* ── EcoAlert — Scanare tab ──────────────────────────────────────────────── */

function detectApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    selectedFile: null,
    previewSrc: null,
    isDragging: false,
    detConf: 0.50,
    loading: false,
    uploadProgress: 0,
    result: null,
    showAnnotated: true,
    originalImageUrl: null,

    // GPS
    geoLat: null,
    geoLng: null,
    geoStatus: '',      // '' | 'loading' | 'ok' | 'error'
    geoAccuracy: null,

    // Rerun
    rerunConf: 0.50,
    isRerunning: false,

    // User note
    userNote: '',

    // Batch state
    batchMode: false,
    batchFiles: [],
    batchLoading: false,
    batchProgress: 0,
    batchResults: null,

    /* ── Init ──────────────────────────────────────────────────────────── */
    initDetect() {
      // Auto-request GPS on init for mobile users
      this._autoGPS();
    },

    async _autoGPS() {
      try {
        const pos = await requestGPS();
        this.geoLat = pos.lat;
        this.geoLng = pos.lng;
        this.geoAccuracy = pos.accuracy;
        this.geoStatus = 'ok';
      } catch (_) {
        this.geoStatus = 'error';
      }
    },

    // Force a truly fresh GPS reading (maximumAge=0) — called when user picks a photo
    async _freshGPS() {
      this.geoStatus = 'loading';
      try {
        const pos = await requestGPS({ maximumAge: 0 });
        this.geoLat = pos.lat;
        this.geoLng = pos.lng;
        this.geoAccuracy = pos.accuracy;
        this.geoStatus = 'ok';
        showToast(`GPS capturat (±${Math.round(pos.accuracy || 0)}m) — locația va fi salvată cu raportul`, 'success');
      } catch (_) {
        this.geoStatus = 'error';
        showToast('GPS indisponibil — raportul va fi salvat fără locație', 'warning');
      }
    },

    /* ── GPS ────────────────────────────────────────────────────────────── */
    async requestGeo() {
      await this._freshGPS();
    },

    get geoLabel() {
      if (this.geoStatus === 'loading') return 'Se caută GPS…';
      if (this.geoStatus === 'ok') return `GPS (±${Math.round(this.geoAccuracy || 0)}m)`;
      if (this.geoStatus === 'error') return 'GPS indisponibil';
      return 'Activează GPS';
    },

    /* ── File handling ──────────────────────────────────────────────────── */
    handleDrop(ev) {
      this.isDragging = false;
      const f = ev.dataTransfer?.files?.[0];
      if (f) { this.setFile(f); this._freshGPS(); }
    },
    handleFileSelect(ev) {
      const f = ev.target?.files?.[0];
      if (f) { this.setFile(f); this._freshGPS(); }
    },
    setFile(file) {
      this.selectedFile = file;
      this.result = null;
      this.originalImageUrl = null;
      const r = new FileReader();
      r.onload = (e) => { this.previewSrc = e.target.result; };
      r.readAsDataURL(file);
    },
    clearFile() {
      this.selectedFile = null;
      this.previewSrc = null;
      this.result = null;
      this.originalImageUrl = null;
      const inp = document.getElementById('fileInput');
      if (inp) inp.value = '';
    },

    /* ── Single detection ───────────────────────────────────────────────── */
    async runDetection() {
      if (!this.selectedFile) return;
      this.loading = true;
      this.result = null;
      this.uploadProgress = 0;

      // Animated progress
      let prog = 0;
      const interval = setInterval(() => {
        if (prog < 88) {
          prog += (88 - prog) * 0.12 + 0.8;
          this.uploadProgress = Math.min(Math.round(prog), 88);
        }
      }, 180);

      try {
        // If GPS is still fetching (triggered at file-select), wait up to 3s
        if (this.geoStatus === 'loading') {
          for (let i = 0; i < 15; i++) {
            await new Promise(r => setTimeout(r, 200));
            if (this.geoStatus !== 'loading') break;
          }
        }

        const fd = new FormData();
        fd.append('file', this.selectedFile);

        let url = `/api/detect?det_conf=${this.detConf}`;
        if (this.geoLat !== null && this.geoLng !== null) {
          url += `&latitude=${this.geoLat}&longitude=${this.geoLng}`;
        }
        if (this.userNote.trim()) {
          url += `&user_note=${encodeURIComponent(this.userNote.trim())}`;
        }

        const data = await fetchAPI(url, { method: 'POST', body: fd });
        clearInterval(interval);
        this.uploadProgress = 100;

        this.result = data;
        this.rerunConf = this.detConf;
        this.showAnnotated = true;
        this.originalImageUrl = `/api/sessions/${data.session_id}/original`;

        const sev = getSeverity(severityFromCount(data.total_objects));
        showToast(`${sev.icon} ${data.total_objects} obiecte detectate — ${data.inference_ms.toFixed(0)} ms`);

        // Notify other tabs to refresh
        window.dispatchEvent(new CustomEvent('eco:newReport'));

      } catch (e) {
        clearInterval(interval);
        showToast(e.message, 'error');
      } finally {
        this.loading = false;
        setTimeout(() => { this.uploadProgress = 0; }, 900);
      }
    },

    /* ── Rerun ──────────────────────────────────────────────────────────── */
    async rerunDetection() {
      if (!this.result) return;
      this.isRerunning = true;
      try {
        const data = await fetchAPI(
          `/api/sessions/${this.result.session_id}/rerun?det_conf=${this.rerunConf}`,
          { method: 'POST' }
        );
        data.annotated_url = data.annotated_url + '?t=' + Date.now();
        this.result = data;
        this.showAnnotated = true;
        showToast(`Re-detectare: ${data.total_objects} obiecte (conf ${this.rerunConf.toFixed(2)})`);
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.isRerunning = false;
      }
    },

    /* ── Batch ──────────────────────────────────────────────────────────── */
    handleBatchSelect(ev) {
      this.batchFiles = Array.from(ev.target?.files || []);
      this.batchResults = null;
    },

    async runBatch() {
      if (!this.batchFiles.length) return;
      this.batchLoading = true;
      this.batchProgress = 0;
      this.batchResults = null;
      try {
        const fd = new FormData();
        this.batchFiles.forEach(f => fd.append('files', f));
        const data = await fetchAPI(`/api/detect/batch?det_conf=${this.detConf}`, { method: 'POST', body: fd });
        this.batchResults = data;
        this.batchProgress = this.batchFiles.length;
        showToast(`Batch: ${data.total_files} imagini, ${data.total_objects} obiecte`);
        window.dispatchEvent(new CustomEvent('eco:newReport'));
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.batchLoading = false;
      }
    },

    /* ── Computed helpers ───────────────────────────────────────────────── */
    get resultSeverity() {
      if (!this.result) return null;
      return getSeverity(severityFromCount(this.result.total_objects));
    },

    materialBadgeStyle(m) {
      return materialBadgeStyle(m);
    },
  };
}
