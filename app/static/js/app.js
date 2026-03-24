/* ── Alpine.js app logic for Trash Detection System ──────────────────── */

function app() {
  return {
    tabs: [
      { id: 'detect',  label: '🔍 Detectare',  icon: '🔍', shortLabel: 'Detectare'  },
      { id: 'history', label: '📋 Istoric',    icon: '📋', shortLabel: 'Istoric'    },
      { id: 'stats',   label: '📊 Statistici', icon: '📊', shortLabel: 'Statistici' },
      { id: 'export',  label: '📥 Export',     icon: '📥', shortLabel: 'Export'     },
    ],
    activeTab: 'detect',

    // Dark mode
    darkMode: false,

    // Single detect
    selectedFile: null,
    previewSrc: null,
    isDragging: false,
    detConf: 0.25,
    loading: false,
    uploadProgress: 0,
    result: null,
    showAnnotated: true,
    originalImageUrl: null,

    // Rerun
    rerunConf: 0.25,
    isRerunning: false,

    // Batch
    batchMode: false,
    batchFiles: [],
    batchLoading: false,
    batchProgress: 0,
    batchResults: null,

    // Toast & Lightbox
    toasts: [],
    lightboxSrc: null,

    // History
    history: { total: 0, skip: 0, limit: 10, items: [] },
    historyPage: 0,
    historyLimit: 10,
    sessionDetail: null,
    isLoadingHistory: false,
    searchQuery: '',
    filterMaterial: '',
    filterMinObjects: null,

    // Stats
    stats: {},
    pieChart: null,
    barChart: null,
    stackedChart: null,

    // ── Init ────────────────────────────────────────────────────────────
    async init() {
      this.darkMode = localStorage.getItem('darkMode') === 'true' ||
        (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches);

      await Promise.all([this.loadStats(), this.loadHistory()]);

      this.$watch('activeTab', async (tab) => {
        if (tab === 'stats') {
          await this.loadStats();
          setTimeout(() => this.renderCharts(), 120);
        }
        if (tab === 'history') await this.loadHistory();
      });
    },

    // ── Dark mode ────────────────────────────────────────────────────────
    toggleDark() {
      this.darkMode = !this.darkMode;
      localStorage.setItem('darkMode', this.darkMode);
    },

    // ── Toast ────────────────────────────────────────────────────────────
    showToast(message, type = 'success') {
      const id = Date.now();
      this.toasts.push({ id, message, type, visible: true });
      setTimeout(() => {
        const t = this.toasts.find(x => x.id === id);
        if (t) t.visible = false;
        setTimeout(() => { this.toasts = this.toasts.filter(x => x.id !== id); }, 250);
      }, 3500);
    },

    // ── File handling ────────────────────────────────────────────────────
    handleDrop(ev) { this.isDragging = false; const f = ev.dataTransfer.files[0]; if (f) this.setFile(f); },
    handleFileSelect(ev) { const f = ev.target.files[0]; if (f) this.setFile(f); },
    setFile(file) {
      this.selectedFile = file;
      this.result = null;
      this.originalImageUrl = null;
      const r = new FileReader(); r.onload = (e) => this.previewSrc = e.target.result; r.readAsDataURL(file);
    },
    clearFile() { this.selectedFile = null; this.previewSrc = null; this.result = null; this.originalImageUrl = null; this.$refs.fileInput.value = ''; },

    // ── Single detection ─────────────────────────────────────────────────
    async runDetection() {
      if (!this.selectedFile) return;
      this.loading = true; this.result = null; this.uploadProgress = 0;
      let prog = 0;
      const interval = setInterval(() => {
        if (prog < 88) { prog += (88 - prog) * 0.12 + 0.8; this.uploadProgress = Math.min(Math.round(prog), 88); }
      }, 180);
      try {
        const fd = new FormData(); fd.append('file', this.selectedFile);
        const resp = await fetch(`/api/detect?det_conf=${this.detConf}`, { method: 'POST', body: fd });
        clearInterval(interval); this.uploadProgress = 100;
        if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || `Eroare server (${resp.status})`); }
        this.result = await resp.json();
        this.rerunConf = this.detConf;
        this.showAnnotated = true;
        this.originalImageUrl = `/api/sessions/${this.result.session_id}/original`;
        this.showToast(`${this.result.total_objects} obiecte detectate — ${this.result.inference_ms.toFixed(0)} ms`);
        this.loadStats(); this.loadHistory();
      } catch (e) { clearInterval(interval); this.showToast(e.message, 'error'); }
      finally { this.loading = false; setTimeout(() => { this.uploadProgress = 0; }, 900); }
    },

    // ── Rerun with different conf ────────────────────────────────────────
    async rerunDetection() {
      if (!this.result) return;
      this.isRerunning = true;
      try {
        const resp = await fetch(`/api/sessions/${this.result.session_id}/rerun?det_conf=${this.rerunConf}`, { method: 'POST' });
        if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || 'Eroare'); }
        const newResult = await resp.json();
        newResult.annotated_url = newResult.annotated_url + '?t=' + Date.now();
        this.result = newResult;
        this.showAnnotated = true;
        this.showToast(`Re-detectare: ${newResult.total_objects} obiecte (conf ${this.rerunConf.toFixed(2)})`);
        this.loadStats(); this.loadHistory();
      } catch (e) { this.showToast(e.message, 'error'); }
      finally { this.isRerunning = false; }
    },

    // ── Batch ────────────────────────────────────────────────────────────
    handleBatchSelect(ev) { this.batchFiles = Array.from(ev.target.files); this.batchResults = null; },
    async runBatch() {
      if (!this.batchFiles.length) return;
      this.batchLoading = true; this.batchProgress = 0; this.batchResults = null;
      try {
        const fd = new FormData();
        this.batchFiles.forEach(f => fd.append('files', f));
        const resp = await fetch(`/api/detect/batch?det_conf=${this.detConf}`, { method: 'POST', body: fd });
        if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || 'Eroare batch'); }
        this.batchResults = await resp.json();
        this.batchProgress = this.batchFiles.length;
        this.showToast(`Batch: ${this.batchResults.total_files} imagini, ${this.batchResults.total_objects} obiecte`);
        this.loadStats(); this.loadHistory();
      } catch (e) { this.showToast(e.message, 'error'); }
      finally { this.batchLoading = false; }
    },

    // ── History ──────────────────────────────────────────────────────────
    async loadHistory() {
      this.isLoadingHistory = true;
      try {
        let url = `/api/sessions?skip=${this.historyPage * this.historyLimit}&limit=${this.historyLimit}`;
        if (this.searchQuery) url += `&q=${encodeURIComponent(this.searchQuery)}`;
        if (this.filterMaterial) url += `&material=${encodeURIComponent(this.filterMaterial)}`;
        if (this.filterMinObjects != null && this.filterMinObjects !== '' && this.filterMinObjects >= 0) url += `&min_objects=${this.filterMinObjects}`;
        const r = await fetch(url);
        if (r.ok) this.history = await r.json();
      } finally { this.isLoadingHistory = false; }
    },
    async openSessionDetail(id) { const r = await fetch(`/api/sessions/${id}`); if (r.ok) this.sessionDetail = await r.json(); },
    async deleteSession(id) {
      if (!confirm(`Ștergi sesiunea #${id}?`)) return;
      const r = await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
      if (r.ok) { this.showToast('Sesiunea a fost ștearsă.'); await this.loadHistory(); await this.loadStats(); }
      else this.showToast('Eroare la ștergere.', 'error');
    },
    getAnnotatedUrl(path) { return `/annotated/${path.split(/[\\/]/).pop()}`; },

    // ── Stats ────────────────────────────────────────────────────────────
    async loadStats() {
      const r = await fetch('/api/stats');
      if (r.ok) { this.stats = await r.json(); if (this.activeTab === 'stats') setTimeout(() => this.renderCharts(), 120); }
    },
    renderCharts() { this.renderPie(); this.renderBar(); this.renderStackedBar(); },

    renderPie() {
      const ctx = document.getElementById('pieChart'); if (!ctx) return;
      const dist = this.stats.material_distribution || []; if (!dist.length) return;
      if (this.pieChart) { this.pieChart.destroy(); this.pieChart = null; }
      const palette = ['#16a34a','#3b82f6','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#ec4899'];
      this.pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: dist.map(d => d.material),
          datasets: [{
            data: dist.map(d => d.count),
            backgroundColor: palette,
            borderWidth: 3,
            borderColor: this.darkMode ? '#1f2937' : '#fff'
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right', labels: { font: { size: 11 }, padding: 14, color: this.darkMode ? '#d1d5db' : '#6b7280' } },
            tooltip: { callbacks: { label: (c) => { const tot = c.dataset.data.reduce((a,b)=>a+b,0); return ` ${c.label}: ${c.parsed} (${((c.parsed/tot)*100).toFixed(1)}%)`; }}}
          }
        }
      });
    },

    renderBar() {
      const ctx = document.getElementById('barChart'); if (!ctx) return;
      const tl = this.stats.timeline || []; if (!tl.length) return;
      if (this.barChart) { this.barChart.destroy(); this.barChart = null; }
      this.barChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: tl.map(t => t.day),
          datasets: [{
            label: 'Obiecte', data: tl.map(t => t.total),
            backgroundColor: '#4ade80', borderColor: '#16a34a', borderWidth: 1, borderRadius: 5
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, ticks: { stepSize: 1, color: this.darkMode ? '#9ca3af' : '#6b7280' }, grid: { color: this.darkMode ? '#374151' : '#f3f4f6' } },
            x: { ticks: { font: { size: 10 }, color: this.darkMode ? '#9ca3af' : '#6b7280' }, grid: { display: false } }
          }
        }
      });
    },

    renderStackedBar() {
      const ctx = document.getElementById('stackedBarChart'); if (!ctx) return;
      const mpd = this.stats.material_per_day || []; if (!mpd.length) return;
      if (this.stackedChart) { this.stackedChart.destroy(); this.stackedChart = null; }

      const days = [...new Set(mpd.map(r => r.day))].sort();
      const materials = [...new Set(mpd.map(r => r.material))];
      const palette = { plastic:'#3b82f6', glass:'#06b6d4', metal:'#f59e0b', paper:'#f97316', other:'#6b7280' };
      const datasets = materials.map(mat => {
        const data = days.map(day => { const found = mpd.find(r => r.day === day && r.material === mat); return found ? found.count : 0; });
        return { label: mat, data, backgroundColor: palette[mat.toLowerCase()] || '#8b5cf6', borderRadius: 3 };
      });

      this.stackedChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: days, datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { font: { size: 11 }, color: this.darkMode ? '#d1d5db' : '#6b7280' } } },
          scales: {
            x: { stacked: true, ticks: { font: { size: 10 }, color: this.darkMode ? '#9ca3af' : '#6b7280' }, grid: { display: false } },
            y: { stacked: true, beginAtZero: true, ticks: { stepSize: 1, color: this.darkMode ? '#9ca3af' : '#6b7280' }, grid: { color: this.darkMode ? '#374151' : '#f3f4f6' } }
          }
        }
      });
    },

    // ── Helpers ──────────────────────────────────────────────────────────
    formatDate(iso) {
      if (!iso) return '—';
      return new Date(iso).toLocaleString('ro-RO', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    },
    materialColor(m) {
      return {
        plastic: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-400',
        glass:   'bg-cyan-100 dark:bg-cyan-900/40 text-cyan-700 dark:text-cyan-400',
        metal:   'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-400',
        paper:   'bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-400',
        other:   'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
      }[m?.toLowerCase()] || 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400';
    },
  };
}
