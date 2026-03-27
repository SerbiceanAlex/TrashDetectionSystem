/* ── EcoAlert — Statistici tab ───────────────────────────────────────────── */

function statsApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    stats: null,
    isLoadingStats: false,
    pieChart: null,
    barChart: null,
    stackedChart: null,

    /* ── Init ─────────────────────────────────────────────────────────── */
    initStats() {
      this.loadStats();
      window.addEventListener('eco:statsChanged', () => this.loadStats());
      window.addEventListener('eco:newReport', () => this.loadStats());
    },

    /* ── Load ─────────────────────────────────────────────────────────── */
    async loadStats() {
      this.isLoadingStats = true;
      try {
        this.stats = await fetchAPI('/api/stats');
      } catch (e) {
        // silently ignore — stats are secondary
      } finally {
        this.isLoadingStats = false;
      }
    },

    async renderCharts() {
      if (!this.stats) await this.loadStats();
      await this.$nextTick();
      setTimeout(() => {
        this._renderPie();
        this._renderBar();
        this._renderStackedBar();
      }, 80);
    },

    /* ── Pie chart ────────────────────────────────────────────────────── */
    _renderPie() {
      const ctx = document.getElementById('pieChart');
      if (!ctx) return;
      const dist = this.stats?.material_distribution || [];
      if (!dist.length) return;

      if (this.pieChart) { this.pieChart.destroy(); this.pieChart = null; }

      const dark = document.documentElement.classList.contains('dark');
      const palette = ['#16a34a', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899'];

      this.pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: dist.map(d => d.material),
          datasets: [{
            data: dist.map(d => d.count),
            backgroundColor: palette,
            borderWidth: 3,
            borderColor: dark ? '#111827' : '#ffffff',
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '62%',
          plugins: {
            legend: {
              position: 'right',
              labels: {
                font: { size: 12, family: "'Inter', sans-serif" },
                padding: 16,
                color: dark ? '#d1d5db' : '#374151',
                usePointStyle: true,
                pointStyleWidth: 12,
              },
            },
            tooltip: {
              callbacks: {
                label: (c) => {
                  const tot = c.dataset.data.reduce((a, b) => a + b, 0);
                  return ` ${c.label}: ${c.parsed} (${((c.parsed / tot) * 100).toFixed(1)}%)`;
                },
              },
            },
          },
        },
      });
    },

    /* ── Bar chart (timeline) ─────────────────────────────────────────── */
    _renderBar() {
      const ctx = document.getElementById('barChart');
      if (!ctx) return;
      const tl = this.stats?.timeline || [];
      if (!tl.length) return;

      if (this.barChart) { this.barChart.destroy(); this.barChart = null; }

      const dark = document.documentElement.classList.contains('dark');

      this.barChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: tl.map(t => formatDateShort(t.day + 'T00:00:00')),
          datasets: [{
            label: 'Obiecte detectate',
            data: tl.map(t => t.total),
            backgroundColor: '#4ade80',
            borderColor: '#16a34a',
            borderWidth: 1,
            borderRadius: 6,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: {
              beginAtZero: true,
              ticks: { stepSize: 1, color: dark ? '#9ca3af' : '#6b7280', font: { size: 11 } },
              grid: { color: dark ? '#1f2937' : '#f3f4f6' },
            },
            x: {
              ticks: { font: { size: 10 }, color: dark ? '#9ca3af' : '#6b7280' },
              grid: { display: false },
            },
          },
        },
      });
    },

    /* ── Stacked bar (material per day) ───────────────────────────────── */
    _renderStackedBar() {
      const ctx = document.getElementById('stackedBarChart');
      if (!ctx) return;
      const mpd = this.stats?.material_per_day || [];
      if (!mpd.length) return;

      if (this.stackedChart) { this.stackedChart.destroy(); this.stackedChart = null; }

      const dark = document.documentElement.classList.contains('dark');
      const days = [...new Set(mpd.map(r => r.day))].sort();
      const materials = [...new Set(mpd.map(r => r.material))];
      const palette = {
        plastic: '#3b82f6', glass: '#06b6d4',
        metal: '#f59e0b', paper: '#f97316', other: '#6b7280',
      };

      const datasets = materials.map(mat => ({
        label: mat,
        data: days.map(day => {
          const found = mpd.find(r => r.day === day && r.material === mat);
          return found ? found.count : 0;
        }),
        backgroundColor: palette[mat.toLowerCase()] || '#8b5cf6',
        borderRadius: 4,
      }));

      this.stackedChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: days.map(d => formatDateShort(d + 'T00:00:00')), datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: {
                font: { size: 11, family: "'Inter', sans-serif" },
                color: dark ? '#d1d5db' : '#374151',
                usePointStyle: true,
              },
            },
          },
          scales: {
            x: {
              stacked: true,
              ticks: { font: { size: 10 }, color: dark ? '#9ca3af' : '#6b7280' },
              grid: { display: false },
            },
            y: {
              stacked: true,
              beginAtZero: true,
              ticks: { stepSize: 1, color: dark ? '#9ca3af' : '#6b7280', font: { size: 11 } },
              grid: { color: dark ? '#1f2937' : '#f3f4f6' },
            },
          },
        },
      });
    },

    /* ── Computed ─────────────────────────────────────────────────────── */
    get topMaterial() {
      const dist = this.stats?.material_distribution || [];
      return dist.length ? dist[0] : null;
    },

    materialBadgeStyle(m) { return materialBadgeStyle(m); },

    /* ── Export ───────────────────────────────────────────────────────── */
    downloadCsv() {
      window.location.href = '/api/export/csv';
    },

    downloadPdf() {
      window.location.href = '/api/export/pdf';
    },
  };
}
