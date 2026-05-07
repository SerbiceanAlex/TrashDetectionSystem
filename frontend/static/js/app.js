/* ── TrashDet — Core app shell ────────────────────────────────────────────
   Community trash reporting platform powered by YOLOv8 AI.
   ─────────────────────────────────────────────────────────────────────── */

function ecoApp() {
  return {
    /* ── Spread all sub-modules ──────────────────────────────────────── */
    ...authApp(),
    ...detectApp(),
    ...videoApp(),
    ...adminApp(),

    /* ── Global state ─────────────────────────────────────────────────── */
    activeTab: 'dashboard',
    darkMode: false,
    sidebarOpen: false,

    // Toast system
    toasts: [],

    // Lightbox
    lightboxSrc: null,

    // Dashboard stats (legacy)
    dashStats: null,
    dashLoading: false,

    // Dashboard B2B
    dashB2B: null,
    dashB2BLoading: false,
    dashB2BTrendChart: null,

    // Locations (B2B multi-camera)
    locations: [],
    locationsLoading: false,
    locationModalOpen: false,
    locationEdit: null,
    locationForm: { name: '', address: '', lat: null, lng: null, rtsp_url: '', alert_email: '', is_active: true },

    // Reports
    reportStats: null,
    reportPeriod: 'week',  // 'today' | 'week' | 'month' | 'year' | 'custom'
    reportFrom: '',
    reportTo: '',

    /* ── Nav tabs B2B (6 tabs — Heroicons SVG paths) ─────────────────── */
    tabs: [
      {
        id: 'dashboard', label: 'Dashboard', short: 'Dashboard',
        svgPath: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1h-2z'
      },
      {
        id: 'scan', label: 'Monitor', short: 'Monitor',
        svgPath: 'M15 10l4.553-2.069A1 1 0 0121 8.82v6.36a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z'
      },
      {
        id: 'incidents', label: 'Incidente', short: 'Incidente',
        svgPath: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z'
      },
      {
        id: 'locations', label: 'Locații', short: 'Locații',
        svgPath: 'M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0zM15 11a3 3 0 11-6 0 3 3 0 016 0z'
      },
      {
        id: 'reports', label: 'Rapoarte', short: 'Rapoarte',
        svgPath: 'M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z'
      },
      {
        id: 'more', label: 'Setări', short: 'Setări',
        svgPath: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065zM15 12a3 3 0 11-6 0 3 3 0 016 0z'
      },
    ],

    // Admin-only tab (accessed from "Mai mult" page, not in bottom nav)
    adminTab: {
      id: 'admin', label: 'Admin', short: 'Admin',
      svgPath: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z'
    },


    /* ── Init ─────────────────────────────────────────────────────────── */
    async init() {
      this.darkMode = localStorage.getItem('eco_dark') === 'true' ||
        (!localStorage.getItem('eco_dark') && window.matchMedia('(prefers-color-scheme: dark)').matches);

      if (this.darkMode) document.documentElement.classList.add('dark');

      registerToastAlpine(this);

      this.initAuth();
      this.initDetect();
      this.initVideo();
      this.initAdmin();

      if (this.isLoggedIn) {
        await this.loadDashboard();
        this._setupAdminTab();
        this.loadNotifications();
        this._notifInterval = setInterval(() => this.loadNotifications(), 30000);
      }

      window.addEventListener('eco:authChanged', async () => {
        if (this.isLoggedIn) {
          await this.loadDashboard();
          this._setupAdminTab();
          this.loadNotifications();
          if (!this._notifInterval) {
            this._notifInterval = setInterval(() => this.loadNotifications(), 30000);
          }
        } else {
          clearInterval(this._notifInterval);
          this._notifInterval = null;
          this.notifications = [];
          this.unreadNotifications = 0;
        }
      });

      window.addEventListener('eco:newReport', () => {
        if (this.activeTab === 'dashboard') this.loadDashboard();
      });

      this.$watch('activeTab', (tab) => {
        if (tab === 'dashboard') this.loadDashboard();
        if (tab === 'incidents') this.loadIncidents();
        if (tab === 'locations') this.loadLocations();
        if (tab === 'reports') this.loadReportStats();
        if (tab === 'admin') this.loadAdminAll();
        if (tab !== 'scan' && this.monitorActive) this.stopMonitor();
        this.sidebarOpen = false;
        this.refreshIcons();
      });

      this.$watch('darkMode', (dark) => {
        document.documentElement.classList.toggle('dark', dark);
        localStorage.setItem('eco_dark', dark);
      });

      // Initialize Lucide icons after Alpine renders
      this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });
    },

    /** Re-render any new Lucide <i data-lucide> tags */
    refreshIcons() {
      this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });
    },

    /* ── Dashboard ────────────────────────────────────────────────────── */
    dashMyStats: null,

    /* ── Notifications ────────────────────────────────────────────────── */
    notifOpen: false,
    notifications: [],
    unreadNotifications: 0,
    _notifInterval: null,

    async loadNotifications() {
      if (!this.token) return;
      try {
        const data = await fetch('/api/me/notifications', {
          headers: { Authorization: 'Bearer ' + this.token }
        }).then(r => r.ok ? r.json() : null);
        if (data) {
          this.notifications = data.notifications;
          this.unreadNotifications = data.unread;
          this.refreshIcons();
        }
      } catch (_) {}
    },

    async markRead(notif) {
      if (notif.is_read) return;
      notif.is_read = 1;
      this.unreadNotifications = Math.max(0, this.unreadNotifications - 1);
      try {
        await fetch(`/api/me/notifications/${notif.id}/read`, {
          method: 'POST',
          headers: { Authorization: 'Bearer ' + this.token },
        });
      } catch (_) {}
    },

    async markAllRead() {
      this.notifications.forEach(n => { n.is_read = 1; });
      this.unreadNotifications = 0;
      try {
        await fetch('/api/me/notifications/read-all', {
          method: 'POST',
          headers: { Authorization: 'Bearer ' + this.token },
        });
      } catch (_) {}
    },

    formatNotifDate(iso) {
      if (!iso) return '';
      const d = new Date(iso);
      const now = new Date();
      const diff = Math.floor((now - d) / 1000);
      if (diff < 60) return 'just now';
      if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
      if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
      return d.toLocaleDateString('en-US', { day: 'numeric', month: 'short' });
    },
    dashWeeklyChart: null,
    myProfile: null,

    async loadDashboard() {
      this.dashB2BLoading = true;
      try {
        const data = await fetch('/api/dashboard/b2b', {
          headers: { Authorization: 'Bearer ' + this.token }
        }).then(r => r.ok ? r.json() : null);
        if (data) this.dashB2B = data;
        await this.$nextTick();
        setTimeout(() => this._renderB2BTrendChart(), 100);
      } catch (e) { console.error('loadDashboard B2B', e); }
      this.dashB2BLoading = false;
    },

    _renderB2BTrendChart() {
      const canvas = document.getElementById('dashB2BTrendChart');
      if (!canvas || !this.dashB2B?.trend_30d) return;
      if (this.dashB2BTrendChart) { this.dashB2BTrendChart.destroy(); this.dashB2BTrendChart = null; }

      const trend = this.dashB2B.trend_30d || [];
      const labels = trend.map(p => {
        const dt = new Date(p.day + 'T12:00:00');
        return dt.toLocaleDateString('ro-RO', { day: '2-digit', month: 'short' });
      });
      const counts = trend.map(p => p.count);
      const dark = document.documentElement.classList.contains('dark');

      this.dashB2BTrendChart = new Chart(canvas, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Incidente',
            data: counts,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16,185,129,0.12)',
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: '#10b981',
            borderWidth: 2.5,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, ticks: { stepSize: 1, precision: 0, color: dark ? '#9ca3af' : '#6b7280', font: { size: 11 } }, grid: { color: dark ? '#1f2937' : '#f3f4f6' } },
            x: { ticks: { color: dark ? '#9ca3af' : '#6b7280', font: { size: 10 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }, grid: { display: false } },
          },
        },
      });
    },

    /* ── B2B Locations ────────────────────────────────────────────────── */
    async loadLocations() {
      this.locationsLoading = true;
      try {
        const data = await fetch('/api/locations', {
          headers: { Authorization: 'Bearer ' + this.token }
        }).then(r => r.ok ? r.json() : null);
        if (data) this.locations = data.locations || data;
      } catch (e) { console.error('loadLocations', e); }
      this.locationsLoading = false;
      this.$nextTick(() => { if (window.lucide) window.lucide.createIcons(); });
    },

    async saveLocation() {
      try {
        const url = this.locationEdit
          ? '/api/locations/' + this.locationEdit.id
          : '/api/locations';
        const method = this.locationEdit ? 'PATCH' : 'POST';
        const r = await fetch(url, {
          method,
          headers: {
            'Authorization': 'Bearer ' + this.token,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(this.locationForm),
        });
        if (!r.ok) throw new Error(await r.text());
        this.toast('Locație salvată', 'success');
        this.locationModalOpen = false;
        this.locationForm = { name: '', address: '', lat: null, lng: null, rtsp_url: '', alert_email: '', is_active: true };
        await this.loadLocations();
      } catch (e) {
        this.toast('Eroare: ' + e.message, 'error');
      }
    },

    async toggleLocation(loc) {
      try {
        await fetch('/api/locations/' + loc.id, {
          method: 'PATCH',
          headers: {
            'Authorization': 'Bearer ' + this.token,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ is_active: !loc.is_active }),
        });
        await this.loadLocations();
      } catch (e) { this.toast('Eroare', 'error'); }
    },

    async deleteLocation(loc) {
      if (!confirm('Ștergi locația "' + loc.name + '"?')) return;
      try {
        await fetch('/api/locations/' + loc.id, {
          method: 'DELETE',
          headers: { 'Authorization': 'Bearer ' + this.token },
        });
        this.toast('Locație ștearsă', 'success');
        await this.loadLocations();
      } catch (e) { this.toast('Eroare', 'error'); }
    },

    /* ── Reports / Export ─────────────────────────────────────────────── */
    async loadReportStats() {
      try {
        const params = new URLSearchParams({ period: this.reportPeriod });
        if (this.reportPeriod === 'custom') {
          if (this.reportFrom) params.set('from', this.reportFrom);
          if (this.reportTo)   params.set('to', this.reportTo);
        }
        const data = await fetch('/api/reports/stats?' + params, {
          headers: { Authorization: 'Bearer ' + this.token }
        }).then(r => r.ok ? r.json() : null);
        if (data) this.reportStats = data;
      } catch (e) { console.error('loadReportStats', e); }
    },

    async exportReport(format) {
      try {
        const params = new URLSearchParams({ period: this.reportPeriod, format });
        if (this.reportPeriod === 'custom') {
          if (this.reportFrom) params.set('from', this.reportFrom);
          if (this.reportTo)   params.set('to', this.reportTo);
        }
        await this._downloadWithAuth('/api/reports/export?' + params, `raport_${this.reportPeriod}.csv`);
      } catch (e) { this.toast('Eroare export', 'error'); }
    },

    async exportIncidentsCSV() {
      try {
        await this._downloadWithAuth('/api/reports/export?format=csv&period=month', 'incidente.csv');
        this.toast('CSV descărcat cu succes', 'success');
      } catch (e) { this.toast('Eroare export: ' + e.message, 'error'); }
    },

    async _downloadWithAuth(url, filename) {
      const resp = await fetch(url, { headers: { Authorization: 'Bearer ' + this.token } });
      if (!resp.ok) throw new Error(await resp.text());
      const blob = await resp.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
    },

    /* ── Legacy dashboard (păstrat pentru compatibilitate) ────────────── */
    async _legacy_loadDashboard() {
      this.dashLoading = true;
      try {
        const [global, personal, profile] = await Promise.all([
          fetch('/api/stats').then(r => r.ok ? r.json() : null),
          fetch('/api/me/stats', {
            headers: { Authorization: 'Bearer ' + this.token }
          }).then(r => r.ok ? r.json() : null).catch(() => null),
          fetch('/api/me/profile', {
            headers: { Authorization: 'Bearer ' + this.token }
          }).then(r => r.ok ? r.json() : null).catch(() => null),
        ]);
        if (global)   this.dashStats   = global;
        if (personal) this.dashMyStats  = personal;
        if (profile)  this.myProfile    = profile;
      } catch (_) {}
      this.dashLoading = false;
    },

    _renderWeeklyChart() {
      const canvas = document.getElementById('dashWeeklyChart');
      if (!canvas || !this.dashMyStats?.weekly_activity) return;
      if (this.dashWeeklyChart) { this.dashWeeklyChart.destroy(); this.dashWeeklyChart = null; }

      // Build last-7-days labels
      const days = [];
      for (let i = 6; i >= 0; i--) {
        const d = new Date(); d.setDate(d.getDate() - i);
        days.push(d.toISOString().slice(0, 10));
      }
      const actMap = {};
      (this.dashMyStats.weekly_activity || []).forEach(r => { actMap[r.day] = r.reports; });
      const counts = days.map(d => actMap[d] || 0);
      const labels = days.map(d => {
        const dt = new Date(d + 'T12:00:00');
        return dt.toLocaleDateString('en-US', { weekday: 'short', day: 'numeric' });
      });
      const dark = document.documentElement.classList.contains('dark');

      this.dashWeeklyChart = new Chart(canvas, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Reports',
            data: counts,
            backgroundColor: counts.map((v, i) => i === 6 ? '#16a34a' : 'rgba(74,222,128,.55)'),
            borderRadius: 6,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, ticks: { stepSize: 1, precision: 0, color: dark ? '#9ca3af' : '#6b7280', font: { size: 11 } }, grid: { color: dark ? '#1f2937' : '#f3f4f6' } },
            x: { ticks: { color: dark ? '#9ca3af' : '#6b7280', font: { size: 10 } }, grid: { display: false } },
          },
        },
      });
    },

    /* ── Tab navigation ───────────────────────────────────────────────── */
    goTo(tab) { this.activeTab = tab; },

    /* ── Admin check (admin accessible from "Mai mult" page) ────────── */
    get isAdmin() { return this.user?.role === 'admin'; },
    _setupAdminTab() {
      // Admin is now accessed from the "Mai mult" page, no tab injection needed
    },

    /* ── Dark mode ────────────────────────────────────────────────────── */
    toggleDark() { this.darkMode = !this.darkMode; },

    /* ── Toast ────────────────────────────────────────────────────────── */
    addToast(message, type = 'success', duration = 3500) {
      const id = Date.now() + Math.random();
      this.toasts.push({ id, message, type, visible: true });
      setTimeout(() => {
        const t = this.toasts.find(x => x.id === id);
        if (t) t.visible = false;
        setTimeout(() => { this.toasts = this.toasts.filter(x => x.id !== id); }, 300);
      }, duration);
    },

    /* ── Lightbox ─────────────────────────────────────────────────────── */
    openLightbox(src) { this.lightboxSrc = src; },
    closeLightbox() { this.lightboxSrc = null; },
  };
}
