/* ── TrashDet — Core app shell ────────────────────────────────────────────
   Community trash reporting platform powered by YOLOv8 AI.
   ─────────────────────────────────────────────────────────────────────── */

function ecoApp() {
  return {
    /* ── Spread all sub-modules ──────────────────────────────────────── */
    ...authApp(),
    ...detectApp(),
    ...mapApp(),
    ...historyApp(),
    ...statsApp(),
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

    // Dashboard stats
    dashStats: null,
    dashLoading: false,

    /* ── Nav tabs (with Heroicons SVG paths) ──────────────────────────── */
    tabs: [
      {
        id: 'dashboard', label: 'Dashboard', short: 'Acasă',
        svgPath: 'M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z'
      },
      {
        id: 'scan', label: 'Scanează', short: 'Scanează',
        svgPath: 'M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z M15 13a3 3 0 11-6 0 3 3 0 016 0z'
      },
      {
        id: 'map', label: 'Hartă', short: 'Hartă',
        svgPath: 'M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7'
      },
      {
        id: 'reports', label: 'Rapoarte', short: 'Rapoarte',
        svgPath: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z'
      },
      {
        id: 'leaderboard', label: 'Leaderboard', short: 'Top',
        svgPath: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'
      },
      {
        id: 'about', label: 'Despre', short: 'Info',
        svgPath: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'
      },
      {
        id: 'settings', label: 'Setări', short: 'Setări',
        svgPath: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z'
      },
    ],

    // Admin-only tab (appended dynamically after login based on role)
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
      this.initHistory();
      this.initStats();
      this.initMap();
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
        if (tab === 'map') this.ensureMap();
        if (tab === 'stats') setTimeout(() => this.renderCharts(), 150);
        if (tab === 'reports') this.loadHistory();
        if (tab === 'dashboard') this.loadDashboard();
        if (tab === 'leaderboard') this.loadLeaderboard();
        if (tab === 'admin') this.loadAdminAll();
        if (tab !== 'live' && this.webcamActive) this.stopWebcam();
        this.sidebarOpen = false;
      });

      this.$watch('darkMode', (dark) => {
        document.documentElement.classList.toggle('dark', dark);
        localStorage.setItem('eco_dark', dark);
      });
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
      if (diff < 60) return 'acum câteva secunde';
      if (diff < 3600) return `acum ${Math.floor(diff / 60)} min`;
      if (diff < 86400) return `acum ${Math.floor(diff / 3600)} ore`;
      return d.toLocaleDateString('ro-RO', { day: 'numeric', month: 'short' });
    },
    dashWeeklyChart: null,

    async loadDashboard() {
      this.dashLoading = true;
      try {
        const [global, personal] = await Promise.all([
          fetch('/api/stats').then(r => r.ok ? r.json() : null),
          fetch('/api/me/stats', {
            headers: { Authorization: 'Bearer ' + this.token }
          }).then(r => r.ok ? r.json() : null).catch(() => null),
        ]);
        if (global)   this.dashStats   = global;
        if (personal) this.dashMyStats  = personal;
        await this.$nextTick();
        this._renderWeeklyChart();
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
        return dt.toLocaleDateString('ro-RO', { weekday: 'short', day: 'numeric' });
      });
      const dark = document.documentElement.classList.contains('dark');

      this.dashWeeklyChart = new Chart(canvas, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Rapoarte',
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

    /* ── Admin tab injection (only for admin role) ────────────────────── */
    _setupAdminTab() {
      const hasAdmin = this.tabs.some(t => t.id === 'admin');
      if (this.user?.role === 'admin' && !hasAdmin) {
        // Insert admin tab before settings
        const settingsIdx = this.tabs.findIndex(t => t.id === 'settings');
        if (settingsIdx >= 0) {
          this.tabs.splice(settingsIdx, 0, this.adminTab);
        } else {
          this.tabs.push(this.adminTab);
        }
      } else if (this.user?.role !== 'admin' && hasAdmin) {
        this.tabs = this.tabs.filter(t => t.id !== 'admin');
      }
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
