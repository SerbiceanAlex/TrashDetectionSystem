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
      }

      window.addEventListener('eco:authChanged', async () => {
        if (this.isLoggedIn) {
          await this.loadDashboard();
          this._setupAdminTab();
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
    async loadDashboard() {
      this.dashLoading = true;
      try {
        const r = await fetch('/api/stats');
        if (r.ok) this.dashStats = await r.json();
      } catch (_) {}
      this.dashLoading = false;
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
