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
    ...videoApp(),
    ...adminApp(),
    ...communityApp(),

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

    /* ── Nav tabs (6 tabs — with Heroicons SVG paths) ────────────────── */
    tabs: [
      {
        id: 'dashboard', label: 'Acasă', short: 'Acasă',
        svgPath: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1h-2z'
      },
      {
        id: 'scan', label: 'Scanare', short: 'Scanare',
        svgPath: 'M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z M15 13a3 3 0 11-6 0 3 3 0 016 0z'
      },
      {
        id: 'map', label: 'Hartă', short: 'Hartă',
        svgPath: 'M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7'
      },
      {
        id: 'community', label: 'Comunitate', short: 'Comunitate',
        svgPath: 'M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z'
      },
      {
        id: 'reports', label: 'Rapoarte', short: 'Rapoarte',
        svgPath: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2'
      },
      {
        id: 'more', label: 'Mai mult', short: 'Mai mult',
        svgPath: 'M4 6h16M4 12h16M4 18h16'
      },
    ],

    // Admin-only tab (accessed from "Mai mult" page, not in bottom nav)
    adminTab: {
      id: 'admin', label: 'Admin', short: 'Admin',
      svgPath: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z'
    },

    // Sub-navigation state
    scanMode: 'foto',             // 'foto' | 'live' | 'batch'
    communitySubTab: 'feed',      // 'feed' | 'top' | 'announcements' | 'campaigns'

    // Onboarding
    onboardingOpen: false,
    onboardingStep: 1,

    // Impact metrics
    impactMetrics: null,

    // Avatar upload
    avatarUploading: false,

    // Photo gallery
    sessionPhotos: [],
    sessionPhotosLoading: false,
    photoUploading: false,

    /* ── Init ─────────────────────────────────────────────────────────── */
    async init() {
      this.darkMode = localStorage.getItem('eco_dark') === 'true' ||
        (!localStorage.getItem('eco_dark') && window.matchMedia('(prefers-color-scheme: dark)').matches);

      if (this.darkMode) document.documentElement.classList.add('dark');

      registerToastAlpine(this);

      this.initAuth();
      this.initDetect();
      this.initHistory();
      this.initMap();
      this.initVideo();
      this.initAdmin();
      this.initCommunity();

      if (this.isLoggedIn) {
        await this.loadDashboard();
        this._setupAdminTab();
        this.loadNotifications();
        this._notifInterval = setInterval(() => this.loadNotifications(), 30000);
        this.loadImpactMetrics();
        // Check onboarding
        if (this.myProfile && !this.myProfile.onboarding_done) {
          this.onboardingOpen = true;
        }
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
        if (tab === 'reports') this.loadHistory();
        if (tab === 'dashboard') this.loadDashboard();
        if (tab === 'community') {
          if (this.communitySubTab === 'top') this.loadLeaderboard();
          else this.loadCommunityFeed();
        }
        if (tab === 'admin') this.loadAdminAll();
        if (tab === 'more') this.loadPrivacySettings();
        // Stop webcam when leaving scan tab or switching away from live mode
        if (tab !== 'scan' && this.webcamActive) this.stopWebcam();
        this.sidebarOpen = false;
      });

      this.$watch('scanMode', (mode) => {
        if (mode !== 'live' && this.webcamActive) this.stopWebcam();
      });

      this.$watch('communitySubTab', (sub) => {
        if (this.activeTab !== 'community') return;
        if (sub === 'top') this.loadLeaderboard();
        else if (sub === 'feed') this.loadCommunityFeed();
        else if (sub === 'campaigns') this.loadCampaigns();
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
    myProfile: null,

    async loadDashboard() {
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

    /* ── Impact metrics ───────────────────────────────────────────────── */
    async loadImpactMetrics() {
      try {
        this.impactMetrics = await fetch('/api/impact').then(r => r.ok ? r.json() : null);
      } catch (_) {}
    },

    /* ── Onboarding ───────────────────────────────────────────────────── */
    async finishOnboarding() {
      this.onboardingOpen = false;
      try {
        await fetchAPI('/api/me/onboarding-done', { method: 'POST' });
      } catch (_) {}
    },

    /* ── Avatar upload ────────────────────────────────────────────────── */
    async uploadAvatar(event) {
      const file = event.target.files?.[0];
      if (!file) return;
      this.avatarUploading = true;
      try {
        const form = new FormData();
        form.append('file', file);
        const res = await fetchAPI('/api/me/avatar', { method: 'POST', body: form });
        if (this.myProfile) this.myProfile.avatar_url = res.avatar_url;
        showToast('Avatar actualizat ✓');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.avatarUploading = false;
      }
    },

    /* ── Photo gallery ────────────────────────────────────────────────── */
    async loadSessionPhotos(sessionId) {
      this.sessionPhotosLoading = true;
      try {
        this.sessionPhotos = await fetchAPI(`/api/sessions/${sessionId}/photos`);
      } catch (_) {
        this.sessionPhotos = [];
      } finally {
        this.sessionPhotosLoading = false;
      }
    },

    async uploadSessionPhoto(event, sessionId) {
      const file = event.target.files?.[0];
      if (!file) return;
      this.photoUploading = true;
      try {
        const form = new FormData();
        form.append('file', file);
        const photo = await fetchAPI(`/api/sessions/${sessionId}/photos`, { method: 'POST', body: form });
        this.sessionPhotos.push(photo);
        showToast('Foto adăugată ✓');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.photoUploading = false;
      }
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
