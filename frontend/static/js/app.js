/* ── TrashDet — Core app shell ────────────────────────────────────────────
   Illegal littering monitoring platform powered by YOLOv8 AI.
   ─────────────────────────────────────────────────────────────────────── */

function ecoApp() {
  return {
    /* ── Spread all sub-modules ──────────────────────────────────────── */
    ...authApp(),
    ...videoApp(),
    ...adminApp(),

    /* ── Global state ─────────────────────────────────────────────────── */
    activeTab: 'dashboard',
    darkMode: false,
    sidebarOpen: false,

    // Toast system
    toasts: [],

    // Confirm modal universal
    confirmModal: {
      open: false, title: '', message: '',
      icon: 'alert-triangle', iconColor: '#ef4444',
      confirmText: 'Confirmă', confirmColor: '#ef4444',
      cancelText: 'Anulează', resolve: null,
    },

    // Lightbox
    lightboxSrc: null,

    // Organization / trial
    org: null,

    // Payment modal
    payModalOpen: false,
    payModalPlan: null,
    payModalStep: 'confirm', // 'confirm' | 'processing' | 'success'

    // Dashboard B2B
    dashB2B: null,
    dashB2BLoading: false,
    get matTotal() {
      return (this.dashB2B?.material_distribution || []).reduce((s, m) => s + m.count, 0);
    },

    /* ── Nav tabs — minimal demo (Dashboard + Monitor + Incidente) ──── */
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

      await this.initAuth();
      this.initVideo();
      this.initAdmin();

      // Handle auth and billing URL params
      const _params = new URLSearchParams(window.location.search);
      const _action = _params.get('action');
      const _checkout = _params.get('checkout');

      if (this.isLoggedIn) {
        await this.loadOrg();
        await this.loadDashboard();
        this._setupAdminTab();
        this.loadNotifications();
        this._notifInterval = setInterval(() => this.loadNotifications(), 30000);
        if (_checkout === 'success') {
          this.payModalStep = 'success';
          this.payModalOpen = true;
        }
        if (_action === 'login' || _action === 'register') {
          this.$nextTick(() => showToast(`Ești deja conectat ca ${this.user?.username || 'utilizator'}`, 'info'));
        }
        history.replaceState({}, '', '/app');
      } else if (_action === 'register') {
        this.$nextTick(() => this.openAuth('register'));
        history.replaceState({}, '', '/app');
      } else if (_action === 'login') {
        this.$nextTick(() => this.openAuth('login'));
        history.replaceState({}, '', '/app');
      }

      window.addEventListener('eco:authChanged', async () => {
        if (this.isLoggedIn) {
          await this.loadOrg();
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
          this.org = null;
        }
      });

      window.addEventListener('eco:newReport', () => {
        if (this.activeTab === 'dashboard') this.loadDashboard();
      });

      this.$watch('activeTab', (tab) => {
        if (tab === 'dashboard') this.loadDashboard();
        if (tab === 'incidents') this.loadIncidents();
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

    // Confirm modal — returnează Promise<boolean>
    showConfirm(title, message, opts = {}) {
      return new Promise(resolve => {
        this.confirmModal = {
          open: true,
          title,
          message,
          icon:         opts.icon         ?? 'alert-triangle',
          iconColor:    opts.iconColor     ?? '#ef4444',
          confirmText:  opts.confirmText   ?? 'Confirmă',
          confirmColor: opts.confirmColor  ?? '#ef4444',
          cancelText:   opts.cancelText    ?? 'Anulează',
          resolve,
        };
        this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });
      });
    },
    confirmModalAccept() {
      if (this.confirmModal.resolve) this.confirmModal.resolve(true);
      this.confirmModal.open = false;
    },
    confirmModalCancel() {
      if (this.confirmModal.resolve) this.confirmModal.resolve(false);
      this.confirmModal.open = false;
    },

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
    async loadDashboard() {
      this.dashB2BLoading = true;
      try {
        const data = await fetch('/api/dashboard/b2b', {
          headers: { Authorization: 'Bearer ' + this.token }
        }).then(r => r.ok ? r.json() : null);
        if (data) this.dashB2B = data;
      } catch (e) { console.error('loadDashboard B2B', e); }
      this.dashB2BLoading = false;
    },

    async loadOrg() {
      if (!this.token) return;
      try {
        this.org = await fetchAPI('/api/me/organization');
      } catch (e) {
        console.error('loadOrg', e);
      }
    },

    get planFeatures() {
      const plan = this.org?.plan || 'trial';
      return {
        webhooks: ['pro', 'enterprise'].includes(plan),
      };
    },

    activatePlan(plan = 'pro') {
      this.payModalPlan = plan;
      this.payModalStep = 'confirm';
      this.payModalOpen = true;
      this.refreshIcons();
    },

    async confirmPayment() {
      if (!this.payModalPlan) return;
      this.payModalStep = 'processing';
      try {
        const result = await fetchAPI('/api/billing/checkout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ plan: this.payModalPlan }),
        });
        if (result.checkout_url) {
          window.location.href = result.checkout_url;
          return;
        }
        await this.loadOrg();
        this.payModalStep = 'success';
        showToast(result.message || 'Plan activat');
      } catch (e) {
        this.payModalStep = 'confirm';
        showToast(e.message, 'error');
      }
    },

    /* ── Tab navigation ───────────────────────────────────────────────── */
    goTo(tab) { this.activeTab = tab; },

    async openIncidentById(id) {
      this.activeTab = 'incidents';
      await this.$nextTick();
      this.incidentStatusFilter = '';
      this.incidentPage = 0;
      await this.loadIncidents();
      const found = this.incidents.find(i => i.id === id);
      if (found) {
        // Dispatch event — incidentModal e in x-data local al tab-ului incidents
        window.dispatchEvent(new CustomEvent('eco:openIncident', { detail: found }));
      } else {
        // Incident nu e in pagina curenta — fetch direct din API
        try {
          const evt = await fetchAPI(`/api/littering/events/${id}`);
          if (evt) window.dispatchEvent(new CustomEvent('eco:openIncident', { detail: evt }));
        } catch (_) {}
      }
    },

    /* ── Admin check ─────────────────────────────────────────────────── */
    get isAdmin() { return this.user?.role === 'admin'; },

    /* ── Toate tab-urile vizibile pentru toti utilizatorii ──────────── */
    get visibleTabs() { return this.tabs; },
    _setupAdminTab() {},

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
