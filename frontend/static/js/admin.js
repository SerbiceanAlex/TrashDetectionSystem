/* ── TrashDet — Admin Panel module ───────────────────────────────────────── */

function adminApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    adminUsers: [],
    adminUsersLoading: false,
    adminStats: null,
    adminStatsLoading: false,
    leaderboard: [],
    leaderboardLoading: false,

    // Confirm delete modal
    adminConfirmUser: null,
    adminConfirmOpen: false,

    // Sub-tab navigation inside admin panel
    adminSubTab: 'overview',  // 'overview' | 'users' | 'reports' | 'activity' | 'authorities' | 'webhooks'

    // Reports management
    adminReports: [],
    adminReportsLoading: false,
    adminReportsTotal: 0,
    adminReportsPage: 0,
    adminReportsFilter: 'all',     // 'all' | 'resolved' | 'unresolved'
    adminReportsSearch: '',

    // Activity feed
    adminActivity: [],
    adminActivityLoading: false,

    // Charts
    adminCharts: null,
    adminChartsLoading: false,
    _adminChartInstances: {},

    // Broadcast
    adminBroadcastMsg: '',
    adminBroadcastSending: false,

    // Authorities
    adminAuthorities: [],
    adminAuthoritiesLoading: false,
    adminNewAuthority: { name: '', email: '', area_description: '' },
    adminForwardingId: null,
    adminForwardSending: false,

    // Webhooks
    adminWebhooks: [],
    adminWebhooksLoading: false,
    adminNewWebhook: { url: '', secret: '', events: 'report.verified,report.cleaned' },
    adminWebhookTesting: null,

    // Storage
    adminStorage: null,
    adminStorageLoading: false,

    /* ── Init ─────────────────────────────────────────────────────────── */
    initAdmin() {
      // Loaded on demand when tab is opened
    },

    _refreshAdminIcons() {
      this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });
    },

    /* ── Load admin stats KPIs ────────────────────────────────────────── */
    async loadAdminStats() {
      this.adminStatsLoading = true;
      try {
        this.adminStats = await fetchAPI('/api/admin/stats');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminStatsLoading = false;
      }
    },

    /* ── Load users table ─────────────────────────────────────────────── */
    async loadAdminUsers() {
      this.adminUsersLoading = true;
      try {
        this.adminUsers = await fetchAPI('/api/admin/users');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminUsersLoading = false;
        this._refreshAdminIcons();
      }
    },

    /* ── Load leaderboard ─────────────────────────────────────────────── */
    async loadLeaderboard() {
      this.leaderboardLoading = true;
      try {
        this.leaderboard = await fetchAPI('/api/leaderboard?limit=10');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.leaderboardLoading = false;
        this._refreshAdminIcons();
      }
    },

    /* ── Toggle user role admin ↔ user ────────────────────────────────── */
    async adminToggleRole(userId, currentRole) {
      const newRole = currentRole === 'admin' ? 'user' : 'admin';
      try {
        await fetchAPI(`/api/admin/users/${userId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ role: newRole }),
        });
        const u = this.adminUsers.find(x => x.id === userId);
        if (u) u.role = newRole;
        showToast(`Rol schimbat → ${newRole}`);
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    /* ── Adjust user points ───────────────────────────────────────────── */
    async adminAdjustPoints(userId, delta) {
      const u = this.adminUsers.find(x => x.id === userId);
      if (!u) return;
      const newPoints = Math.max(0, u.points + delta);
      try {
        await fetchAPI(`/api/admin/users/${userId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ points: newPoints }),
        });
        u.points = newPoints;
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    /* ── Delete user (with confirm) ────────────────────────────────────── */
    adminAskDelete(user) {
      this.adminConfirmUser = user;
      this.adminConfirmOpen = true;
    },

    async adminDeleteUser() {
      if (!this.adminConfirmUser) return;
      try {
        await fetchAPI(`/api/admin/users/${this.adminConfirmUser.id}`, { method: 'DELETE' });
        this.adminUsers = this.adminUsers.filter(u => u.id !== this.adminConfirmUser.id);
        showToast(`Utilizatorul "${this.adminConfirmUser.username}" a fost șters.`);
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminConfirmOpen = false;
        this.adminConfirmUser = null;
      }
    },

    /* ── Resolve/unresolve a detection report ─────────────────────────── */
    async resolveSession(sessionId, currentStatus) {
      const isCurrentlyResolved = currentStatus === 1 || currentStatus === true;
      const msg = isCurrentlyResolved
        ? `Marchezi raportul #${sessionId} ca NEREZOLVAT?`
        : `Marchezi raportul #${sessionId} ca CURĂȚAT?`;
      if (!confirm(msg)) return;
      try {
        const res = await fetchAPI(`/api/sessions/${sessionId}/resolve`, { method: 'POST' });
        const action = res.is_resolved === 1 ? 'marcat curățat' : 'marcat nerezolvat';
        showToast(`Raportul ${sessionId} ${action}`);
        // Refresh reports list
        this.loadAdminReports();
        this.loadAdminStats();
        // Notify history tab to refresh local state
        window.dispatchEvent(new CustomEvent('eco:resolveChanged', { detail: { sessionId, is_resolved: res.is_resolved } }));
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    /* ── Reports management ───────────────────────────────────────────── */
    async loadAdminReports() {
      this.adminReportsLoading = true;
      try {
        const params = new URLSearchParams({
          skip: String(this.adminReportsPage * 20),
          limit: '20',
        });
        if (this.adminReportsFilter !== 'all') params.set('status', this.adminReportsFilter);
        if (this.adminReportsSearch.trim()) params.set('search', this.adminReportsSearch.trim());
        const data = await fetchAPI(`/api/admin/reports?${params}`);
        this.adminReports = data.items;
        this.adminReportsTotal = data.total;
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminReportsLoading = false;
        this._refreshAdminIcons();
      }
    },

    adminReportsPrev() {
      if (this.adminReportsPage > 0) {
        this.adminReportsPage--;
        this.loadAdminReports();
      }
    },

    adminReportsNext() {
      if ((this.adminReportsPage + 1) * 20 < this.adminReportsTotal) {
        this.adminReportsPage++;
        this.loadAdminReports();
      }
    },

    adminFilterReports(filter) {
      this.adminReportsFilter = filter;
      this.adminReportsPage = 0;
      this.loadAdminReports();
    },

    adminSearchReports() {
      this.adminReportsPage = 0;
      this.loadAdminReports();
    },

    async adminDeleteReport(sessionId) {
      if (!confirm(`Ștergi raportul #${sessionId}?`)) return;
      try {
        await fetchAPI(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        this.adminReports = this.adminReports.filter(r => r.id !== sessionId);
        this.adminReportsTotal--;
        this.loadAdminStats();
        showToast(`Raportul #${sessionId} a fost șters.`);
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    /* ── Activity feed ────────────────────────────────────────────────── */
    async loadAdminActivity() {
      this.adminActivityLoading = true;
      try {
        this.adminActivity = await fetchAPI('/api/admin/activity?limit=20');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminActivityLoading = false;
        this._refreshAdminIcons();
      }
    },

    /* ── Charts ───────────────────────────────────────────────────────── */
    async loadAdminCharts() {
      this.adminChartsLoading = true;
      try {
        this.adminCharts = await fetchAPI('/api/admin/charts');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminChartsLoading = false;
        this.$nextTick(() => {
          try { this._renderAdminCharts(); } catch (err) { console.warn('Chart render error:', err); }
        });
      }
    },

    _renderAdminCharts() {
      if (!this.adminCharts) return;

      // Destroy old instances
      Object.values(this._adminChartInstances).forEach(c => c.destroy());
      this._adminChartInstances = {};

      const isDark = document.documentElement.classList.contains('dark');
      const gridColor = isDark ? 'rgba(255,255,255,.08)' : 'rgba(0,0,0,.06)';
      const textColor = isDark ? '#9ca3af' : '#6b7280';

      // Reports timeline (bar chart)
      const rCanvas = document.getElementById('adminChartReports');
      if (rCanvas) {
        this._adminChartInstances.reports = new Chart(rCanvas, {
          type: 'bar',
          data: {
            labels: this.adminCharts.reports_timeline.map(r => r.day.slice(5)),
            datasets: [{
              label: 'Rapoarte',
              data: this.adminCharts.reports_timeline.map(r => r.count),
              backgroundColor: 'rgba(5,150,105,.6)',
              borderRadius: 4,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { grid: { display: false }, ticks: { color: textColor, font: { size: 10 } } },
              y: { beginAtZero: true, grid: { color: gridColor }, ticks: { color: textColor, precision: 0 } },
            },
          },
        });
      }

      // Users timeline (line chart)
      const uCanvas = document.getElementById('adminChartUsers');
      if (uCanvas) {
        this._adminChartInstances.users = new Chart(uCanvas, {
          type: 'line',
          data: {
            labels: this.adminCharts.users_timeline.map(r => r.month),
            datasets: [{
              label: 'Utilizatori noi',
              data: this.adminCharts.users_timeline.map(r => r.count),
              borderColor: '#3b82f6',
              backgroundColor: 'rgba(59,130,246,.15)',
              fill: true, tension: .4, pointRadius: 3,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { grid: { display: false }, ticks: { color: textColor, font: { size: 10 } } },
              y: { beginAtZero: true, grid: { color: gridColor }, ticks: { color: textColor, precision: 0 } },
            },
          },
        });
      }

      // Materials distribution (doughnut)
      const mCanvas = document.getElementById('adminChartMaterials');
      if (mCanvas) {
        const colors = ['#059669', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#6b7280'];
        this._adminChartInstances.materials = new Chart(mCanvas, {
          type: 'doughnut',
          data: {
            labels: this.adminCharts.material_distribution.map(m => m.material),
            datasets: [{
              data: this.adminCharts.material_distribution.map(m => m.count),
              backgroundColor: colors.slice(0, this.adminCharts.material_distribution.length),
              borderWidth: 0,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
              legend: { position: 'bottom', labels: { color: textColor, padding: 12, usePointStyle: true, font: { size: 11 } } },
            },
          },
        });
      }

      // Resolution rate (doughnut)
      const resCanvas = document.getElementById('adminChartResolution');
      if (resCanvas) {
        const rr = this.adminCharts.resolution_rate;
        this._adminChartInstances.resolution = new Chart(resCanvas, {
          type: 'doughnut',
          data: {
            labels: ['Rezolvate', 'Nerezolvate'],
            datasets: [{
              data: [rr.resolved, rr.unresolved],
              backgroundColor: ['#059669', '#ef4444'],
              borderWidth: 0,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
              legend: { position: 'bottom', labels: { color: textColor, padding: 12, usePointStyle: true, font: { size: 11 } } },
            },
          },
        });
      }
    },

    /* ── Broadcast notification ────────────────────────────────────────── */
    async adminSendBroadcast() {
      if (!this.adminBroadcastMsg.trim()) return;
      this.adminBroadcastSending = true;
      try {
        const res = await fetchAPI('/api/admin/broadcast', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: this.adminBroadcastMsg.trim() }),
        });
        showToast(`Notificare trimisă la ${res.sent_to} utilizatori`);
        this.adminBroadcastMsg = '';
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminBroadcastSending = false;
      }
    },

    /* ── Export users CSV ─────────────────────────────────────────────── */
    adminExportUsersCSV() {
      const token = localStorage.getItem('eco_token');
      const a = document.createElement('a');
      a.href = `/api/admin/export/users?token=${encodeURIComponent(token)}`;
      a.download = 'users_export.csv';
      a.click();
    },

    /* ── Authorities CRUD ─────────────────────────────────────────────── */
    async loadAuthorities() {
      this.adminAuthoritiesLoading = true;
      try {
        this.adminAuthorities = await fetchAPI('/api/admin/authorities');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminAuthoritiesLoading = false;
        this._refreshAdminIcons();
      }
    },

    async addAuthority() {
      const { name, email, area_description } = this.adminNewAuthority;
      if (!name.trim() || !email.trim()) return showToast('Nume și email sunt obligatorii', 'error');
      try {
        const auth = await fetchAPI('/api/admin/authorities', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim(), email: email.trim(), area_description: area_description.trim() }),
        });
        this.adminAuthorities.push(auth);
        this.adminNewAuthority = { name: '', email: '', area_description: '' };
        showToast('Autoritate adăugată');
        this._refreshAdminIcons();
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    async deleteAuthority(id) {
      if (!confirm('Ștergi acest contact de autoritate?')) return;
      try {
        await fetchAPI(`/api/admin/authorities/${id}`, { method: 'DELETE' });
        this.adminAuthorities = this.adminAuthorities.filter(a => a.id !== id);
        showToast('Contact șters');
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    async forwardReport(sessionId) {
      this.adminForwardSending = true;
      try {
        const res = await fetchAPI(`/api/admin/forward/${sessionId}`, { method: 'POST' });
        showToast(res.detail || 'Raport trimis la autoritate');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminForwardSending = false;
      }
    },

    /* ── Webhooks CRUD ────────────────────────────────────────────────── */
    async loadWebhooks() {
      this.adminWebhooksLoading = true;
      try {
        this.adminWebhooks = await fetchAPI('/api/admin/webhooks');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminWebhooksLoading = false;
        this._refreshAdminIcons();
      }
    },

    async addWebhook() {
      const { url, secret, events } = this.adminNewWebhook;
      if (!url.trim()) return showToast('URL-ul este obligatoriu', 'error');
      try {
        const wh = await fetchAPI('/api/admin/webhooks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: url.trim(),
            secret: secret.trim() || null,
            events: events.split(',').map(e => e.trim()).filter(Boolean),
            active: true,
          }),
        });
        this.adminWebhooks.push(wh);
        this.adminNewWebhook = { url: '', secret: '', events: 'report.verified,report.cleaned' };
        showToast('Webhook adăugat');
        this._refreshAdminIcons();
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    async toggleWebhook(id) {
      const wh = this.adminWebhooks.find(w => w.id === id);
      if (!wh) return;
      try {
        const updated = await fetchAPI(`/api/admin/webhooks/${id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ active: !wh.active }),
        });
        Object.assign(wh, updated);
        this._refreshAdminIcons();
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    async deleteWebhook(id) {
      if (!confirm('Ștergi acest webhook?')) return;
      try {
        await fetchAPI(`/api/admin/webhooks/${id}`, { method: 'DELETE' });
        this.adminWebhooks = this.adminWebhooks.filter(w => w.id !== id);
        showToast('Webhook șters');
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    async testWebhook(id) {
      this.adminWebhookTesting = id;
      try {
        const res = await fetchAPI(`/api/admin/webhooks/${id}/test`, { method: 'POST' });
        showToast(res.status === 'ok' ? `Test OK (${res.status_code})` : `Test eșuat: ${res.error}`, res.status === 'ok' ? 'success' : 'error');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminWebhookTesting = null;
      }
    },

    /* ── Storage stats ────────────────────────────────────────────────── */
    async loadStorage() {
      this.adminStorageLoading = true;
      try {
        this.adminStorage = await fetchAPI('/api/admin/storage');
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminStorageLoading = false;
        this._refreshAdminIcons();
      }
    },

    _fmtBytes(bytes) {
      if (!bytes || bytes === 0) return '0 B';
      const units = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(1024));
      return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
    },

    /* ── Switch admin sub-tab ─────────────────────────────────────────── */
    switchAdminSubTab(tab) {
      this.adminSubTab = tab;
      if (tab === 'overview') {
        this.loadAdminStats();
        this.loadLeaderboard();
        this.loadAdminCharts();
        this.loadStorage();
      } else if (tab === 'users') {
        this.loadAdminUsers();
      } else if (tab === 'reports') {
        this.loadAdminReports();
      } else if (tab === 'activity') {
        this.loadAdminActivity();
      } else if (tab === 'authorities') {
        this.loadAuthorities();
      } else if (tab === 'webhooks') {
        this.loadWebhooks();
      }
      this._refreshAdminIcons();
    },

    /* ── Refresh all admin data ───────────────────────────────────────── */
    async loadAdminAll() {
      this.adminSubTab = 'overview';
      try {
        await Promise.all([this.loadAdminStats(), this.loadAdminUsers(), this.loadLeaderboard(), this.loadAdminCharts()]);
      } catch (e) {
        console.error('[ADMIN] loadAdminAll error:', e);
      }
    },

    /* ── Helpers ──────────────────────────────────────────────────────── */
    _fmtAdminDate(iso) {
      if (!iso) return '—';
      const d = new Date(iso);
      return d.toLocaleDateString('ro-RO', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    },

    _fmtAdminTimeAgo(iso) {
      if (!iso) return '';
      const diff = Date.now() - new Date(iso).getTime();
      const mins = Math.floor(diff / 60000);
      if (mins < 1) return 'acum';
      if (mins < 60) return `${mins} min`;
      const hrs = Math.floor(mins / 60);
      if (hrs < 24) return `${hrs}h`;
      const days = Math.floor(hrs / 24);
      return `${days}z`;
    },
  };
}
