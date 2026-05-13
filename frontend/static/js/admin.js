/* ── TrashDet — Admin Panel module ───────────────────────────────────────── */

function adminApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    adminUsers: [],
    adminUsersLoading: false,
    adminStats: null,
    adminStatsLoading: false,

    // Confirm delete modal
    adminConfirmUser: null,
    adminConfirmOpen: false,

    // Team invite (org-scoped)
    inviteModalOpen: false,
    inviteForm: { username: '', email: '', role: 'user' },
    inviteSending: false,
    inviteResult: null,

    // Sub-tab navigation inside admin panel
    adminSubTab: 'overview',  // 'overview' | 'users' | 'authorities' | 'incidents' | 'webhooks'

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

    // Webhooks
    adminWebhooks: [],
    adminWebhooksLoading: false,
    adminNewWebhook: { url: '', secret: '', events: 'report.verified,report.cleaned' },
    adminWebhookTesting: null,

    // Storage
    adminStorage: null,
    adminStorageLoading: false,

    // Littering Incidents
    incidents: [],
    incidentsLoading: false,
    incidentTotal: 0,
    incidentPage: 0,
    incidentLimit: 20,
    incidentStatusFilter: '',
    incidentMaterialFilter: '',
    incidentPending: 0,
    incidentReviewed: 0,
    incidentForwarded: 0,
    incidentTotalAll: 0,

    /* ── Init ─────────────────────────────────────────────────────────── */
    initAdmin() {
      window.addEventListener('eco:litteringAlert', () => {
        this.incidentPending += 1;
        if (this.adminSubTab === 'incidents') {
          this.loadIncidents();
        }
      });
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
        showToast(`Role changed → ${newRole}`);
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    /* ── Delete user (with confirm) ────────────────────────────────────── */
    adminAskDelete(user) {
      this.adminConfirmUser = user;
      this.adminConfirmOpen = true;
    },

    openInviteModal() {
      this.inviteForm = { username: '', email: '', role: 'user' };
      this.inviteResult = null;
      this.inviteModalOpen = true;
    },

    async sendInvite() {
      if (!this.inviteForm.username.trim() || !this.inviteForm.email.trim()) {
        return showToast('Username și email obligatorii', 'error');
      }
      this.inviteSending = true;
      try {
        const res = await fetchAPI('/api/admin/users/invite', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.inviteForm),
        });
        this.inviteResult = res;
        await this.loadAdminUsers();
        showToast(res.message || 'Invitație trimisă', 'success');
      } catch (e) {
        showToast('Eroare: ' + e.message, 'error');
      } finally {
        this.inviteSending = false;
      }
    },

    async adminDeleteUser() {
      if (!this.adminConfirmUser) return;
      try {
        await fetchAPI(`/api/admin/users/${this.adminConfirmUser.id}`, { method: 'DELETE' });
        this.adminUsers = this.adminUsers.filter(u => u.id !== this.adminConfirmUser.id);
        showToast(`User "${this.adminConfirmUser.username}" has been deleted.`);
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.adminConfirmOpen = false;
        this.adminConfirmUser = null;
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

      Object.values(this._adminChartInstances).forEach(c => c.destroy());
      this._adminChartInstances = {};

      const isDark = document.documentElement.classList.contains('dark');
      const gridColor = isDark ? 'rgba(255,255,255,.08)' : 'rgba(0,0,0,.06)';
      const textColor = isDark ? '#9ca3af' : '#6b7280';

      const rCanvas = document.getElementById('adminChartReports');
      if (rCanvas) {
        this._adminChartInstances.reports = new Chart(rCanvas, {
          type: 'bar',
          data: {
            labels: this.adminCharts.reports_timeline.map(r => r.day.slice(5)),
            datasets: [{
              label: 'Reports',
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

      const uCanvas = document.getElementById('adminChartUsers');
      if (uCanvas) {
        this._adminChartInstances.users = new Chart(uCanvas, {
          type: 'line',
          data: {
            labels: this.adminCharts.users_timeline.map(r => r.month),
            datasets: [{
              label: 'New users',
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

      const resCanvas = document.getElementById('adminChartResolution');
      if (resCanvas) {
        const rr = this.adminCharts.resolution_rate;
        this._adminChartInstances.resolution = new Chart(resCanvas, {
          type: 'doughnut',
          data: {
            labels: ['Resolved', 'Unresolved'],
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
        showToast(`Notification sent to ${res.sent_to} users`);
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
      if (!name.trim() || !email.trim()) return showToast('Numele și emailul sunt obligatorii', 'error');
      try {
        const auth = await fetchAPI('/api/admin/authorities', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim(), email: email.trim(), area_description: area_description.trim() }),
        });
        this.adminAuthorities.push(auth);
        this.adminNewAuthority = { name: '', email: '', area_description: '' };
        showToast('Contact autoritate adăugat');
        this._refreshAdminIcons();
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    async deleteAuthority(id) {
      const ok = await this.showConfirm(
        'Șterge contact autoritate',
        'Contactul va fi șters permanent. Incidentele trimise anterior rămân înregistrate.',
        { confirmText: 'Șterge', icon: 'trash-2' }
      );
      if (!ok) return;
      try {
        await fetchAPI(`/api/admin/authorities/${id}`, { method: 'DELETE' });
        this.adminAuthorities = this.adminAuthorities.filter(a => a.id !== id);
        showToast('Contact șters');
      } catch (e) {
        showToast(e.message, 'error');
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
      if (!url.trim()) return showToast('URL-ul endpoint-ului este obligatoriu', 'error');
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
      const ok = await this.showConfirm(
        'Șterge webhook',
        'Webhook-ul va fi dezactivat și șters permanent.',
        { confirmText: 'Șterge', icon: 'trash-2' }
      );
      if (!ok) return;
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
        showToast(res.status === 'ok' ? `Test OK (${res.status_code})` : `Test failed: ${res.error}`, res.status === 'ok' ? 'success' : 'error');
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

    /* ── Littering Incidents ───────────────────────────────────────────── */
    async loadIncidents() {
      this.incidentsLoading = true;
      try {
        const skip = this.incidentPage * this.incidentLimit;
        let url = `/api/littering/events?skip=${skip}&limit=${this.incidentLimit}`;
        if (this.incidentStatusFilter) url += `&status=${encodeURIComponent(this.incidentStatusFilter)}`;
        if (this.incidentMaterialFilter) url += `&material=${encodeURIComponent(this.incidentMaterialFilter)}`;

        const data = await fetchAPI(url);
        this.incidents = data.items;
        this.incidentTotal = data.total;

        try {
          // Incarca totalurile globale asincron pentru KPI cards
          const p1 = fetchAPI('/api/littering/events?status=pending&limit=1').then(d => this.incidentPending = d.total);
          const p2 = fetchAPI('/api/littering/events?status=reviewed&limit=1').then(d => this.incidentReviewed = d.total);
          const p3 = fetchAPI('/api/littering/events?status=forwarded&limit=1').then(d => this.incidentForwarded = d.total);
          const p4 = fetchAPI('/api/littering/events?limit=1').then(d => this.incidentTotalAll = d.total);
          await Promise.all([p1, p2, p3, p4]);
        } catch (_) {}
      } catch (e) {
        showToast('Eroare la încărcarea incidentelor: ' + e.message, 'error');
      } finally {
        this.incidentsLoading = false;
      }
    },

    async markIncidentReviewed(id) {
      try {
        const updated = await fetchAPI(`/api/littering/events/${id}/status`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'reviewed' }),
        });
        const idx = this.incidents.findIndex(e => e.id === id);
        if (idx !== -1) this.incidents[idx] = updated;
        this.incidentPending = Math.max(0, this.incidentPending - 1);
        this.incidentReviewed += 1;
        showToast('Incident marcat ca verificat');
      } catch (e) {
        showToast('Eroare: ' + e.message, 'error');
      }
    },

    async forwardIncident(id) {
      const ok = await this.showConfirm(
        'Trimite la autoritate',
        'Evidența acestui incident (clip + metadata + hash SHA-256) va fi marcată ca trimisă autorității responsabile.',
        { confirmText: 'Trimite', confirmColor: '#2563eb', iconColor: '#2563eb', icon: 'send' }
      );
      if (!ok) return;
      try {
        const updated = await fetchAPI(`/api/littering/events/${id}/status`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'forwarded' }),
        });
        const idx = this.incidents.findIndex(e => e.id === id);
        if (idx !== -1) {
          const oldStatus = this.incidents[idx].status;
          this.incidents[idx] = updated;
          if (oldStatus === 'reviewed') this.incidentReviewed = Math.max(0, this.incidentReviewed - 1);
          if (oldStatus === 'pending') this.incidentPending = Math.max(0, this.incidentPending - 1);
          this.incidentForwarded += 1;
        }
        showToast('Incident trimis la autoritate');
      } catch (e) {
        showToast('Eroare: ' + e.message, 'error');
      }
    },

    async dismissIncident(id) {
      const ok = await this.showConfirm(
        'Respinge incident',
        'Incidentul va fi marcat ca fals pozitiv și arhivat. Acțiunea poate fi revizuită ulterior.',
        { confirmText: 'Respinge', confirmColor: '#6b7280', iconColor: '#6b7280', icon: 'x-circle' }
      );
      if (!ok) return;
      try {
        const updated = await fetchAPI(`/api/littering/events/${id}/status`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'dismissed' }),
        });
        const idx = this.incidents.findIndex(e => e.id === id);
        if (idx !== -1) this.incidents[idx] = updated;
        showToast('Incident respins');
      } catch (e) {
        showToast('Eroare: ' + e.message, 'error');
      }
    },

    /* ── Switch admin sub-tab ─────────────────────────────────────────── */
    switchAdminSubTab(tab) {
      this.adminSubTab = tab;
      if (tab === 'overview') {
        this.loadAdminStats();
        this.loadAdminCharts();
        this.loadStorage();
      } else if (tab === 'users') {
        this.loadAdminUsers();
      } else if (tab === 'authorities') {
        this.loadAuthorities();
      } else if (tab === 'incidents') {
        this.loadIncidents();
      } else if (tab === 'webhooks') {
        this.loadWebhooks();
      }
      this._refreshAdminIcons();
    },

    /* ── Refresh all admin data ───────────────────────────────────────── */
    async loadAdminAll() {
      this.adminSubTab = 'overview';
      try {
        await Promise.all([this.loadAdminStats(), this.loadAdminCharts(), this.loadStorage()]);
      } catch (e) {
        console.error('[ADMIN] loadAdminAll error:', e);
      }
    },

    /* ── Helpers ──────────────────────────────────────────────────────── */
    _fmtAdminDate(iso) {
      if (!iso) return '—';
      const d = new Date(iso);
      return d.toLocaleDateString('en-US', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    },

    _fmtAdminTimeAgo(iso) {
      if (!iso) return '';
      const diff = Date.now() - new Date(iso).getTime();
      const mins = Math.floor(diff / 60000);
      if (mins < 1) return 'now';
      if (mins < 60) return `${mins} min`;
      const hrs = Math.floor(mins / 60);
      if (hrs < 24) return `${hrs}h`;
      const days = Math.floor(hrs / 24);
      return `${days}d`;
    },
  };
}
