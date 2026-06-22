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
    adminSubTab: 'overview',  // 'overview' | 'users' | 'incidents'

    // Charts
    adminCharts: {
      reports_timeline: [],
      users_timeline: [],
      material_distribution: [],
      resolution_rate: {},
    },
    adminChartsLoading: false,
    _adminChartInstances: {},

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
    incidentReporterFilter: '',
    incidentPending: 0,
    incidentReviewed: 0,
    incidentForwarded: 0,
    incidentTotalAll: 0,
    incidentForwardingIds: [],
    incidentSelectedIds: [],
    incidentBulkDeleting: false,

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
        showToast(`Rol schimbat: ${this.roleLabel(newRole)}`);
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
        return showToast('Nume utilizator și email obligatorii', 'error');
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
        showToast(`Utilizatorul "${this.adminConfirmUser.username}" a fost șters.`);
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
      // Charts are rendered natively in the template. Keeping this hook lets
      // older flows call it safely after data refreshes.
      this._refreshAdminIcons();
    },

    _chartTotal(rows) {
      return (rows || []).reduce((sum, r) => sum + Number(r.count || 0), 0);
    },

    _chartMax(rows) {
      return Math.max(1, ...(rows || []).map(r => Number(r.count || 0)));
    },

    _chartBarHeight(value, rows) {
      const pct = (Number(value || 0) / this._chartMax(rows)) * 100;
      return Math.max(value ? 8 : 2, Math.round(pct));
    },

    _resolutionPct() {
      const rr = this.adminCharts?.resolution_rate || {};
      const resolved = Number(rr.resolved || 0);
      const total = resolved + Number(rr.unresolved || 0);
      return total ? Math.round((resolved / total) * 100) : 0;
    },

    _materialLabel(material) {
      return material === 'paper' ? 'Hârtie'
        : material === 'glass' ? 'Sticlă'
        : material === 'plastic' ? 'Plastic'
        : material === 'metal' ? 'Metal'
        : material === 'other' ? 'Altele'
        : 'Necunoscut';
    },

    /* ── Export users CSV ─────────────────────────────────────────────── */
    adminExportUsersCSV() {
      const token = getAuthToken();
      const a = document.createElement('a');
      a.href = `/api/admin/export/users?token=${encodeURIComponent(token)}`;
      a.download = 'users_export.csv';
      a.click();
    },

    async exportIncidentsCSV() {
      const token = getAuthToken();
      if (!token) {
        showToast('Autentifica-te pentru export CSV.', 'error');
        return;
      }

      const params = new URLSearchParams({ format: 'csv', period: 'all' });
      if (this.incidentStatusFilter) params.set('status', this.incidentStatusFilter);
      if (this.incidentMaterialFilter) params.set('material', this.incidentMaterialFilter);

      try {
        const response = await fetch(`/api/reports/export?${params.toString()}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) {
          let detail = `Eroare export CSV (${response.status})`;
          try {
            const err = await response.json();
            detail = err?.detail || detail;
          } catch (_) {
            const text = await response.text();
            if (text) detail = text;
          }
          throw new Error(detail);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const date = new Date().toISOString().slice(0, 10);
        const suffix = [
          this.incidentStatusFilter || 'toate',
          this.incidentMaterialFilter || null,
        ].filter(Boolean).join('_');
        const a = document.createElement('a');
        a.href = url;
        a.download = `incidente_trashdet_${suffix}_${date}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        showToast('Raport CSV descarcat.', 'success');
      } catch (e) {
        showToast('Export CSV esuat: ' + e.message, 'error');
      }
    },

    /* ── Storage stats ────────────────────────────────────────────────── */
    async loadStorage() {
      if (!this.token || this.user?.role !== 'admin') {
        this.adminStorage = null;
        this.adminStorageLoading = false;
        return;
      }
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
        if (this.user?.role === 'admin' && this.incidentReporterFilter) {
          url += `&reporter_id=${encodeURIComponent(this.incidentReporterFilter)}`;
        }

        const data = await fetchAPI(url);
        this.incidents = data.items;
        this.incidentTotal = data.total;
        // Resetează selecția la fiecare (re)încărcare a listei: la refresh,
        // schimbarea paginii, a filtrului sau a tab-ului nu rămâne nimic bifat.
        this.incidentSelectedIds = [];

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
        this._refreshAdminIcons();
      }
    },

    isIncidentSelected(id) {
      return this.incidentSelectedIds.includes(id);
    },

    areAllIncidentsOnPageSelected() {
      return this.incidents.length > 0 && this.incidents.every(e => this.incidentSelectedIds.includes(e.id));
    },

    toggleIncidentSelection(id) {
      if (this.isIncidentSelected(id)) {
        this.incidentSelectedIds = this.incidentSelectedIds.filter(x => x !== id);
      } else {
        this.incidentSelectedIds = [...this.incidentSelectedIds, id];
      }
      this._refreshAdminIcons();
    },

    toggleIncidentPageSelection() {
      const pageIds = this.incidents.map(e => e.id);
      if (pageIds.length === 0) return;
      if (this.areAllIncidentsOnPageSelected()) {
        this.incidentSelectedIds = this.incidentSelectedIds.filter(id => !pageIds.includes(id));
      } else {
        this.incidentSelectedIds = Array.from(new Set([...this.incidentSelectedIds, ...pageIds]));
      }
      this._refreshAdminIcons();
    },

    clearIncidentSelection() {
      this.incidentSelectedIds = [];
      this._refreshAdminIcons();
    },

    async markIncidentReviewed(id) {
      const ok = await this.showConfirm(
        'Confirmă incident',
        'Confirmi că dovada indică un act real de aruncare ilegală. Incidentul rămâne salvat pentru raport și analiză.',
        { confirmText: 'Confirmă', confirmColor: '#10b981', iconColor: '#10b981', icon: 'check-circle' }
      );
      if (!ok) return false;
      try {
        const updated = await fetchAPI(`/api/littering/events/${id}/status`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'reviewed' }),
        });
        const idx = this.incidents.findIndex(e => e.id === id);
        if (idx !== -1) {
          const oldStatus = this.incidents[idx].status;
          this.incidents[idx] = updated;
          if (oldStatus === 'pending') this.incidentPending = Math.max(0, this.incidentPending - 1);
          if (oldStatus === 'forwarded') this.incidentForwarded = Math.max(0, this.incidentForwarded - 1);
          if (oldStatus !== 'reviewed') this.incidentReviewed += 1;
        }
        showToast('Incident confirmat ca aruncare ilegală');
        await this.loadStorage();
        this._refreshAdminIcons();
        return true;
      } catch (e) {
        showToast('Eroare: ' + e.message, 'error');
        return false;
      }
    },

    async forwardIncident(id) {
      if (this.incidentForwardingIds.includes(id)) return false;
      const ok = await this.showConfirm(
        'Arhivează dovada',
        'Incidentul va fi marcat ca arhivat: clipul, thumbnailul, hash-ul și notele rămân salvate local pentru raport. Nu se trimite email real către autorități în această versiune locală.',
        { confirmText: 'Arhivează', confirmColor: '#2563eb', iconColor: '#2563eb', icon: 'archive' }
      );
      if (!ok) return false;
      this.incidentForwardingIds.push(id);
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
          if (oldStatus !== 'forwarded') this.incidentForwarded += 1;
        }
        showToast('Dovada a fost arhivată local.');
        await this.loadStorage();
        this._refreshAdminIcons();
        return true;
      } catch (e) {
        showToast('Eroare: ' + e.message, 'error');
        return false;
      } finally {
        this.incidentForwardingIds = this.incidentForwardingIds.filter(x => x !== id);
      }
    },

    async dismissIncident(id) {
      const ok = await this.showConfirm(
        'Marchează fals pozitiv',
        'Incidentul va fi păstrat în istoric, dar marcat ca nerelevant pentru evaluarea finală.',
        { confirmText: 'Fals pozitiv', confirmColor: '#6b7280', iconColor: '#6b7280', icon: 'x-circle' }
      );
      if (!ok) return false;
      try {
        const updated = await fetchAPI(`/api/littering/events/${id}/status`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'dismissed' }),
        });
        const idx = this.incidents.findIndex(e => e.id === id);
        if (idx !== -1) {
          const oldStatus = this.incidents[idx].status;
          this.incidents[idx] = updated;
          if (oldStatus === 'pending') this.incidentPending = Math.max(0, this.incidentPending - 1);
          if (oldStatus === 'reviewed') this.incidentReviewed = Math.max(0, this.incidentReviewed - 1);
          if (oldStatus === 'forwarded') this.incidentForwarded = Math.max(0, this.incidentForwarded - 1);
        }
        showToast('Incident marcat ca fals pozitiv');
        await this.loadStorage();
        this._refreshAdminIcons();
        return true;
      } catch (e) {
        showToast('Eroare: ' + e.message, 'error');
        return false;
      }
    },

    async deleteIncident(id) {
      const ok = await this.showConfirm(
        'Șterge definitiv',
        'Acest incident și videoclipul/imaginile asociate vor fi șterse permanent pentru a elibera spațiu. Acțiunea este ireversibilă.',
        { confirmText: 'Șterge definitiv', confirmColor: '#dc2626', iconColor: '#dc2626', icon: 'trash-2' }
      );
      if (!ok) return false;
      try {
        await fetchAPI(`/api/littering/events/${id}`, { method: 'DELETE' });

        // Elimină din lista curentă
        const idx = this.incidents.findIndex(e => e.id === id);
        if (idx !== -1) {
          const oldStatus = this.incidents[idx].status;
          this.incidents.splice(idx, 1);

          // Actualizează KPI-urile locale
          this.incidentTotalAll = Math.max(0, this.incidentTotalAll - 1);
          if (oldStatus === 'pending') this.incidentPending = Math.max(0, this.incidentPending - 1);
          if (oldStatus === 'reviewed') this.incidentReviewed = Math.max(0, this.incidentReviewed - 1);
          if (oldStatus === 'forwarded') this.incidentForwarded = Math.max(0, this.incidentForwarded - 1);
        }

        // Închide modalul dacă era deschis
        if (typeof this.incidentModal !== 'undefined' && this.incidentModal?.id === id) {
           this.incidentModal = null;
        }

        showToast('Incident șters definitiv.');
        await this.loadStorage();
        this._refreshAdminIcons();
        return true;
      } catch (e) {
        showToast('Eroare: ' + e.message, 'error');
        return false;
      }
    },

    async deleteSelectedIncidents() {
      const ids = [...this.incidentSelectedIds];
      if (ids.length === 0 || this.incidentBulkDeleting) return;
      const ok = await this.showConfirm(
        'Șterge incidente selectate',
        `Vor fi șterse definitiv ${ids.length} incident(e), împreună cu clipurile și imaginile asociate. Acțiunea este ireversibilă.`,
        { confirmText: `Șterge ${ids.length}`, confirmColor: '#dc2626', iconColor: '#dc2626', icon: 'trash-2' }
      );
      if (!ok) return;
      this.incidentBulkDeleting = true;
      let deleted = 0;
      let failed = 0;
      try {
        for (const id of ids) {
          try {
            await fetchAPI(`/api/littering/events/${id}`, { method: 'DELETE' });
            deleted += 1;
          } catch (e) {
            failed += 1;
            console.warn('deleteSelectedIncidents', id, e);
          }
        }
        this.incidentSelectedIds = [];
        await this.loadIncidents();
        await this.loadStorage();
        if (deleted > 0) showToast(`${deleted} incident(e) șterse definitiv.`, 'success');
        if (failed > 0) showToast(`${failed} incident(e) nu au putut fi șterse.`, 'error');
      } finally {
        this.incidentBulkDeleting = false;
        this._refreshAdminIcons();
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
      } else if (tab === 'incidents') {
        if (this.adminUsers.length === 0) this.loadAdminUsers();
        this.loadIncidents();
      }
      this._refreshAdminIcons();
    },

    /* ── Refresh all admin data ───────────────────────────────────────── */
    async loadAdminAll() {
      if (!this.token || this.user?.role !== 'admin') return;
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
