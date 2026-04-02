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

    /* ── Init ─────────────────────────────────────────────────────────── */
    initAdmin() {
      // Loaded on demand when tab is opened
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
      try {
        const res = await fetchAPI(`/api/sessions/${sessionId}/resolve`, { method: 'POST' });
        const action = res.is_resolved === 1 ? 'marcat curățat ✓' : 'marcat nerezolvat';
        showToast(`Raportul ${sessionId} ${action}`);
        window.dispatchEvent(new CustomEvent('eco:resolveChanged', { detail: { sessionId, is_resolved: res.is_resolved } }));
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    /* ── Refresh all admin data ───────────────────────────────────────── */
    async loadAdminAll() {
      await Promise.all([this.loadAdminStats(), this.loadAdminUsers(), this.loadLeaderboard()]);
    },
  };
}
