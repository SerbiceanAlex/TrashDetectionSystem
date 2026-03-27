/* ── EcoAlert — Rapoarte (History) tab ──────────────────────────────────── */

function historyApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    history: { total: 0, skip: 0, limit: 12, items: [] },
    historyPage: 0,
    historyLimit: 12,
    isLoadingHistory: false,
    sessionDetail: null,

    // Filters
    searchQuery: '',
    filterMaterial: '',
    filterMinObjects: null,

    /* ── Init ─────────────────────────────────────────────────────────── */
    initHistory() {
      this.loadHistory();
      // Refresh when a new report comes in from detect tab
      window.addEventListener('eco:newReport', () => this.loadHistory());
    },

    /* ── Load ─────────────────────────────────────────────────────────── */
    async loadHistory() {
      this.isLoadingHistory = true;
      try {
        let url = `/api/sessions?skip=${this.historyPage * this.historyLimit}&limit=${this.historyLimit}`;
        if (this.searchQuery) url += `&q=${encodeURIComponent(this.searchQuery)}`;
        if (this.filterMaterial) url += `&material=${encodeURIComponent(this.filterMaterial)}`;
        if (this.filterMinObjects != null && this.filterMinObjects !== '' && this.filterMinObjects >= 0) {
          url += `&min_objects=${this.filterMinObjects}`;
        }
        const data = await fetchAPI(url);
        this.history = data;
      } catch (e) {
        showToast('Eroare la încărcarea istoricului', 'error');
      } finally {
        this.isLoadingHistory = false;
      }
    },

    async openSessionDetail(id) {
      try {
        this.sessionDetail = await fetchAPI(`/api/sessions/${id}`);
      } catch (e) {
        showToast('Eroare la încărcarea sesiunii', 'error');
      }
    },

    async deleteSession(id) {
      if (!confirm(`Ștergi sesiunea #${id}?`)) return;
      try {
        await fetchAPI(`/api/sessions/${id}`, { method: 'DELETE' });
        showToast('Sesiunea a fost ștearsă.');
        await this.loadHistory();
        window.dispatchEvent(new CustomEvent('eco:statsChanged'));
      } catch (e) {
        showToast('Eroare la ștergere.', 'error');
      }
    },

    /* ── Pagination ───────────────────────────────────────────────────── */
    prevPage() {
      if (this.historyPage > 0) {
        this.historyPage--;
        this.loadHistory();
      }
    },

    nextPage() {
      if ((this.historyPage + 1) * this.historyLimit < this.history.total) {
        this.historyPage++;
        this.loadHistory();
      }
    },

    get totalPages() {
      return Math.max(Math.ceil(this.history.total / this.historyLimit), 1);
    },

    get hasPrev() { return this.historyPage > 0; },
    get hasNext() { return (this.historyPage + 1) * this.historyLimit < this.history.total; },

    /* ── Helpers ──────────────────────────────────────────────────────── */
    getAnnotatedUrl(path) { return getAnnotatedUrl(path); },
    formatDate(iso) { return formatDate(iso); },
    timeAgo(iso) { return timeAgo(iso); },
    materialBadgeStyle(m) { return materialBadgeStyle(m); },

    sessionSeverity(session) {
      return getSeverity(severityFromCount(session.total_objects));
    },
  };
}
