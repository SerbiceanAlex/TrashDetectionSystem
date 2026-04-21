/* ── EcoAlert — Rapoarte (History) tab ──────────────────────────────────── */

function historyApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    history: { total: 0, skip: 0, limit: 12, items: [] },
    historyPage: 0,
    historyLimit: 12,
    isLoadingHistory: false,
    sessionDetail: null,
    voteSummary: null,
    isVoting: false,
    isClaiming: false,
    isCleanUploading: false,

    // Filters
    searchQuery: '',
    filterMaterial: '',
    filterMinObjects: null,

    // Before/After view mode
    detailViewMode: 'annotated', // 'annotated' | 'original' | 'cleaned' | 'compare'

    // Comments
    sessionComments: [],
    commentsLoading: false,
    newCommentText: '',
    commentPosting: false,

    // User note editing
    editNoteText: '',

    /* ── Init ─────────────────────────────────────────────────────────── */
    initHistory() {
      this.loadHistory();
      // Refresh when a new report comes in from detect tab
      window.addEventListener('eco:newReport', () => this.loadHistory());
      // Refresh when a session is resolved/unresolved
      window.addEventListener('eco:resolveChanged', (e) => {
        const { sessionId, is_resolved } = e.detail;
        if (this.sessionDetail?.id === sessionId) {
          this.sessionDetail.is_resolved = is_resolved;
        }
        this.loadHistory();
      });
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
        this.voteSummary = null;
        this.sessionComments = [];
        this.newCommentText = '';
        this.editNoteText = '';
        this.detailViewMode = 'annotated';
        // Load vote summary + comments in parallel
        const [voteData, comments] = await Promise.all([
          fetchAPI(`/api/sessions/${id}/votes`).catch(() => null),
          fetchAPI(`/api/sessions/${id}/comments`).catch(() => []),
        ]);
        if (voteData) this.voteSummary = voteData;
        this.sessionComments = comments || [];
      } catch (e) {
        showToast('Eroare la încărcarea sesiunii', 'error');
      }
    },

    async voteOnSession(voteType) {
      if (!this.sessionDetail || this.isVoting) return;
      this.isVoting = true;
      try {
        const data = await fetchAPI(`/api/sessions/${this.sessionDetail.id}/vote`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ vote_type: voteType }),
        });
        this.voteSummary = data;
        if (data.status) this.sessionDetail.status = data.status;
        showToast(voteType === 'confirm' ? 'Vot confirmat!' : 'Vot înregistrat!', 'success');
        this.loadHistory();
      } catch (e) {
        showToast(e.message || 'Eroare la vot', 'error');
      } finally {
        this.isVoting = false;
      }
    },

    async claimSession() {
      if (!this.sessionDetail || this.isClaiming) return;
      this.isClaiming = true;
      try {
        const data = await fetchAPI(`/api/sessions/${this.sessionDetail.id}/claim`, { method: 'POST' });
        this.sessionDetail.status = data.status;
        this.sessionDetail.claimed_by = data.claimed_by;
        showToast('Ai revendicat curățarea! Du-te la locație și fă o poză după.', 'success');
        this.loadHistory();
      } catch (e) {
        showToast(e.message || 'Eroare la revendicare', 'error');
      } finally {
        this.isClaiming = false;
      }
    },

    async uploadCleanProof(event) {
      const file = event.target.files?.[0];
      if (!file || !this.sessionDetail) return;
      this.isCleanUploading = true;
      try {
        const fd = new FormData();
        fd.append('file', file);
        const data = await fetchAPI(`/api/sessions/${this.sessionDetail.id}/clean`, {
          method: 'POST',
          body: fd,
        });
        this.sessionDetail.status = 'cleaned';
        this.sessionDetail.is_resolved = 1;
        showToast(`Zona curățată! +${data.eco_score_awarded} EcoScore`, 'success');
        this.loadHistory();
        window.dispatchEvent(new CustomEvent('eco:newReport'));
      } catch (e) {
        showToast(e.message || 'Eroare la upload', 'error');
      } finally {
        this.isCleanUploading = false;
      }
    },

    async resolveSession(id) {
      if (!confirm(`Ești sigur că vrei să marchezi acest focar ca fiind curățat?`)) return;
      try {
        await fetchAPI(`/api/sessions/${id}/resolve`, { method: 'POST' });
        showToast('Murdăria a fost marcată ca fiind curățată!', 'success');
        // Update local state instantly
        if (this.sessionDetail && this.sessionDetail.id === id) {
          this.sessionDetail.is_resolved = 1;
        }
        await this.loadHistory();
        window.dispatchEvent(new CustomEvent('eco:newReport')); // force map update
      } catch (e) {
        showToast('Eroare la actualizarea statusului.', 'error');
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

    /* ── Comments ─────────────────────────────────────────────────────── */
    async loadComments(sessionId) {
      this.commentsLoading = true;
      try {
        this.sessionComments = await fetchAPI(`/api/sessions/${sessionId}/comments`);
      } catch (_) {}
      this.commentsLoading = false;
    },

    async postComment() {
      if (!this.newCommentText.trim() || !this.sessionDetail || this.commentPosting) return;
      this.commentPosting = true;
      try {
        const comment = await fetchAPI(`/api/sessions/${this.sessionDetail.id}/comments`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: this.newCommentText.trim() }),
        });
        this.sessionComments.push(comment);
        this.newCommentText = '';
        showToast('Comentariu postat!', 'success');
      } catch (e) {
        showToast(e.message || 'Eroare la postare', 'error');
      } finally {
        this.commentPosting = false;
      }
    },

    async deleteComment(commentId) {
      if (!confirm('Ștergi acest comentariu?')) return;
      try {
        await fetchAPI(`/api/comments/${commentId}`, { method: 'DELETE' });
        this.sessionComments = this.sessionComments.filter(c => c.id !== commentId);
        showToast('Comentariu șters.');
      } catch (e) {
        showToast(e.message || 'Eroare la ștergere', 'error');
      }
    },

    /* ── User Note ────────────────────────────────────────────────────── */
    async saveUserNote() {
      if (!this.editNoteText.trim() || !this.sessionDetail) return;
      try {
        await fetchAPI(`/api/sessions/${this.sessionDetail.id}/note`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_note: this.editNoteText.trim() }),
        });
        this.sessionDetail.user_note = this.editNoteText.trim();
        this.editNoteText = '';
        showToast('Descriere salvată!', 'success');
      } catch (e) {
        showToast(e.message || 'Eroare la salvare', 'error');
      }
    },

    /* ── Share ─────────────────────────────────────────────────────────── */
    shareReport(platform) {
      if (!this.sessionDetail) return;
      const url = `${window.location.origin}/?report=${this.sessionDetail.id}`;
      const text = `Am raportat ${this.sessionDetail.total_objects} deșeuri detectate cu TrashDet AI!`;

      switch (platform) {
        case 'copy':
          navigator.clipboard.writeText(url).then(() => showToast('Link copiat!')).catch(() => showToast('Nu s-a putut copia', 'error'));
          break;
        case 'facebook':
          window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}&quote=${encodeURIComponent(text)}`, '_blank', 'width=600,height=400');
          break;
        case 'twitter':
          window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`, '_blank', 'width=600,height=400');
          break;
        case 'whatsapp':
          window.open(`https://wa.me/?text=${encodeURIComponent(text + ' ' + url)}`, '_blank');
          break;
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

    statusLabel(status) {
      const labels = {
        pending: 'În așteptare',
        verified: 'Verificat',
        in_progress: 'În curs de curățare',
        cleaned: 'Curățat',
        fake: 'Fals',
        expired: 'Expirat',
      };
      return labels[status] || status;
    },

    statusBadgeClass(status) {
      const classes = {
        pending: 'badge--yellow',
        verified: 'badge--green',
        in_progress: 'badge--blue',
        cleaned: 'badge--green',
        fake: 'badge--red',
        expired: 'badge--gray',
      };
      return classes[status] || 'badge--gray';
    },

    async suggestMaterial(recordId, material) {
      if (!material) return;
      try {
        await fetchAPI(`/api/records/${recordId}/suggest-material`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ suggested_material: material }),
        });
        showToast('Sugestie trimisă! Mulțumim', 'success');
        // Refresh detail to show updated material if auto-corrected
        if (this.sessionDetail) {
          this.openSessionDetail(this.sessionDetail.id);
        }
      } catch (e) {
        showToast(e.message || 'Eroare la trimiterea sugestiei', 'error');
      }
    },
  };
}
