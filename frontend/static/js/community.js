/* ── EcoAlert — Community Feed tab ───────────────────────────────────────── */

function communityApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    communityFeed: [],
    communityLoading: false,
    communityPage: 0,
    communityLimit: 20,
    rankDefinitions: [],

    // Privacy settings
    privacyAnonymous: false,
    privacyHideLocation: false,
    privacySaving: false,

    // Campaigns
    campaigns: [],
    campaignsLoading: false,
    campaignLeaderboard: [],
    campaignLeaderboardLoading: false,
    selectedCampaignId: null,
    newCampaign: { title: '', description: '', start_date: '', end_date: '' },
    campaignCreating: false,

    /* ── Init ─────────────────────────────────────────────────────────── */
    initCommunity() {
      // Load rank definitions once
      fetch('/api/ranks').then(r => r.ok ? r.json() : []).then(d => { this.rankDefinitions = d; }).catch(() => {});
    },

    /* ── Feed ─────────────────────────────────────────────────────────── */
    async loadCommunityFeed() {
      this.communityLoading = true;
      try {
        this.communityFeed = await fetchAPI(
          `/api/community/feed?skip=${this.communityPage * this.communityLimit}&limit=${this.communityLimit}`
        );
      } catch (e) {
        showToast('Eroare la încărcarea feed-ului', 'error');
      } finally {
        this.communityLoading = false;
      }
    },

    feedEventIcon(type) {
      const icons = { report: '📸', verified: '✅', cleaned: '🟢', vote: '🗳️' };
      return icons[type] || '📋';
    },

    feedEventLabel(type) {
      const labels = {
        report: 'Raportare nouă',
        verified: 'Raport verificat',
        cleaned: 'Zonă curățată',
        vote: 'Vot comunitate',
      };
      return labels[type] || type;
    },

    /* ── Privacy ──────────────────────────────────────────────────────── */
    async loadPrivacySettings() {
      if (!this.myProfile) return;
      this.privacyAnonymous = this.myProfile.anonymous_reports || false;
      this.privacyHideLocation = this.myProfile.hide_exact_location || false;
    },

    async savePrivacy() {
      this.privacySaving = true;
      try {
        await fetchAPI('/api/me/settings', {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            anonymous_reports: this.privacyAnonymous,
            hide_exact_location: this.privacyHideLocation,
          }),
        });
        showToast('Setările au fost salvate!', 'success');
      } catch (e) {
        showToast('Eroare la salvare', 'error');
      } finally {
        this.privacySaving = false;
      }
    },

    /* ── Campaigns ────────────────────────────────────────────────────── */
    async loadCampaigns() {
      this.campaignsLoading = true;
      try {
        this.campaigns = await fetchAPI('/api/campaigns');
      } catch (e) {
        showToast('Eroare la încărcarea campaniilor', 'error');
      } finally {
        this.campaignsLoading = false;
      }
    },

    async joinCampaign(campaignId) {
      try {
        await fetchAPI(`/api/campaigns/${campaignId}/join`, { method: 'POST' });
        showToast('Te-ai înscris în campanie! 🎉');
        this.loadCampaigns();
        if (this.selectedCampaignId === campaignId) this.loadCampaignLeaderboard(campaignId);
      } catch (e) {
        showToast(e.message, 'error');
      }
    },

    async createCampaign() {
      const { title, description, start_date, end_date } = this.newCampaign;
      if (!title.trim()) return showToast('Titlul este obligatoriu', 'error');
      this.campaignCreating = true;
      try {
        await fetchAPI('/api/campaigns', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: title.trim(),
            description: description.trim() || null,
            start_date: start_date || null,
            end_date: end_date || null,
          }),
        });
        this.newCampaign = { title: '', description: '', start_date: '', end_date: '' };
        showToast('Campanie creată ✓');
        this.loadCampaigns();
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.campaignCreating = false;
      }
    },

    async loadCampaignLeaderboard(campaignId) {
      this.selectedCampaignId = campaignId;
      this.campaignLeaderboardLoading = true;
      try {
        this.campaignLeaderboard = await fetchAPI(`/api/campaigns/${campaignId}/leaderboard`);
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.campaignLeaderboardLoading = false;
      }
    },

    campaignStatus(c) {
      const now = new Date();
      if (c.start_date && new Date(c.start_date) > now) return { label: 'Viitoare', color: '#3b82f6' };
      if (c.end_date && new Date(c.end_date) < now) return { label: 'Încheiată', color: '#6b7280' };
      return { label: 'Activă', color: '#10b981' };
    },

    /* ── Rank helpers ─────────────────────────────────────────────────── */
    rankProgress(ecoScore) {
      const ranks = this.rankDefinitions;
      if (!ranks.length) return 0;
      for (let i = 0; i < ranks.length; i++) {
        const r = ranks[i];
        if (ecoScore >= r.min_score && (r.max_score === null || ecoScore <= r.max_score)) {
          if (r.max_score === null) return 100; // Legend — max rank
          const range = r.max_score - r.min_score + 1;
          return Math.min(100, ((ecoScore - r.min_score) / range) * 100);
        }
      }
      return 0;
    },

    nextRank(currentRank) {
      const ranks = this.rankDefinitions;
      const idx = ranks.findIndex(r => r.name === currentRank);
      if (idx >= 0 && idx < ranks.length - 1) return ranks[idx + 1];
      return null;
    },

    rankColor(rankName) {
      const colors = {
        Novice: '#6b7280', Scout: '#3b82f6', Guardian: '#10b981',
        Ranger: '#f59e0b', Champion: '#ef4444', Legend: '#8b5cf6',
      };
      return colors[rankName] || '#6b7280';
    },

    streakEmoji(days) {
      if (days >= 30) return '🔥🔥🔥';
      if (days >= 14) return '🔥🔥';
      if (days >= 4) return '🔥';
      return '❄️';
    },

    timeAgo(iso) { return timeAgo(iso); },
    formatDate(iso) { return formatDate(iso); },
  };
}
