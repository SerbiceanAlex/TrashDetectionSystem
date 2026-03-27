/* ── EcoAlert — Core app shell ────────────────────────────────────────────
   Combines all sub-modules (detect, map, history, stats) via Alpine.js merge.
   ─────────────────────────────────────────────────────────────────────── */

function ecoApp() {
  return {
    /* ── Spread all sub-modules ──────────────────────────────────────── */
    ...detectApp(),
    ...mapApp(),
    ...historyApp(),
    ...statsApp(),

    /* ── Global state ─────────────────────────────────────────────────── */
    activeTab: 'scan',
    darkMode: false,

    // Toast system
    toasts: [],

    // Lightbox
    lightboxSrc: null,

    /* ── Nav tabs definition ──────────────────────────────────────────── */
    tabs: [
      { id: 'scan',    icon: '📷', label: 'Scanează' },
      { id: 'map',     icon: '🗺️', label: 'Hartă'    },
      { id: 'reports', icon: '📋', label: 'Rapoarte'  },
      { id: 'stats',   icon: '📊', label: 'Statistici'},
    ],

    /* ── Init ─────────────────────────────────────────────────────────── */
    async init() {
      // Dark mode from localStorage or system preference
      this.darkMode = localStorage.getItem('eco_dark') === 'true' ||
        (!localStorage.getItem('eco_dark') && window.matchMedia('(prefers-color-scheme: dark)').matches);

      // Apply dark class immediately
      if (this.darkMode) document.documentElement.classList.add('dark');

      // Register toast system
      registerToastAlpine(this);

      // Init sub-modules
      this.initDetect();
      this.initHistory();
      this.initStats();
      this.initMap();

      // Watch tab changes
      this.$watch('activeTab', (tab) => {
        if (tab === 'map') this.ensureMap();
        if (tab === 'stats') setTimeout(() => this.renderCharts(), 150);
        if (tab === 'reports') this.loadHistory();
      });

      // Watch dark mode
      this.$watch('darkMode', (dark) => {
        document.documentElement.classList.toggle('dark', dark);
        localStorage.setItem('eco_dark', dark);
      });
    },

    /* ── Tab navigation ───────────────────────────────────────────────── */
    goTo(tab) {
      this.activeTab = tab;
    },

    /* ── Dark mode ────────────────────────────────────────────────────── */
    toggleDark() {
      this.darkMode = !this.darkMode;
    },

    /* ── Toast system ─────────────────────────────────────────────────── */
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
