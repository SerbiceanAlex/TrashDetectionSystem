/* ── EcoAlert — Shared utilities ─────────────────────────────────────────── */

const MATERIAL_COLORS = {
  plastic: { bg: '#3b82f6', light: '#dbeafe', text: '#1d4ed8' },
  glass:   { bg: '#06b6d4', light: '#cffafe', text: '#0e7490' },
  metal:   { bg: '#f59e0b', light: '#fef3c7', text: '#b45309' },
  paper:   { bg: '#f97316', light: '#ffedd5', text: '#c2410c' },
  other:   { bg: '#6b7280', light: '#f3f4f6', text: '#374151' },
};

const SEVERITY = {
  0: { label: 'Curat',   color: '#22c55e', bg: '#dcfce7', icon: '🟢' },
  1: { label: 'Scăzut',  color: '#eab308', bg: '#fef9c3', icon: '🟡' },
  2: { label: 'Mediu',   color: '#f97316', bg: '#ffedd5', icon: '🟠' },
  3: { label: 'Ridicat', color: '#ef4444', bg: '#fee2e2', icon: '🔴' },
};

/* ── Toast system ──────────────────────────────────────────────────────────── */
const toastQueue = [];
let _toastAlpine = null;

function registerToastAlpine(alpineInstance) {
  _toastAlpine = alpineInstance;
}

function showToast(message, type = 'success', duration = 3500) {
  if (_toastAlpine) {
    _toastAlpine.addToast(message, type, duration);
  }
}

/* ── Date / time helpers ────────────────────────────────────────────────── */
function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('ro-RO', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function formatDateShort(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('ro-RO', {
    day: '2-digit', month: '2-digit',
  });
}

function timeAgo(iso) {
  if (!iso) return '—';
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  const h = Math.floor(m / 60);
  const d = Math.floor(h / 24);
  if (d > 0) return `acum ${d}z`;
  if (h > 0) return `acum ${h}h`;
  if (m > 0) return `acum ${m}m`;
  return 'acum';
}

function formatDuration(sec) {
  if (!sec) return '0s';
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/* ── Material helpers ───────────────────────────────────────────────────── */
function materialColor(m) {
  return MATERIAL_COLORS[m?.toLowerCase()] || MATERIAL_COLORS.other;
}

function materialBadgeStyle(m) {
  const c = materialColor(m);
  return `background:${c.light};color:${c.text};`;
}

/* ── Severity helpers ───────────────────────────────────────────────────── */
function severityFromCount(count) {
  if (count === 0) return 0;
  if (count < 5) return 1;
  if (count < 15) return 2;
  return 3;
}

function getSeverity(level) {
  return SEVERITY[level] || SEVERITY[0];
}

/* ── Fetch wrapper with error handling ──────────────────────────────────── */
async function fetchAPI(url, options = {}) {
  const token = localStorage.getItem('eco_token');
  if (token) {
    options.headers = {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    };
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Eroare server (${response.status})`;
    try {
      const err = await response.json();
      detail = (err && err.detail) ? err.detail : detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

/* ── GPS / geolocation ──────────────────────────────────────────────────── */
function requestGPS() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation indisponibil'));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude, accuracy: pos.coords.accuracy }),
      (err) => reject(err),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 30000 }
    );
  });
}

/* ── Misc helpers ───────────────────────────────────────────────────────── */
function getAnnotatedUrl(path) {
  if (!path) return '';
  const filename = path.replace(/\\/g, '/').split('/').pop();
  return `/annotated/${filename}`;
}

function parseMaterials(jsonStr) {
  try { return JSON.parse(jsonStr || '{}'); } catch (_) { return {}; }
}

function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}
