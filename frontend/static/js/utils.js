/* ── TrashDet — Shared utilities ──────────────────────────────────────────── */

/* ── Toast system ──────────────────────────────────────────────────────────── */
const toastQueue = [];
let _toastAlpine = null;
const AUTH_TOKEN_KEY = 'eco_token';

function registerToastAlpine(alpineInstance) {
  _toastAlpine = alpineInstance;
}

function showToast(message, type = 'success', duration = 3500) {
  if (_toastAlpine) {
    _toastAlpine.addToast(message, type, duration);
  }
}

/* ── Auth token storage ────────────────────────────────────────────────────── */
function getAuthToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY) || sessionStorage.getItem(AUTH_TOKEN_KEY) || '';
}

function setAuthToken(token) {
  if (token) {
    sessionStorage.setItem(AUTH_TOKEN_KEY, token);
    localStorage.setItem(AUTH_TOKEN_KEY, token);
  } else {
    sessionStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_TOKEN_KEY);
  }
}

function clearAuthToken() {
  sessionStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_TOKEN_KEY);
}

function eventMediaUrl(eventOrId, kind = 'thumbnail') {
  const id = typeof eventOrId === 'object'
    ? (eventOrId?.id ?? eventOrId?.event_id)
    : eventOrId;
  const token = getAuthToken();
  if (!id || !token) return '';
  const mediaKind = kind === 'clip' ? 'clip' : 'thumbnail';
  return `/api/littering/events/${encodeURIComponent(id)}/${mediaKind}?token=${encodeURIComponent(token)}`;
}

/* ── Fetch wrapper with error handling ──────────────────────────────────────── */
async function fetchAPI(url, options = {}) {
  const token = getAuthToken();
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

/* ── GPS / geolocation ──────────────────────────────────────────────────────── */
function requestGPS(options = {}) {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error('Geolocation indisponibil'));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude, accuracy: pos.coords.accuracy }),
      (err) => reject(err),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 30000, ...options }
    );
  });
}
