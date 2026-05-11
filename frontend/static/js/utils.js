/* ── TrashDet — Shared utilities ──────────────────────────────────────────── */

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

/* ── Fetch wrapper with error handling ──────────────────────────────────────── */
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
