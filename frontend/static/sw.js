/* EcoAlert — Service Worker */

const CACHE = 'ecoalert-v3';
const SHELL = [
  '/',
  '/static/css/style.css',
  '/static/js/utils.js',
  '/static/js/detect.js',
  '/static/js/map.js',
  '/static/js/history.js',
  '/static/js/video.js',
  '/static/js/admin.js',
  '/static/js/auth.js',
  '/static/js/community.js',
  '/static/js/app.js',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Network-first for everything (API + static assets)
  e.respondWith(
    fetch(e.request)
      .then(response => {
        // Cache successful responses for offline use
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(e.request).then(cached => cached || new Response('offline', { status: 503 })))
  );
});
