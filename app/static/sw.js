// Minimal service worker for PWA install support
const CACHE_NAME = 'trash-detect-v1';

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(clients.claim());
});

self.addEventListener('fetch', (event) => {
  // Network-first strategy — app needs live API access
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
