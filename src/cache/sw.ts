/// <reference lib="webworker" />

import { clientsClaim } from 'workbox-core';
import { precacheAndRoute, cleanupOutdatedCaches } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { CacheFirst, NetworkFirst, StaleWhileRevalidate } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';

declare let self: ServiceWorkerGlobalScope;

self.skipWaiting();
clientsClaim();
cleanupOutdatedCaches();

precacheAndRoute(self.__WB_MANIFEST);

registerRoute(
  ({ request }) => request.destination === 'script' || request.destination === 'style' || request.destination === 'document',
  new NetworkFirst({ cacheName: 'app-shell-runtime' })
);

registerRoute(
  ({ url }) => url.pathname.startsWith('/assets/models/') || url.pathname.endsWith('.onnx'),
  new CacheFirst({
    cacheName: 'models-runtime',
    plugins: [new ExpirationPlugin({ maxEntries: 12, maxAgeSeconds: 60 * 60 * 24 * 14 })],
  })
);

registerRoute(
  ({ url }) =>
    url.pathname.startsWith('/assets/rules/') ||
    url.pathname.startsWith('/assets/dict/') ||
    url.pathname.startsWith('/assets/tokenizer/'),
  new StaleWhileRevalidate({ cacheName: 'assets-runtime' })
);
