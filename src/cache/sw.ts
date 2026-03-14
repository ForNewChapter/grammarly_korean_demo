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
  ({ url }) =>
    url.pathname.includes('/assets/models/') ||
    url.pathname.includes('/assets/kiwi/') ||
    url.pathname.endsWith('.onnx') ||
    url.pathname.endsWith('.wasm'),
  new CacheFirst({
    cacheName: 'models-runtime',
    plugins: [new ExpirationPlugin({ maxEntries: 24, maxAgeSeconds: 60 * 60 * 24 * 14 })],
  })
);

registerRoute(
  ({ url }) =>
    url.pathname.includes('/assets/rules/') ||
    url.pathname.includes('/assets/dict/') ||
    url.pathname.includes('/assets/tokenizer/'),
  new StaleWhileRevalidate({ cacheName: 'assets-runtime' })
);
