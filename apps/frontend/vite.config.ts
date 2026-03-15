import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const webRoot = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = path.resolve(webRoot, '../..');
const base = process.env.VITE_BASE_PATH || '/';

export default defineConfig({
  root: webRoot,
  base,
  publicDir: path.resolve(repoRoot, 'public'),
  cacheDir: path.resolve(repoRoot, 'node_modules/.vite-web'),
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: path.resolve(webRoot, 'dist'),
    emptyOutDir: true,
  },
  plugins: [react()]
});
