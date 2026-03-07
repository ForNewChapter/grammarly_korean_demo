import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  plugins: [
    react(),
    VitePWA({
      strategies: 'injectManifest',
      srcDir: 'src/cache',
      filename: 'sw.ts',
      injectRegister: 'auto',
      registerType: 'autoUpdate',
      injectManifest: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,webmanifest}'],
      },
      includeAssets: ['favicon.ico'],
      manifest: {
        name: 'Korean On-Device Grammar Demo',
        short_name: 'KoProofDemo',
        theme_color: '#0f5f4a',
        background_color: '#f4f2ea',
        display: 'standalone',
        start_url: '/',
        icons: []
      },
      devOptions: {
        enabled: true,
        type: 'module'
      }
    })
  ]
});
