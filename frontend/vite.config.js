import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Generate a unique build ID based on timestamp to ensure cache busting
const BUILD_ID = Date.now().toString();

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './',  // Changed to relative path for Hugging Face Spaces compatibility
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'terser',
    chunkSizeWarningLimit: 900,
    // Vite otherwise injects <link rel="modulepreload"> for async-chunk deps
    // too, so the upload screen would eagerly download plotly/export anyway —
    // defeating the lazy-loading. Keep preload for the genuine first-paint
    // vendors (react/clerk/etc) but strip the heavy lazy chunks.
    modulePreload: {
      resolveDependencies: (_filename, deps) =>
        deps.filter((d) => !d.includes('plotly') && !/(^|\/)export\./.test(d)),
    },
    rollupOptions: {
      output: {
        // Add content hash to filenames to ensure cache busting
        assetFileNames: (assetInfo) => {
          if (assetInfo.name.endsWith('.css')) {
            return `assets/${BUILD_ID}/[name].[hash][extname]`;
          }
          return `assets/${BUILD_ID}/[name].[hash][extname]`;
        },
        chunkFileNames: `assets/${BUILD_ID}/[name].[hash].js`,
        entryFileNames: `assets/${BUILD_ID}/[name].[hash].js`,
        // Split heavy, independently-cacheable vendors into their own chunks
        // so they (a) don't bloat the entry and (b) only download with the
        // route that needs them (dashboard pulls plotly; export pulls jspdf).
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined;
          if (id.includes('plotly') || id.includes('react-plotly')) return 'plotly';
          if (id.includes('jspdf') || id.includes('html-to-image') || id.includes('html2canvas')) return 'export';
          if (id.includes('@clerk')) return 'clerk';
          if (id.includes('react-router') || id.includes('/react-dom/') || id.includes('/react/') || id.includes('scheduler')) return 'react';
          return 'vendor';
        },
      }
    }
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/upload': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/load_external': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/dashboard': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})