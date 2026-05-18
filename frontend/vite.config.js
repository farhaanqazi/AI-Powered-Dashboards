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
    // NO custom manualChunks. A hand-rolled React/vendor split repeatedly
    // produced circular cross-chunk ESM imports (react <-> vendor/plotly),
    // leaving `React` undefined at chunk init ("Cannot read properties of
    // undefined (reading 'useState'/'createContext')"). Cutting one edge just
    // relocated the cycle — the mechanism itself was the defect. Vite's
    // default chunking handles shared deps correctly, and the real load-time
    // win comes from route-level React.lazy + the dynamic-imported export
    // stack (App.jsx / DashboardPage.jsx), which keep plotly/jsPDF out of the
    // initial chunk regardless of vendor grouping.
    rollupOptions: {
      output: {
        // Content-hashed filenames for cache busting.
        assetFileNames: `assets/${BUILD_ID}/[name].[hash][extname]`,
        chunkFileNames: `assets/${BUILD_ID}/[name].[hash].js`,
        entryFileNames: `assets/${BUILD_ID}/[name].[hash].js`,
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