import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Generate a unique build ID based on timestamp to ensure cache busting
const BUILD_ID = Date.now().toString();

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/',  // Important for serving
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'terser',
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