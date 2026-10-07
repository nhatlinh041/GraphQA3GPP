import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    allowedHosts: ['ask3gpp.online', 'www.ask3gpp.online'],
    proxy: {
      '/api': {
        target: `http://localhost:${process.env.RAG_PORT || 8000}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    // Multi-page: each entry produces its own HTML at the output root
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        cypher: resolve(__dirname, 'cypher.html'),
        benchmarks: resolve(__dirname, 'benchmarks.html'),
        run: resolve(__dirname, 'run.html'),
      },
    },
  },
});
