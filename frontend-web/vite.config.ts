import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const isNgrok = process.env.NGROK === '1';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    cors: true,
    allowedHosts: ['.ngrok-free.dev', '.ngrok.io', '.ngrok-free.app'],
    ...(isNgrok && {
      hmr: {
        clientPort: 443,
        protocol: 'wss',
      },
    }),
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
});
