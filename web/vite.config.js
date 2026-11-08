import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: false,
    proxy: {
      // Proxy API calls to backend when not in mock mode
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        configure: (proxy, options) => {
          // Only proxy when VITE_MOCK is not true
          if (process.env.VITE_MOCK === 'true') {
            proxy.options = { ...proxy.options, bypass: () => true }
          }
        },
      },
    },
  },
})