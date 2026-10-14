import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Tailwind v4 is wired through its Vite plugin. There is deliberately no
// tailwind.config.js and no PostCSS config: v4 is configured in CSS via @theme
// in src/index.css.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    // three.js and the drei helpers dominate the bundle; keep them separate so the
    // dashboard shell can paint before the 3D chunk has finished parsing.
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes('node_modules/three')) return 'three'
          if (id.includes('@react-three')) return 'r3f'
          return undefined
        },
      },
    },
    chunkSizeWarningLimit: 1200,
  },
})
