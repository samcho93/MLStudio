# vite.config.js 새로 생성
@"
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    allowedHosts: 'all'
  }
})
"@ | Set-Content D:\Work\WebSharp\vite.config.js -Encoding UTF8