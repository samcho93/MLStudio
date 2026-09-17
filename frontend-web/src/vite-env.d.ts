/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 백엔드 기본 주소 (예: https://xxxx.ngrok-free.app). 비우면 현재 origin 사용 */
  readonly VITE_API_BASE?: string;
  /** '1' 이면 정적 호스팅(GitHub Pages) 빌드 */
  readonly VITE_STATIC_HOST?: string;
}
