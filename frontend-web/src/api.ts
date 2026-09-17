// ── 백엔드 주소 / 정적 데이터 fallback ─────────────────
// 개발 서버(vite)에서는 base가 비어 있어 /api, /ws 프록시를 그대로 사용한다.
// GitHub Pages 같은 정적 호스팅에서는 백엔드가 없으므로
//   1) ?api=https://xxxx.ngrok-free.app 쿼리 또는 툴바의 Backend 설정으로 주소를 지정하거나
//   2) 빌드 시 생성된 static-api/*.json (노드 카탈로그, 예제)으로 fallback 한다.

const STORAGE_KEY = 'mlstudio.apiBase';

/** 정적 호스팅(GitHub Pages) 빌드 여부 */
export const IS_STATIC_HOST = import.meta.env.VITE_STATIC_HOST === '1';

function normalizeBase(value: string | null | undefined): string {
  return (value || '').trim().replace(/\/+$/, '');
}

function readInitialBase(): string {
  try {
    const query = new URLSearchParams(window.location.search).get('api');
    if (query !== null) {
      const base = normalizeBase(query);
      if (base) localStorage.setItem(STORAGE_KEY, base);
      else localStorage.removeItem(STORAGE_KEY);
      return base;
    }
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return normalizeBase(saved);
  } catch {
    // localStorage 접근 불가 (프라이빗 모드 등)
  }
  return normalizeBase(import.meta.env.VITE_API_BASE);
}

let apiBase = readInitialBase();

export function getApiBase(): string {
  return apiBase;
}

export function setApiBase(value: string) {
  apiBase = normalizeBase(value);
  try {
    if (apiBase) localStorage.setItem(STORAGE_KEY, apiBase);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}

/** 백엔드에 연결 가능한 구성인지 (정적 호스팅 + 주소 미지정이면 false) */
export function hasBackend(): boolean {
  return !!apiBase || !IS_STATIC_HOST;
}

export function apiUrl(path: string): string {
  return `${apiBase}${path}`;
}

export function wsUrl(path: string): string {
  if (apiBase) return apiBase.replace(/^http/, 'ws') + path;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}

export function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  if (!hasBackend()) {
    return Promise.reject(new Error('Backend URL is not configured'));
  }
  const headers = new Headers(init.headers);
  // ngrok 무료 터널의 브라우저 경고 페이지 우회
  if (apiBase) headers.set('ngrok-skip-browser-warning', '1');
  return fetch(apiUrl(path), { ...init, headers });
}

/**
 * 백엔드 GET 요청 → 실패 시 빌드에 포함된 정적 JSON(static-api/<file>)으로 대체.
 */
export async function fetchJsonWithFallback<T = any>(path: string, staticFile: string): Promise<T> {
  if (hasBackend()) {
    try {
      const res = await apiFetch(path);
      if (res.ok && (res.headers.get('content-type') || '').includes('application/json')) {
        return await res.json();
      }
    } catch {
      // fallback 으로 진행
    }
  }
  const res = await fetch(`${import.meta.env.BASE_URL}static-api/${staticFile}`);
  if (!res.ok) throw new Error(`Static data not found: ${staticFile}`);
  return res.json();
}

/** 백엔드 health check */
export async function checkBackend(timeoutMs = 4000): Promise<boolean> {
  if (!hasBackend()) return false;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await apiFetch('/api/health', { signal: controller.signal });
    if (!res.ok) return false;
    const data = await res.json();
    return data?.status === 'ok';
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
