import React, { useEffect, useRef, useState } from 'react';
import { IS_STATIC_HOST, checkBackend, getApiBase, hasBackend, setApiBase } from '../api';

type Status = 'checking' | 'online' | 'offline';

/** 백엔드(FastAPI) 주소 설정 + 연결 상태 표시 */
export function BackendSettings() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<Status>('checking');
  const [draft, setDraft] = useState(getApiBase());
  const ref = useRef<HTMLDivElement>(null);

  const refresh = async () => {
    setStatus('checking');
    setStatus((await checkBackend()) ? 'online' : 'offline');
  };

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!open) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as HTMLElement)) setOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open]);

  const apply = (value: string) => {
    setApiBase(value);
    setDraft(getApiBase());
    refresh();
  };

  const mixedContent =
    window.location.protocol === 'https:' && /^http:\/\//i.test(draft.trim()) &&
    !/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?/i.test(draft.trim());

  const dot =
    status === 'online' ? 'bg-green-400' : status === 'checking' ? 'bg-yellow-400' : 'bg-red-500';
  const label =
    status === 'online'
      ? 'Backend'
      : status === 'checking'
      ? 'Backend...'
      : hasBackend()
      ? 'Backend 오프라인'
      : '브라우저 모드';

  return (
    <div className="relative flex items-center" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="px-3 py-1 text-xs rounded bg-gray-600 hover:bg-gray-500 text-gray-300 font-medium flex items-center gap-1.5 whitespace-nowrap"
        title="학습 서버(FastAPI) 연결 설정"
      >
        <span className={`inline-block w-2 h-2 rounded-full ${dot}`} />
        {label}
      </button>
      {open && (
        <div className="absolute top-full right-0 mt-1 w-80 bg-gray-700 border border-gray-600 rounded-lg shadow-xl z-50 p-3 text-xs text-gray-200 space-y-2">
          <div className="font-semibold text-white">학습 서버 주소</div>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && apply(draft)}
            placeholder={IS_STATIC_HOST ? 'https://xxxx.ngrok-free.app' : '비워두면 현재 서버(/api 프록시) 사용'}
            className="w-full px-2 py-1 rounded bg-gray-800 border border-gray-600 text-white focus:outline-none focus:border-blue-500"
          />
          {mixedContent && (
            <div className="text-yellow-300">
              HTTPS 페이지에서는 http:// 서버에 접속할 수 없습니다. ngrok 등 https 주소를 사용하세요.
            </div>
          )}
          <div className="flex gap-2">
            <button onClick={() => apply(draft)} className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white">
              저장 & 연결 확인
            </button>
            <button onClick={() => apply('')} className="px-3 py-1 rounded bg-gray-600 hover:bg-gray-500 text-white">
              초기화
            </button>
          </div>
          <div className="text-gray-400 leading-relaxed">
            {status === 'online'
              ? '✓ 서버에 연결되었습니다. 학습/예측/모델 관리 기능을 사용할 수 있습니다.'
              : IS_STATIC_HOST
              ? '서버 없이 브라우저(TensorFlow.js)에서 학습합니다. 표(CSV) 데이터 + Dense 계열 예제가 대상이며, CNN·LSTM·전이학습 등은 학습 서버가 필요합니다. 서버를 쓰려면 로컬에서 백엔드와 ngrok 터널(포트 8000)을 실행한 뒤 주소를 입력하세요.'
              : '서버에 연결할 수 없습니다. 백엔드(uvicorn, 포트 8000)가 실행 중인지 확인하세요.'}
          </div>
        </div>
      )}
    </div>
  );
}
