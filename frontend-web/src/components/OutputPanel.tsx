import React, { useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';

const LEVEL_COLORS: Record<string, string> = {
  info: 'text-gray-300',
  success: 'text-green-400',
  data: 'text-blue-300',
  error: 'text-red-400',
};

const LEVEL_BADGE: Record<string, string> = {
  info: 'text-blue-400',
  success: 'text-green-500',
  data: 'text-cyan-400',
  error: 'text-red-500',
};

export function OutputPanel() {
  const logs = useStore((s) => s.logs);
  const showOutput = useStore((s) => s.showOutput);
  const clearLogs = useStore((s) => s.clearLogs);
  const setShowOutput = useStore((s) => s.setShowOutput);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 새 로그 추가 시 자동 스크롤
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  if (!showOutput) return null;

  return (
    <div className="bg-[#0d1117] border-t border-gray-700 flex flex-col" style={{ height: 200 }}>
      {/* 헤더 */}
      <div className="flex items-center justify-between px-3 py-1 bg-gray-800 border-b border-gray-700 shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-bold text-gray-300 uppercase tracking-wider">Output</span>
          <span className="text-[10px] text-gray-500">{logs.length} lines</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={clearLogs}
            className="text-[10px] text-gray-500 hover:text-gray-300 px-1.5 py-0.5 rounded hover:bg-gray-700"
            title="Clear"
          >
            Clear
          </button>
          <button
            onClick={() => setShowOutput(false)}
            className="text-[10px] text-gray-500 hover:text-gray-300 px-1.5 py-0.5 rounded hover:bg-gray-700"
            title="Close"
          >
            ✕
          </button>
        </div>
      </div>

      {/* 로그 내용 */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto overflow-x-auto font-mono text-[11px] leading-[18px] px-3 py-1">
        {logs.length === 0 ? (
          <div className="text-gray-600 py-4 text-center">Run a pipeline to see output here</div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className="flex hover:bg-white/[0.02]">
              <span className="text-gray-600 select-none shrink-0 w-16">{log.timestamp}</span>
              <span className={`shrink-0 w-4 text-center ${LEVEL_BADGE[log.level]}`}>
                {log.level === 'error' ? '✕' : log.level === 'success' ? '●' : log.level === 'data' ? '◆' : '·'}
              </span>
              <span className={`${LEVEL_COLORS[log.level]} whitespace-pre-wrap break-all ml-1`}>
                {log.message}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
