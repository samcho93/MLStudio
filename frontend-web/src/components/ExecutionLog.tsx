import React, { useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';

const LEVEL_COLORS: Record<string, string> = {
  info: 'text-gray-300',
  success: 'text-green-400',
  error: 'text-red-400',
  data: 'text-blue-300',
};

const LEVEL_BADGE: Record<string, string> = {
  info: 'text-blue-400',
  success: 'text-green-500',
  error: 'text-red-500',
  data: 'text-cyan-400',
};

const LEVEL_ICON: Record<string, string> = {
  info: '\u00b7',
  success: '\u25cf',
  error: '\u2715',
  data: '\u25c6',
};

export function ExecutionLog() {
  const logs = useStore((s) => s.logs);
  const logPanelOpen = useStore((s) => s.logPanelOpen);
  const setLogPanelOpen = useStore((s) => s.setLogPanelOpen);
  const clearLogs = useStore((s) => s.clearLogs);
  const isTraining = useStore((s) => s.training.isTraining);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevTrainingRef = useRef(false);

  // Auto-expand when training starts
  useEffect(() => {
    if (isTraining && !prevTrainingRef.current) {
      setLogPanelOpen(true);
    }
    prevTrainingRef.current = isTraining;
  }, [isTraining, setLogPanelOpen]);

  // Auto-scroll to bottom on new logs
  useEffect(() => {
    if (scrollRef.current && logPanelOpen) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, logPanelOpen]);

  const hasError = logs.some((l) => l.level === 'error');

  return (
    <div className="bg-gray-900 border-t border-gray-700 flex flex-col shrink-0">
      {/* Collapsed bar / Header */}
      <div
        className="flex items-center justify-between px-3 py-1 bg-gray-800 border-b border-gray-700 cursor-pointer select-none shrink-0"
        onClick={() => setLogPanelOpen(!logPanelOpen)}
      >
        <div className="flex items-center gap-3">
          <span
            className={`text-[10px] transition-transform duration-200 ${logPanelOpen ? 'rotate-90' : ''} text-gray-400`}
          >
            {'\u25b6'}
          </span>
          <span className="text-[11px] font-bold text-gray-300 uppercase tracking-wider">
            Output
          </span>
          {logs.length > 0 && (
            <span
              className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                hasError
                  ? 'bg-red-600/30 text-red-400'
                  : 'bg-gray-700 text-gray-400'
              }`}
            >
              {logs.length}
            </span>
          )}
          {isTraining && (
            <span className="text-[10px] text-yellow-400 animate-pulse">
              running...
            </span>
          )}
        </div>
        <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
          <button
            onClick={clearLogs}
            className="text-[10px] text-gray-500 hover:text-gray-300 px-1.5 py-0.5 rounded hover:bg-gray-700"
            title="Clear logs"
          >
            Clear
          </button>
          {logPanelOpen && (
            <button
              onClick={() => setLogPanelOpen(false)}
              className="text-[10px] text-gray-500 hover:text-gray-300 px-1.5 py-0.5 rounded hover:bg-gray-700"
              title="Collapse"
            >
              {'\u2715'}
            </button>
          )}
        </div>
      </div>

      {/* Expanded log content */}
      {logPanelOpen && (
        <div
          ref={scrollRef}
          className="overflow-y-auto overflow-x-auto font-mono text-[11px] leading-[18px] px-3 py-1 bg-[#0d1117]"
          style={{ height: 150 }}
        >
          {logs.length === 0 ? (
            <div className="text-gray-600 py-4 text-center">
              Run a pipeline to see output here
            </div>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="flex hover:bg-white/[0.02]">
                <span className="text-gray-600 select-none shrink-0 w-16">
                  {log.timestamp}
                </span>
                <span
                  className={`shrink-0 w-4 text-center ${LEVEL_BADGE[log.level]}`}
                >
                  {LEVEL_ICON[log.level] || '\u00b7'}
                </span>
                <span
                  className={`${LEVEL_COLORS[log.level]} whitespace-pre-wrap break-all ml-1`}
                >
                  {log.message}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
