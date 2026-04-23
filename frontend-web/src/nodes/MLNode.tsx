import React, { memo, useState, useRef, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useStore } from '../store/useStore';

const CATEGORY_COLORS: Record<string, string> = {
  data: '#3b82f6',
  layer: '#8b5cf6',
  training: '#f59e0b',
  visualization: '#10b981',
  evaluation: '#10b981',
  output: '#ef4444',
};

const CATEGORY_LABELS: Record<string, string> = {
  data: 'DATA',
  layer: 'LAYER',
  training: 'TRAIN',
  visualization: 'VIZ',
  evaluation: 'EVAL',
  output: 'OUT',
};

interface MLNodeData {
  label: string;
  category: string;
  params: Record<string, any>;
  inputs: { name: string; type: string }[];
  outputs: { name: string; type: string }[];
  [key: string]: unknown;
}

/* ── 레이아웃 상수 ─────────────────────────────── */
const HEADER_H = 32;
const PORT_ROW_H = 22;   // 포트 한 줄 높이
const PORT_PAD_Y = 6;    // 포트 영역 상하 패딩
const PARAM_ROW_H = 20;  // 파라미터 한 줄 높이
const PARAM_PAD_Y = 6;   // 파라미터 영역 상하 패딩

/* ── 요약 값 포맷팅 ────────────────────────────── */
function SummaryContent({ info }: { info: any }) {
  if (!info) return null;

  // 이미지 출력
  if (info.type === 'image_b64' && info.image_b64) {
    return (
      <img
        src={`data:image/png;base64,${info.image_b64}`}
        alt="output"
        className="rounded max-w-[300px] max-h-[200px] object-contain"
      />
    );
  }

  // ndarray / tensor
  if (info.type === 'ndarray' || info.type === 'tensor') {
    const shape = `[${info.shape?.join(', ')}]`;
    let stats = '';
    if (info.min !== undefined) {
      stats = `min=${info.min.toFixed(3)}, max=${info.max.toFixed(3)}, mean=${info.mean.toFixed(3)}`;
    }
    if (info.unique_count !== undefined) {
      stats = `unique=${info.unique_count}`;
      if (info.unique_values) stats += `: ${info.unique_values.join(', ')}`;
    }
    return (
      <div>
        <span className="text-cyan-400">shape={shape}</span>
        {info.dtype && <span className="text-gray-500 ml-1">({info.dtype})</span>}
        {stats && <div className="text-gray-400 mt-0.5">{stats}</div>}
      </div>
    );
  }

  // DataFrame
  if (info.type === 'DataFrame') {
    return (
      <div>
        <span className="text-cyan-400">shape=[{info.shape?.join(', ')}]</span>
        <div className="text-gray-400 mt-0.5">cols: {info.columns?.slice(0, 6).join(', ')}{(info.columns?.length || 0) > 6 ? '...' : ''}</div>
      </div>
    );
  }

  // 레이어 리스트
  if (info.type === 'layers') {
    return (
      <div>
        <span className="text-purple-400">{info.count} layers</span>
        <div className="mt-1 space-y-0.5">
          {(info.layers as string[]).map((l: string, i: number) => (
            <div key={i} className="text-gray-300 flex items-center gap-1">
              <span className="text-gray-600">{i + 1}.</span>
              <span>{l}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // config
  if (info.type === 'config') {
    const preview = info.preview || {};
    return (
      <div className="space-y-0.5">
        {Object.entries(preview).map(([k, v]) => (
          <div key={k}>
            <span className="text-gray-500">{k}:</span>{' '}
            <span className="text-gray-300">{String(v)}</span>
          </div>
        ))}
      </div>
    );
  }

  // scalar
  if (info.value !== undefined) {
    return <span className="text-green-400">{String(info.value)}</span>;
  }

  // list
  if (info.type === 'list') {
    return <span className="text-gray-400">list (length={info.length})</span>;
  }

  return <span className="text-gray-500">{info.type || ''}</span>;
}

function MLNode({ id, data, selected }: NodeProps<any>) {
  const nodeData = data as MLNodeData;
  const color = CATEGORY_COLORS[nodeData.category] || '#6b7280';
  const catLabel = CATEGORY_LABELS[nodeData.category] || 'MISC';

  const inputs = nodeData.inputs || [];
  const outputs = nodeData.outputs || [];
  const params = Object.entries(nodeData.params || {});
  const maxPorts = Math.max(inputs.length, outputs.length);

  /* ── 핸들 top 위치 계산 (포트 영역 기준) ──── */
  const portHandleTop = (idx: number) =>
    HEADER_H + PORT_PAD_Y + idx * PORT_ROW_H + PORT_ROW_H / 2;

  /* ── 호버 말풍선 (1초 지연, portal) ────────── */
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const nodeRef = useRef<HTMLDivElement>(null);
  const summary = useStore((s) => s.nodeOutputSummaries[id]);

  const onMouseEnter = useCallback(() => {
    if (!summary || Object.keys(summary).length === 0) return;
    hoverTimer.current = setTimeout(() => {
      if (nodeRef.current) {
        const rect = nodeRef.current.getBoundingClientRect();
        setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top });
      }
      setShowTooltip(true);
    }, 1000);
  }, [summary]);

  const onMouseLeave = useCallback(() => {
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    hoverTimer.current = null;
    setShowTooltip(false);
  }, []);

  useEffect(() => {
    return () => { if (hoverTimer.current) clearTimeout(hoverTimer.current); };
  }, []);

  return (
    <div
      ref={nodeRef}
      className="relative rounded-lg shadow-lg ml-node-body"
      style={{
        border: `2px solid ${selected ? '#fff' : color}`,
        background: '#1e1e2e',
        minWidth: 200,
      }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {/* ── 헤더 ─────────────────────────────── */}
      <div
        className="px-3 py-1.5 rounded-t-md flex items-center justify-between ml-node-header"
        style={{ background: color + '22' }}
      >
        <span
          className="text-[10px] font-bold px-1.5 py-0.5 rounded"
          style={{ background: color, color: '#fff' }}
        >
          {catLabel}
        </span>
        <span className="text-xs font-semibold text-gray-200 ml-2 ml-node-label">
          {nodeData.label}
        </span>
      </div>

      {/* ── 포트 영역 (입력 왼쪽 / 출력 오른쪽) ─── */}
      {maxPorts > 0 && (
        <div
          className="flex justify-between"
          style={{ paddingTop: PORT_PAD_Y, paddingBottom: PORT_PAD_Y }}
        >
          {/* 입력 포트 이름 */}
          <div className="flex flex-col min-w-0">
            {inputs.map((port) => (
              <div
                key={port.name}
                className="text-[9px] text-gray-400 pl-4 pr-2 truncate ml-node-port"
                style={{ height: PORT_ROW_H, lineHeight: `${PORT_ROW_H}px` }}
              >
                {port.name}
              </div>
            ))}
            {inputs.length < maxPorts &&
              Array.from({ length: maxPorts - inputs.length }).map((_, i) => (
                <div key={`pad-in-${i}`} style={{ height: PORT_ROW_H }} />
              ))}
          </div>

          {/* 출력 포트 이름 */}
          <div className="flex flex-col min-w-0">
            {outputs.map((port) => (
              <div
                key={port.name}
                className="text-[9px] text-gray-400 pr-4 pl-2 text-right truncate ml-node-port"
                style={{ height: PORT_ROW_H, lineHeight: `${PORT_ROW_H}px` }}
              >
                {port.name}
              </div>
            ))}
            {outputs.length < maxPorts &&
              Array.from({ length: maxPorts - outputs.length }).map((_, i) => (
                <div key={`pad-out-${i}`} style={{ height: PORT_ROW_H }} />
              ))}
          </div>
        </div>
      )}

      {/* ── 파라미터 영역 (포트 아래, 구분선) ───── */}
      {params.length > 0 && (
        <div
          className="text-[10px] text-gray-500 mx-2 ml-node-param-area"
          style={{
            borderTop: '1px solid #333',
            paddingTop: PARAM_PAD_Y,
            paddingBottom: PARAM_PAD_Y,
          }}
        >
          {params.map(([key, val]) => (
            <div
              key={key}
              className="flex justify-between px-1"
              style={{ height: PARAM_ROW_H, lineHeight: `${PARAM_ROW_H}px` }}
            >
              <span className="text-gray-500 ml-node-param-key">{key}</span>
              <span className="text-gray-400 ml-3 ml-node-param-val">{String(val)}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── 입력 핸들 (왼쪽 가장자리) ────────── */}
      {inputs.map((port, idx) => (
        <Handle
          key={`in-${port.name}`}
          type="target"
          position={Position.Left}
          id={port.name}
          style={{
            top: portHandleTop(idx),
            background: color,
            width: 10,
            height: 10,
          }}
          title={port.name}
        />
      ))}

      {/* ── 출력 핸들 (오른쪽 가장자리) ───────── */}
      {outputs.map((port, idx) => (
        <Handle
          key={`out-${port.name}`}
          type="source"
          position={Position.Right}
          id={port.name}
          style={{
            top: portHandleTop(idx),
            background: color,
            width: 10,
            height: 10,
          }}
          title={port.name}
        />
      ))}

      {/* ── 호버 말풍선 (portal로 최상위 렌더) ── */}
      {showTooltip && summary && Object.keys(summary).length > 0 &&
        createPortal(
          <div
            className="fixed pointer-events-none"
            style={{
              left: tooltipPos.x,
              top: tooltipPos.y,
              transform: 'translate(-50%, -100%)',
              marginTop: -10,
              zIndex: 99999,
            }}
          >
            <div
              className="bg-gray-900/95 border border-gray-500 rounded-lg shadow-2xl px-3 py-2 text-[10px] backdrop-blur-sm"
              style={{ maxWidth: 420 }}
            >
              <div className="text-[11px] font-bold text-gray-200 mb-1.5 pb-1 border-b border-gray-700">
                {nodeData.label} Output
              </div>
              <div className="space-y-1.5">
                {Object.entries(summary).map(([port, info]) => (
                  <div key={port}>
                    <span className="font-semibold text-blue-400">{port}</span>
                    <div className="ml-2 mt-0.5">
                      <SummaryContent info={info} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            {/* 하단 화살표 */}
            <div className="flex justify-center">
              <div
                style={{
                  width: 0,
                  height: 0,
                  borderLeft: '7px solid transparent',
                  borderRight: '7px solid transparent',
                  borderTop: '7px solid #6b7280',
                }}
              />
            </div>
          </div>,
          document.body
        )
      }
    </div>
  );
}

export default memo(MLNode);
