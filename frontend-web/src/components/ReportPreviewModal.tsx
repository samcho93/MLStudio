import React, { useRef, useState } from 'react';
import { useStore } from '../store/useStore';
// @ts-ignore
import html2pdf from 'html2pdf.js';

interface Props {
  canvasImage: string | null;
  onClose: () => void;
}

/* ── 브라우저 인쇄용 HTML (폴백) ────────────────── */
function buildPrintHtml(
  nodes: any[],
  edges: any[],
  training: any,
  visualizations: Record<string, string>,
  nodeOutputSummaries: Record<string, any>,
  activeExample: any,
  canvasImage: string | null,
) {
  const hasTrained = training.epochs.length > 0;
  const lastEpoch = training.epochs[training.epochs.length - 1];

  const catOrder = ['data', 'layer', 'training', 'visualization', 'evaluation', 'output'];
  const catLabels: Record<string, string> = {
    data: 'Data', layer: 'Layer', training: 'Training',
    visualization: 'Visualization', evaluation: 'Evaluation', output: 'Output',
  };
  const grouped = nodes.reduce((acc: any, n: any) => {
    const cat = n.data?.category || 'etc';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(n);
    return acc;
  }, {} as Record<string, any[]>);

  let secNum = 1;

  // ── 표지 ──
  let html = `
    <div class="cover">
      <div class="cover-badge">ML Node Studio</div>
      <h1 class="cover-title">Pipeline Report</h1>
      ${activeExample ? `
        <div class="cover-example">${activeExample.title_ko || activeExample.title}</div>
        <div class="cover-desc">${activeExample.description_ko || activeExample.description || ''}</div>
      ` : ''}
      <div class="cover-meta">
        <div>${new Date().toLocaleDateString('ko-KR', { year: 'numeric', month: 'long', day: 'numeric' })}</div>
        <div style="margin-top:4px;">Nodes: ${nodes.length} &nbsp;|&nbsp; Connections: ${edges.length}${hasTrained && lastEpoch ? ` &nbsp;|&nbsp; Final Accuracy: ${(lastEpoch.accuracy * 100).toFixed(1)}%` : ''}</div>
      </div>
    </div>
  `;

  // ── 파이프라인 다이어그램 ──
  if (canvasImage) {
    html += `
      <div class="section-title">${secNum}. Pipeline Diagram</div>
      <div class="diagram-box" style="background:#fff; padding:8px;">
        <img src="${canvasImage}" style="width:100%; display:block;" />
      </div>
    `;
    secNum++;
  }

  // ── 파이프라인 구조 ──
  html += `<div class="section-title">${secNum}. Pipeline Structure</div>`;

  catOrder.forEach((cat) => {
    const items = grouped[cat];
    if (!items || items.length === 0) return;
    html += `<div class="cat-label">${catLabels[cat] || cat}</div>`;
    html += `<table class="struct-table"><tbody>`;
    items.forEach((n: any) => {
      const params = n.data?.params || {};
      const paramStr = Object.entries(params)
        .filter(([, v]) => v !== '' && v != null)
        .map(([k, v]) => `${k}=${v}`)
        .join(', ');
      html += `<tr>
        <td class="struct-id">${n.id}</td>
        <td class="struct-type">${n.data?.nodeType || ''}</td>
        <td class="struct-params">${paramStr}</td>
      </tr>`;
    });
    html += `</tbody></table>`;
  });

  // 데이터 흐름
  html += `<div class="cat-label" style="margin-top:12px;">Data Flow</div>`;
  html += `<div class="flow-list">`;
  edges.forEach((e: any) => {
    const src = e.source + (e.sourceHandle ? `[${e.sourceHandle}]` : '');
    const tgt = e.target + (e.targetHandle ? `[${e.targetHandle}]` : '');
    html += `<div class="flow-item">${src} <span class="flow-arrow">→</span> ${tgt}</div>`;
  });
  html += `</div>`;
  secNum++;

  // ── 노드 출력 요약 ──
  if (hasTrained && Object.keys(nodeOutputSummaries).length > 0) {
    html += `<div class="section-title">${secNum}. Node Output Summary</div>`;
    nodes.forEach((n: any) => {
      const summary = nodeOutputSummaries[n.id];
      if (!summary) return;
      html += `<div class="summary-node"><span class="summary-node-id">${n.id}</span> <span class="summary-node-type">(${n.data?.nodeType || ''})</span></div>`;
      Object.entries(summary).forEach(([port, info]: [string, any]) => {
        if (info.type === 'image_b64') return;
        let valStr = '';
        if (info.type === 'ndarray' || info.type === 'tensor') {
          valStr = `shape=[${info.shape?.join(', ')}] dtype=${info.dtype || ''}`;
          if (info.min !== undefined) valStr += ` min=${info.min.toFixed(3)} max=${info.max.toFixed(3)} mean=${info.mean.toFixed(3)}`;
        } else if (info.type === 'layers') {
          valStr = (info.layers as string[]).join(' → ');
        } else if (info.type === 'DataFrame') {
          valStr = `[${info.shape?.join('×')}] cols: ${info.columns?.slice(0, 5).join(', ')}`;
        } else if (info.type === 'config') {
          valStr = Object.entries(info.preview || {}).map(([k, v]) => `${k}=${v}`).join(', ');
        } else if (info.value !== undefined) {
          valStr = String(info.value);
        } else { valStr = info.type || ''; }
        html += `<div class="summary-port"><b>${port}:</b> ${valStr}</div>`;
      });
    });
    secNum++;
  }

  // ── 학습 결과 ──
  if (hasTrained && lastEpoch) {
    html += `<div class="section-title">${secNum}. Training Results</div>`;

    // 메트릭 카드
    const metrics = [
      ['Total Epochs', String(training.epochs.length)],
      ['Final Loss', lastEpoch.loss.toFixed(4)],
      ['Final Accuracy', `${(lastEpoch.accuracy * 100).toFixed(1)}%`],
      ['Val Loss', lastEpoch.val_loss.toFixed(4)],
      ['Val Accuracy', `${(lastEpoch.val_accuracy * 100).toFixed(1)}%`],
    ];
    if (training.modelPath) metrics.push(['Model Path', training.modelPath]);

    html += `<div class="metric-grid">`;
    metrics.forEach(([label, value]) => {
      html += `<div class="metric-card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div></div>`;
    });
    html += `</div>`;

    // 에포크 테이블
    html += `<table class="epoch-table"><thead><tr>
      <th>Epoch</th><th>Loss</th><th>Val Loss</th><th>Accuracy</th><th>Val Acc</th><th>LR</th>
    </tr></thead><tbody>`;
    training.epochs.forEach((ep: any) => {
      html += `<tr>
        <td>${ep.epoch}</td>
        <td>${ep.loss.toFixed(4)}</td>
        <td>${ep.val_loss.toFixed(4)}</td>
        <td>${(ep.accuracy * 100).toFixed(1)}%</td>
        <td>${(ep.val_accuracy * 100).toFixed(1)}%</td>
        <td>${ep.lr?.toFixed(6) || '-'}</td>
      </tr>`;
    });
    html += `</tbody></table>`;
    secNum++;
  }

  // ── 시각화 ──
  const vizEntries = Object.entries(visualizations);
  if (vizEntries.length > 0 && hasTrained) {
    html += `<div class="section-title">${secNum}. Visualizations</div>`;
    html += `<div class="viz-grid">`;
    vizEntries.forEach(([vizType, b64]) => {
      const label = vizType.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      html += `<div class="viz-item">
        <div class="viz-label">${label}</div>
        <img src="data:image/png;base64,${b64}" class="viz-img" />
      </div>`;
    });
    html += `</div>`;
  }

  // ── 전체 HTML 래핑 ──
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ML Node Studio Report</title>
<style>
  @page {
    size: A4;
    margin: 20mm 15mm 25mm 15mm;
    @bottom-center {
      content: counter(page) " / " counter(pages);
      font-size: 9px;
      color: #999;
    }
    @bottom-left {
      content: "ML Node Studio Report";
      font-size: 8px;
      color: #bbb;
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Segoe UI', sans-serif;
    color: #222;
    background: #fff;
    font-size: 10px;
    line-height: 1.5;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  /* 표지 */
  .cover {
    text-align: center;
    padding: 80px 20px 60px;
    page-break-after: always;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 70vh;
  }
  .cover-badge {
    display: inline-block;
    background: #1e40af;
    color: #fff;
    font-size: 12px;
    font-weight: 700;
    padding: 6px 20px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .cover-title {
    font-size: 32px;
    font-weight: 800;
    color: #111;
    margin-top: 20px;
    letter-spacing: -0.5px;
  }
  .cover-example {
    font-size: 18px;
    font-weight: 600;
    color: #1e40af;
    margin-top: 24px;
  }
  .cover-desc {
    font-size: 11px;
    color: #666;
    margin-top: 8px;
    max-width: 400px;
  }
  .cover-meta {
    margin-top: 40px;
    font-size: 10px;
    color: #999;
    border-top: 1px solid #ddd;
    padding-top: 16px;
  }

  /* 섹션 제목 */
  .section-title {
    font-size: 15px;
    font-weight: 700;
    color: #1e40af;
    border-bottom: 2px solid #1e40af;
    padding-bottom: 4px;
    margin-top: 28px;
    margin-bottom: 12px;
    page-break-after: avoid;
  }

  /* 카테고리 라벨 */
  .cat-label {
    font-size: 10px;
    font-weight: 700;
    color: #555;
    text-transform: uppercase;
    margin-top: 8px;
    margin-bottom: 4px;
    letter-spacing: 0.5px;
  }

  /* 구조 테이블 */
  .struct-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 6px;
    font-size: 9px;
  }
  .struct-table td {
    padding: 3px 6px;
    border-bottom: 1px solid #f0f0f0;
  }
  .struct-id {
    font-weight: 600;
    color: #1e40af;
    width: 30%;
    font-family: 'Consolas', 'Courier New', monospace;
  }
  .struct-type {
    color: #888;
    width: 25%;
  }
  .struct-params {
    color: #aaa;
    font-size: 8px;
  }

  /* 데이터 흐름 */
  .flow-list {
    font-size: 8px;
    font-family: 'Consolas', 'Courier New', monospace;
    color: #777;
    padding-left: 8px;
  }
  .flow-item { line-height: 1.8; }
  .flow-arrow { color: #1e40af; font-weight: 700; }

  /* 노드 출력 요약 */
  .summary-node {
    margin-top: 8px;
    margin-bottom: 2px;
  }
  .summary-node-id {
    font-weight: 700;
    color: #1e40af;
    font-size: 10px;
  }
  .summary-node-type {
    color: #999;
    font-size: 9px;
  }
  .summary-port {
    padding-left: 16px;
    font-size: 8px;
    color: #555;
    line-height: 1.7;
    font-family: 'Consolas', 'Courier New', monospace;
  }

  /* 메트릭 카드 */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 16px;
  }
  .metric-card {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 8px 10px;
    background: #fafbfc;
  }
  .metric-label {
    font-size: 8px;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .metric-value {
    font-size: 14px;
    font-weight: 700;
    color: #111;
    margin-top: 2px;
  }

  /* 에포크 테이블 */
  .epoch-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8px;
    margin-top: 8px;
  }
  .epoch-table th {
    background: #f5f7fa;
    border-bottom: 2px solid #ccc;
    padding: 5px 6px;
    text-align: right;
    font-weight: 700;
    color: #555;
    font-size: 8px;
  }
  .epoch-table th:first-child { text-align: center; }
  .epoch-table td {
    padding: 3px 6px;
    text-align: right;
    border-bottom: 1px solid #f0f0f0;
    font-family: 'Consolas', 'Courier New', monospace;
    color: #444;
  }
  .epoch-table td:first-child {
    text-align: center;
    font-weight: 600;
    color: #1e40af;
  }
  .epoch-table tr:nth-child(even) { background: #fafbfc; }

  /* 시각화 (세로 배치) */
  .viz-grid {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 8px;
  }
  .viz-item { page-break-inside: avoid; }
  .viz-label {
    font-size: 11px;
    font-weight: 700;
    color: #555;
    margin-bottom: 6px;
    text-transform: capitalize;
  }
  .viz-img {
    width: 100%;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
  }

  /* 다이어그램 */
  .diagram-box {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    overflow: hidden;
    page-break-inside: avoid;
  }

  @media print {
    body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  }
</style>
</head>
<body>
${html}
</body>
</html>`;
}

/* ── SVG 파이프라인 다이어그램 ─────────────────────── */
const SVG_CAT_COLORS: Record<string, string> = {
  data: '#3b82f6', layer: '#8b5cf6', training: '#f59e0b',
  visualization: '#10b981', evaluation: '#10b981', output: '#ef4444',
};
const SVG_CAT_LABELS: Record<string, string> = {
  data: 'DATA', layer: 'LAYER', training: 'TRAIN',
  visualization: 'VIZ', evaluation: 'EVAL', output: 'OUT',
};

function PipelineSvgDiagram({ nodes, edges }: { nodes: any[]; edges: any[] }) {
  if (nodes.length === 0) return null;

  const NODE_W = 150;
  const HEADER_H = 22;
  const PORT_H = 14;
  const PARAM_H = 13;
  const PAD = 6;

  // 노드 크기 계산
  const nodeBoxes = nodes.map((n) => {
    const inputs = n.data?.inputs || [];
    const outputs = n.data?.outputs || [];
    const params = Object.entries(n.data?.params || {});
    const maxPorts = Math.max(inputs.length, outputs.length);
    const portSectionH = maxPorts > 0 ? maxPorts * PORT_H + PAD * 2 : 0;
    const paramSectionH = params.length > 0 ? params.length * PARAM_H + PAD * 2 : 0;
    const h = HEADER_H + portSectionH + paramSectionH;
    return {
      ...n,
      x: n.position.x,
      y: n.position.y,
      w: NODE_W,
      h,
      inputs, outputs, params,
      maxPorts,
      portSectionH,
      paramSectionH,
      color: SVG_CAT_COLORS[n.data?.category] || '#6b7280',
      catLabel: SVG_CAT_LABELS[n.data?.category] || 'MISC',
    };
  });

  // 바운딩 박스
  const minX = Math.min(...nodeBoxes.map((n) => n.x));
  const minY = Math.min(...nodeBoxes.map((n) => n.y));
  const maxX = Math.max(...nodeBoxes.map((n) => n.x + n.w));
  const maxY = Math.max(...nodeBoxes.map((n) => n.y + n.h));
  const margin = 20;
  const vbW = maxX - minX + margin * 2;
  const vbH = maxY - minY + margin * 2;

  // 포트 위치 계산
  const portPos = (nodeId: string, handleName: string, side: 'input' | 'output') => {
    const nb = nodeBoxes.find((n) => n.id === nodeId);
    if (!nb) return { x: 0, y: 0 };
    const list = side === 'input' ? nb.inputs : nb.outputs;
    const idx = list.findIndex((p: any) => p.name === handleName);
    const py = nb.y + HEADER_H + PAD + (idx >= 0 ? idx : 0) * PORT_H + PORT_H / 2;
    const px = side === 'input' ? nb.x : nb.x + nb.w;
    return { x: px, y: py };
  };

  return (
    <div className="border border-gray-200 rounded-md overflow-hidden bg-white">
      <svg
        viewBox={`${minX - margin} ${minY - margin} ${vbW} ${vbH}`}
        style={{ width: '100%', height: 'auto' }}
        xmlns="http://www.w3.org/2000/svg"
        fontFamily="'Inter', 'Segoe UI', sans-serif"
      >
        {/* 엣지 */}
        {edges.map((e: any, i: number) => {
          const src = portPos(e.source, e.sourceHandle, 'output');
          const tgt = portPos(e.target, e.targetHandle, 'input');
          const dx = (tgt.x - src.x) * 0.4;
          return (
            <path
              key={i}
              d={`M${src.x},${src.y} C${src.x + dx},${src.y} ${tgt.x - dx},${tgt.y} ${tgt.x},${tgt.y}`}
              fill="none"
              stroke="#9ca3af"
              strokeWidth={1.2}
            />
          );
        })}

        {/* 노드 */}
        {nodeBoxes.map((nb) => (
          <g key={nb.id}>
            {/* 노드 배경 */}
            <rect
              x={nb.x} y={nb.y} width={nb.w} height={nb.h}
              rx={5} ry={5}
              fill="#fff" stroke={nb.color} strokeWidth={1.5}
            />
            {/* 헤더 배경 */}
            <rect
              x={nb.x} y={nb.y} width={nb.w} height={HEADER_H}
              rx={5} ry={5}
              fill={nb.color + '18'}
            />
            <rect
              x={nb.x} y={nb.y + HEADER_H - 5} width={nb.w} height={5}
              fill={nb.color + '18'}
            />
            {/* 카테고리 뱃지 */}
            <rect
              x={nb.x + 6} y={nb.y + 4} width={34} height={14}
              rx={3} fill={nb.color}
            />
            <text
              x={nb.x + 23} y={nb.y + 14}
              textAnchor="middle" fontSize={7} fontWeight={700} fill="#fff"
            >{nb.catLabel}</text>
            {/* 노드 이름 */}
            <text
              x={nb.x + nb.w - 6} y={nb.y + 15}
              textAnchor="end" fontSize={9} fontWeight={600} fill="#111"
            >{nb.data?.label}</text>

            {/* 포트 영역 */}
            {nb.inputs.map((p: any, pi: number) => {
              const py = nb.y + HEADER_H + PAD + pi * PORT_H + PORT_H / 2;
              return (
                <g key={`in-${p.name}`}>
                  <circle cx={nb.x} cy={py} r={3} fill={nb.color} stroke="#fff" strokeWidth={0.5} />
                  <text x={nb.x + 8} y={py + 3} fontSize={7} fill="#374151">{p.name}</text>
                </g>
              );
            })}
            {nb.outputs.map((p: any, pi: number) => {
              const py = nb.y + HEADER_H + PAD + pi * PORT_H + PORT_H / 2;
              return (
                <g key={`out-${p.name}`}>
                  <circle cx={nb.x + nb.w} cy={py} r={3} fill={nb.color} stroke="#fff" strokeWidth={0.5} />
                  <text x={nb.x + nb.w - 8} y={py + 3} fontSize={7} fill="#374151" textAnchor="end">{p.name}</text>
                </g>
              );
            })}

            {/* 파라미터 영역 */}
            {nb.params.length > 0 && (
              <>
                <line
                  x1={nb.x + 6} y1={nb.y + HEADER_H + nb.portSectionH}
                  x2={nb.x + nb.w - 6} y2={nb.y + HEADER_H + nb.portSectionH}
                  stroke="#e5e7eb" strokeWidth={0.5}
                />
                {nb.params.map((entry: any, pi: number) => {
                  const py = nb.y + HEADER_H + nb.portSectionH + PAD + pi * PARAM_H + PARAM_H / 2 + 3;
                  return (
                    <g key={entry[0]}>
                      <text x={nb.x + 8} y={py} fontSize={6.5} fill="#6b7280">{entry[0]}</text>
                      <text x={nb.x + nb.w - 8} y={py} fontSize={6.5} fill="#111" textAnchor="end">{String(entry[1])}</text>
                    </g>
                  );
                })}
              </>
            )}
          </g>
        ))}

        {/* 포트 연결점 위 엣지 끝점 */}
      </svg>
    </div>
  );
}

/* ── 미리보기 컴포넌트 ──────────────────────────── */
export function ReportPreviewModal({ canvasImage, onClose }: Props) {
  const nodes = useStore((s) => s.nodes);
  const edges = useStore((s) => s.edges);
  const training = useStore((s) => s.training);
  const visualizations = useStore((s) => s.visualizations);
  const activeExample = useStore((s) => s.activeExample);
  const nodeOutputSummaries = useStore((s) => s.nodeOutputSummaries);
  const contentRef = useRef<HTMLDivElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [saving, setSaving] = useState(false);

  const hasTrained = training.epochs.length > 0;
  const lastEpoch = training.epochs[training.epochs.length - 1];

  const catOrder = ['data', 'layer', 'training', 'visualization', 'evaluation', 'output'];
  const catLabels: Record<string, string> = {
    data: 'Data', layer: 'Layer', training: 'Training',
    visualization: 'Visualization', evaluation: 'Evaluation', output: 'Output',
  };
  const grouped = nodes.reduce((acc, n) => {
    const cat = (n.data.category as string) || 'etc';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(n);
    return acc;
  }, {} as Record<string, typeof nodes>);

  const handleSavePdf = async () => {
    if (!contentRef.current || saving) return;
    setSaving(true);
    try {
      const fileName = activeExample
        ? `report_${activeExample.title.replace(/\s+/g, '_')}.pdf`
        : `report_${new Date().toISOString().slice(0, 10)}.pdf`;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const opts: any = {
        margin: [10, 10, 15, 10],
        filename: fileName,
        image: { type: 'jpeg', quality: 0.95 },
        html2canvas: { scale: 2, useCORS: true, logging: false },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
        pagebreak: { mode: ['avoid-all', 'css', 'legacy'] },
      };
      await html2pdf().set(opts).from(contentRef.current).save();
    } catch (err) {
      console.error('PDF save failed:', err);
      // 폴백: 브라우저 인쇄
      const printHtml = buildPrintHtml(
        nodes, edges, training, visualizations,
        nodeOutputSummaries, activeExample, canvasImage,
      );
      const iframe = iframeRef.current;
      if (iframe) {
        const doc = iframe.contentDocument || iframe.contentWindow?.document;
        if (doc) {
          doc.open();
          doc.write(printHtml);
          doc.close();
          setTimeout(() => iframe.contentWindow?.print(), 500);
        }
      }
    } finally {
      setSaving(false);
    }
  };

  let secNum = 1;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/70 backdrop-blur-sm overflow-y-auto py-6"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      {/* 인쇄용 숨겨진 iframe */}
      <iframe ref={iframeRef} className="hidden" style={{ position: 'absolute', width: 0, height: 0, border: 'none' }} />

      <div className="bg-white rounded-xl shadow-2xl w-[720px] flex flex-col overflow-hidden">
        {/* 헤더 */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-200 bg-gray-50 sticky top-0 z-10">
          <h2 className="text-sm font-bold text-gray-800">Report Preview</h2>
          <div className="flex gap-2">
            <button
              onClick={handleSavePdf}
              disabled={saving}
              className="px-4 py-1.5 text-xs rounded bg-blue-600 hover:bg-blue-500 text-white font-medium disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save PDF'}
            </button>
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-xs rounded bg-gray-200 hover:bg-gray-300 text-gray-700"
            >
              Close
            </button>
          </div>
        </div>

        {/* 보고서 미리보기 (흰 배경, A4 비율) */}
        <div className="p-8 space-y-6 bg-gray-100 max-h-[80vh] overflow-y-auto">
          <div ref={contentRef} className="bg-white shadow-lg mx-auto" style={{ maxWidth: 640, padding: '40px 36px' }}>

            {/* ── 표지 ── */}
            <div className="text-center pb-8 mb-8 border-b-2 border-blue-700">
              <span className="inline-block bg-blue-700 text-white text-[10px] font-bold px-4 py-1 rounded-full tracking-wider uppercase">
                ML Node Studio
              </span>
              <h1 className="text-2xl font-extrabold text-gray-900 mt-4 tracking-tight">Pipeline Report</h1>
              {activeExample && (
                <div className="mt-5">
                  <span className="text-lg font-semibold text-blue-700">{activeExample.title_ko || activeExample.title}</span>
                  <p className="text-xs text-gray-500 mt-1 max-w-md mx-auto">{activeExample.description_ko || activeExample.description}</p>
                </div>
              )}
              <div className="mt-6 text-[10px] text-gray-400 border-t border-gray-200 pt-3">
                {new Date().toLocaleDateString('ko-KR', { year: 'numeric', month: 'long', day: 'numeric' })}
                <span className="mx-2">|</span>
                Nodes: {nodes.length} / Connections: {edges.length}
                {hasTrained && lastEpoch && (
                  <>
                    <span className="mx-2">|</span>
                    Accuracy: {(lastEpoch.accuracy * 100).toFixed(1)}%
                  </>
                )}
              </div>
            </div>

            {/* ── 파이프라인 다이어그램 (SVG 벡터) ── */}
            {nodes.length > 0 && (
              <section className="mb-8">
                <h2 className="text-sm font-bold text-blue-700 border-b-2 border-blue-700 pb-1 mb-3">{secNum++}. Pipeline Diagram</h2>
                <PipelineSvgDiagram nodes={nodes} edges={edges} />
              </section>
            )}

            {/* ── 파이프라인 구조 ── */}
            <section className="mb-8">
              <h2 className="text-sm font-bold text-blue-700 border-b-2 border-blue-700 pb-1 mb-3">{secNum++}. Pipeline Structure</h2>
              {catOrder.map((cat) => {
                const items = grouped[cat];
                if (!items || items.length === 0) return null;
                return (
                  <div key={cat} className="mb-3">
                    <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wide mb-1">{catLabels[cat]}</h3>
                    <table className="w-full text-[10px]">
                      <tbody>
                        {items.map((n) => {
                          const params = n.data.params as Record<string, any> || {};
                          const paramStr = Object.entries(params)
                            .filter(([, v]) => v !== '' && v != null)
                            .map(([k, v]) => `${k}=${v}`)
                            .join(', ');
                          return (
                            <tr key={n.id} className="border-b border-gray-100">
                              <td className="py-1 pr-2 font-mono font-semibold text-blue-700 w-[30%]">{n.id}</td>
                              <td className="py-1 pr-2 text-gray-400 w-[25%]">{n.data.nodeType as string}</td>
                              <td className="py-1 text-gray-400 text-[9px]">{paramStr}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                );
              })}
              <div className="mt-3">
                <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wide mb-1">Data Flow</h3>
                <div className="ml-1 text-[9px] font-mono text-gray-500 space-y-0.5">
                  {edges.map((e, i) => (
                    <div key={i}>
                      {e.source}[{(e as any).sourceHandle || 'out'}]
                      <span className="text-blue-700 font-bold mx-1">→</span>
                      {e.target}[{(e as any).targetHandle || 'in'}]
                    </div>
                  ))}
                </div>
              </div>
            </section>

            {/* ── 노드 출력 요약 ── */}
            {hasTrained && Object.keys(nodeOutputSummaries).length > 0 && (
              <section className="mb-8">
                <h2 className="text-sm font-bold text-blue-700 border-b-2 border-blue-700 pb-1 mb-3">{secNum++}. Node Output Summary</h2>
                <div className="space-y-2">
                  {nodes.map((n) => {
                    const summary = nodeOutputSummaries[n.id];
                    if (!summary) return null;
                    return (
                      <div key={n.id}>
                        <div className="text-[10px]">
                          <span className="font-bold text-blue-700">{n.id}</span>
                          <span className="text-gray-400 ml-1">({n.data.nodeType as string})</span>
                        </div>
                        <div className="ml-4 text-[9px] font-mono text-gray-500">
                          {Object.entries(summary).map(([port, info]: [string, any]) => {
                            if (info.type === 'image_b64') return null;
                            let valStr = '';
                            if (info.type === 'ndarray' || info.type === 'tensor') {
                              valStr = `shape=[${info.shape?.join(', ')}] dtype=${info.dtype}`;
                              if (info.min !== undefined) valStr += ` min=${info.min.toFixed(3)} max=${info.max.toFixed(3)}`;
                            } else if (info.type === 'layers') {
                              valStr = (info.layers as string[]).join(' → ');
                            } else if (info.type === 'DataFrame') {
                              valStr = `[${info.shape?.join('×')}] cols: ${info.columns?.slice(0, 5).join(', ')}`;
                            } else if (info.type === 'config') {
                              valStr = Object.entries(info.preview || {}).map(([k, v]) => `${k}=${v}`).join(', ');
                            } else if (info.value !== undefined) {
                              valStr = String(info.value);
                            } else { valStr = info.type; }
                            return <div key={port}><b>{port}:</b> {valStr}</div>;
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}

            {/* ── 학습 결과 ── */}
            {hasTrained && lastEpoch && (
              <section className="mb-8">
                <h2 className="text-sm font-bold text-blue-700 border-b-2 border-blue-700 pb-1 mb-3">{secNum++}. Training Results</h2>
                {/* 메트릭 카드 */}
                <div className="grid grid-cols-3 gap-2 mb-4">
                  {[
                    ['Total Epochs', String(training.epochs.length)],
                    ['Final Loss', lastEpoch.loss.toFixed(4)],
                    ['Accuracy', `${(lastEpoch.accuracy * 100).toFixed(1)}%`],
                    ['Val Loss', lastEpoch.val_loss.toFixed(4)],
                    ['Val Accuracy', `${(lastEpoch.val_accuracy * 100).toFixed(1)}%`],
                    ['Model', training.modelPath || '-'],
                  ].map(([label, val]) => (
                    <div key={label} className="border border-gray-200 rounded-md p-2 bg-gray-50/50">
                      <div className="text-[8px] text-gray-400 uppercase tracking-wide">{label}</div>
                      <div className="text-sm font-bold text-gray-800 mt-0.5">{val}</div>
                    </div>
                  ))}
                </div>

                {/* 에포크 테이블 */}
                <table className="w-full text-[9px] border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b-2 border-gray-300">
                      <th className="py-1.5 px-2 text-center font-bold text-gray-500">Epoch</th>
                      <th className="py-1.5 px-2 text-right font-bold text-gray-500">Loss</th>
                      <th className="py-1.5 px-2 text-right font-bold text-gray-500">Val Loss</th>
                      <th className="py-1.5 px-2 text-right font-bold text-gray-500">Accuracy</th>
                      <th className="py-1.5 px-2 text-right font-bold text-gray-500">Val Acc</th>
                      <th className="py-1.5 px-2 text-right font-bold text-gray-500">LR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {training.epochs.map((ep: any, i: number) => (
                      <tr key={i} className={`border-b border-gray-100 ${i % 2 === 0 ? '' : 'bg-gray-50/50'}`}>
                        <td className="py-1 px-2 text-center font-semibold text-blue-700">{ep.epoch}</td>
                        <td className="py-1 px-2 text-right font-mono text-gray-600">{ep.loss.toFixed(4)}</td>
                        <td className="py-1 px-2 text-right font-mono text-gray-600">{ep.val_loss.toFixed(4)}</td>
                        <td className="py-1 px-2 text-right font-mono text-gray-600">{(ep.accuracy * 100).toFixed(1)}%</td>
                        <td className="py-1 px-2 text-right font-mono text-gray-600">{(ep.val_accuracy * 100).toFixed(1)}%</td>
                        <td className="py-1 px-2 text-right font-mono text-gray-400">{ep.lr?.toFixed(6) || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}

            {/* ── 시각화 (2단 세로 배치) ── */}
            {hasTrained && Object.keys(visualizations).length > 0 && (
              <section className="mb-8">
                <h2 className="text-sm font-bold text-blue-700 border-b-2 border-blue-700 pb-1 mb-3">{secNum++}. Visualizations</h2>
                <div className="flex flex-col gap-4">
                  {Object.entries(visualizations).map(([vizType, b64]) => (
                    <div key={vizType} style={{ pageBreakInside: 'avoid' }}>
                      <div className="text-xs font-bold text-gray-600 mb-2 capitalize">{vizType.replace(/_/g, ' ')}</div>
                      <img src={`data:image/png;base64,${b64}`} alt={vizType} className="w-full border border-gray-200 rounded" />
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* 푸터 */}
            <div className="text-center text-[9px] text-gray-400 pt-4 border-t border-gray-200">
              Generated by ML Node Studio
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
