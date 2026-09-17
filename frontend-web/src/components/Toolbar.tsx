import React, { useEffect, useState, useRef } from 'react';
import { useReactFlow } from '@xyflow/react';
import { useStore } from '../store/useStore';
import { HelpModal } from './HelpModal';
import { ReportPreviewModal } from './ReportPreviewModal';
import { ModelRecommendModal } from './ModelRecommendModal';
import { toPng } from 'html-to-image';
import { apiFetch, hasBackend, wsUrl } from '../api';
import { BackendSettings } from './BackendSettings';

export function Toolbar() {
  const { fitView } = useReactFlow();
  const { nodes, edges, training, ws, setWs, setTraining, resetTraining, fetchExamples } = useStore();
  const setModelManagerOpen = useStore((s) => s.setModelManagerOpen);
  const setTestResults = useStore((s) => s.setTestResults);
  const addLog = useStore((s) => s.addLog);
  const clearLogs = useStore((s) => s.clearLogs);
  const logPanelOpen = useStore((s) => s.logPanelOpen);
  const setLogPanelOpen = useStore((s) => s.setLogPanelOpen);
  const logs = useStore((s) => s.logs);

  // 편집 기능
  const newCanvas = useStore((s) => s.newCanvas);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const undoStack = useStore((s) => s.undoStack);
  const redoStack = useStore((s) => s.redoStack);
  const deleteSelected = useStore((s) => s.deleteSelected);
  const copySelected = useStore((s) => s.copySelected);
  const cutSelected = useStore((s) => s.cutSelected);
  const pasteClipboard = useStore((s) => s.pasteClipboard);
  const selectAll = useStore((s) => s.selectAll);
  const clipboard = useStore((s) => s.clipboard);

  const [editMenuOpen, setEditMenuOpen] = useState(false);
  const editMenuRef = useRef<HTMLDivElement>(null);

  // 편집 메뉴 외부 클릭 시 닫기
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (editMenuRef.current && !editMenuRef.current.contains(e.target as HTMLElement)) {
        setEditMenuOpen(false);
      }
    };
    if (editMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [editMenuOpen]);

  const hasSelection = nodes.some((n) => n.selected) || edges.some((e) => e.selected);

  // 예제 목록 로드
  useEffect(() => {
    fetchExamples();
  }, [fetchExamples]);

  const handleRun = () => {
    if (!hasBackend()) {
      alert('학습을 실행하려면 학습 서버(백엔드)가 필요합니다.\n툴바 오른쪽의 [데모 모드] 버튼에서 서버 주소를 입력하세요.');
      return;
    }
    const pipelineNodes = nodes.map((n) => ({
      id: n.id,
      type: n.data.nodeType,
      params: n.data.params || {},
    }));

    const pipelineEdges = edges.map((e) => ({
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle || 'output',
      targetHandle: e.targetHandle || 'input',
    }));

    resetTraining();
    clearLogs();
    setTraining({ isTraining: true });
    setLogPanelOpen(true);
    addLog('info', `\u{1F680} Pipeline started \u2014 ${pipelineNodes.length} nodes, ${pipelineEdges.length} edges`);

    const socket = new WebSocket(wsUrl('/ws/train'));

    socket.onopen = () => {
      addLog('info', 'Connected to backend');
      socket.send(
        JSON.stringify({
          type: 'START_TRAINING',
          pipeline: { nodes: pipelineNodes, edges: pipelineEdges },
        })
      );
    };

    let currentNodeId: string | null = null;

    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const store = useStore.getState();

      switch (msg.type) {
        case 'PIPELINE_START':
          store.clearNodeOutputSummaries();
          currentNodeId = null;
          store.addLog('info', `\u{1F680} Pipeline started \u2014 execution order: ${(msg.execution_order || []).join(' \u2192 ')}`);
          break;

        case 'NODE_START':
          currentNodeId = msg.node_id;
          store.addLog('info', `\u25b6 [${msg.node_type}] ${msg.node_id} starting...`);
          break;

        case 'NODE_COMPLETE':
          store.addLog('success', `\u2713 [${msg.node_type || ''}] ${msg.node_id} complete`);
          if (msg.output_summary) {
            // 이미 VISUALIZATION으로 저장된 이미지 데이터 보존
            const existing = store.nodeOutputSummaries[msg.node_id] || {};
            const merged = { ...msg.output_summary };
            for (const [key, val] of Object.entries(existing)) {
              if ((val as any)?.image_b64 && !(merged[key] as any)?.image_b64) {
                merged[key] = val;
              }
            }
            store.setNodeOutputSummary(msg.node_id, merged);
          }
          break;

        case 'EPOCH_UPDATE':
          store.addEpochData(msg);
          if (msg.epoch === 1 || msg.epoch === msg.total_epochs || msg.epoch % 5 === 0) {
            store.addLog(
              'data',
              `\u{1F4CA} Epoch ${msg.epoch}/${msg.total_epochs} \u2014 loss: ${msg.loss.toFixed(4)}, acc: ${(msg.accuracy * 100).toFixed(1)}%, val_loss: ${msg.val_loss.toFixed(4)}, val_acc: ${(msg.val_accuracy * 100).toFixed(1)}%`
            );
          }
          break;

        case 'TRAINING_COMPLETE': {
          store.setTraining({ isTraining: false, modelPath: msg.model_path });
          const metrics = msg.metrics || {};
          const metricStr = Object.entries(metrics)
            .map(([k, v]) => `${k}=${typeof v === 'number' ? (v as number).toFixed(4) : v}`)
            .join('  ');
          store.addLog('success', `\u{1F389} Training complete! ${metricStr}`);
          store.addLog('success', `\u{1F389} Model saved: ${msg.model_path}`);
          break;
        }

        case 'PIPELINE_COMPLETE':
          store.addLog('success', `\u2713 Pipeline finished successfully`);
          break;

        case 'VISUALIZATION':
          store.setVisualization(msg.viz_type, msg.image_b64);
          store.addLog('info', `\u{1F4C8} Visualization generated: ${msg.viz_type}`);
          // 현재 실행 중인 노드에 이미지 연결
          if (currentNodeId && msg.image_b64) {
            const prev = store.nodeOutputSummaries[currentNodeId] || {};
            store.setNodeOutputSummary(currentNodeId, {
              ...prev,
              image_b64: { type: 'image_b64', image_b64: msg.image_b64 },
            });
          }
          break;

        case 'MODEL_READY':
          store.addLog('success', `🧪 Model ready for testing! ${msg.is_regression ? '(Regression)' : `(${(msg.class_names || []).length} classes)`}`);
          break;

        case 'DEVICE_INFO':
          store.addLog('info', `\u{1F5A5} Device: ${msg.gpu_name || msg.device}`);
          break;

        case 'MODEL_SUMMARY':
          store.addLog('data', `Model Summary:\n${msg.summary}`);
          break;

        case 'METRICS':
          store.addLog('success', `\u{1F4CA} Metrics: ${JSON.stringify(msg.metrics)}`);
          break;

        case 'DATA_INSPECT': {
          const info = msg.info || {};
          store.addLog('data', `Data: shape=${JSON.stringify(info.shape)} dtype=${info.dtype}`);
          break;
        }

        case 'TEST_RESULTS': {
          store.setTestResults({
            metrics: msg.metrics || {},
            samples: msg.samples || [],
            num_predictions: msg.num_predictions || 0,
            prediction_summary: msg.prediction_summary,
          });
          const metricsStr = msg.metrics?.type === 'regression'
            ? `MSE=${msg.metrics.mse?.toFixed(4)} R2=${msg.metrics.r2?.toFixed(4)}`
            : `Accuracy=${(msg.metrics?.accuracy * 100)?.toFixed(2)}%`;
          store.addLog('success', `Test complete: ${metricsStr} (${msg.num_predictions} samples)`);
          break;
        }

        case 'MODEL_LOADED':
          store.addLog('success', `Model "${msg.name}" loaded (${msg.framework})`);
          break;

        case 'ERROR':
          store.setTraining({ isTraining: false, error: msg.message });
          store.addLog('error', `\u274c Error: ${msg.message}`);
          if (msg.traceback) {
            const tbLines = msg.traceback.trim().split('\n');
            const last = tbLines.slice(-3).join('\n');
            store.addLog('error', last);
          }
          break;

        default:
          store.addLog('info', `Unknown message: ${msg.type}`);
      }
    };

    socket.onerror = () => {
      setTraining({ isTraining: false, error: 'WebSocket connection failed' });
      addLog('error', 'WebSocket connection failed — is the backend running?');
    };

    socket.onclose = () => {
      setTraining({ isTraining: false });
    };

    setWs(socket);
  };

  const handleStop = () => {
    if (ws) {
      ws.send(JSON.stringify({ type: 'STOP_TRAINING' }));
      ws.close();
      setWs(null);
    }
    setTraining({ isTraining: false });
    addLog('info', 'Training stopped by user');
  };

  const handleSave = async () => {
    const pipelineNodes = nodes.map((n) => ({
      id: n.id,
      type: n.data.nodeType,
      params: n.data.params || {},
      position: n.position,
    }));
    const pipelineEdges = edges.map((e) => ({
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
    }));
    const name = prompt('Pipeline name:', 'my-pipeline');
    if (!name) return;
    try {
      const res = await apiFetch('/api/pipeline/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, nodes: pipelineNodes, edges: pipelineEdges }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch {
      alert('파이프라인 저장에 실패했습니다. 학습 서버(백엔드) 연결을 확인하세요.');
    }
  };

  const handleLoad = async () => {
    let list: any[];
    try {
      const res = await apiFetch('/api/pipeline/list');
      list = await res.json();
    } catch {
      alert('파이프라인 목록을 불러오지 못했습니다. 학습 서버(백엔드) 연결을 확인하세요.');
      return;
    }
    if (list.length === 0) { alert('No saved pipelines'); return; }
    const name = prompt(`Pipelines:\n${list.map((p: any) => p.name).join('\n')}\n\nEnter name to load:`);
    if (!name) return;
    const pRes = await apiFetch(`/api/pipeline/load/${name}`);
    if (!pRes.ok) { alert('Pipeline not found'); return; }
    const pipeline = await pRes.json();
    const { catalog } = useStore.getState();
    const loadedNodes = (pipeline.nodes || []).map((n: any) => {
      const catalogItem = catalog.find((c: any) => c.type === n.type);
      return {
        id: n.id,
        type: 'mlNode',
        position: n.position || { x: Math.random() * 400, y: Math.random() * 400 },
        data: {
          label: n.type,
          category: catalogItem?.category || 'misc',
          params: n.params || {},
          inputs: catalogItem?.inputs || [{ name: 'input', type: 'any' }],
          outputs: catalogItem?.outputs || [{ name: 'output', type: 'any' }],
          nodeType: n.type,
        },
      };
    });
    useStore.getState().setNodes(loadedNodes);
    useStore.getState().setEdges(
      (pipeline.edges || []).map((e: any, i: number) => ({ id: `e-${i}`, ...e }))
    );
    setTimeout(() => fitView({ padding: 0.1, duration: 300 }), 100);
  };

  const [showHelp, setShowHelp] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [showRecommend, setShowRecommend] = useState(false);
  const [canvasImage, setCanvasImage] = useState<string | null>(null);
  const hasError = logs.some((l) => l.level === 'error');

  const handleReport = () => {
    const rfEl = document.querySelector('.react-flow') as HTMLElement;
    if (rfEl && rfEl.clientWidth > 200 && rfEl.clientHeight > 200) {
      // 캡처 전: 컨트롤 숨기기 + 라이트 테마 적용
      const controls = rfEl.querySelector('.react-flow__controls') as HTMLElement;
      if (controls) controls.style.display = 'none';
      rfEl.classList.add('report-capture');

      // 배경 도트 숨기기
      const bgPattern = rfEl.querySelector('.react-flow__background') as HTMLElement;
      if (bgPattern) bgPattern.style.display = 'none';

      toPng(rfEl, {
        backgroundColor: '#ffffff',
        pixelRatio: 2,
      })
        .then((img) => { setCanvasImage(img); setShowReport(true); })
        .catch((err) => { console.warn('Canvas capture failed:', err); setCanvasImage(null); setShowReport(true); })
        .finally(() => {
          rfEl.classList.remove('report-capture');
          if (controls) controls.style.display = '';
          if (bgPattern) bgPattern.style.display = '';
        });
    } else {
      setCanvasImage(null);
      setShowReport(true);
    }
  };

  return (
    <>
    <div className="h-10 bg-gray-800 border-b border-gray-700 flex items-center px-4 gap-2 [&_button]:leading-none">
      <span className="text-sm font-bold text-white mr-4">ML Node Studio</span>

      {/* ── File 메뉴 ─────────────────────────── */}
      <button
        onClick={() => {
          if (nodes.length > 0 && !confirm('현재 파이프라인을 지우고 새로 만드시겠습니까?')) return;
          newCanvas();
        }}
        className="px-3 py-1 text-xs rounded bg-gray-600 hover:bg-gray-500 text-white"
        title="새로 만들기 (New)"
      >
        New
      </button>
      <button onClick={handleSave} className="px-3 py-1 text-xs rounded bg-gray-600 hover:bg-gray-500 text-white">
        Save
      </button>
      <button onClick={handleLoad} className="px-3 py-1 text-xs rounded bg-gray-600 hover:bg-gray-500 text-white">
        Load
      </button>

      <div className="w-px h-5 bg-gray-600 mx-1" />

      {/* ── Edit 메뉴 (드롭다운) ──────────────── */}
      <div className="relative flex items-center" ref={editMenuRef}>
        <button
          onClick={() => setEditMenuOpen(!editMenuOpen)}
          className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
            editMenuOpen ? 'bg-blue-600 text-white' : 'bg-gray-600 hover:bg-gray-500 text-white'
          }`}
        >
          Edit ▾
        </button>
        {editMenuOpen && (
          <div className="absolute top-full left-0 mt-1 w-56 bg-gray-700 border border-gray-600 rounded-lg shadow-xl z-50 py-1">
            <EditMenuItem
              label="Undo"
              shortcut="Ctrl+Z"
              disabled={undoStack.length === 0}
              onClick={() => { undo(); setEditMenuOpen(false); }}
            />
            <EditMenuItem
              label="Redo"
              shortcut="Ctrl+Shift+Z"
              disabled={redoStack.length === 0}
              onClick={() => { redo(); setEditMenuOpen(false); }}
            />
            <div className="border-t border-gray-600 my-1" />
            <EditMenuItem
              label="Cut"
              shortcut="Ctrl+X"
              disabled={!hasSelection}
              onClick={() => { cutSelected(); setEditMenuOpen(false); }}
            />
            <EditMenuItem
              label="Copy"
              shortcut="Ctrl+C"
              disabled={!hasSelection}
              onClick={() => { copySelected(); setEditMenuOpen(false); }}
            />
            <EditMenuItem
              label="Paste"
              shortcut="Ctrl+V"
              disabled={!clipboard}
              onClick={() => { pasteClipboard(); setEditMenuOpen(false); }}
            />
            <div className="border-t border-gray-600 my-1" />
            <EditMenuItem
              label="Delete"
              shortcut="Delete"
              disabled={!hasSelection}
              onClick={() => { deleteSelected(); setEditMenuOpen(false); }}
            />
            <EditMenuItem
              label="Select All"
              shortcut="Ctrl+A"
              disabled={nodes.length === 0}
              onClick={() => { selectAll(); setEditMenuOpen(false); }}
            />
          </div>
        )}
      </div>

      <div className="w-px h-5 bg-gray-600 mx-1" />

      {/* ── Models ────────────────────────────── */}
      <button
        onClick={() => setModelManagerOpen(true)}
        className="px-3 py-1 text-xs rounded bg-gray-600 hover:bg-gray-500 text-white"
      >
        Models
      </button>
      <button
        onClick={() => setShowRecommend(true)}
        className="px-3 py-1 text-xs rounded bg-indigo-600 hover:bg-indigo-500 text-white leading-none whitespace-nowrap"
        title="데이터 형식에 따른 모델 추천"
      >
        🤖 추천
      </button>

      <div className="w-px h-5 bg-gray-600 mx-1" />

      {/* ── Run/Stop ──────────────────────────── */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleRun}
          disabled={training.isTraining}
          className="px-3 py-[5px] text-xs rounded bg-green-600 hover:bg-green-700 disabled:opacity-50 text-white font-medium leading-none"
        >
          {training.isTraining ? 'Running...' : 'Run'}
        </button>

        <button
          onClick={handleStop}
          disabled={!training.isTraining}
          className="px-3 py-[5px] text-xs rounded bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white font-medium leading-none"
        >
          Stop
        </button>
      </div>

      {training.isTraining && (
        <span className="text-xs text-yellow-400 ml-4">
          Epoch {training.currentEpoch}/{training.totalEpochs}
        </span>
      )}
      {training.modelPath && !training.isTraining && (
        <span className="text-xs text-green-400 ml-4 flex items-center gap-2">
          Saved: {training.modelPath}
          <button
            onClick={() => setModelManagerOpen(true)}
            className="px-2 py-0.5 text-[10px] rounded bg-blue-600 hover:bg-blue-500 text-white font-medium"
          >
            Save Model
          </button>
        </span>
      )}

      {/* 오른쪽 */}
      <div className="flex-1" />

      <BackendSettings />
      <button
        onClick={handleReport}
        disabled={nodes.length === 0}
        className="px-3 py-1 text-xs rounded bg-gray-600 hover:bg-gray-500 text-gray-300 font-medium disabled:opacity-30"
        title="파이프라인 보고서 미리보기 / PDF 저장"
      >
        Report
      </button>
      <button
        onClick={() => setShowHelp(true)}
        className="px-3 py-1 text-xs rounded bg-gray-600 hover:bg-gray-500 text-gray-300 font-medium"
      >
        📘 Help
      </button>
      <button
        onClick={() => setLogPanelOpen(!logPanelOpen)}
        className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
          hasError
            ? 'bg-red-600/80 text-white hover:bg-red-600'
            : logPanelOpen
            ? 'bg-blue-600 text-white hover:bg-blue-500'
            : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
        }`}
      >
        Output {logs.length > 0 ? `(${logs.length})` : ''}
      </button>
    </div>
    {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
    {showReport && <ReportPreviewModal canvasImage={canvasImage} onClose={() => setShowReport(false)} />}
    {showRecommend && (
      <ModelRecommendModal
        onClose={() => setShowRecommend(false)}
        onLoadExample={async (id) => {
          await useStore.getState().loadExample(id);
          setTimeout(() => fitView({ padding: 0.1, duration: 300 }), 100);
        }}
      />
    )}
    </>
  );
}

// ── 편집 메뉴 아이템 ──────────────────────────────
function EditMenuItem({
  label,
  shortcut,
  disabled,
  onClick,
}: {
  label: string;
  shortcut: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-full flex items-center justify-between px-3 py-1.5 text-xs transition-colors ${
        disabled
          ? 'text-gray-500 cursor-default'
          : 'text-gray-200 hover:bg-gray-600'
      }`}
    >
      <span>{label}</span>
      <span className="text-gray-500 text-[10px] ml-4">{shortcut}</span>
    </button>
  );
}
