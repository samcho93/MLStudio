import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,

  useReactFlow,
  type Node,
  type Edge,
  type OnSelectionChangeParams,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useStore } from './store/useStore';
import { nodeTypes } from './nodes/nodeTypes';
import { Sidebar } from './components/Sidebar';
import { PropertiesPanel } from './components/PropertiesPanel';
import { Toolbar } from './components/Toolbar';
import { TrainingPanel } from './components/TrainingPanel';
import { ExecutionLog } from './components/ExecutionLog';
import { ModelManagerPanel } from './components/ModelManagerPanel';
import { NODE_DESCRIPTIONS } from './components/HelpModal';

let nodeCounter = 1000; // drag-drop용 카운터 (Sidebar와 충돌 방지)

export default function App() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    setCatalog,
    setSelectedNodeId,
    updateGuideForSelectedNode,
    training,
    deleteSelected,
    copySelected,
    cutSelected,
    pasteClipboard,
    selectAll,
    undo,
    redo,
    pushSnapshot,
    addNode,
    catalog,
  } = useStore();

  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, fitView } = useReactFlow();

  // ── 패널 접기/펼치기 ─────────────────────────────
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [propertiesCollapsed, setPropertiesCollapsed] = useState(false);
  const [trainingCollapsed, setTrainingCollapsed] = useState(false);

  // 패널 토글·학습 결과 창 등 레이아웃 변경 → fitView
  const showTrainingPanel = training.epochs.length > 0;
  useEffect(() => {
    const timer = setTimeout(() => fitView({ padding: 0.05, duration: 200 }), 50);
    return () => clearTimeout(timer);
  }, [sidebarCollapsed, propertiesCollapsed, trainingCollapsed, showTrainingPanel, fitView]);

  // ── 드래그앤드롭: 사이드바 → 캔버스 ─────────────────
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const raw = e.dataTransfer.getData('application/ml-node');
      if (!raw) return;

      const item = JSON.parse(raw);
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });

      pushSnapshot();
      nodeCounter++;
      const id = `${item.type}_${nodeCounter}`;
      const defaults: Record<string, any> = {};
      for (const p of item.params) {
        defaults[p.name] = p.default ?? '';
      }

      addNode({
        id,
        type: 'mlNode',
        position,
        data: {
          label: item.type,
          category: item.category,
          params: defaults,
          inputs: item.inputs,
          outputs: item.outputs,
          nodeType: item.type,
        },
      });
    },
    [screenToFlowPosition, pushSnapshot, addNode]
  );

  // ── 우클릭 컨텍스트 메뉴 ────────────────────────────
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    nodeId: string | null; // null = 캔버스(pane) 우클릭
  } | null>(null);

  // 노드 설명 대화상자
  const [descriptionDialog, setDescriptionDialog] = useState<{
    nodeType: string;
    desc_ko: string;
    detail: string;
    tips: string;
  } | null>(null);

  const onNodeContextMenu = useCallback(
    (e: React.MouseEvent, node: Node) => {
      e.preventDefault();
      const selectedCount = nodes.filter((n) => n.selected).length;
      // 우클릭한 노드가 이미 선택된 다중 선택의 일부면 선택 유지
      if (!node.selected || selectedCount <= 1) {
        setSelectedNodeId(node.id);
      }
      setContextMenu({ x: e.clientX, y: e.clientY, nodeId: node.id });
    },
    [nodes, setSelectedNodeId]
  );

  const onPaneContextMenu = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      e.preventDefault();
      setContextMenu({ x: e.clientX, y: e.clientY, nodeId: null });
    },
    []
  );

  // 다중 선택(드래그 블록) 후 우클릭
  const onSelectionContextMenu = useCallback(
    (e: React.MouseEvent, selectedNodes: Node[]) => {
      e.preventDefault();
      if (selectedNodes.length > 0) {
        setContextMenu({ x: e.clientX, y: e.clientY, nodeId: selectedNodes[0].id });
      }
    },
    []
  );

  // 메뉴 외부 클릭 시 닫기
  useEffect(() => {
    if (!contextMenu) return;
    const handleClick = () => setContextMenu(null);
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [contextMenu]);

  // 노드 카탈로그 로드
  useEffect(() => {
    fetch('/api/node-catalog')
      .then((res) => res.json())
      .then((data) => setCatalog(data))
      .catch(() => {
        // 오프라인 시 기본 카탈로그
        setCatalog(getDefaultCatalog());
      });
  }, [setCatalog]);

  // ── 키보드 단축키 ────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // input/textarea 내부에서는 무시
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      const ctrl = e.ctrlKey || e.metaKey;

      // Delete / Backspace: 선택 삭제
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        deleteSelected();
        return;
      }

      // Ctrl+Z: Undo
      if (ctrl && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
        return;
      }

      // Ctrl+Shift+Z or Ctrl+Y: Redo
      if ((ctrl && e.key === 'z' && e.shiftKey) || (ctrl && e.key === 'y')) {
        e.preventDefault();
        redo();
        return;
      }

      // Ctrl+C: Copy
      if (ctrl && e.key === 'c') {
        e.preventDefault();
        copySelected();
        return;
      }

      // Ctrl+X: Cut
      if (ctrl && e.key === 'x') {
        e.preventDefault();
        cutSelected();
        return;
      }

      // Ctrl+V: Paste
      if (ctrl && e.key === 'v') {
        e.preventDefault();
        pasteClipboard();
        return;
      }

      // Ctrl+A: Select All
      if (ctrl && e.key === 'a') {
        e.preventDefault();
        selectAll();
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [deleteSelected, copySelected, cutSelected, pasteClipboard, selectAll, undo, redo]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
      updateGuideForSelectedNode(node.id);
    },
    [setSelectedNodeId, updateGuideForSelectedNode]
  );

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    updateGuideForSelectedNode(null);
  }, [setSelectedNodeId, updateGuideForSelectedNode]);

  // 드래그 시작 감지용 ref
  const isDraggingRef = useRef(false);

  // 노드/엣지 변경 시 undo 스냅샷
  const onNodesChangeWrapped: typeof onNodesChange = useCallback(
    (changes) => {
      for (const c of changes) {
        // 드래그 시작 시 (이동 전) 스냅샷 저장
        if (c.type === 'position' && c.dragging === true && !isDraggingRef.current) {
          isDraggingRef.current = true;
          pushSnapshot();
        }
        // 드래그 종료
        if (c.type === 'position' && c.dragging === false) {
          isDraggingRef.current = false;
        }
        // 노드 삭제 시 스냅샷
        if (c.type === 'remove') {
          pushSnapshot();
        }
      }
      onNodesChange(changes);
    },
    [onNodesChange, pushSnapshot]
  );

  const onConnectWrapped: typeof onConnect = useCallback(
    (connection) => {
      pushSnapshot();
      onConnect(connection);
    },
    [onConnect, pushSnapshot]
  );

  // 선택 변경 감지 — 노드 선택 시 속성 패널 연동
  const onSelectionChange = useCallback(
    ({ nodes: selectedNodes }: OnSelectionChangeParams) => {
      if (selectedNodes.length === 1) {
        setSelectedNodeId(selectedNodes[0].id);
        updateGuideForSelectedNode(selectedNodes[0].id);
      } else if (selectedNodes.length === 0) {
        // 엣지만 선택되었거나 아무것도 없을 때
        setSelectedNodeId(null);
        updateGuideForSelectedNode(null);
      }
    },
    [setSelectedNodeId, updateGuideForSelectedNode]
  );

  const showTraining = showTrainingPanel;

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      <Toolbar />

      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* 왼쪽 사이드바 + 토글 */}
        {!sidebarCollapsed && <Sidebar />}
        <button
          onClick={() => setSidebarCollapsed((v) => !v)}
          className="flex-shrink-0 w-5 bg-gray-800 border-r border-gray-700 flex items-center justify-center hover:bg-gray-700 transition-colors group"
          title={sidebarCollapsed ? '노드 팔레트 열기' : '노드 팔레트 닫기'}
        >
          <span className="text-gray-500 group-hover:text-white text-xs select-none">
            {sidebarCollapsed ? '▶' : '◀'}
          </span>
        </button>

        {/* 가운데: 캔버스 + 하단 출력 */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 relative min-h-0" ref={reactFlowWrapper}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChangeWrapped}
              onEdgesChange={onEdgesChange}
              onConnect={onConnectWrapped}
              onNodeClick={onNodeClick}
              onPaneClick={(e) => { onPaneClick(); setContextMenu(null); }}
              onSelectionChange={onSelectionChange}
              onNodeContextMenu={onNodeContextMenu}
              onPaneContextMenu={onPaneContextMenu}
              onSelectionContextMenu={onSelectionContextMenu}
              onDragOver={onDragOver}
              onDrop={onDrop}
              nodeTypes={nodeTypes as any}
              fitView
              deleteKeyCode={null}
              selectionOnDrag
              panOnDrag={[1, 2]}
              selectionKeyCode={null}
              multiSelectionKeyCode="Ctrl"
              elementsSelectable
              edgesFocusable
              proOptions={{ hideAttribution: true }}
              defaultEdgeOptions={{
                style: { strokeWidth: 2, stroke: '#6b7280' },
              }}
              edgesReconnectable
            >
              <Background color="#374151" gap={20} />
              <Controls />

            </ReactFlow>

            {/* 우클릭 컨텍스트 메뉴 */}
            {contextMenu && (
              <div
                className="fixed bg-gray-800 border border-gray-600 rounded shadow-xl py-1 z-50 min-w-[160px]"
                style={{ top: contextMenu.y, left: contextMenu.x }}
              >
                {contextMenu.nodeId ? (
                  (() => {
                    const selectedNodes = nodes.filter((n) => n.selected);
                    const isMulti = selectedNodes.length > 1;
                    const targetIds = isMulti
                      ? new Set(selectedNodes.map((n) => n.id))
                      : new Set([contextMenu.nodeId]);
                    return <>
                    {/* ── 노드 메뉴 ── */}
                    {!isMulti && (
                      <>
                        <button
                          className="w-full text-left px-3 py-1.5 text-xs text-gray-200 hover:bg-blue-600 hover:text-white transition-colors"
                          onClick={() => {
                            setSelectedNodeId(contextMenu.nodeId);
                            updateGuideForSelectedNode(contextMenu.nodeId);
                            setContextMenu(null);
                          }}
                        >
                          Edit Properties
                        </button>
                        <button
                          className="w-full text-left px-3 py-1.5 text-xs text-gray-200 hover:bg-blue-600 hover:text-white transition-colors"
                          onClick={() => {
                            const nd = nodes.find((n) => n.id === contextMenu.nodeId);
                            const nt = nd ? (nd.data.nodeType as string) : null;
                            if (nt) {
                              const desc = NODE_DESCRIPTIONS[nt];
                              if (desc) {
                                setDescriptionDialog({ nodeType: nt, ...desc });
                              } else {
                                const catItem = catalog.find((c) => c.type === nt);
                                setDescriptionDialog({
                                  nodeType: nt,
                                  desc_ko: catItem?.description || nt,
                                  detail: '이 노드에 대한 상세 설명이 아직 등록되지 않았습니다.',
                                  tips: '',
                                });
                              }
                            }
                            setContextMenu(null);
                          }}
                        >
                          Description
                        </button>
                        <div className="border-t border-gray-600 my-1" />
                      </>
                    )}
                    {isMulti && (
                      <div className="px-3 py-1 text-[10px] text-gray-500">
                        {selectedNodes.length} nodes selected
                      </div>
                    )}
                    <button
                      className="w-full text-left px-3 py-1.5 text-xs text-gray-200 hover:bg-blue-600 hover:text-white transition-colors"
                      onClick={() => {
                        copySelected();
                        setContextMenu(null);
                      }}
                    >
                      Copy
                      <span className="float-right text-gray-500">Ctrl+C</span>
                    </button>
                    <button
                      className="w-full text-left px-3 py-1.5 text-xs text-gray-200 hover:bg-blue-600 hover:text-white transition-colors"
                      onClick={() => {
                        cutSelected();
                        setContextMenu(null);
                      }}
                    >
                      Cut
                      <span className="float-right text-gray-500">Ctrl+X</span>
                    </button>
                    <button
                      className="w-full text-left px-3 py-1.5 text-xs text-gray-200 hover:bg-blue-600 hover:text-white transition-colors"
                      onClick={() => {
                        pasteClipboard();
                        setContextMenu(null);
                      }}
                    >
                      Paste
                      <span className="float-right text-gray-500">Ctrl+V</span>
                    </button>
                    <div className="border-t border-gray-600 my-1" />
                    <button
                      className="w-full text-left px-3 py-1.5 text-xs text-red-400 hover:bg-red-600 hover:text-white transition-colors"
                      onClick={() => {
                        pushSnapshot();
                        useStore.getState().setNodes(
                          nodes.filter((n) => !targetIds.has(n.id))
                        );
                        useStore.getState().setEdges(
                          edges.filter(
                            (e) => !targetIds.has(e.source) && !targetIds.has(e.target)
                          )
                        );
                        setSelectedNodeId(null);
                        setContextMenu(null);
                      }}
                    >
                      Delete{isMulti ? ` (${selectedNodes.length})` : ''}
                      <span className="float-right text-gray-500">Del</span>
                    </button>
                  </>;
                  })()
                ) : (
                  <>
                    {/* ── 캔버스(Pane) 메뉴 ── */}
                    <button
                      className="w-full text-left px-3 py-1.5 text-xs text-gray-200 hover:bg-blue-600 hover:text-white transition-colors"
                      onClick={() => {
                        pasteClipboard();
                        setContextMenu(null);
                      }}
                    >
                      Paste
                      <span className="float-right text-gray-500">Ctrl+V</span>
                    </button>
                    <div className="border-t border-gray-600 my-1" />
                    <button
                      className="w-full text-left px-3 py-1.5 text-xs text-gray-200 hover:bg-blue-600 hover:text-white transition-colors"
                      onClick={() => {
                        selectAll();
                        setContextMenu(null);
                      }}
                    >
                      Select All
                      <span className="float-right text-gray-500">Ctrl+A</span>
                    </button>
                    <div className="border-t border-gray-600 my-1" />
                    <button
                      className="w-full text-left px-3 py-1.5 text-xs text-gray-200 hover:bg-blue-600 hover:text-white transition-colors"
                      onClick={() => {
                        undo();
                        setContextMenu(null);
                      }}
                    >
                      Undo
                      <span className="float-right text-gray-500">Ctrl+Z</span>
                    </button>
                    <button
                      className="w-full text-left px-3 py-1.5 text-xs text-gray-200 hover:bg-blue-600 hover:text-white transition-colors"
                      onClick={() => {
                        redo();
                        setContextMenu(null);
                      }}
                    >
                      Redo
                      <span className="float-right text-gray-500">Ctrl+Y</span>
                    </button>
                  </>
                )}
              </div>
            )}
          </div>

          {/* 하단 실행 로그 패널 */}
          <ExecutionLog />
        </div>

        {/* 오른쪽 Properties 토글 + 패널 */}
        <button
          onClick={() => setPropertiesCollapsed((v) => !v)}
          className="flex-shrink-0 w-5 bg-gray-800 border-l border-gray-700 flex items-center justify-center hover:bg-gray-700 transition-colors group"
          title={propertiesCollapsed ? 'Properties 열기' : 'Properties 닫기'}
        >
          <span className="text-gray-500 group-hover:text-white text-xs select-none">
            {propertiesCollapsed ? '◀' : '▶'}
          </span>
        </button>
        {!propertiesCollapsed && <PropertiesPanel />}
        {showTraining && (
          <>
            <button
              onClick={() => setTrainingCollapsed((v) => !v)}
              className="flex-shrink-0 w-5 bg-gray-800 border-l border-gray-700 flex items-center justify-center hover:bg-gray-700 transition-colors group"
              title={trainingCollapsed ? 'Training Monitor 열기' : 'Training Monitor 닫기'}
            >
              <span className="text-gray-500 group-hover:text-white text-xs select-none">
                {trainingCollapsed ? '◀' : '▶'}
              </span>
            </button>
            {!trainingCollapsed && <TrainingPanel />}
          </>
        )}
      </div>

      {/* 모델 관리 모달 */}
      <ModelManagerPanel />

      {/* 노드 설명 대화상자 */}
      {descriptionDialog && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={(e) => { if (e.target === e.currentTarget) setDescriptionDialog(null); }}
        >
          <div className="bg-gray-800 border border-gray-600 rounded-xl shadow-2xl w-[480px] max-h-[70vh] flex flex-col overflow-hidden">
            {/* 헤더 */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
              <div className="flex items-center gap-2">
                <span className="text-base">📖</span>
                <h3 className="text-sm font-bold text-white">{descriptionDialog.nodeType}</h3>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/15 text-blue-300 border border-blue-500/20">
                  {descriptionDialog.desc_ko}
                </span>
              </div>
              <button
                onClick={() => setDescriptionDialog(null)}
                className="text-gray-400 hover:text-white text-lg leading-none px-1"
              >
                ✕
              </button>
            </div>

            {/* 내용 */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* 상세 설명 */}
              <div>
                <h4 className="text-[11px] font-bold text-gray-500 uppercase mb-1.5">설명</h4>
                <p className="text-xs text-gray-300 leading-relaxed">
                  {descriptionDialog.detail}
                </p>
              </div>

              {/* 팁 */}
              {descriptionDialog.tips && (
                <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className="text-xs">💡</span>
                    <h4 className="text-[11px] font-bold text-yellow-400">Tip</h4>
                  </div>
                  <p className="text-[11px] text-yellow-200/80 leading-relaxed">
                    {descriptionDialog.tips}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// 백엔드 미연결 시 기본 카탈로그
function getDefaultCatalog() {
  const io = (name: string, type: string) => ({ name, type });
  const layerIO = { inputs: [io('input', 'layer_config')], outputs: [io('output', 'layer_config')] };
  return [
    // ── 데이터 (11) ─────────────────────────────────
    { type: 'CSVLoader', category: 'data', description: 'CSV 파일 로드', params: [{ name: 'file_path', type: 'string', label: '파일 경로' }, { name: 'target_column', type: 'string', label: '타겟 컬럼', default: '' }, { name: 'separator', type: 'string', label: '구분자', default: ',' }], inputs: [], outputs: [io('features', 'dataset'), io('labels', 'dataset'), io('dataframe', 'dataframe')] },
    { type: 'NumpyInput', category: 'data', description: '내장 데이터셋 로드', params: [{ name: 'dataset', type: 'select', label: '데이터셋', options: ['mnist', 'fashion_mnist', 'cifar10', 'iris', 'boston'], default: 'mnist' }], inputs: [], outputs: [io('train_features', 'tensor'), io('train_labels', 'tensor'), io('test_features', 'tensor'), io('test_labels', 'tensor')] },
    { type: 'ImageFolder', category: 'data', description: '이미지 폴더 로드', params: [{ name: 'folder_path', type: 'string', label: '폴더 경로' }, { name: 'image_size', type: 'number', label: '이미지 크기', default: 224 }, { name: 'batch_size', type: 'number', label: '배치 크기', default: 32 }], inputs: [], outputs: [io('features', 'dataset'), io('labels', 'dataset'), io('class_names', 'list')] },
    { type: 'TrainValSplit', category: 'data', description: '데이터 분할', params: [{ name: 'val_ratio', type: 'number', label: '검증 비율', default: 0.2 }, { name: 'test_ratio', type: 'number', label: '테스트 비율', default: 0.0 }, { name: 'random_seed', type: 'number', label: '시드', default: 42 }], inputs: [io('features', 'dataset'), io('labels', 'dataset')], outputs: [io('train_features', 'dataset'), io('train_labels', 'dataset'), io('val_features', 'dataset'), io('val_labels', 'dataset'), io('test_features', 'dataset'), io('test_labels', 'dataset')] },
    { type: 'DataInspector', category: 'data', description: '데이터 미리보기', params: [], inputs: [io('input', 'any')], outputs: [io('output', 'any')] },
    { type: 'StandardScaler', category: 'data', description: 'Z-score 정규화', params: [], inputs: [io('input', 'dataset')], outputs: [io('output', 'dataset')] },
    { type: 'MinMaxScaler', category: 'data', description: 'Min-Max 정규화', params: [], inputs: [io('input', 'dataset')], outputs: [io('output', 'dataset')] },
    { type: 'LabelEncoder', category: 'data', description: '레이블 인코딩', params: [], inputs: [io('input', 'dataset')], outputs: [io('output', 'dataset')] },
    { type: 'OneHotEncoder', category: 'data', description: 'One-Hot 인코딩', params: [], inputs: [io('input', 'dataset')], outputs: [io('output', 'dataset')] },
    { type: 'PCA', category: 'data', description: '주성분 분석', params: [{ name: 'n_components', type: 'number', label: '주성분 수', default: 2 }], inputs: [io('input', 'dataset')], outputs: [io('output', 'dataset')] },
    { type: 'Augmentation', category: 'data', description: '이미지 증강', params: [{ name: 'rotation', type: 'number', label: '회전 각도', default: 20 }, { name: 'horizontal_flip', type: 'boolean', label: '수평 뒤집기', default: true }], inputs: [io('input', 'dataset')], outputs: [io('output', 'dataset')] },
    // ── 레이어 (19) ─────────────────────────────────
    { type: 'Dense', category: 'layer', description: '완전연결층', params: [{ name: 'units', type: 'number', label: '유닛 수', default: 128 }, { name: 'activation', type: 'select', label: '활성화', options: ['relu', 'sigmoid', 'tanh', 'softmax', 'linear'], default: 'relu' }], ...layerIO },
    { type: 'Conv2D', category: 'layer', description: '2D 합성곱층', params: [{ name: 'filters', type: 'number', label: '필터 수', default: 32 }, { name: 'kernel_size', type: 'number', label: '커널 크기', default: 3 }, { name: 'activation', type: 'select', label: '활성화', options: ['relu', 'sigmoid', 'tanh', 'linear'], default: 'relu' }], ...layerIO },
    { type: 'Conv1D', category: 'layer', description: '1D 합성곱층', params: [{ name: 'filters', type: 'number', label: '필터 수', default: 64 }, { name: 'kernel_size', type: 'number', label: '커널 크기', default: 3 }], ...layerIO },
    { type: 'Conv2DTranspose', category: 'layer', description: '전치 합성곱', params: [{ name: 'filters', type: 'number', label: '필터 수', default: 32 }, { name: 'kernel_size', type: 'number', label: '커널 크기', default: 3 }], ...layerIO },
    { type: 'MaxPooling2D', category: 'layer', description: '최대 풀링', params: [{ name: 'pool_size', type: 'number', label: '풀 크기', default: 2 }], ...layerIO },
    { type: 'Flatten', category: 'layer', description: 'Flatten', params: [], ...layerIO },
    { type: 'GlobalAveragePooling2D', category: 'layer', description: '공간 평균 풀링 2D', params: [], ...layerIO },
    { type: 'GlobalAveragePooling1D', category: 'layer', description: '글로벌 평균 풀링 1D', params: [], ...layerIO },
    { type: 'BatchNorm', category: 'layer', description: '배치 정규화', params: [], ...layerIO },
    { type: 'Dropout', category: 'layer', description: '드롭아웃', params: [{ name: 'rate', type: 'number', label: '비율', default: 0.5 }], ...layerIO },
    { type: 'Embedding', category: 'layer', description: '임베딩', params: [{ name: 'input_dim', type: 'number', label: '어휘 크기', default: 10000 }, { name: 'output_dim', type: 'number', label: '임베딩 차원', default: 128 }, { name: 'max_length', type: 'number', label: '최대 길이', default: 100 }], ...layerIO },
    { type: 'LSTM', category: 'layer', description: 'LSTM', params: [{ name: 'units', type: 'number', label: '유닛 수', default: 64 }, { name: 'return_sequences', type: 'boolean', label: '시퀀스 반환', default: false }, { name: 'bidirectional', type: 'boolean', label: '양방향', default: false }], ...layerIO },
    { type: 'GRU', category: 'layer', description: 'GRU', params: [{ name: 'units', type: 'number', label: '유닛 수', default: 64 }, { name: 'return_sequences', type: 'boolean', label: '시퀀스 반환', default: false }], ...layerIO },
    { type: 'Reshape', category: 'layer', description: 'Reshape', params: [{ name: 'target_shape', type: 'string', label: '목표 shape', default: '-1, 1' }], ...layerIO },
    { type: 'Add', category: 'layer', description: 'Add (잔차 연결)', params: [], inputs: [io('input_a', 'layer_config'), io('input_b', 'layer_config')], outputs: [io('output', 'layer_config')] },
    { type: 'Concat', category: 'layer', description: 'Concatenate', params: [{ name: 'axis', type: 'number', label: '축', default: -1 }], inputs: [io('input_a', 'layer_config'), io('input_b', 'layer_config')], outputs: [io('output', 'layer_config')] },
    { type: 'PretrainedModel', category: 'layer', description: '사전학습 모델', params: [{ name: 'model_name', type: 'select', label: '모델', options: ['MobileNetV2', 'ResNet50', 'EfficientNetB0', 'VGG16'], default: 'MobileNetV2' }, { name: 'trainable', type: 'boolean', label: '파인튜닝', default: false }], ...layerIO },
    { type: 'MultiHeadAttention', category: 'layer', description: '멀티 헤드 어텐션', params: [{ name: 'num_heads', type: 'number', label: '헤드 수', default: 8 }, { name: 'key_dim', type: 'number', label: '키 차원', default: 64 }], ...layerIO },
    { type: 'TransformerBlock', category: 'layer', description: 'Transformer 블록', params: [{ name: 'num_heads', type: 'number', label: '헤드 수', default: 4 }, { name: 'ff_dim', type: 'number', label: 'FFN 차원', default: 128 }], ...layerIO },
    // ── 학습 (6) ────────────────────────────────────
    { type: 'Optimizer', category: 'training', description: '옵티마이저', params: [{ name: 'type', type: 'select', label: '옵티마이저', options: ['adam', 'sgd', 'rmsprop', 'adamw'], default: 'adam' }, { name: 'learning_rate', type: 'number', label: '학습률', default: 0.001 }], inputs: [], outputs: [io('optimizer_config', 'config')] },
    { type: 'LossFunction', category: 'training', description: '손실 함수', params: [{ name: 'type', type: 'select', label: '손실 함수', options: ['sparse_categorical_crossentropy', 'categorical_crossentropy', 'binary_crossentropy', 'mse', 'mae'], default: 'sparse_categorical_crossentropy' }], inputs: [], outputs: [io('loss_config', 'config')] },
    { type: 'EarlyStopping', category: 'training', description: '조기 종료', params: [{ name: 'patience', type: 'number', label: '인내 횟수', default: 5 }], inputs: [], outputs: [io('callback_config', 'config')] },
    { type: 'ModelCheckpoint', category: 'training', description: '모델 체크포인트', params: [{ name: 'filepath', type: 'string', label: '저장 경로', default: 'outputs/best_model.keras' }, { name: 'monitor', type: 'string', label: '모니터 메트릭', default: 'val_loss' }], inputs: [], outputs: [io('callback_config', 'config')] },
    { type: 'LRScheduler', category: 'training', description: '학습률 스케줄러', params: [{ name: 'type', type: 'select', label: '스케줄러', options: ['StepLR', 'CosineAnnealing', 'ReduceOnPlateau'], default: 'ReduceOnPlateau' }], inputs: [], outputs: [io('scheduler_config', 'config')] },
    { type: 'Trainer', category: 'training', description: '학습 실행', params: [{ name: 'epochs', type: 'number', label: '에포크', default: 10 }, { name: 'batch_size', type: 'number', label: '배치 크기', default: 32 }, { name: 'framework', type: 'select', label: '프레임워크', options: ['pytorch', 'tensorflow'], default: 'pytorch' }], inputs: [io('layers', 'layer_config'), io('train_features', 'dataset'), io('train_labels', 'dataset'), io('val_features', 'dataset'), io('val_labels', 'dataset'), io('optimizer_config', 'config'), io('loss_config', 'config'), io('callback_config', 'config'), io('scheduler_config', 'config')], outputs: [io('model', 'model'), io('history', 'history'), io('model_path', 'string')] },
    // ── 시각화 (3) ──────────────────────────────────
    { type: 'LossCurve', category: 'visualization', description: 'Loss 곡선', params: [], inputs: [io('history', 'history')], outputs: [io('image_b64', 'string')] },
    { type: 'AccuracyCurve', category: 'visualization', description: 'Accuracy 곡선', params: [], inputs: [io('history', 'history')], outputs: [io('image_b64', 'string')] },
    { type: 'ModelSummary', category: 'visualization', description: '모델 구조 요약', params: [], inputs: [io('model', 'model')], outputs: [io('summary', 'string')] },
    // ── 평가 (2) ────────────────────────────────────
    { type: 'ClassificationMetrics', category: 'evaluation', description: '분류 메트릭', params: [], inputs: [io('model', 'model'), io('test_features', 'dataset'), io('test_labels', 'dataset')], outputs: [io('metrics', 'dict')] },
    { type: 'ConfusionMatrix', category: 'evaluation', description: '혼동 행렬', params: [], inputs: [io('model', 'model'), io('test_features', 'dataset'), io('test_labels', 'dataset')], outputs: [io('image_b64', 'string')] },
    // ── 평가 - ModelTester ─────────────────────────────
    { type: 'ModelTester', category: 'evaluation', description: '모델 테스트', params: [{ name: 'batch_size', type: 'number', label: 'Batch Size', default: 32 }, { name: 'show_samples', type: 'number', label: 'Show Samples', default: 10 }], inputs: [io('model', 'model'), io('test_features', 'dataset'), io('test_labels', 'dataset')], outputs: [io('predictions', 'dataset'), io('metrics', 'any')] },
    // ── 출력 ──────────────────────────────────────────
    { type: 'ModelSave', category: 'output', description: '모델 저장', params: [{ name: 'filepath', type: 'string', label: '저장 경로', default: 'outputs/model.keras' }], inputs: [io('model', 'model')], outputs: [io('model_path', 'string')] },
    { type: 'ModelLoader', category: 'output', description: '저장된 모델 로드', params: [{ name: 'model_name', type: 'select', label: 'Model Name', options: [], default: '' }, { name: 'framework', type: 'select', label: 'Framework', options: ['tensorflow', 'pytorch'], default: 'tensorflow' }], inputs: [], outputs: [io('model', 'model'), io('model_info', 'any')] },
  ];
}
