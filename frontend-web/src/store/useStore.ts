import { create } from 'zustand';
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react';
import { apiFetch, fetchJsonWithFallback } from '../api';

// ── Undo/Redo 스냅샷 ─────────────────────────────
interface Snapshot {
  nodes: Node[];
  edges: Edge[];
}

const MAX_HISTORY = 50;

// ── 노드 카탈로그 타입 ──────────────────────────────
export interface NodeParam {
  name: string;
  type: string;
  label: string;
  default?: any;
  options?: string[];
}

export interface NodeCatalogItem {
  type: string;
  category: string;
  description: string;
  params: NodeParam[];
  inputs: { name: string; type: string }[];
  outputs: { name: string; type: string }[];
}

// ── 학습 상태 ────────────────────────────────────────
export interface EpochData {
  epoch: number;
  total_epochs: number;
  loss: number;
  val_loss: number;
  accuracy: number;
  val_accuracy: number;
  lr: number;
  elapsed_sec: number;
  [key: string]: any;
}

export interface TrainingState {
  isTraining: boolean;
  epochs: EpochData[];
  currentEpoch: number;
  totalEpochs: number;
  modelPath: string | null;
  error: string | null;
}

// ── 로그 ──────────────────────────────────────────
export interface LogEntry {
  timestamp: string;  // HH:MM:SS
  level: 'info' | 'success' | 'error' | 'data';
  message: string;
}

// ── 예제 메타데이터 ──────────────────────────────────
export interface GuideStep {
  step: number;
  node: string;
  text: string;
}

export interface ExampleMeta {
  id: number;
  title: string;
  title_ko: string;
  category: string;
  difficulty: string;
  description: string;
  description_ko: string;
  tags: string[];
  estimated_time: string;
  dataset_info?: {
    name: string;
    source: string;
    samples: number;
    features: number | string;
    classes?: number;
  };
  learning_objectives: string[];
  guide_steps: GuideStep[];
  requires_phase?: number;
}

// ── 예제 인덱스 (목록 표시용) ────────────────────────
export interface ExampleIndex {
  id: number;
  title: string;
  title_ko: string;
  category: string;
  difficulty: string;
  tags: string[];
  estimated_time: string;
  requires_phase: number;
}

// ── 저장된 모델 ──────────────────────────────────────
export interface SavedModelInfo {
  name: string;
  framework: string;
  saved_at: string;
  is_regression: boolean;
  class_names: string[] | null;
  input_shape: number[] | null;
}

// ── 테스트 결과 ──────────────────────────────────────
export interface TestResults {
  metrics: Record<string, any>;
  samples: Array<Record<string, any>>;
  num_predictions: number;
  prediction_summary?: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
}

// ── 스토어 ───────────────────────────────────────────
interface AppState {
  // React Flow
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  addNode: (node: Node) => void;
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  updateNodeParams: (nodeId: string, params: Record<string, any>) => void;

  // 노드 카탈로그
  catalog: NodeCatalogItem[];
  setCatalog: (catalog: NodeCatalogItem[]) => void;

  // 학습 상태
  training: TrainingState;
  setTraining: (state: Partial<TrainingState>) => void;
  addEpochData: (data: EpochData) => void;
  resetTraining: () => void;

  // 시각화
  visualizations: Record<string, string>; // viz_type → base64
  setVisualization: (type: string, b64: string) => void;

  // 노드 중간 출력 요약 (파이프라인 실행 후)
  nodeOutputSummaries: Record<string, Record<string, any>>; // node_id → port summaries
  setNodeOutputSummary: (nodeId: string, summary: Record<string, any>) => void;
  clearNodeOutputSummaries: () => void;

  // WebSocket
  ws: WebSocket | null;
  setWs: (ws: WebSocket | null) => void;

  // 선택된 노드
  selectedNodeId: string | null;
  setSelectedNodeId: (id: string | null) => void;

  // 예제
  examples: ExampleIndex[];
  activeExample: ExampleMeta | null;
  activeExampleGuide: string | null;
  activeGuideStep: number;
  fetchExamples: () => Promise<void>;
  loadExample: (id: number) => Promise<void>;
  updateGuideForSelectedNode: (nodeId: string | null) => void;
  setActiveExample: (example: ExampleMeta | null) => void;

  // 사이드바 탭
  sidebarTab: 'nodes' | 'examples';
  setSidebarTab: (tab: 'nodes' | 'examples') => void;

  // 출력 로그
  logs: LogEntry[];
  logPanelOpen: boolean;
  showOutput: boolean;
  addLog: (level: LogEntry['level'], message: string) => void;
  clearLogs: () => void;
  setLogPanelOpen: (open: boolean) => void;
  setShowOutput: (show: boolean) => void;

  // ── 편집 기능 ────────────────────────────────────
  // Undo/Redo
  undoStack: Snapshot[];
  redoStack: Snapshot[];
  pushSnapshot: () => void;
  undo: () => void;
  redo: () => void;

  // 새로 작성
  newCanvas: () => void;

  // 선택된 노드/엣지 삭제
  deleteSelected: () => void;

  // 클립보드 (복사/잘라내기/붙여넣기)
  clipboard: { nodes: Node[]; edges: Edge[] } | null;
  copySelected: () => void;
  cutSelected: () => void;
  pasteClipboard: () => void;

  // 전체 선택
  selectAll: () => void;

  // ── 모델 관리 ─────────────────────────────────────
  savedModels: SavedModelInfo[];
  modelManagerOpen: boolean;
  testResults: TestResults | null;
  fetchSavedModels: () => Promise<void>;
  saveCurrentModel: (name: string) => Promise<boolean>;
  loadSavedModel: (name: string) => Promise<boolean>;
  deleteSavedModel: (name: string) => Promise<boolean>;
  setModelManagerOpen: (open: boolean) => void;
  setTestResults: (results: TestResults | null) => void;
}

const initialTraining: TrainingState = {
  isTraining: false,
  epochs: [],
  currentEpoch: 0,
  totalEpochs: 0,
  modelPath: null,
  error: null,
};

export const useStore = create<AppState>((set, get) => ({
  // ── React Flow ────────────────────────────────────
  nodes: [],
  edges: [],

  onNodesChange: (changes) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) });
  },

  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },

  onConnect: (connection) => {
    set({ edges: addEdge(connection, get().edges) });
  },

  addNode: (node) => {
    set({ nodes: [...get().nodes, node] });
  },

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),

  updateNodeParams: (nodeId, params) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, params: { ...(n.data.params as Record<string, any>), ...params } } } : n
      ),
    });
  },

  // ── 카탈로그 ──────────────────────────────────────
  catalog: [],
  setCatalog: (catalog) => set({ catalog }),

  // ── 학습 ──────────────────────────────────────────
  training: initialTraining,
  setTraining: (state) =>
    set({ training: { ...get().training, ...state } }),
  addEpochData: (data) =>
    set({
      training: {
        ...get().training,
        epochs: [...get().training.epochs, data],
        currentEpoch: data.epoch,
        totalEpochs: data.total_epochs,
      },
    }),
  resetTraining: () => set({ training: initialTraining }),

  // ── 시각화 ────────────────────────────────────────
  visualizations: {},
  setVisualization: (type, b64) =>
    set({ visualizations: { ...get().visualizations, [type]: b64 } }),

  nodeOutputSummaries: {},
  setNodeOutputSummary: (nodeId, summary) =>
    set({ nodeOutputSummaries: { ...get().nodeOutputSummaries, [nodeId]: summary } }),
  clearNodeOutputSummaries: () => set({ nodeOutputSummaries: {} }),

  // ── WebSocket ─────────────────────────────────────
  ws: null,
  setWs: (ws) => set({ ws }),

  // ── 선택 ──────────────────────────────────────────
  selectedNodeId: null,
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  // ── 예제 ──────────────────────────────────────────
  examples: [],
  activeExample: null,
  activeExampleGuide: null,
  activeGuideStep: 0,

  setActiveExample: (example) => set({ activeExample: example }),

  fetchExamples: async () => {
    try {
      const data = await fetchJsonWithFallback('/api/examples', 'examples.json');
      set({ examples: data });
    } catch {
      set({ examples: [] });
    }
  },

  loadExample: async (id: number) => {
    try {
      const data = await fetchJsonWithFallback(`/api/examples/${id}`, `examples/${id}.json`);
      const { catalog } = get();

      const meta: ExampleMeta = data.meta;

      const pipelineNodes: Node[] = (data.nodes || []).map((n: any) => {
        const catalogItem = catalog.find((c) => c.type === n.type);
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

      const pipelineEdges: Edge[] = (data.edges || []).map((e: any, i: number) => ({
        id: `e-${i}`,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
      }));

      set({
        nodes: pipelineNodes,
        edges: pipelineEdges,
        activeExample: meta,
        activeExampleGuide: null,
        activeGuideStep: 0,
        selectedNodeId: null,
        training: initialTraining,
        visualizations: {},
        nodeOutputSummaries: {},
      });
    } catch (err) {
      console.error('Failed to load example:', err);
    }
  },

  updateGuideForSelectedNode: (nodeId: string | null) => {
    const { activeExample } = get();
    if (!activeExample || !nodeId) {
      set({ activeExampleGuide: null, activeGuideStep: 0 });
      return;
    }
    const step = activeExample.guide_steps.find((s) => s.node === nodeId);
    if (step) {
      set({ activeExampleGuide: step.text, activeGuideStep: step.step });
    } else {
      set({ activeExampleGuide: null, activeGuideStep: 0 });
    }
  },

  // ── 사이드바 탭 ───────────────────────────────────
  sidebarTab: 'nodes',
  setSidebarTab: (tab) => set({ sidebarTab: tab }),

  // ── 출력 로그 ──────────────────────────────────────
  logs: [],
  logPanelOpen: false,
  showOutput: false,
  addLog: (level, message) => {
    const now = new Date();
    const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    const logs = [...get().logs, { timestamp, level, message }];
    // 최대 500줄 유지
    set({ logs: logs.length > 500 ? logs.slice(-500) : logs });
  },
  clearLogs: () => set({ logs: [] }),
  setLogPanelOpen: (open) => set({ logPanelOpen: open }),
  setShowOutput: (show) => set({ showOutput: show }),

  // ── 편집 기능 ──────────────────────────────────────

  // Undo/Redo
  undoStack: [],
  redoStack: [],

  pushSnapshot: () => {
    const { nodes, edges, undoStack } = get();
    const snapshot: Snapshot = {
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    };
    const newStack = [...undoStack, snapshot];
    if (newStack.length > MAX_HISTORY) newStack.shift();
    set({ undoStack: newStack, redoStack: [] });
  },

  undo: () => {
    const { undoStack, redoStack, nodes, edges } = get();
    if (undoStack.length === 0) return;
    const prev = undoStack[undoStack.length - 1];
    const currentSnapshot: Snapshot = {
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    };
    set({
      nodes: prev.nodes,
      edges: prev.edges,
      undoStack: undoStack.slice(0, -1),
      redoStack: [...redoStack, currentSnapshot],
      selectedNodeId: null,
    });
  },

  redo: () => {
    const { undoStack, redoStack, nodes, edges } = get();
    if (redoStack.length === 0) return;
    const next = redoStack[redoStack.length - 1];
    const currentSnapshot: Snapshot = {
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    };
    set({
      nodes: next.nodes,
      edges: next.edges,
      undoStack: [...undoStack, currentSnapshot],
      redoStack: redoStack.slice(0, -1),
      selectedNodeId: null,
    });
  },

  // 새로 작성
  newCanvas: () => {
    const { pushSnapshot } = get();
    pushSnapshot();
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      activeExample: null,
      activeExampleGuide: null,
      activeGuideStep: 0,
      training: initialTraining,
      visualizations: {},
      nodeOutputSummaries: {},
    });
  },

  // 선택된 노드/엣지 삭제
  deleteSelected: () => {
    const { nodes, edges, pushSnapshot } = get();
    const selectedNodeIds = nodes.filter((n) => n.selected).map((n) => n.id);
    const selectedEdgeIds = edges.filter((e) => e.selected).map((e) => e.id);
    if (selectedNodeIds.length === 0 && selectedEdgeIds.length === 0) return;

    pushSnapshot();

    // 노드 삭제 시 연결된 엣지도 함께 삭제
    const newNodes = nodes.filter((n) => !n.selected);
    const newEdges = edges.filter(
      (e) =>
        !e.selected &&
        !selectedNodeIds.includes(e.source) &&
        !selectedNodeIds.includes(e.target)
    );
    set({ nodes: newNodes, edges: newEdges, selectedNodeId: null });
  },

  // 클립보드
  clipboard: null,

  copySelected: () => {
    const { nodes, edges } = get();
    const selectedNodes = nodes.filter((n) => n.selected);
    if (selectedNodes.length === 0) return;
    const selectedIds = new Set(selectedNodes.map((n) => n.id));
    const selectedEdges = edges.filter(
      (e) => selectedIds.has(e.source) && selectedIds.has(e.target)
    );
    set({
      clipboard: {
        nodes: JSON.parse(JSON.stringify(selectedNodes)),
        edges: JSON.parse(JSON.stringify(selectedEdges)),
      },
    });
  },

  cutSelected: () => {
    const { copySelected, deleteSelected } = get();
    copySelected();
    deleteSelected();
  },

  pasteClipboard: () => {
    const { clipboard, nodes, edges, pushSnapshot } = get();
    if (!clipboard || clipboard.nodes.length === 0) return;

    pushSnapshot();

    // ID 매핑 (중복 방지)
    const idMap: Record<string, string> = {};
    const ts = Date.now();
    clipboard.nodes.forEach((n, i) => {
      idMap[n.id] = `${n.id}_copy_${ts}_${i}`;
    });

    const newNodes: Node[] = clipboard.nodes.map((n, i) => ({
      ...n,
      id: idMap[n.id],
      position: { x: n.position.x + 50, y: n.position.y + 50 },
      selected: true,
    }));

    const newEdges: Edge[] = clipboard.edges.map((e, i) => ({
      ...e,
      id: `e-paste-${ts}-${i}`,
      source: idMap[e.source] || e.source,
      target: idMap[e.target] || e.target,
      selected: false,
    }));

    // 기존 노드들의 선택 해제
    const updatedNodes = nodes.map((n) => ({ ...n, selected: false }));

    set({
      nodes: [...updatedNodes, ...newNodes],
      edges: [...edges, ...newEdges],
    });
  },

  // 전체 선택
  selectAll: () => {
    const { nodes, edges } = get();
    set({
      nodes: nodes.map((n) => ({ ...n, selected: true })),
      edges: edges.map((e) => ({ ...e, selected: true })),
    });
  },

  // ── 모델 관리 ─────────────────────────────────────
  savedModels: [],
  modelManagerOpen: false,
  testResults: null,

  fetchSavedModels: async () => {
    try {
      const res = await apiFetch('/api/model/list');
      const data = await res.json();
      set({ savedModels: data });
    } catch {
      set({ savedModels: [] });
    }
  },

  saveCurrentModel: async (name: string) => {
    try {
      const res = await apiFetch('/api/model/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (data.error) {
        get().addLog('error', `Model save failed: ${data.error}`);
        return false;
      }
      get().addLog('success', `Model "${name}" saved successfully`);
      get().fetchSavedModels();
      return true;
    } catch (e: any) {
      get().addLog('error', `Model save failed: ${e.message}`);
      return false;
    }
  },

  loadSavedModel: async (name: string) => {
    try {
      const res = await apiFetch(`/api/model/load/${name}`, { method: 'POST' });
      const data = await res.json();
      if (data.error) {
        get().addLog('error', `Model load failed: ${data.error}`);
        return false;
      }
      get().addLog('success', `Model "${name}" loaded (${data.framework})`);
      return true;
    } catch (e: any) {
      get().addLog('error', `Model load failed: ${e.message}`);
      return false;
    }
  },

  deleteSavedModel: async (name: string) => {
    try {
      const res = await apiFetch(`/api/model/delete/${name}`, { method: 'DELETE' });
      const data = await res.json();
      if (data.error) {
        get().addLog('error', `Model delete failed: ${data.error}`);
        return false;
      }
      get().addLog('success', `Model "${name}" deleted`);
      get().fetchSavedModels();
      return true;
    } catch (e: any) {
      get().addLog('error', `Model delete failed: ${e.message}`);
      return false;
    }
  },

  setModelManagerOpen: (open) => set({ modelManagerOpen: open }),
  setTestResults: (results) => set({ testResults: results }),
}));

// 외부 스크립트(Selenium 등)에서 store 접근 가능하도록 노출
if (typeof window !== 'undefined') {
  (window as any).__store = useStore;
}
