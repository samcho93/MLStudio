import React, { useState, useMemo } from 'react';
import { useStore } from '../store/useStore';
import { FilePreviewModal } from './FilePreviewModal';
import { NODE_DESCRIPTIONS } from './HelpModal';

const DIFFICULTY_COLORS: Record<string, string> = {
  beginner: '#10b981',
  intermediate: '#f59e0b',
  advanced: '#ef4444',
};

export function PropertiesPanel() {
  const selectedNodeId = useStore((s) => s.selectedNodeId);
  const nodes = useStore((s) => s.nodes);
  const updateNodeParams = useStore((s) => s.updateNodeParams);
  const catalog = useStore((s) => s.catalog);
  const activeExample = useStore((s) => s.activeExample);
  const setSelectedNodeId = useStore((s) => s.setSelectedNodeId);

  const node = nodes.find((n) => n.id === selectedNodeId);
  const nodeType = node ? (node.data.nodeType as string) : null;
  const catalogItem = nodeType ? catalog.find((c) => c.type === nodeType) : null;
  const paramSchemas = catalogItem?.params || [];
  const params = node ? ((node.data.params as Record<string, any>) || {}) : {};

  // 선택된 노드의 가이드 스텝
  const guideStep = activeExample?.guide_steps.find(
    (s) => s.node === selectedNodeId
  );

  const [previewPath, setPreviewPath] = useState<string | null>(null);

  // 파일 경로 파라미터인지 판별
  const isPathParam = (name: string) =>
    ['file_path', 'folder_path', 'model_path', 'model_name'].includes(name);

  const handleChange = (name: string, value: any) => {
    if (node) updateNodeParams(node.id, { [name]: value });
  };

  return (
    <div className="w-72 bg-gray-800 border-l border-gray-700 overflow-y-auto flex flex-col">
      {/* ── 예제 개요 (노드 미선택 + 예제 로드 시) ───── */}
      {activeExample && !node && (
        <ExampleOverview
          example={activeExample}
          onStepClick={(nodeId) => setSelectedNodeId(nodeId)}
        />
      )}

      {/* ── 노드 속성 편집 ───────────────────────────── */}
      {node ? (
        <>
          <div className="p-3 border-b border-gray-700">
            <h3 className="text-sm font-bold text-white">{nodeType}</h3>
            <p className="text-[10px] text-gray-500">{node.id}</p>
          </div>

          {/* 해당 노드의 가이드 */}
          {guideStep && (
            <div className="mx-3 mt-3 p-2.5 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <div className="flex items-center gap-1.5 mb-1">
                <span className="text-[10px] bg-blue-500 text-white w-4 h-4 rounded-full flex items-center justify-center font-bold">
                  {guideStep.step}
                </span>
                <span className="text-[10px] font-semibold text-blue-400">Guide</span>
              </div>
              <p className="text-[11px] text-blue-200 leading-relaxed">
                {guideStep.text}
              </p>
            </div>
          )}

          {/* 파라미터 편집 */}
          <div className="p-3 space-y-3">
            {paramSchemas.length === 0 && (
              <p className="text-xs text-gray-500">No parameters</p>
            )}
            {paramSchemas.map((schema) => (
              <div key={schema.name}>
                <label className="block text-[11px] text-gray-400 mb-1">
                  {schema.label || schema.name}
                </label>

                {schema.type === 'select' ? (
                  <select
                    value={params[schema.name] ?? schema.default ?? ''}
                    onChange={(e) => handleChange(schema.name, e.target.value)}
                    className="w-full bg-gray-700 text-gray-200 text-xs rounded px-2 py-1.5 border border-gray-600 focus:border-blue-500 focus:outline-none"
                  >
                    {(schema.options || []).map((opt: string) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                ) : schema.type === 'boolean' ? (
                  <input
                    type="checkbox"
                    checked={params[schema.name] ?? schema.default ?? false}
                    onChange={(e) => handleChange(schema.name, e.target.checked)}
                    className="rounded bg-gray-700 border-gray-600"
                  />
                ) : schema.type === 'number' ? (
                  <input
                    type="number"
                    value={params[schema.name] ?? schema.default ?? 0}
                    onChange={(e) => handleChange(schema.name, parseFloat(e.target.value) || 0)}
                    className="w-full bg-gray-700 text-gray-200 text-xs rounded px-2 py-1.5 border border-gray-600 focus:border-blue-500 focus:outline-none"
                  />
                ) : (
                  <div className={isPathParam(schema.name) ? 'flex gap-1' : ''}>
                    <input
                      type="text"
                      value={params[schema.name] ?? schema.default ?? ''}
                      onChange={(e) => handleChange(schema.name, e.target.value)}
                      className={`bg-gray-700 text-gray-200 text-xs rounded px-2 py-1.5 border border-gray-600 focus:border-blue-500 focus:outline-none ${
                        isPathParam(schema.name) ? 'flex-1 min-w-0' : 'w-full'
                      }`}
                    />
                    {isPathParam(schema.name) && (
                      <button
                        type="button"
                        title="미리보기"
                        onClick={() => {
                          const val = params[schema.name] ?? schema.default ?? '';
                          if (val) setPreviewPath(val);
                        }}
                        className="shrink-0 w-7 h-7 flex items-center justify-center rounded bg-gray-700 border border-gray-600 text-gray-400 hover:text-blue-400 hover:border-blue-500 hover:bg-blue-500/10 transition-colors text-xs"
                      >
                        🔍
                      </button>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* 전체 스텝 네비게이션 */}
          {activeExample && (
            <div className="border-t border-gray-700 p-3">
              <h4 className="text-[10px] font-bold text-gray-500 uppercase mb-2">
                All Steps
              </h4>
              <div className="space-y-1">
                {activeExample.guide_steps.map((gs) => (
                  <button
                    key={gs.step}
                    onClick={() => setSelectedNodeId(gs.node)}
                    className={`w-full text-left flex items-center gap-1.5 px-1.5 py-1 rounded text-[10px] transition-colors ${
                      gs.node === selectedNodeId
                        ? 'bg-blue-500/20 text-blue-300'
                        : 'text-gray-500 hover:text-gray-300 hover:bg-gray-700/50'
                    }`}
                  >
                    <span
                      className={`w-3.5 h-3.5 rounded-full flex items-center justify-center text-[8px] font-bold ${
                        gs.node === selectedNodeId
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-600 text-gray-400'
                      }`}
                    >
                      {gs.step}
                    </span>
                    <span className="truncate">{gs.node.replace(/_\d+$/, '')}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      ) : !activeExample ? (
        <div className="p-3">
          <p className="text-xs text-gray-500">Select a node to edit properties</p>
          <p className="text-[10px] text-gray-600 mt-2">
            Or load an example from the Examples tab
          </p>
        </div>
      ) : null}

      {/* 파일 미리보기 모달 */}
      {previewPath && (
        <FilePreviewModal
          filePath={previewPath}
          onClose={() => setPreviewPath(null)}
        />
      )}
    </div>
  );
}

// ── 예제 개요 서브컴포넌트 ──────────────────────────
function ExampleOverview({
  example,
  onStepClick,
}: {
  example: NonNullable<ReturnType<typeof useStore.getState>['activeExample']>;
  onStepClick: (nodeId: string) => void;
}) {
  const [showPipeline, setShowPipeline] = useState(false);

  return (
    <div className="flex flex-col">
      {/* 헤더 */}
      <div className="p-3 border-b border-gray-700">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] text-gray-500">Example #{example.id}</span>
          <span
            className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
            style={{
              backgroundColor: (DIFFICULTY_COLORS[example.difficulty] || '#6b7280') + '22',
              color: DIFFICULTY_COLORS[example.difficulty] || '#6b7280',
            }}
          >
            {example.difficulty}
          </span>
        </div>
        <h3 className="text-sm font-bold text-white">{example.title_ko}</h3>
        <p className="text-[10px] text-gray-500">{example.title}</p>
      </div>

      {/* 설명 */}
      <div className="p-3 border-b border-gray-700">
        <p className="text-[11px] text-gray-300 leading-relaxed">
          {example.description_ko}
        </p>
        {/* Pipeline 보기 버튼 */}
        <button
          type="button"
          onClick={() => setShowPipeline(true)}
          className="mt-2 w-full py-1.5 rounded-lg bg-purple-500/15 border border-purple-500/30 text-purple-300 text-[11px] font-medium hover:bg-purple-500/25 hover:border-purple-500/50 transition-colors flex items-center justify-center gap-1.5"
        >
          <span>🔗</span>
          Pipeline 구조 보기
        </button>
      </div>

      {/* Pipeline 구조 대화상자 */}
      {showPipeline && (
        <PipelineOverviewDialog onClose={() => setShowPipeline(false)} />
      )}

      {/* 데이터셋 정보 */}
      {example.dataset_info && (
        <div className="p-3 border-b border-gray-700">
          <h4 className="text-[10px] font-bold text-gray-500 uppercase mb-2">Dataset</h4>
          <div className="grid grid-cols-2 gap-1.5 text-[10px]">
            <div className="bg-gray-700/50 rounded px-2 py-1">
              <div className="text-gray-500">Name</div>
              <div className="text-gray-300">{example.dataset_info.name}</div>
            </div>
            <div className="bg-gray-700/50 rounded px-2 py-1">
              <div className="text-gray-500">Source</div>
              <div className="text-gray-300">{example.dataset_info.source}</div>
            </div>
            <div className="bg-gray-700/50 rounded px-2 py-1">
              <div className="text-gray-500">Samples</div>
              <div className="text-gray-300">{example.dataset_info.samples.toLocaleString()}</div>
            </div>
            <div className="bg-gray-700/50 rounded px-2 py-1">
              <div className="text-gray-500">Features</div>
              <div className="text-gray-300">{example.dataset_info.features}</div>
            </div>
            {example.dataset_info.classes && (
              <div className="bg-gray-700/50 rounded px-2 py-1 col-span-2">
                <div className="text-gray-500">Classes</div>
                <div className="text-gray-300">{example.dataset_info.classes}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 학습 목표 */}
      <div className="p-3 border-b border-gray-700">
        <h4 className="text-[10px] font-bold text-gray-500 uppercase mb-2">Learning Objectives</h4>
        <ul className="space-y-1">
          {example.learning_objectives.map((obj, i) => (
            <li key={i} className="text-[11px] text-gray-300 flex gap-1.5">
              <span className="text-green-500 mt-0.5 shrink-0">&#10003;</span>
              <span>{obj}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* 가이드 스텝 */}
      <div className="p-3">
        <h4 className="text-[10px] font-bold text-gray-500 uppercase mb-2">Step-by-Step Guide</h4>
        <div className="space-y-2">
          {example.guide_steps.map((gs) => (
            <button
              key={gs.step}
              onClick={() => onStepClick(gs.node)}
              className="w-full text-left group"
            >
              <div className="flex gap-2">
                <span className="text-[10px] bg-blue-500 text-white w-5 h-5 rounded-full flex items-center justify-center font-bold shrink-0 mt-0.5 group-hover:bg-blue-400">
                  {gs.step}
                </span>
                <div>
                  <div className="text-[10px] font-semibold text-blue-400 group-hover:text-blue-300">
                    {gs.node.replace(/_\d+$/, '')}
                  </div>
                  <div className="text-[10px] text-gray-400 leading-relaxed group-hover:text-gray-300">
                    {gs.text}
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* 실행 안내 */}
        <div className="mt-4 p-2 bg-green-500/10 border border-green-500/30 rounded-lg">
          <p className="text-[11px] text-green-300 text-center font-medium">
            Run 버튼을 클릭하여 파이프라인을 실행하세요!
          </p>
        </div>
        <div className="mt-2 text-center text-[10px] text-gray-500">
          예상 소요 시간: {example.estimated_time}
        </div>
      </div>
    </div>
  );
}

// ── 카테고리 색상 & 아이콘 ────────────────────────────
const CATEGORY_STYLES: Record<string, { icon: string; color: string; bg: string; border: string }> = {
  data: { icon: '📂', color: 'text-emerald-300', bg: 'bg-emerald-500/15', border: 'border-emerald-500/30' },
  layer: { icon: '🧱', color: 'text-blue-300', bg: 'bg-blue-500/15', border: 'border-blue-500/30' },
  training: { icon: '⚙️', color: 'text-orange-300', bg: 'bg-orange-500/15', border: 'border-orange-500/30' },
  visualization: { icon: '📈', color: 'text-purple-300', bg: 'bg-purple-500/15', border: 'border-purple-500/30' },
  evaluation: { icon: '📊', color: 'text-pink-300', bg: 'bg-pink-500/15', border: 'border-pink-500/30' },
  output: { icon: '💾', color: 'text-yellow-300', bg: 'bg-yellow-500/15', border: 'border-yellow-500/30' },
};

// ── Pipeline 구조 대화상자 ─────────────────────────────
function PipelineOverviewDialog({ onClose }: { onClose: () => void }) {
  const nodes = useStore((s) => s.nodes);
  const edges = useStore((s) => s.edges);
  const catalog = useStore((s) => s.catalog);

  // 노드를 카테고리별로 그룹화하고 실행 순서(토폴로지 정렬)를 계산
  const { stages, connections } = useMemo(() => {
    // 카테고리별 그룹
    const catOrder = ['data', 'layer', 'training', 'visualization', 'evaluation', 'output'];
    const grouped: Record<string, typeof nodes> = {};
    for (const node of nodes) {
      const cat = (node.data.category as string) || 'data';
      if (!grouped[cat]) grouped[cat] = [];
      grouped[cat].push(node);
    }

    const stages = catOrder
      .filter((cat) => grouped[cat]?.length)
      .map((cat) => ({
        category: cat,
        nodes: grouped[cat],
      }));

    // 연결 요약: source 노드 → target 노드 (중복 제거)
    const connSet = new Set<string>();
    const conns: { from: string; to: string; port: string }[] = [];
    for (const edge of edges) {
      const key = `${edge.source}->${edge.target}`;
      if (!connSet.has(key)) {
        connSet.add(key);
        conns.push({
          from: edge.source,
          to: edge.target,
          port: (edge.sourceHandle as string) || '',
        });
      }
    }

    return { stages, connections: conns };
  }, [nodes, edges]);

  const getNodeType = (nodeId: string) => {
    const node = nodes.find((n) => n.id === nodeId);
    return node ? (node.data.nodeType as string) : nodeId;
  };

  const getNodeDesc = (nodeType: string) => {
    const desc = NODE_DESCRIPTIONS[nodeType];
    return desc?.desc_ko || catalog.find((c) => c.type === nodeType)?.description || '';
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-gray-800 border border-gray-600 rounded-xl shadow-2xl w-[780px] max-h-[85vh] flex flex-col overflow-hidden">
        {/* 헤더 */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-700">
          <div className="flex items-center gap-2.5">
            <span className="text-xl">🔗</span>
            <h3 className="text-lg font-bold text-white">Pipeline 구조</h3>
            <span className="text-sm text-gray-500">
              {nodes.length}개 노드 / {edges.length}개 연결
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-xl leading-none px-1"
          >
            ✕
          </button>
        </div>

        {/* 내용 */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* 단계별 노드 */}
          {stages.map((stage, si) => {
            const style = CATEGORY_STYLES[stage.category] || CATEGORY_STYLES.data;
            return (
              <div key={stage.category}>
                {/* 카테고리 헤더 */}
                <div className="flex items-center gap-2 mb-2.5">
                  <span className="text-base">{style.icon}</span>
                  <h4 className={`text-base font-bold uppercase ${style.color}`}>
                    {stage.category}
                  </h4>
                  <span className="text-xs text-gray-600">({stage.nodes.length})</span>
                </div>

                {/* 노드 카드들 */}
                <div className="space-y-2 ml-5">
                  {stage.nodes.map((node) => {
                    const nodeType = node.data.nodeType as string;
                    const params = node.data.params as Record<string, any> || {};
                    const desc = getNodeDesc(nodeType);
                    const paramKeys = Object.entries(params).filter(
                      ([, v]) => v !== '' && v !== null && v !== undefined
                    );

                    return (
                      <div
                        key={node.id}
                        className={`rounded-lg px-4 py-2.5 ${style.bg} border ${style.border}`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className={`text-sm font-bold ${style.color}`}>{nodeType}</span>
                            <span className="text-xs text-gray-500">{node.id}</span>
                          </div>
                        </div>
                        {desc && (
                          <p className="text-sm text-gray-400 mt-0.5">{desc}</p>
                        )}
                        {paramKeys.length > 0 && (
                          <div className="flex flex-wrap gap-1.5 mt-1.5">
                            {paramKeys.map(([k, v]) => (
                              <span
                                key={k}
                                className="text-xs px-2 py-0.5 rounded bg-gray-700/60 text-gray-400"
                              >
                                {k}=<span className="text-gray-300">{String(v)}</span>
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* 단계 간 화살표 */}
                {si < stages.length - 1 && (
                  <div className="flex justify-center py-1.5">
                    <span className="text-gray-600 text-sm">↓</span>
                  </div>
                )}
              </div>
            );
          })}

          {/* 데이터 흐름 */}
          {connections.length > 0 && (
            <div className="border-t border-gray-700 pt-4">
              <h4 className="text-base font-bold text-gray-500 uppercase mb-2.5">
                데이터 흐름 ({connections.length}개 연결)
              </h4>
              <div className="space-y-1.5">
                {connections.map((conn, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span className="text-gray-300 font-mono">{getNodeType(conn.from)}</span>
                    <span className="text-gray-600">→</span>
                    <span className="text-gray-300 font-mono">{getNodeType(conn.to)}</span>
                    {conn.port && (
                      <span className="text-xs text-gray-600 ml-1">({conn.port})</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
