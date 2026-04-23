import React from 'react';
import { useStore, type NodeCatalogItem } from '../store/useStore';
import { ExampleBrowser } from './ExampleBrowser';

const CATEGORY_ORDER = ['data', 'layer', 'training', 'visualization', 'evaluation', 'output'];
const CATEGORY_LABELS: Record<string, string> = {
  data: 'Data',
  layer: 'Layer',
  training: 'Training',
  visualization: 'Visualization',
  evaluation: 'Evaluation',
  output: 'Output',
};
const CATEGORY_ICONS: Record<string, string> = {
  data: '\u{1F4C2}',
  layer: '\u{1F9F1}',
  training: '\u2699\uFE0F',
  visualization: '\u{1F4C8}',
  evaluation: '\u{1F4CA}',
  output: '\u{1F4BE}',
};

let nodeCounter = 0;

export function Sidebar() {
  const catalog = useStore((s) => s.catalog);
  const addNode = useStore((s) => s.addNode);
  const pushSnapshot = useStore((s) => s.pushSnapshot);
  const sidebarTab = useStore((s) => s.sidebarTab);
  const setSidebarTab = useStore((s) => s.setSidebarTab);

  const grouped = CATEGORY_ORDER.reduce(
    (acc, cat) => {
      acc[cat] = catalog.filter((item) => item.category === cat);
      return acc;
    },
    {} as Record<string, NodeCatalogItem[]>
  );

  const handleAddNode = (item: NodeCatalogItem) => {
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
      position: { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 },
      data: {
        label: item.type,
        category: item.category,
        params: defaults,
        inputs: item.inputs,
        outputs: item.outputs,
        nodeType: item.type,
      },
    });
  };

  const handleDragStart = (e: React.DragEvent, item: NodeCatalogItem) => {
    e.dataTransfer.setData('application/ml-node', JSON.stringify(item));
    e.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="w-60 bg-gray-800 border-r border-gray-700 flex flex-col overflow-hidden">
      {/* 탭 헤더 */}
      <div className="flex border-b border-gray-700">
        <button
          onClick={() => setSidebarTab('nodes')}
          className={`flex-1 py-2 text-xs font-bold transition-colors ${
            sidebarTab === 'nodes'
              ? 'text-white border-b-2 border-blue-500 bg-gray-800'
              : 'text-gray-500 hover:text-gray-300 bg-gray-800/50'
          }`}
        >
          Nodes
        </button>
        <button
          onClick={() => setSidebarTab('examples')}
          className={`flex-1 py-2 text-xs font-bold transition-colors ${
            sidebarTab === 'examples'
              ? 'text-white border-b-2 border-green-500 bg-gray-800'
              : 'text-gray-500 hover:text-gray-300 bg-gray-800/50'
          }`}
        >
          Examples
        </button>
      </div>

      {/* 탭 콘텐츠 */}
      {sidebarTab === 'nodes' ? (
        <div className="flex-1 overflow-y-auto">
          {CATEGORY_ORDER.map((cat) => {
            const items = grouped[cat];
            if (!items || items.length === 0) return null;
            return (
              <div key={cat} className="px-2 py-1">
                <div className="text-[10px] font-bold text-gray-500 uppercase tracking-wider px-1 py-1">
                  {CATEGORY_ICONS[cat]} {CATEGORY_LABELS[cat]}
                </div>
                {items.map((item) => (
                  <button
                    key={item.type}
                    onClick={() => handleAddNode(item)}
                    draggable
                    onDragStart={(e) => handleDragStart(e, item)}
                    className="w-full text-left px-2 py-1.5 text-xs text-gray-300 hover:bg-gray-700 rounded transition-colors cursor-grab active:cursor-grabbing"
                    title={item.description}
                  >
                    {item.type}
                  </button>
                ))}
              </div>
            );
          })}
        </div>
      ) : (
        <ExampleBrowser />
      )}
    </div>
  );
}
