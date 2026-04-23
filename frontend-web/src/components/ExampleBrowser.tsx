import React, { useState } from 'react';
import { useReactFlow } from '@xyflow/react';
import { useStore } from '../store/useStore';

const CATEGORY_LABELS: Record<string, string> = {
  tutorial: '튜토리얼',
  classification: '분류',
  regression: '회귀',
  unsupervised: '비지도',
  image: '이미지',
  nlp: 'NLP',
  timeseries: '시계열',
  advanced: '고급',
};

const DIFFICULTY_COLORS: Record<string, string> = {
  beginner: '#10b981',
  intermediate: '#f59e0b',
  advanced: '#ef4444',
};

const DIFFICULTY_LABELS: Record<string, string> = {
  beginner: '초급',
  intermediate: '중급',
  advanced: '고급',
};

export function ExampleBrowser() {
  const examples = useStore((s) => s.examples);
  const loadExample = useStore((s) => s.loadExample);
  const { fitView } = useReactFlow();
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [loading, setLoading] = useState(false);

  const filtered =
    selectedCategory === 'all'
      ? examples
      : examples.filter((e) => e.category === selectedCategory);

  const categories = ['all', ...new Set(examples.map((e) => e.category))];

  const handleLoad = async (id: number) => {
    setLoading(true);
    try {
      await loadExample(id);
      setTimeout(() => fitView({ padding: 0.1, duration: 300 }), 100);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* 카테고리 필터 */}
      <div className="px-2 py-2 flex flex-wrap gap-1">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-2 py-0.5 text-[10px] rounded-full font-medium transition-colors ${
              selectedCategory === cat
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
          >
            {cat === 'all' ? '전체' : CATEGORY_LABELS[cat] || cat}
          </button>
        ))}
      </div>

      {/* 예제 목록 */}
      <div className="flex-1 overflow-y-auto px-2 space-y-1.5 pb-2">
        {loading && (
          <div className="text-center text-xs text-gray-500 py-4">
            Loading...
          </div>
        )}

        {filtered.map((ex) => (
          <button
            key={ex.id}
            onClick={() => handleLoad(ex.id)}
            disabled={loading}
            className="w-full text-left bg-gray-700/50 hover:bg-gray-700 rounded-lg p-2.5 transition-colors border border-transparent hover:border-gray-600 disabled:opacity-50"
          >
            {/* 헤더 */}
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] font-bold text-gray-500">
                #{ex.id}
              </span>
              <span
                className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
                style={{
                  backgroundColor: (DIFFICULTY_COLORS[ex.difficulty] || '#6b7280') + '22',
                  color: DIFFICULTY_COLORS[ex.difficulty] || '#6b7280',
                }}
              >
                {DIFFICULTY_LABELS[ex.difficulty] || ex.difficulty}
              </span>
            </div>

            {/* 제목 */}
            <div className="text-xs font-semibold text-gray-200 mb-0.5">
              {ex.title_ko}
            </div>
            <div className="text-[10px] text-gray-500 mb-1.5">
              {ex.title}
            </div>

            {/* 태그 + 시간 */}
            <div className="flex items-center justify-between">
              <div className="flex flex-wrap gap-1">
                {ex.tags.slice(0, 3).map((tag) => (
                  <span
                    key={tag}
                    className="text-[9px] bg-gray-600/50 text-gray-400 px-1 py-0.5 rounded"
                  >
                    {tag}
                  </span>
                ))}
              </div>
              <span className="text-[9px] text-gray-500">{ex.estimated_time}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
