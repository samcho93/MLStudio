import React, { useState, useEffect } from 'react';
import { useStore } from '../store/useStore';

interface ModelInfo {
  ready: boolean;
  feature_names: string[] | null;
  class_names: string[] | null;
  is_regression: boolean;
  is_nlp: boolean;
  input_shape: number[] | null;
}

interface PredictionResult {
  type: 'classification' | 'regression';
  predicted_class?: number;
  predicted_label?: string;
  confidences?: number[];
  class_names?: string[];
  prediction?: number;
  error?: string;
}

export function PredictionPanel() {
  const training = useStore((s) => s.training);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [inputValues, setInputValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 모델 정보 가져오기
  useEffect(() => {
    if (!training.isTraining && training.modelPath) {
      fetch('/api/model-info')
        .then((r) => r.json())
        .then((info) => {
          if (info.ready) {
            setModelInfo(info);
            // 초기 입력값 설정
            if (info.feature_names) {
              const initial: Record<string, string> = {};
              info.feature_names.forEach((name: string) => {
                initial[name] = '0';
              });
              setInputValues(initial);
            }
          }
        })
        .catch(() => {});
    }
  }, [training.isTraining, training.modelPath]);

  if (!modelInfo || !modelInfo.ready) return null;
  if (modelInfo.is_nlp) return null; // NLP는 별도 UI 필요

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const values = modelInfo.feature_names
        ? modelInfo.feature_names.map((name) => parseFloat(inputValues[name] || '0'))
        : Object.values(inputValues).map((v) => parseFloat(v || '0'));

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: values }),
      });

      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (e: any) {
      setError(e.message || '예측 실패');
    } finally {
      setLoading(false);
    }
  };

  const handleRandom = () => {
    if (modelInfo.feature_names) {
      const random: Record<string, string> = {};
      modelInfo.feature_names.forEach((name) => {
        random[name] = (Math.random() * 10 - 2).toFixed(2);
      });
      setInputValues(random);
    }
  };

  return (
    <div className="border-t border-gray-700 bg-gray-850">
      <div className="p-3 border-b border-gray-700 flex items-center justify-between">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          🧪 Test / Predict
        </h3>
        <div className="flex gap-1">
          <button
            onClick={handleRandom}
            className="px-2 py-0.5 text-[10px] rounded bg-gray-600 hover:bg-gray-500 text-gray-300"
          >
            Random
          </button>
          <button
            onClick={handlePredict}
            disabled={loading}
            className="px-2 py-0.5 text-[10px] rounded bg-blue-600 hover:bg-blue-500 text-white font-medium disabled:opacity-50"
          >
            {loading ? '...' : 'Predict'}
          </button>
        </div>
      </div>

      {/* 입력 필드 */}
      <div className="p-2 max-h-48 overflow-y-auto">
        {modelInfo.feature_names ? (
          <div className="grid grid-cols-2 gap-1">
            {modelInfo.feature_names.map((name) => (
              <div key={name} className="flex items-center gap-1">
                <label className="text-[10px] text-gray-400 truncate w-20" title={name}>
                  {name}
                </label>
                <input
                  type="number"
                  step="any"
                  value={inputValues[name] || '0'}
                  onChange={(e) =>
                    setInputValues((prev) => ({ ...prev, [name]: e.target.value }))
                  }
                  className="flex-1 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-[10px] text-white w-0 min-w-0"
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[10px] text-gray-400">
            Input shape: {JSON.stringify(modelInfo.input_shape)}
            <div className="mt-1 grid grid-cols-4 gap-1">
              {Array.from({ length: modelInfo.input_shape?.[0] || 4 }, (_, i) => (
                <input
                  key={i}
                  type="number"
                  step="any"
                  value={inputValues[String(i)] || '0'}
                  onChange={(e) =>
                    setInputValues((prev) => ({ ...prev, [String(i)]: e.target.value }))
                  }
                  className="bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-[10px] text-white"
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 에러 */}
      {error && (
        <div className="px-3 pb-2">
          <div className="text-[10px] text-red-400 bg-red-900/30 rounded p-1.5">
            {error}
          </div>
        </div>
      )}

      {/* 결과 */}
      {result && (
        <div className="px-3 pb-3">
          {result.type === 'classification' ? (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">🎯</span>
                <span className="text-sm font-bold text-green-400">
                  {result.predicted_label}
                </span>
              </div>
              {/* 확률 바 */}
              {result.confidences && result.confidences.length <= 20 && (
                <div className="space-y-0.5">
                  {result.confidences.map((conf, i) => {
                    const label = result.class_names?.[i] || String(i);
                    const isTop = i === result.predicted_class;
                    return (
                      <div key={i} className="flex items-center gap-1">
                        <span
                          className={`text-[9px] w-16 truncate text-right ${
                            isTop ? 'text-green-400 font-bold' : 'text-gray-400'
                          }`}
                          title={label}
                        >
                          {label}
                        </span>
                        <div className="flex-1 bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${isTop ? 'bg-green-500' : 'bg-blue-500/50'}`}
                            style={{ width: `${Math.max(conf * 100, 1)}%` }}
                          />
                        </div>
                        <span className={`text-[9px] w-10 ${isTop ? 'text-green-400' : 'text-gray-500'}`}>
                          {(conf * 100).toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-lg">📊</span>
              <span className="text-sm text-gray-400">Prediction:</span>
              <span className="text-lg font-bold text-blue-400">
                {result.prediction?.toFixed(4)}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
