import React from 'react';
import { useStore } from '../store/useStore';
import { LossChart, AccuracyChart } from '../charts/TrainingChart';
import { PredictionPanel } from './PredictionPanel';
import { TestResultsPanel } from './TestResultsPanel';

export function TrainingPanel() {
  const training = useStore((s) => s.training);
  const visualizations = useStore((s) => s.visualizations);
  const testResults = useStore((s) => s.testResults);

  if (training.epochs.length === 0 && Object.keys(visualizations).length === 0 && !testResults) {
    return null;
  }

  const lastEpoch = training.epochs[training.epochs.length - 1];

  return (
    <div className="w-80 bg-gray-800 border-l border-gray-700 overflow-y-auto flex flex-col">
      <div className="p-3 border-b border-gray-700">
        <h3 className="text-sm font-bold text-white">Training Monitor</h3>
      </div>

      {/* 현재 메트릭 */}
      {lastEpoch && (
        <div className="p-3 grid grid-cols-2 gap-2 text-xs">
          <div className="bg-gray-700 rounded p-2">
            <div className="text-gray-500">Loss</div>
            <div className="text-yellow-400 font-mono">{lastEpoch.loss.toFixed(4)}</div>
          </div>
          <div className="bg-gray-700 rounded p-2">
            <div className="text-gray-500">Val Loss</div>
            <div className="text-red-400 font-mono">{lastEpoch.val_loss.toFixed(4)}</div>
          </div>
          <div className="bg-gray-700 rounded p-2">
            <div className="text-gray-500">Accuracy</div>
            <div className="text-blue-400 font-mono">{(lastEpoch.accuracy * 100).toFixed(2)}%</div>
          </div>
          <div className="bg-gray-700 rounded p-2">
            <div className="text-gray-500">Val Acc</div>
            <div className="text-green-400 font-mono">{(lastEpoch.val_accuracy * 100).toFixed(2)}%</div>
          </div>
          <div className="bg-gray-700 rounded p-2 col-span-2">
            <div className="text-gray-500">Elapsed</div>
            <div className="text-gray-300 font-mono">{lastEpoch.elapsed_sec}s</div>
          </div>
        </div>
      )}

      {/* 차트 */}
      <div className="p-3 space-y-3">
        <LossChart />
        <AccuracyChart />
      </div>

      {/* 테스트/예측 패널 */}
      {!training.isTraining && training.modelPath && <PredictionPanel />}

      {/* 테스트 결과 패널 */}
      <TestResultsPanel />

      {/* 시각화 이미지 */}
      {Object.entries(visualizations).map(([type, b64]) => (
        <div key={type} className="p-3">
          <h4 className="text-xs text-gray-400 mb-1">{type}</h4>
          <img
            src={`data:image/png;base64,${b64}`}
            alt={type}
            className="w-full rounded"
          />
        </div>
      ))}
    </div>
  );
}
