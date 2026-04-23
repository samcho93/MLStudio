import React from 'react';
import { useStore, type TestResults } from '../store/useStore';

export function TestResultsPanel() {
  const testResults = useStore((s) => s.testResults);
  const setTestResults = useStore((s) => s.setTestResults);

  if (!testResults) return null;

  const { metrics, samples, num_predictions, prediction_summary } = testResults;
  const isRegression = metrics?.type === 'regression';

  return (
    <div className="border-t border-gray-700">
      <div className="p-3 border-b border-gray-700 flex items-center justify-between">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          {'🧪'} Test Results
          <span className="text-[10px] text-gray-500 font-normal">
            ({num_predictions} predictions)
          </span>
        </h3>
        <button
          onClick={() => setTestResults(null)}
          className="text-gray-500 hover:text-white text-xs"
        >
          x
        </button>
      </div>

      {/* Metrics */}
      <div className="p-3">
        {isRegression ? (
          <RegressionMetrics metrics={metrics} summary={prediction_summary} />
        ) : (
          <ClassificationMetrics metrics={metrics} />
        )}
      </div>

      {/* Sample Predictions */}
      {samples && samples.length > 0 && (
        <div className="px-3 pb-3">
          <h4 className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-1">
            Sample Predictions
          </h4>
          <div className="max-h-48 overflow-y-auto rounded border border-gray-700">
            <table className="w-full text-[10px]">
              <thead className="bg-gray-700 sticky top-0">
                <tr>
                  <th className="px-2 py-1 text-left text-gray-400">#</th>
                  <th className="px-2 py-1 text-left text-gray-400">Predicted</th>
                  {samples[0]?.true_label !== undefined && (
                    <th className="px-2 py-1 text-left text-gray-400">True</th>
                  )}
                  {samples[0]?.confidence !== undefined && (
                    <th className="px-2 py-1 text-left text-gray-400">Conf.</th>
                  )}
                  <th className="px-2 py-1 text-center text-gray-400">Match</th>
                </tr>
              </thead>
              <tbody>
                {samples.map((s, i) => {
                  const pred = s.predicted_class ?? s.predicted_value;
                  const trueVal = s.true_label;
                  const isMatch = trueVal !== undefined && pred !== undefined &&
                    (typeof pred === 'number' && typeof trueVal === 'number'
                      ? Math.abs(pred - trueVal) < 0.001
                      : String(pred) === String(trueVal));

                  return (
                    <tr
                      key={i}
                      className={`border-t border-gray-700/50 ${
                        isMatch ? 'bg-green-900/10' : trueVal !== undefined ? 'bg-red-900/10' : ''
                      }`}
                    >
                      <td className="px-2 py-1 text-gray-500">{s.index}</td>
                      <td className="px-2 py-1 text-white font-mono">
                        {typeof pred === 'number' && !Number.isInteger(pred)
                          ? pred.toFixed(4)
                          : pred}
                      </td>
                      {trueVal !== undefined && (
                        <td className="px-2 py-1 text-gray-300 font-mono">
                          {typeof trueVal === 'number' && !Number.isInteger(trueVal)
                            ? trueVal.toFixed(4)
                            : trueVal}
                        </td>
                      )}
                      {s.confidence !== undefined && (
                        <td className="px-2 py-1 text-gray-400 font-mono">
                          {(s.confidence * 100).toFixed(1)}%
                        </td>
                      )}
                      <td className="px-2 py-1 text-center">
                        {trueVal !== undefined ? (
                          isMatch ? (
                            <span className="text-green-400">O</span>
                          ) : (
                            <span className="text-red-400">X</span>
                          )
                        ) : (
                          <span className="text-gray-600">-</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Confusion Matrix */}
      {metrics?.confusion_matrix && (
        <div className="px-3 pb-3">
          <h4 className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-1">
            Confusion Matrix
          </h4>
          <ConfusionMatrixGrid matrix={metrics.confusion_matrix} />
        </div>
      )}
    </div>
  );
}

function ClassificationMetrics({ metrics }: { metrics: Record<string, any> }) {
  const accuracy = metrics?.accuracy;
  const report = metrics?.classification_report;

  return (
    <div className="space-y-2">
      {accuracy !== undefined && (
        <div className="flex items-center gap-3">
          <div className="bg-gray-700 rounded-lg p-2.5 flex-1">
            <div className="text-[10px] text-gray-500 uppercase">Accuracy</div>
            <div className={`text-lg font-bold font-mono ${
              accuracy > 0.9 ? 'text-green-400' : accuracy > 0.7 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {(accuracy * 100).toFixed(2)}%
            </div>
          </div>
          {report?.['weighted avg'] && (
            <>
              <div className="bg-gray-700 rounded-lg p-2.5 flex-1">
                <div className="text-[10px] text-gray-500 uppercase">Precision</div>
                <div className="text-sm font-bold font-mono text-blue-400">
                  {(report['weighted avg'].precision * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-gray-700 rounded-lg p-2.5 flex-1">
                <div className="text-[10px] text-gray-500 uppercase">Recall</div>
                <div className="text-sm font-bold font-mono text-purple-400">
                  {(report['weighted avg'].recall * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-gray-700 rounded-lg p-2.5 flex-1">
                <div className="text-[10px] text-gray-500 uppercase">F1</div>
                <div className="text-sm font-bold font-mono text-cyan-400">
                  {(report['weighted avg']['f1-score'] * 100).toFixed(1)}%
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function RegressionMetrics({
  metrics,
  summary,
}: {
  metrics: Record<string, any>;
  summary?: { mean: number; std: number; min: number; max: number };
}) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {metrics?.mse !== undefined && (
        <div className="bg-gray-700 rounded-lg p-2">
          <div className="text-[10px] text-gray-500 uppercase">MSE</div>
          <div className="text-sm font-bold font-mono text-yellow-400">
            {metrics.mse.toFixed(4)}
          </div>
        </div>
      )}
      {metrics?.rmse !== undefined && (
        <div className="bg-gray-700 rounded-lg p-2">
          <div className="text-[10px] text-gray-500 uppercase">RMSE</div>
          <div className="text-sm font-bold font-mono text-orange-400">
            {metrics.rmse.toFixed(4)}
          </div>
        </div>
      )}
      {metrics?.mae !== undefined && (
        <div className="bg-gray-700 rounded-lg p-2">
          <div className="text-[10px] text-gray-500 uppercase">MAE</div>
          <div className="text-sm font-bold font-mono text-blue-400">
            {metrics.mae.toFixed(4)}
          </div>
        </div>
      )}
      {metrics?.r2 !== undefined && (
        <div className="bg-gray-700 rounded-lg p-2">
          <div className="text-[10px] text-gray-500 uppercase">R2 Score</div>
          <div className={`text-sm font-bold font-mono ${
            metrics.r2 > 0.8 ? 'text-green-400' : metrics.r2 > 0.5 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {metrics.r2.toFixed(4)}
          </div>
        </div>
      )}
    </div>
  );
}

function ConfusionMatrixGrid({ matrix }: { matrix: number[][] }) {
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="overflow-x-auto">
      <div className="inline-grid gap-px bg-gray-600 rounded overflow-hidden" style={{
        gridTemplateColumns: `repeat(${matrix[0].length}, minmax(28px, 1fr))`,
      }}>
        {matrix.map((row, i) =>
          row.map((val, j) => {
            const intensity = maxVal > 0 ? val / maxVal : 0;
            const isDiag = i === j;
            return (
              <div
                key={`${i}-${j}`}
                className="text-center py-1 text-[10px] font-mono"
                style={{
                  backgroundColor: isDiag
                    ? `rgba(34, 197, 94, ${0.1 + intensity * 0.5})`
                    : `rgba(239, 68, 68, ${intensity * 0.3})`,
                  color: val > 0 ? 'white' : '#6b7280',
                }}
              >
                {val}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
