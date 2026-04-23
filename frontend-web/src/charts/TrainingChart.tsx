import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useStore } from '../store/useStore';

export function LossChart() {
  const epochs = useStore((s) => s.training.epochs);

  if (epochs.length === 0) return null;

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h3 className="text-sm font-semibold text-gray-300 mb-2">Loss</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={epochs}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="epoch" stroke="#9ca3af" fontSize={11} />
          <YAxis stroke="#9ca3af" fontSize={11} />
          <Tooltip
            contentStyle={{ background: '#1f2937', border: '1px solid #374151' }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            name="Train Loss"
          />
          <Line
            type="monotone"
            dataKey="val_loss"
            stroke="#ef4444"
            strokeWidth={2}
            dot={false}
            name="Val Loss"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function AccuracyChart() {
  const epochs = useStore((s) => s.training.epochs);

  if (epochs.length === 0) return null;

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h3 className="text-sm font-semibold text-gray-300 mb-2">Accuracy</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={epochs}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="epoch" stroke="#9ca3af" fontSize={11} />
          <YAxis stroke="#9ca3af" fontSize={11} domain={[0, 1]} />
          <Tooltip
            contentStyle={{ background: '#1f2937', border: '1px solid #374151' }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            name="Train Acc"
          />
          <Line
            type="monotone"
            dataKey="val_accuracy"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
            name="Val Acc"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
