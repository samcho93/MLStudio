import React, { useEffect, useState } from 'react';
import { useStore, type SavedModelInfo } from '../store/useStore';

export function ModelManagerPanel() {
  const modelManagerOpen = useStore((s) => s.modelManagerOpen);
  const setModelManagerOpen = useStore((s) => s.setModelManagerOpen);
  const savedModels = useStore((s) => s.savedModels);
  const fetchSavedModels = useStore((s) => s.fetchSavedModels);
  const saveCurrentModel = useStore((s) => s.saveCurrentModel);
  const loadSavedModel = useStore((s) => s.loadSavedModel);
  const deleteSavedModel = useStore((s) => s.deleteSavedModel);
  const training = useStore((s) => s.training);

  const [saveName, setSaveName] = useState('');
  const [saving, setSaving] = useState(false);
  const [loadingModel, setLoadingModel] = useState<string | null>(null);
  const [tab, setTab] = useState<'save' | 'load'>('load');

  useEffect(() => {
    if (modelManagerOpen) {
      fetchSavedModels();
    }
  }, [modelManagerOpen, fetchSavedModels]);

  if (!modelManagerOpen) return null;

  const canSave = !training.isTraining && training.modelPath;

  const handleSave = async () => {
    if (!saveName.trim()) return;
    setSaving(true);
    await saveCurrentModel(saveName.trim());
    setSaving(false);
    setSaveName('');
  };

  const handleLoad = async (name: string) => {
    setLoadingModel(name);
    await loadSavedModel(name);
    setLoadingModel(null);
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete model "${name}"?`)) return;
    await deleteSavedModel(name);
  };

  const formatDate = (iso: string) => {
    if (!iso) return '';
    try {
      const d = new Date(iso);
      return d.toLocaleString();
    } catch {
      return iso;
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl border border-gray-600 shadow-2xl w-[560px] max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <span className="text-lg">{'🗂'}</span> Model Manager
          </h2>
          <button
            onClick={() => setModelManagerOpen(false)}
            className="text-gray-400 hover:text-white text-lg w-7 h-7 flex items-center justify-center rounded hover:bg-gray-700"
          >
            x
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700">
          <button
            onClick={() => setTab('save')}
            className={`flex-1 py-2 text-xs font-bold transition-colors ${
              tab === 'save'
                ? 'text-white border-b-2 border-blue-500'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            Save Model
          </button>
          <button
            onClick={() => setTab('load')}
            className={`flex-1 py-2 text-xs font-bold transition-colors ${
              tab === 'load'
                ? 'text-white border-b-2 border-green-500'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            Saved Models ({savedModels.length})
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {tab === 'save' ? (
            <div>
              {canSave ? (
                <div className="space-y-3">
                  <p className="text-xs text-gray-400">
                    Save the currently trained model to disk. It can be loaded later
                    or used with a ModelLoader node.
                  </p>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={saveName}
                      onChange={(e) => setSaveName(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleSave()}
                      placeholder="Model name (e.g., iris-classifier)"
                      className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
                    />
                    <button
                      onClick={handleSave}
                      disabled={saving || !saveName.trim()}
                      className="px-4 py-2 text-xs rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium"
                    >
                      {saving ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                  <div className="text-[10px] text-gray-500">
                    Saved to: outputs/saved_models/{saveName || '...'}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="text-3xl mb-2">{'🧠'}</div>
                  <p className="text-sm text-gray-400">No trained model in memory</p>
                  <p className="text-xs text-gray-500 mt-1">
                    Train a pipeline first, then come back to save the model.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div>
              {savedModels.length === 0 ? (
                <div className="text-center py-8">
                  <div className="text-3xl mb-2">{'📭'}</div>
                  <p className="text-sm text-gray-400">No saved models yet</p>
                  <p className="text-xs text-gray-500 mt-1">
                    Train a model and save it to see it here.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {savedModels.map((m) => (
                    <ModelCard
                      key={m.name}
                      model={m}
                      loading={loadingModel === m.name}
                      onLoad={() => handleLoad(m.name)}
                      onDelete={() => handleDelete(m.name)}
                      formatDate={formatDate}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ModelCard({
  model,
  loading,
  onLoad,
  onDelete,
  formatDate,
}: {
  model: SavedModelInfo;
  loading: boolean;
  onLoad: () => void;
  onDelete: () => void;
  formatDate: (s: string) => string;
}) {
  const fw = model.framework === 'pytorch' ? 'PyTorch' : 'TensorFlow';
  const fwColor = model.framework === 'pytorch' ? 'text-orange-400' : 'text-blue-400';
  const taskType = model.is_regression ? 'Regression' : 'Classification';
  const taskColor = model.is_regression ? 'bg-purple-600/20 text-purple-400' : 'bg-green-600/20 text-green-400';

  return (
    <div className="bg-gray-700/50 rounded-lg border border-gray-600 p-3 hover:border-gray-500 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-bold text-white truncate">{model.name}</span>
            <span className={`text-[10px] font-mono ${fwColor}`}>{fw}</span>
          </div>
          <div className="flex items-center gap-2 text-[10px]">
            <span className={`px-1.5 py-0.5 rounded ${taskColor}`}>{taskType}</span>
            {model.class_names && (
              <span className="text-gray-500">{model.class_names.length} classes</span>
            )}
            {model.input_shape && (
              <span className="text-gray-500">shape: [{model.input_shape.join(', ')}]</span>
            )}
          </div>
          <div className="text-[10px] text-gray-500 mt-1">{formatDate(model.saved_at)}</div>
        </div>

        <div className="flex gap-1 ml-2">
          <button
            onClick={onLoad}
            disabled={loading}
            className="px-2.5 py-1 text-[10px] rounded bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white font-medium"
          >
            {loading ? '...' : 'Load'}
          </button>
          <button
            onClick={onDelete}
            className="px-2 py-1 text-[10px] rounded bg-red-600/30 hover:bg-red-600 text-red-400 hover:text-white"
          >
            Del
          </button>
        </div>
      </div>
    </div>
  );
}
