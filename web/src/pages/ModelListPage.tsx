import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Trash2, CheckCircle, AlertCircle, Loader2, Plus, PowerOff, FileCog } from 'lucide-react';
import { useAppStore, type ModelInfo } from '@/stores/appStore';
import { fetchModels, downloadModel, deleteModel, unloadModel, convertModel } from '@/utils/api';

export default function ModelListPage() {
  const navigate = useNavigate();
  const { models, setModels, setSelectedModelId, loadedModelId } = useAppStore();
  const [loading, setLoading] = useState(true);
  const [downloadId, setDownloadId] = useState('');
  const [downloadType, setDownloadType] = useState('sd15');
  const [downloading, setDownloading] = useState(false);
  const [showDownload, setShowDownload] = useState(false);
  const [unloadingId, setUnloadingId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  // The backend downloads/converts in the background — poll while anything
  // is still in progress.
  useEffect(() => {
    const busy = models.some((m) => m.status === 'downloading' || m.status === 'converting');
    if (!busy) return;
    const t = setInterval(() => loadModels(true), 5000);
    return () => clearInterval(t);
  }, [models]);

  async function loadModels(silent = false) {
    try {
      if (!silent) setLoading(true);
      const data = await fetchModels();
      if (data?.models) {
        setModels(data.models as ModelInfo[]);
        // Only apply the default when nothing is selected yet — otherwise
        // the 5s status poll would keep resetting a model the user picked.
        if (data.default_model && !useAppStore.getState().selectedModelId) {
          setSelectedModelId(data.default_model);
        }
      }
    } catch {
      if (!silent) setModels([]);
    } finally {
      if (!silent) setLoading(false);
    }
  }

  async function handleDownload() {
    if (!downloadId.trim()) return;
    setDownloading(true);
    setActionError(null);
    try {
      // Returns immediately — actual download runs in the background and
      // the list polls for status updates.
      await downloadModel(downloadId.trim(), downloadType);
      setShowDownload(false);
      setDownloadId('');
      await loadModels();
    } catch (e) {
      setActionError(`下载启动失败：${(e as Error).message}`);
    } finally {
      setDownloading(false);
    }
  }

  async function handleConvert(modelId: string) {
    setActionError(null);
    try {
      // Returns immediately — conversion runs in the background and the
      // list polls for status updates.
      await convertModel(modelId);
      await loadModels(true);
    } catch (e) {
      setActionError(`转换启动失败：${(e as Error).message}`);
    }
  }

  async function handleDelete(modelId: string) {
    if (!window.confirm(`确定删除模型「${modelId}」？该操作不可恢复。`)) return;
    setActionError(null);
    try {
      await deleteModel(modelId);
      await loadModels();
    } catch (e) {
      setActionError(`删除失败：${(e as Error).message}`);
    }
  }

  async function handleUnload(modelId: string) {
    if (unloadingId) return;
    setUnloadingId(modelId);
    try {
      await unloadModel(modelId);
    } catch {
      // handled by UI
    } finally {
      setUnloadingId(null);
    }
  }

  // Single-file checkpoints can generate directly (loaded via from_single_file
  // on the backend); "ready" Diffusers dirs are the fast path.
  const selectable = (status: string) => status === 'ready' || status === 'single_file';

  function handleSelect(model: ModelInfo) {
    if (selectable(model.status)) {
      setSelectedModelId(model.id);
      navigate(`/generate/${encodeURIComponent(model.id)}`);
    }
  }

  const statusIcon = (status: string) => {
    switch (status) {
      case 'ready': return <CheckCircle size={14} className="text-neon-cyan" />;
      case 'downloading': return <Loader2 size={14} className="text-neon-purple animate-spin" />;
      case 'converting': return <Loader2 size={14} className="text-neon-purple animate-spin" />;
      case 'single_file': return <FileCog size={14} className="text-neon-cyan" />;
      case 'error': return <AlertCircle size={14} className="text-red-400" />;
      default: return null;
    }
  };

  const statusText = (status: string) => {
    switch (status) {
      case 'ready': return '就绪';
      case 'downloading': return '下载中';
      case 'converting': return '转换中';
      case 'single_file': return '单文件';
      case 'error': return '错误';
      default: return '未知';
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-display text-2xl font-bold text-white tracking-wider">
            模型管理
          </h1>
          <p className="text-gray-500 text-sm mt-1">管理你的 Stable Diffusion 模型</p>
        </div>
        <button
          onClick={() => setShowDownload(true)}
          className="flex items-center gap-2 px-4 py-2 bg-neon-purple/10 border border-neon-purple/30 rounded-lg text-neon-purple text-sm hover:bg-neon-purple/20 hover:shadow-glow-purple transition-all duration-300"
        >
          <Plus size={16} />
          添加模型
        </button>
      </div>

      {showDownload && (
        <div className="mb-6 p-4 bg-surface-light border border-surface-border rounded-lg animate-slide-up">
          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <label className="block text-xs text-gray-500 mb-1">HuggingFace 模型 ID</label>
              <input
                type="text"
                value={downloadId}
                onChange={(e) => setDownloadId(e.target.value)}
                placeholder="runwayml/stable-diffusion-v1-5"
                className="w-full bg-surface border border-surface-border rounded px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-neon-purple transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">类型</label>
              <select
                value={downloadType}
                onChange={(e) => setDownloadType(e.target.value)}
                className="bg-surface border border-surface-border rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-neon-purple transition-colors"
              >
                <option value="sd15">SD 1.5</option>
                <option value="sdxl">SDXL</option>
              </select>
            </div>
            <button
              onClick={handleDownload}
              disabled={downloading || !downloadId.trim()}
              className="px-4 py-2 bg-neon-purple text-white rounded text-sm hover:bg-neon-purple-glow disabled:opacity-50 transition-all duration-300"
            >
              {downloading ? '下载中...' : '下载'}
            </button>
            <button
              onClick={() => setShowDownload(false)}
              className="px-4 py-2 text-gray-500 hover:text-white text-sm transition-colors"
            >
              取消
            </button>
          </div>
        </div>
      )}

      {actionError && (
        <div className="mb-6 p-3 bg-red-500/10 border border-red-500/30 rounded-lg animate-slide-up">
          <p className="text-xs text-red-400">{actionError}</p>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 size={32} className="text-neon-purple animate-spin" />
        </div>
      ) : models.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 text-gray-500">
          <BoxIcon />
          <p className="mt-4 text-sm">暂无已安装的模型</p>
          <p className="text-xs mt-1">点击"添加模型"从 HuggingFace 下载</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              onClick={() => handleSelect(model)}
              className={`group relative p-4 bg-surface-light border rounded-lg transition-all duration-300 ${
                selectable(model.status)
                  ? 'border-surface-border hover:border-neon-purple/50 hover:shadow-glow-purple cursor-pointer'
                  : 'border-surface-border opacity-60'
              }`}
            >
              <div className="flex items-center gap-4">
                <div className="w-20 h-20 rounded-lg bg-surface-lighter border border-surface-border flex items-center justify-center overflow-hidden flex-shrink-0">
                  {model.preview_url ? (
                    <img src={model.preview_url} alt={model.name} className="w-full h-full object-cover" />
                  ) : (
                    <div className="w-8 h-8 rounded bg-neon-purple/20" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 min-w-0">
                      <h3 className="text-sm font-semibold text-white truncate">
                        {model.name}
                      </h3>
                      {statusIcon(model.status)}
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {model.status === 'single_file' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleConvert(model.id);
                          }}
                          className="p-1.5 rounded text-gray-500 hover:text-neon-cyan hover:bg-neon-cyan/10 transition-all duration-200"
                          title="转换为 Diffusers 格式（拆分为目录，加快后续加载）"
                        >
                          <FileCog size={14} />
                        </button>
                      )}
                      {loadedModelId === model.id && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleUnload(model.id);
                          }}
                          disabled={unloadingId === model.id}
                          className="p-1.5 rounded text-gray-500 hover:text-yellow-400 hover:bg-yellow-400/10 transition-all duration-200"
                          title="卸载模型"
                        >
                          {unloadingId === model.id ? (
                            <Loader2 size={14} className="animate-spin" />
                          ) : (
                            <PowerOff size={14} />
                          )}
                        </button>
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(model.id);
                        }}
                        className="p-1.5 rounded text-gray-500 hover:text-red-400 hover:bg-red-400/10 transition-all duration-200"
                        title="删除模型"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1 truncate">{model.id}</p>
                  <div className="flex items-center gap-3 mt-2">
                    <span className="text-xs text-gray-500 uppercase bg-surface-lighter px-2 py-0.5 rounded">
                      {model.type}
                    </span>
                    <span className="text-xs text-gray-500">
                      {model.size_mb > 1000
                        ? `${(model.size_mb / 1000).toFixed(1)}GB`
                        : `${model.size_mb}MB`}
                    </span>
                    <span className="text-xs text-gray-600">{statusText(model.status)}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function BoxIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-gray-700">
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
      <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
      <line x1="12" y1="22.08" x2="12" y2="12" />
    </svg>
  );
}