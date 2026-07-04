import { useState, useEffect } from 'react';
import { Trash2, ChevronLeft, ChevronRight, X, Loader2 } from 'lucide-react';
import { fetchHistory, deleteHistoryItem } from '@/utils/api';
import { useAppStore } from '@/stores/appStore';

interface HistoryItem {
  id: number;
  prompt: string;
  negative_prompt: string;
  seed: number;
  steps: number;
  cfg: number;
  width: number;
  height: number;
  model_id: string;
  scheduler: string;
  mode: string;
  image_url: string;
  thumbnail_url: string;
  created_at: string;
}

const modeLabels: Record<string, string> = {
  txt2img: '文生图',
  img2img: '图生图',
  inpaint: '局部重绘',
};

export default function HistoryPage() {
  const baseUrl = useAppStore((s) => s.settings.api_base_url);
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);
  const pageSize = 20;

  useEffect(() => {
    loadHistory();
  }, [page]);

  async function loadHistory() {
    setLoading(true);
    try {
      const data = await fetchHistory(page, pageSize);
      if (data) {
        setItems(data.items || []);
        setTotal(data.total || 0);
      }
    } catch {
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id: number) {
    try {
      await deleteHistoryItem(id);
      if (selectedItem?.id === id) setSelectedItem(null);
      await loadHistory();
    } catch {
      // handled by UI
    }
  }

  const totalPages = Math.ceil(total / pageSize);

  const getImageUrl = (path: string) => `${baseUrl}${path}`;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-display text-2xl font-bold text-white tracking-wider">
            历史记录
          </h1>
          <p className="text-gray-500 text-sm mt-1">共 {total} 条记录</p>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 size={32} className="text-neon-purple animate-spin" />
        </div>
      ) : items.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 text-gray-500">
          <p className="text-sm">暂无历史记录</p>
          <p className="text-xs mt-1">生成的图像将会显示在这里</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
            {items.map((item) => (
              <div
                key={item.id}
                onClick={() => setSelectedItem(item)}
                className="group relative aspect-square bg-surface-light border border-surface-border rounded-lg overflow-hidden cursor-pointer hover:border-neon-purple/50 hover:shadow-glow-purple transition-all duration-300"
              >
                <img
                  src={getImageUrl(item.thumbnail_url || item.image_url)}
                  alt={item.prompt}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(item.id);
                  }}
                  className="absolute top-2 right-2 p-1.5 rounded bg-surface/80 text-gray-500 hover:text-red-400 hover:bg-red-400/10 transition-all duration-200 opacity-0 group-hover:opacity-100"
                  title="删除记录"
                >
                  <Trash2 size={14} />
                </button>
                <div className="absolute inset-0 bg-gradient-to-t from-surface/90 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-end p-2">
                  <p className="text-xs text-white line-clamp-2">{item.prompt}</p>
                </div>
              </div>
            ))}
          </div>

          {/* 分页 */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-4 mt-6">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="p-2 rounded text-gray-500 hover:text-neon-purple disabled:opacity-30 transition-colors"
              >
                <ChevronLeft size={18} />
              </button>
              <span className="text-sm text-gray-400">
                {page} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="p-2 rounded text-gray-500 hover:text-neon-purple disabled:opacity-30 transition-colors"
              >
                <ChevronRight size={18} />
              </button>
            </div>
          )}
        </>
      )}

      {/* 详情弹窗 */}
      {selectedItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in">
          <div className="relative max-w-4xl w-full mx-4 max-h-[90vh] flex flex-col md:flex-row bg-surface-light border border-surface-border rounded-xl overflow-hidden">
            <button
              onClick={() => setSelectedItem(null)}
              className="absolute top-3 right-3 z-10 p-1.5 rounded-full bg-surface/80 text-gray-400 hover:text-white transition-colors"
            >
              <X size={16} />
            </button>

            <div className="flex-1 flex items-center justify-center bg-surface p-4 min-h-[300px]">
              <img
                src={getImageUrl(selectedItem.image_url)}
                alt={selectedItem.prompt}
                className="max-w-full max-h-[70vh] object-contain rounded"
              />
            </div>

            <div className="w-80 flex-shrink-0 p-5 border-l border-surface-border overflow-y-auto">
              <h3 className="text-sm text-white font-semibold mb-4">生成详情</h3>

              <div className="space-y-3">
                <DetailRow label="提示词" value={selectedItem.prompt} />
                <DetailRow label="负向提示词" value={selectedItem.negative_prompt || '-'} />
                <DetailRow label="随机种子" value={String(selectedItem.seed)} />
                <DetailRow label="步数" value={String(selectedItem.steps)} />
                <DetailRow label="引导系数" value={String(selectedItem.cfg)} />
                <DetailRow label="分辨率" value={`${selectedItem.width}x${selectedItem.height}`} />
                <DetailRow label="调度器" value={selectedItem.scheduler} />
                <DetailRow label="模式" value={modeLabels[selectedItem.mode] || selectedItem.mode} />
                <DetailRow label="模型" value={selectedItem.model_id} />
                <DetailRow label="生成时间" value={new Date(selectedItem.created_at).toLocaleString()} />
              </div>

              <div className="mt-6 flex gap-2">
                <a
                  href={getImageUrl(selectedItem.image_url)}
                  download={`local-dream-${selectedItem.id}.jpg`}
                  className="flex-1 text-center py-2 rounded bg-neon-purple/10 border border-neon-purple/30 text-neon-purple text-xs hover:bg-neon-purple/20 transition-all duration-200"
                >
                  下载
                </a>
                <button
                  onClick={() => handleDelete(selectedItem.id)}
                  className="flex-1 py-2 rounded bg-red-500/10 border border-red-500/30 text-red-400 text-xs hover:bg-red-500/20 transition-all duration-200"
                >
                  删除
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-xs text-gray-500 uppercase">{label}</span>
      <p className="text-sm text-gray-300 mt-0.5 break-all">{value}</p>
    </div>
  );
}