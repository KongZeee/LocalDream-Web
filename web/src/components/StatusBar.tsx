import { useEffect, useState } from 'react';
import { Cpu, Activity, HardDrive } from 'lucide-react';
import { fetchSystemStatus } from '@/utils/api';

export default function StatusBar() {
  const [status, setStatus] = useState<{
    gpu_available: boolean;
    gpu_name: string;
    gpu_memory_used_mb: number | null;
    gpu_memory_total_mb: number | null;
    cpu_memory_used_mb: number | null;
    cpu_memory_total_mb: number | null;
    loaded_models: string[];
    current_model: string | null;
  } | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const data = await fetchSystemStatus();
        setStatus(data);
      } catch {
        // ignore
      }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  if (!status) {
    return (
      <div className="flex items-center gap-2 text-[10px] text-gray-600">
        <Activity size={10} className="animate-pulse" />
        <span>检测中...</span>
      </div>
    );
  }

  const pct = (used: number, total: number) =>
    total > 0 ? Math.round((used / total) * 100) : 0;

  const gpuPct = status.gpu_memory_used_mb != null && status.gpu_memory_total_mb != null
    ? pct(status.gpu_memory_used_mb, status.gpu_memory_total_mb)
    : null;

  const cpuPct = status.cpu_memory_used_mb != null && status.cpu_memory_total_mb != null
    ? pct(status.cpu_memory_used_mb, status.cpu_memory_total_mb)
    : null;

  const fmt = (mb: number | null) => {
    if (mb == null) return '—';
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
  };

  const barColor = (p: number) => {
    if (p < 60) return 'bg-neon-cyan';
    if (p < 85) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="flex flex-col gap-2 w-full text-[10px]">
      {/* 当前模型 */}
      {status.current_model && (
        <div className="flex items-center gap-1.5 text-gray-400">
          <Cpu size={10} className="text-neon-purple" />
          <span className="truncate max-w-[140px] text-gray-300">
            {status.current_model.split('/').pop()}
          </span>
        </div>
      )}

      {/* GPU 显存 */}
      {status.gpu_available && gpuPct != null && (
        <div>
          <div className="flex items-center justify-between mb-0.5">
            <div className="flex items-center gap-1">
              <Activity size={10} className="text-neon-cyan" />
              <span className="text-gray-500">GPU {status.gpu_name.split('(')[0].trim()}</span>
            </div>
            <span className="text-gray-400">
              {fmt(status.gpu_memory_used_mb!)} / {fmt(status.gpu_memory_total_mb!)}
            </span>
          </div>
          <div className="h-1 bg-surface-border rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${barColor(gpuPct)}`}
              style={{ width: `${gpuPct}%` }}
            />
          </div>
        </div>
      )}

      {/* CPU 内存 */}
      {cpuPct != null && (
        <div>
          <div className="flex items-center justify-between mb-0.5">
            <div className="flex items-center gap-1">
              <HardDrive size={10} className="text-gray-500" />
              <span className="text-gray-500">CPU 内存</span>
            </div>
            <span className="text-gray-400">
              {fmt(status.cpu_memory_used_mb!)} / {fmt(status.cpu_memory_total_mb!)}
            </span>
          </div>
          <div className="h-1 bg-surface-border rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${barColor(cpuPct)}`}
              style={{ width: `${cpuPct}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
