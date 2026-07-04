import { useState } from 'react';
import { Upload, Zap, Loader2, ImageIcon, Download } from 'lucide-react';
import { upscaleImage } from '@/utils/api';

export default function UpscalePage() {
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [upscaleModel, setUpscaleModel] = useState('RealESRGAN_x4plus');
  const [tileSize, setTileSize] = useState(512);
  const [upscaling, setUpscaling] = useState(false);
  const [result, setResult] = useState<{ image: string; format: string; width: number; height: number; duration_ms: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    setSourceFile(file);
    setSourcePreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleUpscale = async () => {
    if (!sourceFile) return;
    setUpscaling(true);
    setError(null);
    try {
      const data = await upscaleImage(sourceFile, upscaleModel, tileSize);
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setUpscaling(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-display text-2xl font-bold text-white tracking-wider">
            超分辨率
          </h1>
          <p className="text-gray-500 text-sm mt-1">使用 AI 将图像放大 4 倍</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 原图 */}
        <div>
          <p className="text-xs text-gray-500 mb-2">原图</p>
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="relative aspect-square bg-surface-light border-2 border-dashed border-surface-border rounded-xl flex items-center justify-center hover:border-neon-purple/50 transition-colors cursor-pointer overflow-hidden"
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleUpload}
              className="absolute inset-0 opacity-0 cursor-pointer"
            />
            {sourcePreview ? (
              <img src={sourcePreview} alt="原图" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center text-gray-600">
                <Upload size={48} className="mx-auto mb-3 opacity-30" />
                <p className="text-sm">拖放图片到此处</p>
                <p className="text-xs mt-1">或点击选择文件</p>
              </div>
            )}
          </div>
        </div>

        {/* 结果 */}
        <div>
          <p className="text-xs text-gray-500 mb-2">放大结果</p>
          <div className="aspect-square bg-surface-light border border-surface-border rounded-xl flex items-center justify-center overflow-hidden relative">
            {result ? (
              <img
                src={`data:image/${result.format};base64,${result.image}`}
                alt="放大结果"
                className="w-full h-full object-contain animate-fade-in"
              />
            ) : upscaling ? (
              <div className="text-center">
                <Loader2 size={48} className="text-neon-purple animate-spin mx-auto mb-3" />
                <p className="text-sm text-neon-purple">正在放大...</p>
              </div>
            ) : (
              <div className="text-center text-gray-600">
                <Zap size={48} className="mx-auto mb-3 opacity-30" />
                <p className="text-sm">放大结果</p>
                <p className="text-xs mt-1">上传图片后点击"开始放大"</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 控制面板 */}
      <div className="mt-6 p-4 bg-surface-light border border-surface-border rounded-xl">
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">超分模型</label>
            <select
              value={upscaleModel}
              onChange={(e) => setUpscaleModel(e.target.value)}
              className="bg-surface border border-surface-border rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-neon-purple transition-colors"
            >
              <option value="RealESRGAN_x4plus">Real-ESRGAN 4x+</option>
              <option value="RealESRGAN_x4plus_anime">Real-ESRGAN 4x+ 动漫</option>
              <option value="UltraSharpV2">UltraSharp V2</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">分块大小</label>
            <input
              type="number"
              value={tileSize}
              onChange={(e) => setTileSize(parseInt(e.target.value) || 512)}
              className="w-24 bg-surface border border-surface-border rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-neon-purple transition-colors"
            />
          </div>
          <button
            onClick={handleUpscale}
            disabled={!sourceFile || upscaling}
            className="flex items-center gap-2 px-6 py-2 bg-neon-cyan/10 border border-neon-cyan/30 rounded-lg text-neon-cyan text-sm hover:bg-neon-cyan/20 hover:shadow-glow-cyan disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-300"
          >
            <Zap size={16} />
            开始放大
          </button>
        </div>

        {result && (
          <div className="flex items-center gap-3 mt-4 pt-4 border-t border-surface-border">
            <span className="text-xs text-gray-400">
              {result.width}x{result.height}
            </span>
            <span className="text-xs text-gray-500">
              {(result.duration_ms / 1000).toFixed(1)}秒
            </span>
            <div className="flex-1" />
            <a
              href={`data:image/${result.format};base64,${result.image}`}
              download={`upscaled-${Date.now()}.${result.format}`}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-surface-lighter border border-surface-border text-gray-400 hover:text-neon-cyan text-xs transition-all duration-200"
            >
              <Download size={14} />
              下载
            </a>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
            <p className="text-xs text-red-400">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}