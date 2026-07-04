import { useState, useRef, useCallback, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Play, Square, RefreshCw, Upload, ImageIcon, Paintbrush, Type, X,
  Loader2, Download, Save, Cpu, ChevronDown, Zap, Clock, Gauge,
  Layers, Plus,
} from 'lucide-react';
import { useAppStore } from '@/stores/appStore';
import {
  generateImage,
  type GenerateRequest,
  type SSEProgressEvent,
  type SSECompleteEvent,
  fetchLoadedModels,
  unloadModel as apiUnloadModel,
  fetchLoras,
  preloadModel as apiPreloadModel,
} from '@/utils/api';

type Mode = 'txt2img' | 'img2img' | 'inpaint';

export default function GeneratePage() {
  const { modelId: paramModelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();
  const {
    settings, selectedModelId, models, loadedModelId, setLoadedModelId,
    loras, setLoras, activeLoras, addActiveLora, removeActiveLora,
    updateLoraWeight, clearActiveLoras,
  } = useAppStore();

  const modelId = paramModelId || selectedModelId || '';

  const [switchingModel, setSwitchingModel] = useState(false);

  useEffect(() => {
    const pollLoaded = async () => {
      try {
        const data = await fetchLoadedModels();
        setLoadedModelId(data.current_model);
      } catch {
        // ignore
      }
    };
    pollLoaded();
    const interval = setInterval(pollLoaded, 5000);
    return () => clearInterval(interval);
  }, [setLoadedModelId]);

  useEffect(() => {
    fetchLoras().then((data) => {
      setLoras(data.loras);
    }).catch(() => {});
  }, [setLoras]);

  const handlePreloadModel = async () => {
    if (!modelId || loadedModelId === modelId) return;
    setPreloadingModel(true);
    try {
      await apiPreloadModel(modelId);
      const data = await fetchLoadedModels();
      setLoadedModelId(data.current_model);
    } catch {
      // ignore
    } finally {
      setPreloadingModel(false);
    }
  };

  const handleModelSwitch = async (newModelId: string) => {
    if (newModelId === modelId) return;
    if (loadedModelId && loadedModelId !== newModelId) {
      setSwitchingModel(true);
      try {
        await apiUnloadModel(loadedModelId);
      } catch {
        // ignore
      }
      setSwitchingModel(false);
    }
    navigate(`/generate/${newModelId}`);
  };

  const handleUnloadModel = async () => {
    if (!loadedModelId) return;
    setSwitchingModel(true);
    try {
      await apiUnloadModel(loadedModelId);
      setLoadedModelId(null);
    } catch {
      // ignore
    }
    setSwitchingModel(false);
  };

  const [mode, setMode] = useState<Mode>('txt2img');
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState(settings.default_negative_prompt);
  const [steps, setSteps] = useState(settings.default_steps);
  const [cfg, setCfg] = useState(settings.default_cfg);
  const [seed, setSeed] = useState(-1);
  const [width, setWidth] = useState(settings.default_width);
  const [height, setHeight] = useState(settings.default_height);
  const [scheduler, setScheduler] = useState(settings.default_scheduler);
  const [speedMode, setSpeedMode] = useState(settings.default_speed_mode ?? 'balanced');
  const [denoiseStrength, setDenoiseStrength] = useState(0.6);
  const [showPreview, setShowPreview] = useState(true);
  const [previewStride, setPreviewStride] = useState(2);

  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [maskBase64, setMaskBase64] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [maskPreview, setMaskPreview] = useState<string | null>(null);

  const [generating, setGenerating] = useState(false);
  const [loraMenuOpen, setLoraMenuOpen] = useState(false);
  const [preloadingModel, setPreloadingModel] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [intermediateImage, setIntermediateImage] = useState<string | null>(null);
  const [finalImage, setFinalImage] = useState<string | null>(null);
  const [finalFormat, setFinalFormat] = useState('jpeg');
  const [generationTime, setGenerationTime] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const SPEED_PRESETS: Record<string, { steps: number; cfg: number; scheduler: string; label: string; icon: typeof Zap; desc: string }> = {
    fast: { steps: 10, cfg: 4.0, scheduler: 'dpm', label: '快速', icon: Zap, desc: '~30s' },
    balanced: { steps: 15, cfg: 7.0, scheduler: 'dpm', label: '标准', icon: Gauge, desc: '~45s' },
    quality: { steps: 25, cfg: 7.5, scheduler: 'dpm_karras', label: '高质量', icon: Clock, desc: '~75s' },
  };

  const handleSpeedModeChange = (mode: string) => {
    setSpeedMode(mode);
    const preset = SPEED_PRESETS[mode];
    if (preset) {
      setSteps(preset.steps);
      setCfg(preset.cfg);
      setScheduler(preset.scheduler);
    }
  };

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result.split(',')[1]);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>, isMask: boolean) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const base64 = await fileToBase64(file);
    if (isMask) {
      setMaskBase64(base64);
      setMaskPreview(URL.createObjectURL(file));
    } else {
      setImageBase64(base64);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleDrop = useCallback(async (e: React.DragEvent, isMask: boolean) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith('image/')) return;
    const base64 = await fileToBase64(file);
    if (isMask) {
      setMaskBase64(base64);
      setMaskPreview(URL.createObjectURL(file));
    } else {
      setImageBase64(base64);
      setImagePreview(URL.createObjectURL(file));
    }
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim() || !modelId) return;
    setError(null);
    setFinalImage(null);
    setIntermediateImage(null);
    setGenerating(true);
    setCurrentStep(0);
    setTotalSteps(0);

    const actualSeed = seed === -1 ? Math.floor(Math.random() * 2147483647) : seed;

    const params: GenerateRequest = {
      prompt: prompt.trim(),
      negative_prompt: negativePrompt.trim(),
      model_id: modelId,
      mode,
      steps,
      cfg,
      seed: actualSeed,
      width,
      height,
      scheduler,
      denoise_strength: denoiseStrength,
      show_preview: showPreview,
      preview_stride: previewStride,
      speed_mode: speedMode,
      loras: activeLoras.map((a) => ({
        id: a.lora.id,
        name: a.lora.name,
        path: a.lora.path,
        filename: a.lora.filename,
        weight: a.weight,
      })),
    };

    if (mode !== 'txt2img' && imageBase64) {
      params.image = imageBase64;
    }
    if (mode === 'inpaint' && maskBase64) {
      params.mask = maskBase64;
    }

    abortRef.current = generateImage(
      params,
      (event: SSEProgressEvent) => {
        setCurrentStep(event.step);
        setTotalSteps(event.total_steps);
        if (event.image) {
          const fmt = event.preview_format || 'jpeg';
          setIntermediateImage(`data:image/${fmt};base64,${event.image}`);
        }
      },
      (event: SSECompleteEvent) => {
        setFinalImage(`data:image/${event.format};base64,${event.image}`);
        setFinalFormat(event.format);
        setGenerationTime(event.generation_time_ms);
        setGenerating(false);
        setCurrentStep(event.total_steps);
        if (actualSeed !== seed) setSeed(actualSeed);
      },
      (errMsg: string) => {
        setError(errMsg);
        setGenerating(false);
      },
    );
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setGenerating(false);
  };

  const handleSaveToHistory = () => {
    navigate('/history');
  };

  const progressPercent = totalSteps > 0 ? Math.round((currentStep / totalSteps) * 100) : 0;

  const displayImage = finalImage || intermediateImage;

  return (
    <div className="flex gap-6 h-[calc(100vh-6rem)]">
      {/* 左侧参数面板 */}
      <div className="w-80 flex-shrink-0 flex flex-col gap-4 overflow-y-auto pr-2">
        {/* 模型选择 */}
        <div>
          <label className="block text-xs text-gray-500 mb-1">模型</label>
          <div className="relative">
            <select
              value={modelId}
              onChange={(e) => handleModelSwitch(e.target.value)}
              disabled={generating || switchingModel}
              className="w-full appearance-none bg-surface-light border border-surface-border rounded-lg px-3 py-2 pr-8 text-sm text-white focus:outline-none focus:border-neon-purple transition-colors disabled:opacity-50"
            >
              <option value="" disabled>选择模型...</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.type === 'sdxl' ? 'SDXL' : 'SD 1.5'} · {m.size_mb}MB)
                </option>
              ))}
            </select>
            <ChevronDown size={14} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" />
          </div>
          {/* 预加载按钮 + 加载状态 */}
          {modelId && !generating && (
            <button
              onClick={handlePreloadModel}
              disabled={preloadingModel || loadedModelId === modelId}
              className={`mt-1.5 w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border transition-all duration-200 ${
                loadedModelId === modelId
                  ? 'bg-neon-purple/10 border-neon-purple/20 text-neon-purple cursor-default'
                  : preloadingModel
                  ? 'bg-surface-light border-surface-border text-gray-500 cursor-not-allowed'
                  : 'bg-surface-light border-surface-border text-gray-400 hover:border-neon-purple/40 hover:text-neon-purple'
              }`}
            >
              {preloadingModel ? (
                <>
                  <Loader2 size={11} className="animate-spin" />
                  预加载中...
                </>
              ) : loadedModelId === modelId ? (
                <>
                  <Cpu size={11} />
                  已预加载
                </>
              ) : (
                <>
                  <Layers size={11} />
                  预加载模型
                </>
              )}
            </button>
          )}
          {loadedModelId && loadedModelId !== modelId && (
            <div className="flex items-center gap-2 mt-1">
              <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-neon-purple/10 border border-neon-purple/20">
                <Cpu size={10} className="text-neon-purple" />
                <span className="text-[10px] text-neon-purple">
                  已加载: {loadedModelId.split('/').pop()}
                </span>
              </div>
              {switchingModel ? (
                <Loader2 size={10} className="text-neon-cyan animate-spin" />
              ) : (
                <button
                  onClick={handleUnloadModel}
                  className="text-[10px] text-gray-500 hover:text-red-400 transition-colors"
                >
                  卸载
                </button>
              )}
            </div>
          )}
        </div>

        {/* LoRA 选择 */}
        <div>
          <label className="block text-xs text-gray-500 mb-1">LoRA</label>
          {/* 已选 LoRA 标签 + 权重滑块 */}
          {activeLoras.length > 0 && (
            <div className="flex flex-col gap-2 mb-2">
              {activeLoras.map((active) => (
                <div key={active.lora.id} className="bg-surface-light border border-surface-border rounded-lg px-3 py-2">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs text-white truncate max-w-[140px]">{active.lora.name}</span>
                    <button
                      onClick={() => removeActiveLora(active.lora.id)}
                      className="text-gray-500 hover:text-red-400 transition-colors"
                    >
                      <X size={11} />
                    </button>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min={0}
                      max={1.5}
                      step={0.05}
                      value={active.weight}
                      onChange={(e) => updateLoraWeight(active.lora.id, parseFloat(e.target.value))}
                      className="flex-1"
                    />
                    <span className="text-[10px] text-neon-purple w-8 text-right">{active.weight.toFixed(2)}</span>
                  </div>
                </div>
              ))}
              <button
                onClick={clearActiveLoras}
                className="text-[10px] text-gray-600 hover:text-gray-400 transition-colors self-start"
              >
                清除全部
              </button>
            </div>
          )}
          {/* 添加 LoRA 按钮 */}
          <div className="relative">
            <button
              onClick={() => setLoraMenuOpen((v) => !v)}
              disabled={generating || loras.length === 0}
              className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-surface-light border border-surface-border text-gray-400 hover:border-neon-cyan/40 hover:text-neon-cyan text-xs transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Plus size={13} />
              {loras.length === 0 ? '未检测到 LoRA' : '添加 LoRA'}
              {loras.length > 0 && <ChevronDown size={11} className={`ml-auto transition-transform ${loraMenuOpen ? 'rotate-180' : ''}`} />}
            </button>
            {/* LoRA 下拉菜单 */}
            {loraMenuOpen && loras.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-surface-light border border-surface-border rounded-lg shadow-xl z-50 max-h-48 overflow-y-auto">
                {loras.map((lora) => {
                  const isActive = activeLoras.some((a) => a.lora.id === lora.id);
                  return (
                    <button
                      key={lora.id}
                      onClick={() => {
                        if (!isActive) addActiveLora(lora);
                        setLoraMenuOpen(false);
                      }}
                      disabled={isActive}
                      className={`w-full text-left px-3 py-2 text-xs transition-colors ${
                        isActive
                          ? 'text-gray-600 cursor-default'
                          : 'text-gray-300 hover:bg-neon-purple/10 hover:text-neon-purple'
                      } border-b border-surface-border last:border-b-0`}
                    >
                      <div className="font-medium">{lora.name}</div>
                      <div className="text-[10px] text-gray-600">{lora.size_mb}MB</div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
          {loras.length > 0 && activeLoras.length === 0 && (
            <p className="text-[10px] text-gray-600 mt-1">点击添加 LoRA 叠加层</p>
          )}
        </div>

        {/* 模式切换 */}
        <div className="flex bg-surface-light rounded-lg p-1 border border-surface-border">
          {([
            { key: 'txt2img', icon: Type, label: '文生图' },
            { key: 'img2img', icon: ImageIcon, label: '图生图' },
            { key: 'inpaint', icon: Paintbrush, label: '局部重绘' },
          ] as const).map(({ key, icon: Icon, label }) => (
            <button
              key={key}
              onClick={() => setMode(key)}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded text-xs transition-all duration-200 ${
                mode === key
                  ? 'bg-neon-purple/20 text-neon-purple'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </div>

        {/* 正向提示词 */}
        <div>
          <label className="block text-xs text-gray-500 mb-1">正向提示词</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="描述你想要生成的图像..."
            rows={3}
            className="w-full bg-surface-light border border-surface-border rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-neon-purple transition-colors resize-none"
          />
        </div>

        {/* 负向提示词 */}
        <div>
          <label className="block text-xs text-gray-500 mb-1">负向提示词</label>
          <textarea
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder="想要避免的内容..."
            rows={2}
            className="w-full bg-surface-light border border-surface-border rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-neon-purple transition-colors resize-none"
          />
        </div>

        {/* 图片上传 (图生图/局部重绘) */}
        {mode !== 'txt2img' && (
          <div>
            <label className="block text-xs text-gray-500 mb-1">参考图片</label>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => handleDrop(e, false)}
              className="relative border-2 border-dashed border-surface-border rounded-lg p-4 text-center hover:border-neon-purple/50 transition-colors cursor-pointer"
            >
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleImageUpload(e, false)}
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
              {imagePreview ? (
                <div className="relative">
                  <img src={imagePreview} alt="参考图" className="max-h-32 mx-auto rounded" />
                  <button
                    onClick={() => { setImagePreview(null); setImageBase64(null); }}
                    className="absolute top-1 right-1 p-1 bg-black/60 rounded-full text-white hover:text-red-400"
                  >
                    <X size={12} />
                  </button>
                </div>
              ) : (
                <div className="py-4">
                  <Upload size={24} className="mx-auto text-gray-600 mb-1" />
                  <p className="text-xs text-gray-500">拖放或点击上传</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 蒙版上传 (局部重绘) */}
        {mode === 'inpaint' && (
          <div>
            <label className="block text-xs text-gray-500 mb-1">蒙版图片</label>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => handleDrop(e, true)}
              className="relative border-2 border-dashed border-surface-border rounded-lg p-4 text-center hover:border-neon-pink/50 transition-colors cursor-pointer"
            >
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleImageUpload(e, true)}
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
              {maskPreview ? (
                <div className="relative">
                  <img src={maskPreview} alt="蒙版" className="max-h-32 mx-auto rounded" />
                  <button
                    onClick={() => { setMaskPreview(null); setMaskBase64(null); }}
                    className="absolute top-1 right-1 p-1 bg-black/60 rounded-full text-white hover:text-red-400"
                  >
                    <X size={12} />
                  </button>
                </div>
              ) : (
                <div className="py-4">
                  <Paintbrush size={24} className="mx-auto text-gray-600 mb-1" />
                  <p className="text-xs text-gray-500">拖放或点击上传</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 参数调节 */}
        <div className="space-y-3 pt-2 border-t border-surface-border">
          <ParamSlider label="步数" value={steps} min={1} max={50} step={1} onChange={setSteps} />
          <ParamSlider label="引导系数" value={cfg} min={1} max={30} step={0.5} onChange={setCfg} />
          <ParamSlider label="去噪强度" value={denoiseStrength} min={0.1} max={1} step={0.05} onChange={setDenoiseStrength} />
          <ParamSlider label="预览间隔" value={previewStride} min={1} max={10} step={1} onChange={setPreviewStride} />

          <div>
            <label className="block text-xs text-gray-500 mb-1">随机种子</label>
            <div className="flex gap-2">
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
                className="flex-1 bg-surface-light border border-surface-border rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-neon-purple transition-colors"
              />
              <button
                onClick={() => setSeed(-1)}
                className="p-1.5 rounded bg-surface-light border border-surface-border text-gray-500 hover:text-neon-purple transition-colors"
                title="随机种子"
              >
                <RefreshCw size={14} />
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-500 mb-1">分辨率</label>
            <div className="flex gap-2 items-center">
              <input
                type="number"
                value={width}
                onChange={(e) => setWidth(parseInt(e.target.value) || 512)}
                className="w-20 bg-surface-light border border-surface-border rounded px-2 py-1.5 text-sm text-white focus:outline-none focus:border-neon-purple transition-colors"
              />
              <span className="text-gray-600">x</span>
              <input
                type="number"
                value={height}
                onChange={(e) => setHeight(parseInt(e.target.value) || 512)}
                className="w-20 bg-surface-light border border-surface-border rounded px-2 py-1.5 text-sm text-white focus:outline-none focus:border-neon-purple transition-colors"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-500 mb-1">调度器</label>
            <select
              value={scheduler}
              onChange={(e) => setScheduler(e.target.value)}
              className="w-full bg-surface-light border border-surface-border rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-neon-purple transition-colors"
            >
              <option value="dpm">DPM++ 2M</option>
              <option value="dpm_karras">DPM++ 2M Karras</option>
              <option value="euler">Euler</option>
              <option value="euler_karras">Euler Karras</option>
              <option value="euler_a">Euler Ancestral</option>
              <option value="lcm">LCM</option>
            </select>
          </div>
        </div>

        {/* 生成按钮 */}
        <div className="space-y-2">
          <div>
            <label className="block text-xs text-gray-500 mb-1">生成速度</label>
            <div className="flex bg-surface-light rounded-lg p-1 border border-surface-border">
              {Object.entries(SPEED_PRESETS).map(([key, preset]) => {
                const Icon = preset.icon;
                return (
                  <button
                    key={key}
                    onClick={() => handleSpeedModeChange(key)}
                    className={`flex-1 flex flex-col items-center gap-0.5 py-2 rounded text-xs transition-all duration-200 ${
                      speedMode === key
                        ? 'bg-neon-cyan/10 text-neon-cyan'
                        : 'text-gray-500 hover:text-gray-300'
                    }`}
                  >
                    <Icon size={13} />
                    <span className="text-[10px]">{preset.label}</span>
                    <span className="text-[9px] opacity-60">{preset.desc}</span>
                  </button>
                );
              })}
            </div>
          </div>
        <div className="flex gap-2 pt-2">
          {generating ? (
            <button
              onClick={handleStop}
              className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm hover:bg-red-500/20 transition-all duration-300"
            >
              <Square size={14} />
              停止
            </button>
          ) : (
            <button
              onClick={handleGenerate}
              disabled={!prompt.trim() || !modelId}
              className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-neon-purple border border-neon-purple rounded-lg text-white text-sm hover:bg-neon-purple-glow hover:shadow-glow-purple disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-300"
            >
              <Play size={14} />
              开始生成
            </button>
          )}
        </div>
        </div>

        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
            <p className="text-xs text-red-400">{error}</p>
          </div>
        )}
      </div>

      {/* 右侧预览区 */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 bg-surface-light border border-surface-border rounded-lg flex items-center justify-center relative overflow-hidden">
          {displayImage ? (
            <img
              src={displayImage}
              alt="生成结果"
              className="max-w-full max-h-full object-contain animate-fade-in"
            />
          ) : (
            <div className="text-center text-gray-600">
              <ImageIcon size={64} className="mx-auto mb-4 opacity-30" />
              <p className="text-sm">图像预览区</p>
              <p className="text-xs mt-1">输入提示词并点击"开始生成"</p>
            </div>
          )}

          {/* 进度条 */}
          {generating && (
            <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-surface/80 to-transparent">
              <div className="flex items-center gap-3">
                <Loader2 size={16} className="text-neon-purple animate-spin" />
                <div className="flex-1 h-1.5 bg-surface-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-neon-purple to-neon-cyan rounded-full transition-all duration-300"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400">
                  {currentStep}/{totalSteps}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* 操作栏 */}
        {finalImage && (
          <div className="flex items-center gap-3 mt-3 p-3 bg-surface-light border border-surface-border rounded-lg animate-slide-up">
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span>{width}x{height}</span>
              <span>·</span>
              <span>{(generationTime / 1000).toFixed(1)}秒</span>
              <span>·</span>
              <span>{steps} 步</span>
            </div>
            <div className="flex-1" />
            <a
              href={finalImage}
              download={`local-dream-${Date.now()}.${finalFormat}`}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-surface-lighter border border-surface-border text-gray-400 hover:text-neon-cyan hover:border-neon-cyan/30 text-xs transition-all duration-200"
            >
              <Download size={14} />
              下载
            </a>
            <button
              onClick={handleSaveToHistory}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-surface-lighter border border-surface-border text-gray-400 hover:text-neon-purple hover:border-neon-purple/30 text-xs transition-all duration-200"
            >
              <Save size={14} />
              历史记录
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function ParamSlider({
  label, value, min, max, step, onChange,
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <label className="text-xs text-gray-500">{label}</label>
        <span className="text-xs text-neon-purple">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}