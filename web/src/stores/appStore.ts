import { create } from 'zustand';

export interface ModelInfo {
  id: string;
  name: string;
  type: 'sd15' | 'sdxl';
  /** single_file = single-file checkpoint (directly generatable);
   *  converting = being converted to Diffusers format in the background. */
  status: 'ready' | 'downloading' | 'converting' | 'single_file' | 'error';
  size_mb: number;
  preview_url: string;
}

export interface LoraInfo {
  id: string;
  name: string;
  filename: string;
  path: string;
  size_mb: number;
}

export interface ActiveLora {
  lora: LoraInfo;
  weight: number;
}

export interface GenerationParams {
  prompt: string;
  negative_prompt: string;
  mode: 'txt2img' | 'img2img' | 'inpaint';
  steps: number;
  cfg: number;
  seed: number;
  width: number;
  height: number;
  scheduler: string;
  denoise_strength: number;
  show_preview: boolean;
  preview_stride: number;
  image: string | null;
  mask: string | null;
  speed_mode: string;
}

export interface AppSettings {
  api_base_url: string;
  models_dir: string;
  default_steps: number;
  default_cfg: number;
  default_width: number;
  default_height: number;
  default_scheduler: string;
  default_negative_prompt: string;
  default_speed_mode: string;
}

interface AppState {
  sidebarCollapsed: boolean;
  models: ModelInfo[];
  selectedModelId: string | null;
  loadedModelId: string | null;
  settings: AppSettings;

  // LoRA state
  loras: LoraInfo[];
  activeLoras: ActiveLora[];   // currently selected LoRAs with weights

  // UI actions
  toggleSidebar: () => void;
  setModels: (models: ModelInfo[]) => void;
  setSelectedModelId: (id: string | null) => void;
  setLoadedModelId: (id: string | null) => void;
  setSettings: (settings: Partial<AppSettings>) => void;

  // LoRA actions
  setLoras: (loras: LoraInfo[]) => void;
  setActiveLoras: (loras: ActiveLora[]) => void;
  addActiveLora: (lora: LoraInfo, weight?: number) => void;
  removeActiveLora: (loraId: string) => void;
  updateLoraWeight: (loraId: string, weight: number) => void;
  clearActiveLoras: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  sidebarCollapsed: false,
  models: [],
  selectedModelId: null,
  loadedModelId: null,
  settings: {
    api_base_url: 'http://127.0.0.1:8081',
    models_dir: '',
    default_steps: 15,
    default_cfg: 7.0,
    default_width: 512,
    default_height: 512,
    default_scheduler: 'dpm',
    default_negative_prompt: 'ugly, blurry, low quality, bad anatomy',
    default_speed_mode: 'balanced',
  },

  // LoRA initial state
  loras: [],
  activeLoras: [],

  // UI actions
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  setModels: (models) => set({ models }),
  setSelectedModelId: (id) => set({ selectedModelId: id }),
  setLoadedModelId: (id) => set({ loadedModelId: id }),
  setSettings: (partial) =>
    set((s) => ({ settings: { ...s.settings, ...partial } })),

  // LoRA actions
  setLoras: (loras) => set({ loras }),

  setActiveLoras: (loras) => set({ activeLoras: loras }),

  addActiveLora: (lora, weight = 0.7) =>
    set((s) => {
      if (s.activeLoras.some((a) => a.lora.id === lora.id)) return s;
      return { activeLoras: [...s.activeLoras, { lora, weight }] };
    }),

  removeActiveLora: (loraId) =>
    set((s) => ({
      activeLoras: s.activeLoras.filter((a) => a.lora.id !== loraId),
    })),

  updateLoraWeight: (loraId, weight) =>
    set((s) => ({
      activeLoras: s.activeLoras.map((a) =>
        a.lora.id === loraId ? { ...a, weight } : a
      ),
    })),

  clearActiveLoras: () => set({ activeLoras: [] }),
}));
