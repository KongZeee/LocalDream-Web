import { useAppStore } from '@/stores/appStore';

const getBaseUrl = () => useAppStore.getState().settings.api_base_url;

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const baseUrl = getBaseUrl();
  const res = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(err.message || err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export interface HealthResponse {
  status: string;
  gpu_available: boolean;
  gpu_name: string;
  loaded_model: string | null;
}

export interface GenerateRequest {
  prompt: string;
  negative_prompt: string;
  model_id: string;
  mode: string;
  steps: number;
  cfg: number;
  seed: number;
  width: number;
  height: number;
  scheduler: string;
  denoise_strength: number;
  image?: string;
  mask?: string;
  show_preview: boolean;
  preview_stride: number;
  speed_mode: string;
  loras?: Array<{
    id: string;
    name: string;
    path: string;
    filename: string;
    weight: number;
  }>;
}

export interface SSEProgressEvent {
  type: 'progress';
  step: number;
  total_steps: number;
  image?: string;
  preview_format?: string;
}

export interface SSECompleteEvent {
  type: 'complete';
  image: string;
  format: string;
  seed: number;
  width: number;
  height: number;
  generation_time_ms: number;
  first_step_time_ms: number;
  total_steps: number;
}

export interface SSEStartedEvent {
  type: 'started';
  total_steps: number;
}

export type SSEEvent = SSEStartedEvent | SSEProgressEvent | SSECompleteEvent | { type: 'error'; message: string };

export function generateImage(
  params: GenerateRequest,
  onProgress: (event: SSEProgressEvent) => void,
  onComplete: (event: SSECompleteEvent) => void,
  onError: (error: string) => void,
): AbortController {
  const controller = new AbortController();
  const baseUrl = getBaseUrl();

  fetch(`${baseUrl}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal: controller.signal,
  }).then(async (response) => {
    if (!response.ok) {
      const err = await response.json().catch(() => ({ message: response.statusText }));
      onError(err.message || err.detail || `HTTP ${response.status}`);
      return;
    }
    const reader = response.body?.getReader();
    if (!reader) {
      onError('No response body');
      return;
    }
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'started') {
              if ('total_steps' in data) {
                onProgress({
                  type: 'progress',
                  step: 0,
                  total_steps: data.total_steps,
                });
              }
            }
            else if (data.type === 'progress') onProgress(data as SSEProgressEvent);
            else if (data.type === 'complete') onComplete(data as SSECompleteEvent);
            else if (data.type === 'error') onError(data.message);
          } catch {
            // skip unparseable lines
          }
        }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') {
      onError(err.message);
    }
  });

  return controller;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health');
}

export async function fetchModels() {
  return request<{ models: Array<{
    id: string; name: string; type: string; status: string; size_mb: number; preview_url: string;
  }>; default_model: string }>('/api/models');
}

export async function downloadModel(modelId: string, modelType: string) {
  return request('/api/models/download', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId, model_type: modelType }),
  });
}

export async function deleteModel(modelId: string) {
  return request(`/api/models/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
}

export async function fetchLoadedModels() {
  return request<{
    loaded_models: string[];
    current_model: string | null;
  }>('/api/models/loaded');
}

export async function unloadModel(modelId: string) {
  return request('/api/models/unload', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId }),
  });
}

export async function unloadAllModels() {
  return request('/api/models/unload-all', { method: 'POST' });
}

export async function fetchHistory(page = 1, pageSize = 20) {
  return request<{
    items: Array<{
      id: number; prompt: string; negative_prompt: string; seed: number;
      steps: number; cfg: number; width: number; height: number;
      model_id: string; scheduler: string; mode: string;
      image_url: string; thumbnail_url: string; created_at: string;
    }>;
    total: number; page: number; page_size: number;
  }>(`/api/history?page=${page}&page_size=${pageSize}`);
}

export async function fetchLoras() {
  return request<{
    loras: Array<{
      id: string;
      name: string;
      filename: string;
      path: string;
      size_mb: number;
    }>;
  }>('/api/loras');
}

export async function fetchLoraNames() {
  return request<Array<{ id: string; name: string }>>('/api/loras/names');
}

export async function preloadModel(modelId: string) {
  return request<{ status: string; model_id: string }>(
    `/api/models/${encodeURIComponent(modelId)}/preload`,
    { method: 'POST' }
  );
}

export async function deleteHistoryItem(id: number) {
  return request(`/api/history/${id}`, { method: 'DELETE' });
}

export async function upscaleImage(file: File, modelId: string, tileSize?: number) {
  const baseUrl = getBaseUrl();
  const formData = new FormData();
  formData.append('image', file);
  formData.append('model_id', modelId);
  if (tileSize) formData.append('tile_size', String(tileSize));

  const res = await fetch(`${baseUrl}/api/upscale`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(err.message || err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchSystemStatus(): Promise<{
  gpu_available: boolean;
  gpu_name: string;
  gpu_memory_used_mb: number | null;
  gpu_memory_total_mb: number | null;
  cpu_memory_used_mb: number | null;
  cpu_memory_total_mb: number | null;
  loaded_models: string[];
  current_model: string | null;
}> {
  return request('/api/system/status');
}

export async function fetchSettings() {
  return request<Record<string, unknown>>('/api/settings');
}

export async function saveSettings(settings: Record<string, unknown>) {
  return request('/api/settings', {
    method: 'PUT',
    body: JSON.stringify(settings),
  });
}