import { useState, useEffect } from 'react';
import { Save, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { useAppStore } from '@/stores/appStore';
import { fetchHealth, fetchSettings, saveSettings } from '@/utils/api';

export default function SettingsPage() {
  const { settings, setSettings } = useAppStore();
  const [localSettings, setLocalSettings] = useState({ ...settings });
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saved' | 'error'>('idle');
  const [health, setHealth] = useState<{ status: string; gpu_available: boolean; gpu_name: string; loaded_model: string | null } | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);

  useEffect(() => {
    loadHealth();
    loadSettings();
  }, []);

  async function loadHealth() {
    setHealthLoading(true);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  }

  async function loadSettings() {
    try {
      const data = await fetchSettings();
      if (data) {
        const merged = { ...settings, ...data } as typeof settings;
        setLocalSettings(merged);
        setSettings(merged);
      }
    } catch {
      // use defaults
    }
  }

  async function handleSave() {
    setSaving(true);
    setSaveStatus('idle');
    try {
      await saveSettings(localSettings as unknown as Record<string, unknown>);
      setSettings(localSettings);
      setSaveStatus('saved');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } catch {
      setSaveStatus('error');
    } finally {
      setSaving(false);
    }
  }

  const update = (key: string, value: string | number) => {
    setLocalSettings((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-display text-2xl font-bold text-white tracking-wider">
            API 设置
          </h1>
          <p className="text-gray-500 text-sm mt-1">配置后端服务与默认参数</p>
        </div>
      </div>

      {/* 后端状态 */}
      <div className="mb-8 p-4 bg-surface-light border border-surface-border rounded-xl">
        <h3 className="text-sm text-white font-semibold mb-3">后端状态</h3>
        {healthLoading ? (
          <div className="flex items-center gap-2">
            <Loader2 size={14} className="text-neon-purple animate-spin" />
            <span className="text-xs text-gray-500">检查中...</span>
          </div>
        ) : health ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${health.status === 'ok' ? 'bg-neon-cyan' : 'bg-red-400'}`} />
              <span className="text-xs text-gray-300 uppercase">{health.status}</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500">GPU:</span>
                <span className="ml-2 text-gray-300">
                  {health.gpu_available ? health.gpu_name : '不可用'}
                </span>
              </div>
              <div>
                <span className="text-gray-500">已加载模型:</span>
                <span className="ml-2 text-gray-300">
                  {health.loaded_model || '无'}
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <AlertCircle size={14} className="text-red-400" />
            <span className="text-xs text-red-400">后端无法连接</span>
          </div>
        )}
      </div>

      {/* API 配置 */}
      <div className="mb-8 p-4 bg-surface-light border border-surface-border rounded-xl">
        <h3 className="text-sm text-white font-semibold mb-4">API 配置</h3>
        <div className="space-y-4">
          <SettingField
            label="API 地址"
            value={localSettings.api_base_url}
            onChange={(v) => update('api_base_url', v)}
            placeholder="http://127.0.0.1:8081"
          />
          <SettingField
            label="模型目录"
            value={localSettings.models_dir}
            onChange={(v) => update('models_dir', v)}
            placeholder="D:/models/stable-diffusion"
          />
        </div>
      </div>

      {/* 默认参数 */}
      <div className="mb-8 p-4 bg-surface-light border border-surface-border rounded-xl">
        <h3 className="text-sm text-white font-semibold mb-4">默认参数</h3>
        <div className="grid grid-cols-2 gap-4">
          <SettingField
            label="步数"
            value={String(localSettings.default_steps)}
            onChange={(v) => update('default_steps', parseInt(v) || 20)}
            type="number"
          />
          <SettingField
            label="引导系数"
            value={String(localSettings.default_cfg)}
            onChange={(v) => update('default_cfg', parseFloat(v) || 7.5)}
            type="number"
          />
          <SettingField
            label="宽度"
            value={String(localSettings.default_width)}
            onChange={(v) => update('default_width', parseInt(v) || 512)}
            type="number"
          />
          <SettingField
            label="高度"
            value={String(localSettings.default_height)}
            onChange={(v) => update('default_height', parseInt(v) || 512)}
            type="number"
          />
          <div className="col-span-2">
            <SettingField
              label="默认负向提示词"
              value={localSettings.default_negative_prompt}
              onChange={(v) => update('default_negative_prompt', v)}
            />
          </div>
        </div>
      </div>

      {/* 保存 */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-2 px-6 py-2.5 bg-neon-purple border border-neon-purple rounded-lg text-white text-sm hover:bg-neon-purple-glow hover:shadow-glow-purple disabled:opacity-50 transition-all duration-300"
        >
          <Save size={16} />
          {saving ? '保存中...' : '保存设置'}
        </button>
        {saveStatus === 'saved' && (
          <span className="flex items-center gap-1 text-xs text-neon-cyan animate-fade-in">
            <CheckCircle size={14} />
            已保存
          </span>
        )}
        {saveStatus === 'error' && (
          <span className="flex items-center gap-1 text-xs text-red-400 animate-fade-in">
            <AlertCircle size={14} />
            保存失败
          </span>
        )}
      </div>
    </div>
  );
}

function SettingField({
  label, value, onChange, placeholder, type = 'text',
}: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder?: string; type?: string;
}) {
  return (
    <div>
      <label className="block text-xs text-gray-500 mb-1">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-surface border border-surface-border rounded px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-neon-purple transition-colors"
      />
    </div>
  );
}