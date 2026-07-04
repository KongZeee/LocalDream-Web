# 🎨 LocalDream-Web

> 自托管 Stable Diffusion Web UI —— 基于 React + FastAPI + DirectML，专为 AMD 显卡优化。

一个轻量、现代的 Stable Diffusion 生图 Web 界面，可在本地运行，支持文本生图、图生图、局部重绘、LoRA、历史记录等功能。无需 NVIDIA 显卡，AMD GPU 通过 DirectML 原生加速。

---

## ✨ 功能特性

### 核心生图
| 功能 | 说明 |
|------|------|
| 🖼️ **文生图 (txt2img)** | 输入提示词即可生成高质量图像 |
| 🎭 **图生图 (img2img)** | 基于参考图像进行风格迁移或变形 |
| ✏️ **局部重绘 (inpaint)** | 对图像局部区域进行重新绘制 |
| 🔍 **实时预览** | 生图过程中可看到中间预览画面 |
| ⚡ **速度预设** | 快速 / 均衡 / 质量三档速度模式 |
| 🔄 **多种调度器** | DPM / DPM++ Karras / Euler / Euler A / LCM |

### 模型与 LoRA
| 功能 | 说明 |
|------|------|
| 📦 **多模型切换** | SD 1.5 / SDXL 等，自动缓存预热 |
| 🧩 **LoRA 支持** | 多 LoRA 叠加，实时切换，自动卸载旧 LoRA 显存 |
| 🗑️ **模型卸载** | UI 一键卸载指定模型，释放 GPU 显存 |

### 辅助功能
| 功能 | 说明 |
|------|------|
| 📜 **生图历史** | 自动保存每次生成记录，支持删除和缩略图 |
| 🔼 **图片放大** | 内置超分辨率放大功能 |
| 📊 **GPU 状态栏** | 实时监控 GPU 显存占用 |
| ⚙️ **自定义设置** | 分辨率、步数、CFG 值、调度器等全部可调 |

### AMD 显卡专项优化
- ✅ 通过 **torch_directml** 在 AMD 显卡上原生运行（无需 CUDA）
- ✅ 针对 **RDNA 1**（RX 5000 系列）等 DirectML 存在的兼容性 bug，内置了 `nn.Embedding`、`nn.Linear`、`nn.GroupNorm`、`nn.Conv2d` 的自动 CPU 回退补丁
- ✅ 自动开启 `enable_vae_slicing`、`enable_vae_tiling`、`enable_attention_slicing`，大幅降低大分辨率下的显存压力
- ✅ 内置 **OOM 自动恢复机制**：显存耗尽时自动释放缓存并重试一次
- ✅ **LRU 显存管理**：最多同时保留 2 个 pipeline，超过时自动逐出最旧的，防止显存泄漏

---

## 🖥️ 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10/11 | Windows 11 |
| **GPU** | AMD RX 5700 XT (8GB) 及以上 | AMD RX 6800 XT (16GB) 及以上 |
| **显存** | 8GB | 16GB+ |
| **内存** | 16GB | 32GB |
| **Python** | 3.11 / 3.12 | 3.12 |
| **Node.js** | 18+ | 20+ |

> **NVIDIA 显卡用户**：理论上可以适配（将 `torch_directml` 替换为 CUDA 版本），但本项目主要针对 AMD + DirectML 场景开发和测试。

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/<你的用户名>/LocalDream-Web.git
cd LocalDream-Web
```

### 2. 安装 Python 依赖

本项目需要以下 Python 包（建议在虚拟环境中安装）：

```bash
pip install torch torch-directml fastapi uvicorn aiosqlite diffusers transformers accelerate safetensors pillow peft
```

**最低版本要求：**
```
torch           >= 2.0.0
torch-directml  >= 0.2.0
fastapi         >= 0.100.0
uvicorn         >= 0.20.0
aiosqlite       >= 0.19.0
diffusers       >= 0.25.0
transformers    >= 4.30.0
accelerate      >= 0.20.0
peft            >= 0.5.0
```

### 3. 安装前端依赖

```bash
cd web
npm install
```

### 4. 下载模型文件

将模型文件放入 `web/backend/data/models/` 目录：

```
web/backend/data/models/
├── aom3a1b/          ← SD 1.5 模型（示例）
├── counterfeit-v3.0/ ← SD 1.5 模型（示例）
└── Lora/             ← LoRA 文件(.safetensors)
```

也可通过 `Settings` 页面配置其他模型目录。

### 5. 启动项目

双击 `start.bat` 或在命令行运行：

```cmd
start.bat
```

启动后自动打开：
- 🎨 **前端界面**：http://localhost:5173
- ⚙️ **后端 API**：http://127.0.0.1:8081

---

## 📁 项目结构

```
LocalDream-Web/
├── start.bat / stop.bat          # Windows 一键启停脚本
├── LocalDream-SD.json            # SillyTavern 插件配置（可选）
├── web/
│   ├── backend/
│   │   ├── run.py                # 后端入口（启动 uvicorn）
│   │   ├── app/
│   │   │   ├── main.py           # FastAPI 路由注册
│   │   │   ├── config.py         # 全局配置（路径、端口等）
│   │   │   ├── db.py             # SQLite 历史记录数据库
│   │   │   ├── routes/           # API 路由（生图/LoRA/模型/历史等）
│   │   │   └── services/         # 核心业务逻辑
│   │   │       ├── generator.py  # ⭐ 生图引擎 + OOM 保护 + LRU 缓存
│   │   │       ├── lora_manager.py # LoRA 加载/卸载管理
│   │   │       ├── model_manager.py # 模型扫描/切换
│   │   │       ├── upscaler.py   # 图片放大
│   │   │       └── history.py    # 生图历史管理
│   │   └── data/
│   │       ├── models/           # 放置模型和 LoRA 文件
│   │       ├── outputs/          # 生成图片输出
│   │       └── localdream.db     # SQLite 数据库
│   ├── src/
│   │   ├── pages/               # 页面组件（生图/历史/模型/设置/放大）
│   │   ├── components/          # 公共组件（侧边栏/布局/状态栏）
│   │   ├── stores/              # Zustand 状态管理
│   │   └── utils/               # API 请求封装
│   └── package.json
```

---

## ⚙️ 环境配置

### 环境变量

通过环境变量自定义路径和端口：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LOCAL_DREAM_DATA_DIR` | `web/backend/data` | 数据根目录 |
| `LOCAL_DREAM_MODELS_DIR` | `{DATA_DIR}/models` | 模型目录 |
| `LOCAL_DREAM_OUTPUT_DIR` | `{DATA_DIR}/outputs` | 输出图片目录 |
| `LOCAL_DREAM_DB_PATH` | `{DATA_DIR}/localdream.db` | SQLite 数据库路径 |
| `LOCAL_DREAM_HOST` | `0.0.0.0` | 后端监听地址 |
| `LOCAL_DREAM_PORT` | `8081` | 后端端口 |

**示例：**

```cmd
set LOCAL_DREAM_PORT=9090
set LOCAL_DREAM_MODELS_DIR=D:\SD-Models
python web/backend/run.py
```

---

## 🔧 AMD 显卡常见问题

### RDNA 1（RX 5000 系列）特殊说明

AMD RDNA 1 架构的 DirectML 实现在某些操作（`nn.Embedding`、`nn.Linear`、`F.group_norm`、`F.conv2d`）上存在内部断言失败的问题。本项目已通过以下方式处理：

1. **CPU 回退补丁**：当 DirectML 操作报 `INTERNAL ASSERT` 或 `unbox` 错误时，自动将计算回退到 CPU 执行，再将结果移回 GPU
2. **VAE 解码走 CPU**：SDXL 的 VAE 解码在 CPU fp32 上执行，避免 DirectML 精度问题
3. **VAE Slicing/Tiling**：减少单次 VAE 解码的显存峰值
4. **Attention Slicing**：减少 UNet 注意力计算的显存占用

### 显存不足时的建议

1. 降低生成分辨率（如从 1024×1024 降到 512×512）
2. 使用快速模式（减少步数）
3. 减少同时加载的 LoRA 数量
4. 及时卸载不用的模型
5. 关闭其他占用显存的程序

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **前端** | React 18 + TypeScript + Vite + TailwindCSS + Zustand + Lucide Icons |
| **后端** | FastAPI + Uvicorn + SQLite (aiosqlite) |
| **AI 推理** | Diffusers + HuggingFace Transformers + PEFT (LoRA) |
| **GPU 加速** | PyTorch + torch-directml（AMD 显卡）/ CUDA（NVIDIA 兼容） |
| **图像处理** | Pillow |

---

## 📝 开发说明

### 前后端联调

```bash
# 终端 1：启动后端
cd web/backend
python run.py

# 终端 2：启动前端
cd web
npm run dev
```

### 构建生产版本

```bash
cd web
npm run build
```

构建产物位于 `web/dist/`，可通过任何静态文件服务器托管。

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/generate` | POST | 发起生图请求（SSE 流式返回） |
| `/api/models` | GET | 获取可用模型列表 |
| `/api/models/{id}/preload` | POST | 预加载模型到显存 |
| `/api/models/loaded` | GET | 获取当前已加载的模型 |
| `/api/loras` | GET | 获取可用 LoRA 列表 |
| `/api/history` | GET/POST/DELETE | 生图历史管理 |
| `/api/upscale` | POST | 图片放大 |
| `/api/system/status` | GET | 系统状态（GPU/显存） |
| `/api/health` | GET | 健康检查 |

---

## 🤝 贡献

欢迎提交 Issue 和 PR！请确保：
1. 代码符合现有风格
2. 新功能附上简要说明
3. AMD 显卡兼容性变更请在 PR 描述中注明测试机型

---

## 📄 许可证

本项目代码采用 **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**（署名-非商业性使用 4.0 国际）许可协议。

**这意味着：**
- ✅ 你可以自由使用、修改、分发本项目代码
- ✅ 可以在个人项目中使用本项目
- ❌ **不得用于任何商业目的**
- ❌ 不得声称本项目由你原创

> ⚠️ **注意**：本项目使用的 Stable Diffusion 模型权重（如 SD 1.5、SDXL）及 LoRA 文件均受其各自原始许可证约束。请在下载和使用模型权重前，确认并遵守各模型的许可条款。

---

## 🙏 致谢

- [Stability AI](https://stability.ai/) —— Stable Diffusion
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)
- [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/) —— AMD GPU 推理加速
- [Vite](https://vite.dev/) + [React](https://react.dev/)

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with ❤️ by [Your Name]

</div>
