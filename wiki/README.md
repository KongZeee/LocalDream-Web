# LocalDream-Web Code Wiki

> 自托管 Stable Diffusion Web UI —— 基于 React + FastAPI + DirectML，专为 AMD 显卡优化。

本 Wiki 基于 **LocalDream-Web** 项目源码分析生成，覆盖项目整体架构、模块职责、关键类与函数、依赖关系以及运行部署方式。

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [01-项目概览与架构](./01-项目概览与架构.md) | 项目定位、技术栈、整体架构图、请求链路、目录结构 |
| [02-后端模块详解](./02-后端模块详解.md) | FastAPI 后端各模块职责、关键类与函数逐个说明 |
| [03-前端模块详解](./03-前端模块详解.md) | React 前端页面、组件、状态管理、API 封装说明 |
| [04-API接口文档](./04-API接口文档.md) | 全部 REST 端点、请求/响应模型、SSE 流式协议 |
| [05-数据存储设计](./05-数据存储设计.md) | SQLite 表结构、数据目录布局、历史记录持久化流程 |
| [06-DirectML兼容与显存管理](./06-DirectML兼容与显存管理.md) | AMD RDNA 1 兼容补丁、OOM 自动恢复、LRU 显存缓存（项目核心技术） |
| [07-依赖关系](./07-依赖关系.md) | 前后端依赖清单、模块依赖图、第三方库版本要求 |
| [08-部署与运行指南](./08-部署与运行指南.md) | 环境要求、安装步骤、启动方式、环境变量配置、生产构建 |

---

## 项目速览

**LocalDream-Web** 是一个轻量、现代的本地 Stable Diffusion 生图 Web 界面，核心特点：

- **无需 NVIDIA 显卡**：通过 `torch_directml` 在 AMD 显卡上原生运行（DirectML 后端）
- **完整生图功能**：文生图（txt2img）、图生图（img2img）、局部重绘（inpaint）、实时中间预览（SSE 流式）
- **模型管理**：SD 1.5 / SDXL 多模型切换、预加载预热、一键卸载、HuggingFace 下载
- **LoRA 支持**：多 LoRA 叠加、实时权重调节、切换组合时自动清理旧 pipeline 显存
- **辅助能力**：生图历史（SQLite + 缩略图）、RealESRGAN 4x 超分放大、GPU/内存实时状态栏
- **RDNA 1 专项优化**：针对 RX 5000 系列 DirectML 的算子 bug 内置 CPU 回退补丁
- **显存保护**：VAE Slicing/Tiling、Attention Slicing、OOM 自动重试、最多缓存 2 个 pipeline 的 LRU 逐出策略

## 技术栈一览

| 层级 | 技术 |
|------|------|
| 前端 | React 18 + TypeScript + Vite 6 + TailwindCSS 3 + Zustand 5 + React Router 7 + Lucide Icons |
| 后端 | Python + FastAPI + Uvicorn + Pydantic + aiosqlite (SQLite) |
| AI 推理 | Diffusers + Transformers + PEFT (LoRA) + huggingface_hub |
| GPU 加速 | PyTorch + torch-directml（AMD）/ CUDA（NVIDIA 兼容路径） |
| 图像处理 | Pillow + RealESRGAN / BasicSR（可选超分） |

## 快速启动

```cmd
:: 1. 安装后端依赖（建议虚拟环境）
pip install torch torch-directml fastapi uvicorn aiosqlite diffusers transformers accelerate safetensors pillow peft

:: 2. 安装前端依赖
cd web && npm install

:: 3. 将 SD 模型放入 web/backend/data/models/

:: 4. 一键启动（Windows）
start.bat
```

启动后：
- 前端界面：<http://localhost:5173>
- 后端 API：<http://127.0.0.1:8081>

详见 [08-部署与运行指南](./08-部署与运行指南.md)。
