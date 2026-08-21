# 04 - API 接口文档

> 返回 [Wiki 首页](./README.md) | 上一章 [03-前端模块详解](./03-前端模块详解.md) | 下一章 [05-数据存储设计](./05-数据存储设计.md)

后端基础地址默认 `http://127.0.0.1:8081`（可在前端「API 设置」页修改，或用环境变量 `LOCAL_DREAM_HOST/PORT` 调整后端监听）。所有业务端点挂在 `/api` 前缀下，返回 JSON（除图片端点与 SSE）。

## 端点总览

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/generate` | 生图（SSE 流式响应） |
| GET | `/api/health` | 健康检查 |
| GET | `/api/models` | 模型列表 |
| GET | `/api/models/loaded` | 已加载模型 |
| POST | `/api/models/unload` | 卸载模型 |
| POST | `/api/models/unload-all` | 卸载全部 |
| POST | `/api/models/{model_id}/preload` | 预加载模型 |
| POST | `/api/models/download` | 下载模型 |
| DELETE | `/api/models/{model_id}` | 删除模型 |
| GET | `/api/loras` | LoRA 列表（完整） |
| GET | `/api/loras/names` | LoRA 列表（轻量） |
| GET | `/api/history` | 历史分页 |
| GET | `/api/history/{id}/image` | 历史原图（JPEG） |
| GET | `/api/history/{id}/thumbnail` | 历史缩略图（JPEG） |
| DELETE | `/api/history/{id}` | 删除历史 |
| GET | `/api/settings` | 读取设置 |
| PUT | `/api/settings` | 保存设置 |
| GET | `/api/system/status` | GPU/CPU 状态 |
| POST | `/api/upscale` | 图片 4x 放大（multipart） |
| GET | `/` | 服务信息 |

---

## 1. 生图接口

### `POST /api/generate`

**请求体**（`application/json`，GenerateRequest）：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `prompt` | string | ✅ | — | 正向提示词 |
| `negative_prompt` | string | ❌ | `""` | 负向提示词 |
| `model_id` | string | ✅ | — | 模型 ID；本地模型为目录名（`/` 以 `--` 存储），也接受 HF repo ID |
| `mode` | string | ❌ | `"txt2img"` | `txt2img` / `img2img` / `inpaint` |
| `steps` | int | ❌ | `20` | 推理步数（≤0 时由速度预设接管） |
| `cfg` | float | ❌ | `7.5` | CFG 引导系数（≤0 时由速度预设接管） |
| `seed` | int | ❌ | `42` | 随机种子 |
| `width` / `height` | int | ❌ | `512` | 输出尺寸 |
| `scheduler` | string | ❌ | `"dpm"` | `dpm` / `dpm_karras` / `euler` / `euler_karras` / `euler_a` / `lcm` |
| `denoise_strength` | float | ❌ | `0.6` | img2img / inpaint 降噪强度 |
| `image` | string? | ❌ | null | 参考图 base64（img2img / inpaint 必需） |
| `mask` | string? | ❌ | null | 蒙版 base64（inpaint 必需，白色区域为重绘区） |
| `show_preview` | bool | ❌ | `true` | 是否推送中间预览 |
| `preview_stride` | int | ❌ | `1` | 每 N 步推送一次预览 |
| `speed_mode` | string | ❌ | `"balanced"` | `fast` / `balanced` / `quality` |
| `loras` | list? | ❌ | null | `[{id, name, path, filename, weight}]`，path 为 LoRA 文件绝对路径 |

**响应**：`text/event-stream`（SSE），每条消息 `data: {JSON}\n\n`。响应头含 `Cache-Control: no-cache`、`Connection: keep-alive`、`X-Accel-Buffering: no`。

**事件流协议：**

```
data: {"type": "started", "total_steps": 15}
data: {"type": "progress", "step": 1, "total_steps": 15}
data: {"type": "progress", "step": 2, "total_steps": 15, "image": "<base64 jpeg>", "preview_format": "jpeg"}
...
data: {"type": "complete", "image": "<base64 jpeg>", "format": "jpeg", "seed": 12345,
       "width": 512, "height": 512, "generation_time_ms": 45123,
       "first_step_time_ms": 12000, "total_steps": 15}
```

| 事件类型 | 字段 | 说明 |
|---------|------|------|
| `started` | `total_steps` | 推理开始（pipeline 加载/预热完成） |
| `progress` | `step`, `total_steps`, `image?`, `preview_format?` | 每步回调；`image` 为 ≤256px JPEG 预览（受 `show_preview`/`preview_stride` 控制） |
| `complete` | `image`, `format`, `seed`, `width`, `height`, `generation_time_ms`, `first_step_time_ms`, `total_steps` | 完成；后端此时已自动写入历史 |
| `error` | `message` | 失败（含 OOM 重试后的提示、异常堆栈文本） |

> 行为细节：
> - `complete` 到达后，后端异步写入历史记录（落盘失败不影响已返回的图片）。
> - OOM 时后端自动清缓存重试一次，失败才发 `error`。
> - 预览解码失败仅打日志，事件仍会照常发送（无 `image` 字段）。

### `GET /api/health`

```json
{
  "status": "ok",
  "gpu_available": true,
  "gpu_name": "AMD GPU (DirectML)",
  "loaded_model": "aom3a1b",
  "models_dir": "D:/.../web/backend/data/models"
}
```

> 注意：`gpu_available` 当前固定为 `true`、`gpu_name` 固定为 `"AMD GPU (DirectML)"`，是简化实现，不代表真实探测结果。

## 2. 模型接口

### `GET /api/models`

```json
{
  "models": [
    { "id": "aom3a1b", "name": "aom3a1b", "type": "sd15",
      "status": "ready", "size_mb": 4096, "preview_url": "" }
  ],
  "default_model": "aom3a1b"
}
```

逻辑：优先返回 `model_cache` 表记录；表为空时扫描 `MODELS_DIR` 下含 `model_index.json` 的目录。

### `GET /api/models/loaded`

```json
{ "loaded_models": ["aom3a1b"], "current_model": "aom3a1b" }
```

### `POST /api/models/unload`

请求体：`{"model_id": "aom3a1b"}` → 成功 `{"status": "unloaded", ...}`；未加载返回 **404**。

### `POST /api/models/unload-all`

→ `{"status": "all_unloaded"}`

### `POST /api/models/{model_id}/preload`

`model_id` 为**路径参数**（支持含 `/` 的 HF ID，如 `runwayml/stable-diffusion-v1-5`，需 URL 编码）。加载并预热模型，→ `{"status": "loaded", "model_id": ...}`；失败 500。

### `POST /api/models/download`

请求体：`{"model_id": "runwayml/stable-diffusion-v1-5", "model_type": "sd15"}`

> ⚠️ 该接口**同步等待整个 snapshot_download 完成**才响应（下载写入 `model_cache` 表后返回 `{"status": "downloading"}`）。大模型下载期间 HTTP 请求会长时间挂起，前端按阻塞式 UI 处理。

### `DELETE /api/models/{model_id}`

删除本地模型目录与缓存记录 → `{"status": "deleted"}`。

## 3. LoRA 接口

### `GET /api/loras`

```json
{
  "loras": [
    { "id": "1990s", "name": "1990s", "filename": "1990s.safetensors",
      "path": "D:/.../models/Lora/1990s.safetensors", "size_mb": 144.0 }
  ]
}
```

扫描 `{MODELS_DIR}/Lora/` 下 `.safetensors` / `.pt` / `.ckpt`。

### `GET /api/loras/names`

```json
[ { "id": "1990s", "name": "1990s" } ]
```

## 4. 历史接口

### `GET /api/history?page=1&page_size=20`

```json
{
  "items": [
    { "id": 3, "prompt": "...", "negative_prompt": "...", "seed": 123,
      "steps": 15, "cfg": 7.0, "width": 512, "height": 512,
      "model_id": "aom3a1b", "scheduler": "dpm", "mode": "txt2img",
      "image_url": "/api/history/3/image",
      "thumbnail_url": "/api/history/3/thumbnail",
      "created_at": "2025-01-01 12:00:00" }
  ],
  "total": 3, "page": 1, "page_size": 20
}
```

### `GET /api/history/{id}/image` / `GET /api/history/{id}/thumbnail`

直接返回 `image/jpeg` 二进制；记录或文件不存在返回 **404**。

### `DELETE /api/history/{id}`

删除表记录与磁盘上的原图/缩略图 → `{"status": "deleted"}`。

## 5. 设置接口

### `GET /api/settings`

```json
{
  "default_steps": "15", "default_cfg": "7.0",
  "default_width": "512", "default_height": "512",
  "default_scheduler": "dpm",
  "default_negative_prompt": "ugly, blurry, low quality, bad anatomy"
}
```

（值均为字符串，前端按需转型。）

### `PUT /api/settings`

请求体：任意 `{key: value}` 字典（可含自定义键如 `default_model`、`api_base_url`）→ 逐键 `INSERT OR REPLACE` → `{"status": "saved"}`。

## 6. 系统状态接口

### `GET /api/system/status`

```json
{
  "gpu_available": true,
  "gpu_name": "AMD GPU (DirectML)",
  "gpu_memory_used_mb": 6144,
  "gpu_memory_total_mb": 8192,
  "cpu_memory_used_mb": 10240,
  "cpu_memory_total_mb": 32768,
  "loaded_models": ["aom3a1b"],
  "current_model": "aom3a1b"
}
```

探测链：CUDA 显存 → psutil → Win32 `GlobalMemoryStatusEx`（DirectML 模式下 GPU 字段实际是系统物理内存）。内存不可探测时对应字段为 `null`。

## 7. 超分接口

### `POST /api/upscale`

**multipart/form-data：**

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `image` | File | 必填 | 待放大图片（content-type 须为 `image/*`，否则 400） |
| `model_id` | string | `"RealESRGAN_x4plus"` | RealESRGAN 模型名（权重文件需自备） |
| `tile_size` | int | `512` | 分块大小 |

**响应：**

```json
{ "image": "<base64 jpeg>", "format": "jpeg",
  "width": 2048, "height": 2048, "duration_ms": 3200 }
```

> 未安装 `realesrgan`/`basicsr` 时自动降级为 Pillow LANCZOS 4x 放大。

## 8. 根路径

### `GET /`

```json
{ "name": "Local Dream API", "version": "1.0.0" }
```

---

## 错误格式约定

FastAPI 默认错误体：`{"detail": "..."}`。前端 `request()` 会依次尝试 `message` → `detail` → `HTTP {status}` 生成错误文案。生图过程中的业务错误不走 HTTP 状态码，而是通过 SSE `error` 事件返回。
