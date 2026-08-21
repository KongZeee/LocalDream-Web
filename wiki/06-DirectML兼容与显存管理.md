# 06 - DirectML 兼容与显存管理

> 返回 [Wiki 首页](./README.md) | 上一章 [05-数据存储设计](./05-数据存储设计.md) | 下一章 [07-依赖关系](./07-依赖关系.md)

本章是本项目**最有技术含量的部分**，全部实现在 [services/generator.py](../web/backend/app/services/generator.py)。目标：让 Stable Diffusion 在 **AMD 显卡（DirectML 后端）** 上稳定运行，尤其是存在算子缺陷的 **RDNA 1（RX 5000 系列）**。

## 1. 问题背景

torch-directml 把 PyTorch 张量挂在 `privateuseone:0`（DML）设备上，但与 CUDA 相比存在多处缺陷：

| 问题 | 表现 |
|------|------|
| RDNA 1 fp16 算子内部断言失败 | `nn.Embedding` / `nn.Linear` / `F.group_norm` 抛 `INTERNAL ASSERT` 或 `unbox` 错误 |
| 部分卷积形状回退 CPU | `_slow_conv2d_forward` 输出落在 CPU，设备不一致导致后续报错 |
| 调度器张量设备漂移 | `DPMSolverMultistepScheduler` 在 `set_timesteps` 末尾把 `sigmas` 拉回 CPU，而 UNet 输出在 DML，`sample - sigma_t * model_output` 直接崩 |
| DML 无成熟的显存查询/清空 API | OOM 恢复与状态上报需迂回实现 |
| VAE fp16 解码精度/崩溃 | SDXL 的 VAE 在 DML fp16 上解码易崩或OOM |

## 2. 算子级 CPU 回退补丁

策略：**正常路径直接跑 DML，仅在出错/设备不一致时降级到 CPU 计算，再把结果搬回原设备**。这样只有少数坏算子付出 CPU 搬运代价。

### 2.1 `_patch_nn_embedding(m)` — Embedding 设备对齐

输入与权重设备不同且涉及 DML 时：权重与输入都搬到 CPU 执行 `F.embedding`，输出按目标设备/ dtype 回搬。用于 SDXL 的两个文本编码器（`_install_embedding_patch` 一次性遍历 `text_encoder` / `text_encoder_2` 的全部 `nn.Embedding`，模块级锁 + `_EMBEDDING_PATCH_DONE` 保证只打一次）。

### 2.2 `_patch_nn_linear(m)` / `_patched_linear()` — Linear 断言回退

包裹 forward：先尝试原路径；捕获异常且错误串含 `INTERNAL ASSERT` / `unbox` 时，权重、偏置、输入转 CPU fp32 计算，结果回搬原设备与 dtype。

### 2.3 `_patch_nn_conv2d(m)` — Conv2d 双向对齐

两个方向的问题都处理：

1. **入向**：输入在 CPU 而权重在 DML（如进入 UNet 的 `sample`）→ 先 `input.to(weight.device)`
2. **出向**：DML 输入但底层走了 `_slow_conv2d_forward` 回退 CPU → 检测输出设备≠权重设备时搬回，并限次打印诊断日志（`_n_fallback[0] <= 3`）

SDXL 模式下会遍历 UNet 与 VAE 的全部 `nn.Conv2d` 打补丁。

### 2.4 `_patch_nn_groupnorm(m)` — GroupNorm 断言回退

同 Linear 策略：异常含 `INTERNAL ASSERT`/`unbox` 时权重/输入上 CPU 执行 `F.group_norm`，输出回搬到**输入的设备与 dtype**，同样限次打印。

### 2.5 `_patch_scheduler_step(sched)` — 调度器设备对齐

包装 `set_timesteps` 与 `step`：

- `set_timesteps_wrapper`：调用原函数后，若记录了 `_ld_dml_runtime_device`，把 `sigmas / alphas_cumprod / betas / alphas` 重新搬到该设备（对抗 DPM 调度器内部 `sigmas.to("cpu")` 的行为）
- `step_wrapper`：每次以 `sample.device`（UNet 输出所在设备）刷新 `_ld_dml_runtime_device`；`model_output` 与 `sample` 设备不一致时对齐再进原 `step`

每个调度器实例用 `_ld_dml_step_patched` 标记防止重复包装；pipeline 加载与 `_set_scheduler` 两处都会调用。

## 3. SDXL 专用设备布局

`_get_pipeline()` 中对 SDXL 采取**混合设备布局**，在显存与兼容性间折中：

```
┌────────────────────────── SDXL pipeline ──────────────────────────┐
│  text_encoder / text_encoder_2  →  CPU（fp16 Embedding 在 DML 上易崩）│
│  unet                           →  DML fp16 + channels_last（约5GB）│
│  vae                            →  DML fp16（失败则 CPU fp32）        │
│                                     ↳ 仅为让 _execution_device 判定为 │
│                                        DML，使 latent 建在 DML 上     │
│  VAE 解码（最终出图/预览）        →  CPU fp32 专用 VAE（fp16-fix 权重） │
└────────────────────────────────────────────────────────────────────┘
```

要点：

- VAE 换用社区 **`madebyollin/sdxl-vae-fp16-fix`** 权重（DML 上 fp16 稳定）
- VAE 放 DML 的真正目的：pipeline 的 `_execution_device` 取第一个模块设备，VAE 在 DML ⇒ latent 建在 DML ⇒ 与 UNet 输出一致，避免调度器设备错配
- **最终解码走 CPU**：`_get_cpu_vae_fp32()` 懒加载 fp32 VAE 单例（`_cpu_vae_cache`），生图用 `output_type="latent"` 拿潜变量后由 CPU 解码——彻底绕开 DML VAE 解码崩溃与 OOM
- 开启 `enable_vae_slicing()` + `enable_vae_tiling()` + `enable_attention_slicing("max")`（SD 1.5 路径开启常规 attention slicing），压低显存峰值

## 4. Pipeline LRU 缓存与显存策略

### 4.1 缓存结构

- 缓存键：`"{model_id}:{mode}"`，带 LoRA 时追加 `":<{id:weight,id:weight,...}>"`（组合按 id 排序，权重参与键）→ **不同 LoRA 组合 = 不同 pipeline 实例**，互不污染
- 容量：`_MAX_CACHED_PIPELINES = 2`（最多两个 pipeline 驻留显存）
- 顺序表 `_pipeline_access_order`（旧→新）维护 LRU；`_touch()` 命中即移到末尾

### 4.2 逐出时机（三处）

| 时机 | 函数 | 行为 |
|------|------|------|
| 新建 pipeline 前 | `_evict_lru_unlocked()` | 持锁循环逐出最旧，直到数量 < 2 |
| LoRA 组合切换后 | `_cleanup_stale_lora_pipelines(model_id, mode, fresh_key)` | 前缀匹配 `{model_id}:{mode}:<` 且 ≠ 新键的全部驱逐（**立即**释放旧 LoRA pipeline 显存） |
| 显式卸载 API | `unload_model()` / `unload_all_models()` | 按 `model_id:` 前缀或全量清空，随后 `gc.collect()` |

> 直接 `del pipe` 后依赖引用计数 + GC 回收显存；`_evict_lru_unlocked` 在删完仍超额时再补一次 `gc.collect()`。

### 4.3 OOM 自动恢复

`run_inference()`（线程池内执行）的异常处理链：

```
RuntimeError("not enough GPU video memory")
  └─ 未重试过？
       ├─ _empty_dml_cache()   # torch.cuda.empty_cache + torch_directml.empty_cache + gc.collect
       ├─ 重新 _get_pipeline() + _set_scheduler()   # 可能已触发 LRU 逐出
       └─ 再推理一次
            ├─ 成功 → 正常返回
            └─ 仍 OOM → error_message = "OOM after retry: ... Try reducing resolution ..."
```

每个请求只重试一次（`_oom_retried` 标志），避免死循环。其他异常一律转 `{type(e).__name__}: {e}` + 完整堆栈文本进 `error` 事件。

### 4.4 显存观测

- `_report_vram(label)`：`torch_directml.memory_allocated()` / `get_device_properties(0).total_memory` 计算占用并写 stderr；在预热前后、取 pipeline 前后各打一次
- `routes/system.py` 的状态接口：CUDA 优先，DirectML 场景退化为 Win32 系统内存（DML 无显存查询 API）；`services/test_vram.py` 是独立探测脚本

## 5. 模型预热（Warmup）

`_warmup_model(model_id, loras)`：

- `_warmed_up` 集合去重，每个模型进程内只预热一次
- 预热推理：64×64、2 步、guidance 1.0、`output_type="latent"` —— 以最小代价触发权重传输、shader 编译、内存池分配等一次性开销
- 失败只打日志不阻断（真实生图仍会尝试）
- `preload_model()`（模型管理服务）在预加载后直接把模型加入 `_warmed_up`，跳过后续预热

## 6. 兼容性 Shim 与环境补丁

| 位置 | 内容 |
|------|------|
| `generator.py` 模块顶部 | 导入 peft/diffusers 前给 transformers 补 `EncoderDecoderCache` / `DynamicCache` 占位类并 reload——解决 peft≥0.14 对旧版 transformers（如 4.42）的导入兼容 |
| `backend/_patch_transformers.py` | **独立运维脚本**：把本机 transformers `modeling_utils.py` 中 `safe_open(...).metadata()` 的结果兜底为 `{}`，修复 `metadata.get("format")` 对 None 崩溃。路径硬编码作者机器，其他环境需改路径手动运行 |
| `main.py` 顶部 | `sys.path.insert(0, r"C:\py_pkgs")` 作者本机额外包目录（部署到他处可删） |

## 7. 已知限制与注意点

- DirectML 下 `system.py` 的「GPU 显存」实际显示的是**系统物理内存**（Win32 兜底），仅在 CUDA 下才是真实显存
- 调度器张量强制搬 DML 的 `_force_scheduler_to_dml` 已被**禁用**（代码注释保留）：Euler 系调度器在 `set_timesteps` 里对 sigmas 调 `numpy()`，DML 张量会崩；改由 `_patch_scheduler_step` 的运行时对齐兜底
- LoRA 权重参与缓存键：同 LoRA 不同权重会产生新 pipeline 并逐出旧的（显存友好，但反复微调权重会触发重载）
- NVIDIA 显卡理论可用（代码含 CUDA 探测分支），但布局调优均针对 DirectML，未在 CUDA 上测试
