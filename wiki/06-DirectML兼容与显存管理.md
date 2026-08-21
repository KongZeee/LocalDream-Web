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
- 容量：`_MAX_CACHED_PIPELINES`（默认 2；环境变量 `LOCAL_DREAM_MAX_PIPELINES` 可调，16GB+ 显卡建议调大）
- 顺序表 `_pipeline_access_order`（旧→新）维护 LRU；`_touch()` 命中即移到末尾
- **锁策略**：`_pipeline_lock` 为 `RLock`（可重入——`_get_pipeline` 持锁加载时内部会再调用 `_cleanup_stale_lora_pipelines` 拿同一把锁）；`_inference_lock`（普通 Lock）串行化所有 GPU 推理——pipeline 非线程安全且并发推理必然超显存，并发请求排队等待（客户端收到 `notice` 排队提示）

### 4.2 逐出时机（四处）与显存归还

| 时机 | 函数 | 行为 |
|------|------|------|
| 新建 pipeline 前 | `_evict_lru_unlocked()` | 持锁循环逐出最旧，直到数量 < 上限 |
| LoRA 组合切换后 | `_cleanup_stale_lora_pipelines(model_id, mode, fresh_key)` | 前缀匹配 `{model_id}:{mode}:<` 且 ≠ 新键的全部驱逐（**立即**释放旧 LoRA pipeline 显存） |
| OOM 恢复 L1 | `_unload_other_models(keep_model_id)` | 驱逐所有非当前模型的缓存 pipeline（其它模型常是显存超限元凶） |
| 显式卸载 API | `unload_model()` / `unload_all_models()` | 按 `model_id:` 前缀或全量清空；同时清理对应的 SD1.5 CPU VAE 副本缓存（`_cpu_vae_cache_sd15`，每个 ~335MB 系统内存，避免卸载后泄漏） |

> **显存真正归还**：以上所有逐出路径在 `del pipe` 后都会调用 `_empty_dml_cache()`。仅 `del` + `gc.collect()` 只会把块还给 torch 分配器的缓存池，驱动层显存占用并不下降——`empty_cache` 才真正释放。

### 4.3 OOM 分级降级阶梯

`run_inference()`（线程池内执行，持 `_inference_lock`）捕获 `_is_oom()`（宽匹配 `not enough gpu video memory` / `out of memory` / `0x8007000e` / `e_outofmemory` / `video memory`，大小写不敏感——DirectML 的 OOM 有多种报错面目，只匹配一种会漏掉）后按代价从小到大逐级恢复：

```
OOM 捕获（宽匹配）
  ├─ L1：_empty_dml_cache() + _unload_other_models()（驱逐其它模型）→ 重试
  │        └─ 发 notice：「显存不足，已清理缓存并卸载其它模型，正在重试…」
  ├─ L2：kwargs["callback_on_step_end"] = None（关闭实时预览）→ 重试
  │        └─ 发 notice：「显存不足，已关闭实时预览，正在重试…」
  ├─ L3：txt2img 分辨率 ×0.75（对齐 8 的倍数，下限 256）→ 重试
  │        └─ 发 notice：「显存不足，降级至 W×H 重试…」
  └─ 仍失败 → error 事件（含可操作建议；img2img/inpaint 不做 L3，提示换更小参考图）
```

每次成功降级后照常推理；`complete` 事件中的 `width/height` 反映**实际输出尺寸**（可能小于请求值）。非 OOM 异常一律转 `{type(e).__name__}: {e}` + 完整堆栈文本进 `error` 事件。

### 4.4 显存观测

- `_report_vram(label)`：`torch_directml.memory_allocated()` / `get_device_properties(0).total_memory` 计算占用并写 stderr；在预热前后、取 pipeline 前后各打一次
- `routes/system.py` 的状态接口：CUDA 优先，DirectML 场景退化为 Win32 系统内存（DML 无显存查询 API）；`services/test_vram.py` 是独立探测脚本

## 5. 模型预热（Warmup）

`_warmup_model(model_id, loras)`：

- `_warmed_up` 集合去重，每个模型进程内只预热一次
- 预热推理（持 `_inference_lock`）：64×64、2 步、guidance 1.0、`output_type="latent"` —— 以最小代价触发权重传输、shader 编译、内存池分配等一次性开销
- 失败只打日志不阻断（真实生图仍会尝试）
- `preload_model()`（模型管理服务）现调用 `_warmup_model()` 本身——**真预热**（加载 + 推理）；生图路径中 warmup 亦移入线程池执行，不再冻结事件循环

## 6. 兼容性 Shim 与环境补丁

| 位置 | 内容 |
|------|------|
| `generator.py` 模块顶部 | 导入 peft/diffusers 前给 transformers 补 `EncoderDecoderCache` / `DynamicCache` 占位类并 reload——解决 peft≥0.14 对旧版 transformers（如 4.42）的导入兼容 |
| `backend/_patch_transformers.py` | **独立运维脚本**：把本机 transformers `modeling_utils.py` 中 `safe_open(...).metadata()` 的结果兜底为 `{}`，修复 `metadata.get("format")` 对 None 崩溃。路径硬编码作者机器，其他环境需改路径手动运行 |

## 7. 单文件模型与"无名 XL"模型的 SDXL 识别

- **SDXL 判定**（`model_manager.detect_sdxl`，`generator._is_sdxl_model` 包装）：名字含 "xl" → 目录读 `model_index.json` 的 `_class_name` 含 "XL" → 单文件 checkpoint 用 `peek_checkpoint_type` 读 safetensors header（`label_emb` / `conditioner.embedders.1` 即 SDXL）。Anima 2.9B、Pony、Illustrious 等**名字不含 "xl" 的 SDXL 架构模型**因此能正确走混合设备布局（原实现按名字猜测，误判为 SD 1.5 整管上 DML 必炸显存）
- **单文件直载**：`_get_pipeline` 发现本地路径是文件时走 `pipe_cls.from_single_file()`（SD1.5 追加 `safety_checker=None`）；每次冷加载都要在内存中做一次格式转换，常用模型建议 `POST /api/models/convert` 拆包成 Diffusers 目录

## 8. 已知限制与注意点

- DirectML 下 `system.py` 的「GPU 显存」实际显示的是**系统物理内存**（Win32 兜底），仅在 CUDA 下才是真实显存
- 调度器张量强制搬 DML 的 `_force_scheduler_to_dml` 已被**禁用**（代码注释保留）：Euler 系调度器在 `set_timesteps` 里对 sigmas 调 `numpy()`，DML 张量会崩；改由 `_patch_scheduler_step` 的运行时对齐兜底
- LoRA 权重参与缓存键：同 LoRA 不同权重会产生新 pipeline 并逐出旧的（显存友好，但反复微调权重会触发重载）
- NVIDIA 显卡理论可用（代码含 CUDA 探测分支），但布局调优均针对 DirectML，未在 CUDA 上测试
