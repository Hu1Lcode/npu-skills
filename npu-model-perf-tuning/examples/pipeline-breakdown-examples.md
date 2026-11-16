# 流水线拆解示例（P1）

P1 的目标：从启动脚本追调用链，产出"整条流水线有哪些阶段、每阶段用哪个模型/模块、源码 file:line"。下面给两个真实拆解范例，照着套到新模型。

## 范例 A：AIGC 图像生成 / 编辑（Qwen-Image-Edit）
入口 `qwen-image/diffusers_infer.py` → `QwenImageEditPlusPipeline`。

| # | 阶段 | 模型/模块 | 源码 | 说明 |
|---|------|-----------|------|------|
| 1 | prepare_data / encode_prompt | text_encoder（+ tokenizer） | pipeline 内 `encode_prompt` | 文本编码，一次 |
| 2 | prepare_latents / vae encode | VAE (encoder) | pipeline 内 `vae.encode` | 输入图 → latent |
| 3 | denoise loop | DiT (transformer) | pipeline 内 `transformer(...)` × 采样步 | **最重**，多步循环 |
| 4 | vae_decode | VAE (decoder) | pipeline 内 `vae.decode` | latent → 图像 |

打点建议：
```
with t.stage("encode_prompt"): prompt_emb = encode_prompt(...)
with t.stage("vae_encode"):    latents = vae.encode(image)
for i, ts in enumerate(timesteps):
    with t.stage(f"denoise_step"): latents = transformer(latents, ts, ...)
with t.stage("vae_decode"):    out = vae.decode(latents)
```
DiT 用 fp32 时尤其要单独量出 denoise 占比（参考 `dit_fp32_profiling_report.md`，ReduceSum 曾占 69.3%）。

## 范例 B：说话人分离（DiariZen / pyannote 风格）
入口 `ascend-model/DiariZen/diarizen/pipelines/inference.py:132-197` `__call__`。

| # | 阶段 | 模型/模块 | 源码 | 说明 |
|---|------|-----------|------|------|
| 1 | 读音频 + 重采样 | I/O + resample | `inference.py` __call__ 开头 | 数据准备 |
| 2 | get_segmentations | 分割模型（powerset） | `inference.py` __call__ | 逐 chunk 滑窗，产出 speaker activity |
| 3 | median filter + binarize | 后处理 | `inference.py` | 平滑、二值化 |
| 4 | speaker_count | 计数 | `inference.py` | 估计人数 |
| 5 | get_embeddings | embedding 模型 | `inference.py` | 每段抽 speaker embedding |
| 6 | clustering | 聚类 | `inference.py:166` 附近 | embedding → 说话人聚类 |
| 7 | reconstruct | 重建标注 | `inference.py` | binarize 后处理成 annotation |

对比参照：`pyannote/audio/pipelines/utils/hook.py:135-189` 的 `TimingHook` 已用 `time.time()` 记每步（无 synchronize）——我们照它的粒度，但在打点前补 `torch.npu.synchronize()`。

## 范例 C：训练 step
入口 `trainer_single_opt.py:444-460`。一步内部：

| # | 子阶段 | 说明 |
|---|--------|------|
| 1 | next(dataloader) | 取 batch（含数据加载/预处理，常是空泡来源） |
| 2 | training_step / forward | 前向 |
| 3 | loss.backward() | 反向 |
| 4 | optimizer.step() + zero_grad() | 更新 |

打点建议：train_step 内每个子阶段一个 stage，多步记录，最后报 s/step（与 log 里的 s/step 对齐校验）。

## 拆解时抓什么
1. 启动脚本 → 主函数 → pipeline/runner 类 → `__call__`/`forward`/`train_step`。
2. 每个"模型"调用点记 file:line；AIGC 要区分"一次性"（encode prompt/vae encode）与"循环内"（denoise，乘采样步数）。
3. 训练要单独关注 dataloader 段（是否 host 成瓶颈）。
4. 拆解表直接进最终报告 §3。
