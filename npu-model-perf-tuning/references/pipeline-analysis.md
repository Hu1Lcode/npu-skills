# P1 · 静态分析：调用链拆解（Pipeline / Train-step Decomposition）

目标：在动手优化前，先从**启动脚本**追到真正执行计算的位置，产出一张可打点的「流水线拆解表」。这张表是后续所有打点、profiling 归因、报告调用栈的统一依据。

## 怎么追踪

1. **找入口**：读启动脚本（`*.sh` / `run.py` / `benchmark.py` / `torchrun` 命令行）。记下真正调用的 python 入口文件与参数。
2. **顺藤摸瓜**：从入口 `main()` 往下读，`grep` 出模型构造处（`from_pretrained` / `AutoModel` / `Pipeline(` / `Trainer(`）与执行处（`__call__` / `.generate` / `.train()` / `train_step` / `training_step`）。
3. **切分阶段**：把执行路径切成有序阶段；判定**推理**还是**训练**。
   - 推理：拆出整条流水线**每一部分 + 每部分用的是哪个模型/模块**。
   - 训练：拆出**一个 step** 的各部分。
4. **记录打点位置**：每个阶段"开始前 / 结束后"各一个时间点，标注到 `file:line`。
5. **判定瓶颈先验**：哪个阶段最"重"（大模型 forward、循环 num_steps、逐样本 python 循环、numpy 聚类等），给后续 profiling 一个预期。

## 输出表格（写入 `perf_reports/<日期>/pipeline_breakdown.md`）

| # | 阶段名 | 使用的模型/模块 | 源码 file:line | 输入 → 输出 | 打点位置 |
|---|--------|-----------------|----------------|-------------|----------|
| 1 | ... | ... | ... | ... | before/after 行号 |

再补一列"备注/先验热点"，方便 P2 后对照。

## 三个真实样例

### A. AIGC 扩散推理（Qwen-Image-Edit）
入口 `qwen-image/diffusers_infer.py` → `QwenImageEditPlusPipeline`：

| # | 阶段 | 模型/模块 | 说明 |
|---|------|-----------|------|
| 1 | prepare_data | `text_encoder` + VAE/图像编码 | prompt 文本编码 + 输入图编码成 latent |
| 2 | denoise | DiT / transformer | 循环 `num_steps` 去噪循环（通常最重） |
| 3 | vae_decode | VAE decoder | latent → 最终图像 |

### B. 说话人分离推理（DiariZen）
`ascend-model/DiariZen/diarizen/pipelines/inference.py:132-197` 的 `__call__`：

| # | 阶段 | 模型/模块 | 说明 |
|---|------|-----------|------|
| 1 | read_audio / preprocess | 预处理 | 读音频、重采样 |
| 2 | get_segmentations | 分割模型 | 语音活动/分段 |
| 3 | median filter + binarize | 后处理 | 平滑、二值化 |
| 4 | speaker_count | 估计器 | 估计说话人数量 |
| 5 | get_embeddings | embedding 模型 | 提取说话人 embedding（可能较重） |
| 6 | clustering | 聚类（VBx/AHC 等） | 逐段聚类（可能 CPU numpy 重） |
| 7 | reconstruct + binarize | 组装 | 生成 annotation |

> 参照：`pyannote/audio/pipelines/utils/hook.py:135-189` 的 `TimingHook` 已用 `time.time()` 记每段耗时（但**没有** synchronize），可作为打点位置的现成参照——我们补上同步即可。

### C. 训练 step
`ascend-model/DiariZen/diarizen/trainer_single_opt.py:444-460`：

| # | 阶段 | 说明 |
|---|------|------|
| 1 | dataloader fetch (+H2D) | 取一个 batch 到 device |
| 2 | forward (`training_step`) | 前向 + 计算 loss |
| 3 | loss.backward() | 反向 |
| 4 | optimizer.step() | 更新权重（含梯度累积/DDP 通信） |
| 5 | zero_grad() | 清零 |

## 注意事项
- 拆解要**到"可观测的边界"**：不必拆到单个算子（那是 profiling 的事），但要能回答"时间花在哪一段、这段用哪个模型"。
- 多卡/多进程：标注哪些阶段含通信（DDP all-reduce、HCCL）。
- 循环体（`num_steps`、逐样本 for）要单独标出——它们常是隐蔽热点。
- 记录 batch/seq/dtype/并行配置，报告要用。
