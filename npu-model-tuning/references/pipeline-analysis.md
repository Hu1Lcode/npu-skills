# P1 · 静态分析：调用链拆解与打点位置

目标：从**启动脚本**追到真正执行计算的位置，产出一张可打点的调用链，并画出带打点位置的 Mermaid 流程图。这张图是后续所有打点、profiling 归因和报告的统一依据。

## 怎么追踪

1. **找入口**：读启动脚本（`.sh` / `run.py` / `benchmark.py` / `torchrun` 命令行），记录：
   - 设置了哪些环境变量（尤其是 `ASCEND_RT_VISIBLE_DEVICES`、`TASK_QUEUE_ENABLE`、`PYTORCH_NPU_ALLOC_CONF`）
   - 真正调用的 Python 入口文件/模块与参数
2. **顺藤摸瓜**：从入口 `main()` 往下读：
   - 模型/流水线构造处：`from_pretrained`、`AutoModel`、`Pipeline(...)`、`build_model()`、`load_checkpoint()`
   - 执行处：`__call__`、`.generate()`、`.train()`、`train_step()`、`forward()`
   - 数据处：`DataLoader`、`dataset.__getitem__`、预处理函数
3. **切分阶段**：把执行路径切成有序的、可观测的阶段，判定推理还是训练。
   - 推理：拆出整条流水线每一部分及它使用的模型/模块
   - 训练：拆出一个 step 的各部分
4. **记录打点位置**：每个阶段"开始前 / 结束后"各一个时间点，标到 `file:line`。
5. **判定先验热点**：哪段最重（大模型 forward、循环 `num_steps`、逐样本 Python 循环、numpy 聚类等），给 P4 profiling 一个预期。

## 输出：`pipeline_breakdown.md`

### 1. 调用链表

| # | 阶段名 | 使用的模型/模块 | 源码 file:line | 输入 → 输出 | 打点位置 | 先验热点 |
|---|--------|----------------|----------------|-------------|----------|----------|
| 1 | prompt_encode | text_encoder + tokenizer | `pipeline.py:120` | prompt → embeds | T1: `pipeline.py:118`, T2: `pipeline.py:125` | 低 |
| 2 | image_encode | VAE encoder | `pipeline.py:130` | 参考图 → latent | T3/T4 | 中 |
| 3 | dit_denoise | DiT transformer | `pipeline.py:180` | latent → latent | T5/T6（含循环内单步 T5a/T6a） | 高 |
| 4 | vae_decode | VAE decoder | `pipeline.py:240` | latent → 输出 | T7/T8 | 中 |

### 2. Mermaid 流程图（必须标注打点位置）

在报告中用 Mermaid 画调用链，并用 `probe` class 标注打点位置：

```mermaid
flowchart TD
    A[启动脚本 run.sh] --> B[python main.py --config ...]
    B --> C[构建 Pipeline / 加载模型]
    C --> D["prompt_encode<br/>text_encoder + tokenizer"]
    D --> E["image_encode<br/>VAE encoder"]
    E --> F["dit_denoise<br/>DiT × num_steps"]
    F --> G["vae_decode<br/>VAE decoder"]
    G --> H[后处理 / 保存输出]

    D -.-> T1((T1/T2))
    E -.-> T3((T3/T4))
    F -.-> T5((T5/T6))
    F -.-> T5a((每步 T5a/T6a))
    G -.-> T7((T7/T8))

    classDef probe fill:#ffe,stroke:#e90,stroke-width:2px;
    class T1,T3,T5,T5a,T7 probe;
```

> 如果调用链有分支（如 CFG、不同输入路径、条件分支），把分支画出来；计时点放在"每次都会经过的边界"上。循环体（`num_steps`、逐样本 for）要单独标出，因为它常是隐蔽热点。

## 扩散模型拆解模板（重点）

以视频/图像编辑模型为例，pipeline 一般由三部分组成：

| 阶段 | 包含内容 | 说明 |
|---|---|---|
| `prompt_encode` | tokenizer + text_encoder | prompt / 文本编码，通常每次请求执行一次 |
| `image_encode` | VAE encoder（参考图/源视频编码） | 图像编辑、图生视频、视频编辑场景有；纯文生图可能没有 |
| `dit_denoise` | DiT transformer × `num_inference_steps` | 去噪循环，通常是最重阶段；需记录总耗时和单步平均 |
| `vae_decode` | VAE decoder | latent → 最终图像/视频帧 |

如果模型有额外的音频编码器、视觉编码器（如 VLM 条件编码）、文本/图像交叉注意力预处理，也作为独立阶段列出，不要混进 `prompt_encode`。

## 训练模型拆解模板

| # | 阶段 | 说明 |
|---|---|---|
| 1 | dataloader / batch fetch | 含数据加载、预处理、H2D 拷贝（常是空泡来源） |
| 2 | forward | 前向 + loss 计算 |
| 3 | backward | 反向 |
| 4 | optimizer.step() | 更新权重（含梯度累积 / DDP/HCCL 通信） |
| 5 | zero_grad() | 清零 |

多卡/多进程：标注哪些阶段含通信（DDP all-reduce、HCCL）；把 `torchrun` 启动方式、rank 数、并行策略写进 P0 记录。

## 注意事项

- 拆解要到"可观测的边界"：不必拆到单个算子（那是 profiling 的事），但要能回答"时间花在哪一段、这段用哪个模型"。
- 自研框架（如 cosmos-framework）没有标准 pipeline 类时，从启动脚本的 `main()` 和配置构建处开始追，找到实际执行循环；不要按框架名字猜。
- 静态分析看不清的地方，用 `torch_npu.profiler` 的 `with_stack=True` 抓一次调用栈来校正。
- 记录 batch / seq / dtype / 并行配置，报告和复现都要用。
