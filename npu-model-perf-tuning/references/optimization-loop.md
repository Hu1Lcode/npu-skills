# P5 · 优化迭代：单变量 · 固化 · 侵入性 · 冲突交人工

这是优化的核心循环。每轮：**读证据 → 只改一个变量 → 复测 → 判三态 → 固化进基线 → 回到 P4**。

## 为什么一次只改一个变量
多个改动一起上，无法归因是哪个起了作用、哪个其实有害；也无法在某个改动导致精度回归时定位。单变量 + 每次复测，才能把"有效"沉淀为可复用的确定收益。

## 侵入性分级（决定做不做 / 交不交人工）

| 等级 | 典型手段 | 处理 |
|---|---|---|
| **低** | 环境变量、`torch_npu` 接入、融合算子替换、config 级（dtype/并行/batch 微调） | 直接做，固化 |
| **中** | 局部代码改写、cast 消除、循环/索引优化、少量接口替换 | 做，但需精度+回归校验 |
| **高** | 大规模重构、替换核心模块/算法、改并行范式、重写数据管线 | **只列入"高侵入候选清单"交用户判断是否要做，不自动实施** |

> 用户关注的是"对代码侵入性改动较少"的优化点。高侵入项即使收益大也先列出、给出预估收益与工作量，由用户拍板。

## 实验协议
1. 从**当前固化基线**出发（不是从开箱出发，否则会丢已得收益）。
2. 只应用一个改动（改一处代码 / 加一个环境变量）。
3. 用 P2/P7 的**同口径**复测（同 warmup/步数/batch/机器状态）；建议重复 ≥3 次取中位数，抵消抖动。
4. **增益判定**：相对当前基线提升要显著大于运行间抖动（经验：>3~5% 且重复可复现才算数）；同时过精度校验。
5. 三态处理：
   - ✅ 有效 → 固化；更新"当前基线"为该结果。
   - ❌ 无效/负收益/精度回归/variance 内 → 否决，回滚。
   - ⚠️ 与已有优化冲突、或需大改代码、或需在多个方案间取舍 → 不自行决定，记"待决策"交用户。

## 固化产物
- `optimizations/applied.patch`：所有已固化代码改动的累积 diff（可 `git apply` 回放）。
- `optimizations/env.applied.sh`：所有已固化环境变量（可 `source`）。
- 台账 `optimization_ledger.md`（模板见 `assets/optimization-ledger.template.md`）：每次实验一行，含 before→after、提升%、原因/依据、精度结论、状态。

## 冲突与待决策（一定要交人工）
典型：
- `PYTORCH_ALLOC_CONF` 与 `PYTORCH_NPU_ALLOC_CONF` 互斥（同义，不能同设）。
- 融合算子与 dtype/精度要求冲突（如某融合算子只支持某 dtype）。
- 两个优化互相抵消（一个改善通信、一个破坏重叠）。
- 需要大规模改代码的高侵入方案。
处理：在台账标 `⚠️ 待决策`，**汇总成一个决策清单给用户**（列：冲突点、各方案收益/代价、建议），等用户选择后再动。

## 循环终止
达标（目标指标满足）或持续无更多"低/中侵入"收益时停止；剩下的高侵入/待决策项进入最终报告。

## 一轮的伪代码
```
while not reached_target:
    prof = collect_and_analyze(current_state, steps=3)   # P4
    cand = pick_low_intrusion_candidate(prof)            # 从诊断playbook选
    if not cand: break
    apply_one(cand)
    before, after = remeasure()                           # 同口径, >=3 次中位数
    if significant(after) and precision_ok():
        freeze(cand); update_baseline(after)             # 固化
    else:
        rollback(cand)                                    # 否决
    if conflicting(cand): record_decision_item(cand)      # 交人工
```
