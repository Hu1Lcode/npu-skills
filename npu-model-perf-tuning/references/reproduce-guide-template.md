# 复现指南模板（`reproduce_guide.md`）

把这次调优沉淀成"别人拿到同一模型也能复现"的指南。与 `final_report.md` 配套，附在 `perf_reports/<日期>/` 下。

## 模板

```markdown
# <模型名> 性能调优复现指南

## 0. 前提
- 硬件：<Ascend 型号 x 卡数>
- 软件：CANN <…>、torch <…>、torch_npu <…>、其它依赖 <…>
- 代码：<repo> @ commit <hash>

## 1. 环境准备
```bash
# 依赖安装（按实际填写）
pip install -r requirements.txt
# 环境变量（来自 optimizations/env.applied.sh）
source ./optimizations/env.applied.sh
```

## 2. 复现开箱基线
```bash
# 原始代码，不加任何优化
<baseline 启动命令，含 warmup/测量步数>
# 期望：E2E ≈ <X> ms 或 s/step ≈ <X>（与报告第4节一致）
```

## 3. 应用优化
```bash
# 1) 代码改动（累积 patch）
git apply ./optimizations/applied.patch
# 2) 环境变量
source ./optimizations/env.applied.sh
# 3) 确认入口已 import torch_npu + transfer_to_npu
grep -n "transfer_to_npu" <entry_file>
```

## 4. 复现调优后性能
```bash
<同口径启动命令>
# 期望：E2E ≈ <Y> ms 或 s/step ≈ <Y>（提升 <Z%>）
```

## 5. 复现 profiling（可选）
```bash
# 见 references/profiling-recipes.md；采 3~5 步
<profiling 命令>
# 产物：perf_reports/<日期>/prof/
```

## 6. 校验精度
```bash
<精度对比命令；与开箱输出/loss 比对，容差 <…>>
```

## 7. 已知注意事项
- 首步慢属正常（预热），计时需跳过 warmup。
- <机器/卡号映射、多卡要求、随机种子等>
```

## 要求
- 每步给出**可直接粘贴**的命令，并写**期望数值**，让别人能判断是否复现成功。
- 明确"开箱"与"调优后"两套命令的区别只在于 patch + 环境变量 + 入口 import。
- 若依赖私有数据/权重，注明获取方式或替代。
