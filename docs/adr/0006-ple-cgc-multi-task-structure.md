# PLE-CGC 多任务结构

**Status: accepted**

## Context

现有两轴模型 API 已能自由组合特征编码器与多任务结构，但多任务结构只有 MMoE 与 Shared-Bottom。两者都缺少 PLE 的逐层 shared/task-specific 抽取路径，无法验证渐进式任务分化能否改善当前四任务预测。该扩展必须保持既有输出、保存/加载、实验元数据和验证集选型纪律。

## Decision

在多任务结构轴注册 `ple_cgc`（Progressive Layered Extraction with Customized Gate Control），并允许与 `mlp`、`dcn`、`senet` 三个编码器组合。

- 首版默认 `num_layers=1`，但公开参数保留为正整数 `num_layers=n`；`0`、负数、布尔值和非整数均拒绝。
- 每层参数独立，不跨层共享。默认每层有 2 个 shared experts、每个任务 2 个 task-specific experts，expert units=48。
- 第一层的 shared 与各 task 分支都接收编码器输出；后续层的 shared experts 接上一层 shared 表征，各任务 experts 接对应任务的上一层表征。
- 每层保留一个 shared gate 和每任务一个 task gate，全部使用 softmax。
- task gate 只混合本层 shared experts 与对应任务的 experts；shared gate 混合本层 shared experts 与全部任务的 task-specific experts。
- 最后一层只将各 task-specific 输出送入各自的 `Dense(32, ReLU) → Dense(1, sigmoid)` tower；shared 输出不直接连接预测头。
- 自定义层必须进入统一 `custom_objects()`，并保证 `.keras` 保存后在新进程加载可复现预测；每个 run 的结构 metadata 记录全部生效超参。

## Experiment protocol

先以 `ple_cgc+mlp` 验证结构本身的增量，再与 baseline v2 当前验证集冠军 `mmoe+senet` 比较。正式结论只允许使用验证集、`val_auc_mean` 早停和 seeds 2025/2026/2027；最终确认集保持未加载，直到候选配置正式锁定。工程 smoke 与单元测试只证明可运行，不产生性能结论。

## Consequences

逐层独立参数使 `num_layers` 表示真实的 progressive extraction 深度，但层数增长会线性增加专家与 gate 参数。最后一层 shared 输出不直接进入 tower，可保持任务分化语义：shared experts 只通过 task gate 影响任务输出，shared gate 输出只供更深层的 shared 分支使用。因此单层配置下最后一个 shared gate 没有反向梯度，这是该结构决策的已知结果，而不是训练失败。注册后模型自检矩阵由 2 个结构 × 3 个编码器扩展为 3 × 3，加上 3 个 Single-task 编码器和 Logistic，共 13 个变体。
