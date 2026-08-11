# QSMLA 单核 FP16 完整语义适配设计

日期：2026-08-11

目标仓库：`SuperNPUBench`

目标目录：`benchmark/one-level-arch`

## 1. 背景

当前 SuperNPU QSMLA 实现已完成固定二维规格下的基础分块和 SWA：

- Q、共享 K/V 和输出为连续二维张量；
- QK 沿 D 分块，独立 `TMATMUL` 后在 Vec 上累加；
- FP32 Online Softmax；
- one-pass 以及两个 two-pass 版本；
- 默认规格 `s1=64, s2=128, D=512, Tm=32, Tk=32, Td=64` 已通过精度验证。

transformer 原算子还包含五种模板模式、TND、Paged Attention、变长序列、sinks、softmax LSE、稀疏索引和压缩 KV。当前适配文件虽保留部分参数，但多数参数未被函数体消费。

本设计在不引入 metadata 分核和 HIFLOAT8 的前提下，实现原算子的完整计算语义，为后续多核和量化适配建立稳定核心。

## 2. 目标与非目标

### 2.1 本阶段目标

使用当前 PTO 基本块、单核顺序执行和 FP16 输入/输出，实现：

1. 五种模式：
   - SWA
   - HCA
   - CSA
   - ORI_SPARSE
   - ORI_CMP_SPARSE
2. Q/KV 的 TND 布局。
3. KV 的 PA_BBND 布局及 ori/cmp block table。
4. 多 batch、N1/N2 和 GQA 分组的单核顺序遍历。
5. `cu_seqlens_*`、`seqused_*` 表达的变长序列。
6. NoMask、RightDownCausal 和 Band mask。
7. `ori_win_left/right=-1` 的不受限语义。
8. ori/cmp sparse indices、topk length 和无效索引 `-1`。
9. `cmp_ratio` 和 `cmp_residual_kv`。
10. sinks。
11. 正确的 FP32 softmax LSE：`m + ln(l)`。
12. 固定基本块下的非整除序列尾块正确性。

### 2.2 本阶段非目标

本阶段明确不实现：

- metadata 解析和多核任务分发；
- HIFLOAT8 输入和真实 descale；
- 强制 BF16 输出对齐；
- split-G、流水、双缓冲等性能优化；
- 完整 aclnn host 校验层；
- 与原算子完全一致的性能或硬件资源使用方式。

`metadata` 和 descale 参数可保留在入口签名中，但本阶段不参与任务分发或 FP16 计算。

## 3. 总体架构

采用共享核心架构：

```text
Runtime Params
    |
    v
Sequence Context Builder
    |
    v
Mode Planner -----> KvWorkItem 序列
    |                       |
    |                       v
    +--------------> KV Address Resolver
                            |
                            v
                   Safe FP16 KV Tile
                            |
                            v
              Shared Fused Online Softmax
                            |
                            v
                    Attention Out + LSE
```

边界原则：

- Planner 只决定选取哪些逻辑 token。
- Resolver 只把逻辑 token 映射到连续或 PA 物理地址。
- Online Softmax 不感知 TND、PA 或具体模板模式。
- 五种模式共享 QK、mask、Online Softmax、PV 和输出代码。
- ori 与 cmp 工作项必须进入同一组 `m/l/O` 状态，不能分别归一化后相加。

## 4. 参数和运行时上下文

### 4.1 静态模板参数

静态参数保持基本块设计：

```cpp
D, Tm, Tk, Td, PageBlockSize
```

其中：

- `D` 当前目标为 512；
- QK 沿 `Td` 分块；
- Query 基本块为 `Tm` 行；
- KV 基本块为 `Tk` 行；
- 运行时有效长度不改变 Tile 静态形状。

### 4.2 运行时参数

统一入口需要表达：

- Q、ori KV、cmp KV、attention output、softmax LSE；
- ori/cmp sparse indices；
- ori/cmp block table；
- `cu_seqlens_q/ori/cmp`；
- `seqused_q/ori/cmp`；
- `cmp_residual_kv`；
- ori/cmp topk length；
- sinks；
- `softmax_scale`、`cmp_ratio`；
- ori/cmp mask mode；
- ori window left/right；
- layout、模板模式、topk value mode 和 LSE 开关；
- B、N1、N2、最大序列长度和 block-table stride 等寻址元数据。

本阶段单核入口从这些参数直接生成任务，不读取 metadata。

### 4.3 FP16 scale policy

FP16 输入已经表示实际浮点值，因此：

```text
q_descale      = 1
ori_kv_descale = 1
cmp_kv_descale = 1
```

但 `softmax_scale` 必须保留：

```text
score = Q @ K^T * softmax_scale
```

内部定义 ScalePolicy，以便后续 HIFLOAT8 替换为：

```text
ori QK scale = softmax_scale * q_descale * ori_kv_descale
cmp QK scale = softmax_scale * q_descale * cmp_kv_descale
ori V scale  = ori_kv_descale
cmp V scale  = cmp_kv_descale
```

FP16 policy 的 QK scale 仅返回 `softmax_scale`，V scale 返回 1。

## 5. SequenceContext

每个 batch 生成：

```cpp
struct SequenceContext {
    int batchIdx;
    int qBegin;
    int qLen;
    int oriBegin;
    int oriLen;
    int cmpBegin;
    int cmpLen;
    int restoredCmpLen;
};
```

TND 下：

```text
qBegin = cu_seqlens_q[b]
qCapacity = cu_seqlens_q[b + 1] - qBegin
qLen = seqused_q != nullptr ? seqused_q[b] : qCapacity
```

ori/cmp 同理。有效长度不得超过对应 TND 容量。

压缩前恢复长度：

```text
restoredCmpLen = cmpLen * cmp_ratio + cmp_residual_kv[b]
```

当 residual 未传入时按 0 处理。需要 residual 的原算子参数组合由调用层预检保证。

单核遍历顺序：

```text
for batch
  for kv_head in [0, N2)
    for gqa_head in [0, N1/N2)
      for q tile
```

输出 head 为 `kv_head * G + gqa_head`。

## 6. KvWorkItem 和模式规划

### 6.1 工作项

工作项描述一个 Query 行或 Query 小块需要消费的 KV 集合：

```cpp
enum class KvSource { ORI, CMP };
enum class KvSelection { RANGE, INDEXED };
enum class MaskPolicy { NONE, CAUSAL, BAND };

struct KvWorkItem {
    KvSource source;
    KvSelection selection;
    MaskPolicy maskPolicy;
    int logicalBegin;
    int logicalEnd;
    const int32_t* indices;
    int selectedCount;
};
```

实际 C++ 结构可为满足编译器限制而调整，但必须保持上述职责边界。

Sparse indices 是源 KV 中的逻辑 token 索引，不是 PA 物理 block id。一个 Query tile 内不同 Query 行可以有不同 indices 和 topk length，因此不能假设整块共享索引。

### 6.2 五种模式映射

| 模式 | ORI 工作项 | CMP 工作项 |
|---|---|---|
| SWA | RANGE，应用 ori mask | 无 |
| HCA | RANGE，应用 ori window | RANGE，压缩 causal 前缀 |
| CSA | RANGE，应用 ori window | INDEXED，按 cmp causal 阈值过滤 |
| ORI_SPARSE | INDEXED，按 topk length 取数 | 无 |
| ORI_CMP_SPARSE | INDEXED | INDEXED |

所有工作项依次进入同一个 Online Softmax 状态。

### 6.3 Sparse 有效性

一个 sparse 位置仅在以下条件同时成立时有效：

```text
k < topk_length
index != -1
0 <= index < source_kv_len
满足对应 causal/mask 阈值
```

`topk_length` 缺省时使用 sparse tensor 的 K 维长度，但仍需处理 `-1`。

重复 index 按输入语义重复参与注意力，不做去重。

`topk_value_mode` 第一阶段只实现原算子当前使用并由 golden 覆盖的索引值语义；不支持的取值在调用层拒绝，不能静默按另一语义执行。

## 7. Mask 精确语义

对 batch 内第 `qPos` 个 Query token，定义 ori causal 上界：

```text
oriThreshold = oriLen - qLen + qPos + 1
```

所有区间使用半开区间 `[lo, hi)`。

### 7.1 ORI NoMask

```text
[0, oriLen)
```

### 7.2 ORI RightDownCausal

```text
[0, clamp(oriThreshold, 0, oriLen))
```

### 7.3 ORI Band/SWA

```text
lo = ori_win_left == -1
       ? 0
       : oriThreshold - ori_win_left - 1

hi = ori_win_right == -1
       ? oriLen
       : oriThreshold + ori_win_right

valid = [clamp(lo, 0, oriLen), clamp(hi, 0, oriLen))
```

这与原文档窗口公式一致，并明确支持 `-1` 表示不受限。

### 7.4 CMP threshold

```text
cmpThreshold = floor(oriThreshold / cmp_ratio)
```

HCA 连续消费：

```text
[0, clamp(cmpThreshold, 0, cmpLen))
```

CSA 中仅保留满足以下条件的 sparse index：

```text
0 <= index < min(cmpThreshold, cmpLen)
```

ORI_SPARSE 和 ORI_CMP_SPARSE 的 mask mode 组合遵循原算子校验约束；NoMask 模式主要由 topk length 和 index 有效性确定集合。

## 8. 地址解析

地址解析分为两步。

### 8.1 选取逻辑 token

RANGE：

```text
logicalToken = logicalBegin + k
```

INDEXED：

```text
logicalToken = sparseIndices[query, kvHead, k]
```

必须先完成 topk、`-1`、范围和 mask 校验，再计算物理地址。

### 8.2 布局映射

TND 连续布局：

```text
physicalRow = sequenceBegin + logicalToken
```

PA_BBND：

```text
logicalBlock  = logicalToken / pageBlockSize
offsetInBlock = logicalToken % pageBlockSize
physicalBlock = blockTable[batch, logicalBlock]
physicalRow   = physicalBlock * pageBlockSize + offsetInBlock
```

ORI 和 CMP 使用各自的 cache、block table、block-table stride 和有效长度。

### 8.3 KV tile 装载策略

优先顺序：

1. 验证 one-level-arch 现有 `MGATHER` 的真实字节偏移语义和支持范围。
2. 对连续且物理连续的 RANGE 工作项使用 `TCOPYIN`。
3. 对 sparse 或跨物理页的工作项使用经过独立验证的 gather 路径。
4. 如果 MGATHER 无法安全完成目标布局，先将选中 KV 行收集到连续临时缓冲，再复用现有 QK/PV 路径。

任何路径都必须在访问 block table 和 KV cache 前判定逻辑 token 是否有效。

## 9. 共享 Online Softmax

每个有效 Query 行维护 FP32：

```text
m : 当前最大值
l : 当前指数和
O : 未归一化输出累加
```

### 9.1 初始化

无 sinks：

```text
m = -infinity
l = 0
O = 0
```

有 sinks：

```text
m = sinks[queryHead]
l = 1
O = 0
```

sink 表示一个不贡献 V 的虚拟 softmax 项。

### 9.2 每个 KV tile 的更新

```text
S = Q @ K_tile^T * scale
```

无效 token 的 score 必须在 row max 前置为 `-infinity`。对有效列：

```text
mNew = max(m, rowmax(S))
alpha = exp(m - mNew)
P = exp(S - mNew)
lNew = alpha * l + rowsum(P)
ONew = alpha * O + P @ V_tile
```

然后更新：

```text
m = mNew
l = lNew
O = ONew
```

ORI 和 CMP tile 使用相同的 `m/l/O`。FP16 阶段两侧 V scale 均为 1。

### 9.3 输出和 LSE

存在至少一个有效普通 KV 或 sink 时：

```text
attentionOut = O / l
softmaxLse = m + ln(l)
```

边界约定：

- 无有效 KV 且无 sink：`attentionOut=0`、`softmaxLse=-infinity`；
- 无有效 KV 但有 sink：`attentionOut=0`、`softmaxLse=sink`。

SuperScalarModel 已支持 FP32 自然对数 TLOG，但 one-level-arch 当前没有 QSMLA 可复用示例。实现前必须先完成独立 TLOG 编译和 gfrun smoke，确认结果为自然对数。不得用 denominator `l` 代替 LSE。

## 10. 固定基本块和尾块

本阶段不引入运行时可变 Tile shape，仍使用固定 `Tm/Tk/Td`。

### 10.1 Q 尾块

- 固定加载 `Tm x Td`；
- 越过当前 batch Q 容量的位置必须安全补零，不能越界读取；
- 只为 `validQRows` 更新状态；
- 只写回有效 Query 行。

### 10.2 KV 尾块

- 固定生成 `Tk x Td`；
- 无效行安全补零；
- 对应 score 置为负无穷；
- 无效 V 行不得贡献 PV。

### 10.3 Sparse 尾块

- 固定处理 Tk 个候选位置；
- 超过 topk length、`-1`、越界或不满足 mask 的位置均无效；
- 在地址解析前完成判定。

### 10.4 PA 尾页

- 以逻辑有效长度为准，而不是以物理页容量为准；
- 超出 `seqused_*` 的位置不得查表或读取 cache；
- page block size 不要求等于 Tk，KV tile 可以跨物理页组装。

## 11. 建议文件边界

新增完整入口，同时保留当前三个版本作为回归基线：

```text
benchmark/one-level-arch/kernels/fa/
  quant_sparse_flash_mla_full_pto.hpp
  qsmla/
    quant_sparse_flash_mla_common_pto.hpp
    quant_sparse_flash_mla_address_pto.hpp
    quant_sparse_flash_mla_planner_pto.hpp
    quant_sparse_flash_mla_online_pto.hpp
```

职责：

| 文件 | 职责 |
|---|---|
| common | 枚举、参数、上下文、工作项、ScalePolicy |
| address | TND/PA 地址解析、index 校验、安全 KV tile 加载 |
| planner | 五模式规划、mask/threshold、topk length |
| online | 共享 QK、m/l/O、PV、sinks、LSE、尾块 |
| full entry | 单核 batch/head/query 遍历和组件编排 |

只有在完整入口稳定并完成回归后，才决定是否合并或删除现有 one-pass/two-pass 文件；本阶段不删除它们。

## 12. 输入校验和错误处理

SuperNPUBench benchmark 入口没有完整 aclnn 状态返回机制，因此采用两层策略。

### 12.1 调用/测试层预检

执行 kernel 前拒绝：

- `N2 <= 0`、`N1 % N2 != 0`；
- `cmp_ratio <= 0`；
- 非单调或长度不足 `B+1` 的 cu_seqlens；
- `seqused_*` 超过对应容量；
- PA block size、block-table stride 非法；
- 有效逻辑页对应的 block id 越界；
- 模式要求的 cmp/sparse/block-table 参数缺失；
- mask mode、window、topk value mode 的非法组合。

### 12.2 Kernel 防御

即使调用层已校验，kernel 仍必须保证：

- 无效 sparse index 不触发 KV 读取；
- 尾部 token 不触发 block-table 或 cache 越界读取；
- 空工作项不更新 Online Softmax；
- 只写有效输出区域。

本阶段不为非法参数定义可恢复的 device-side 错误码；非法调用由调用层失败，kernel 防御用于避免内存错误。

## 13. 实施阶段和门禁

### 阶段 0：基本指令和数据搬运烟测

独立验证：

- FP32 TLOG 编译、gfrun 和自然对数精度；
- Q/KV 尾块安全补零加载；
- MGATHER 字节偏移语义；
- logical token 到 PA 物理行映射；
- 跨页 Tk tile 的组装。

门禁：每个原语都有小规格 golden；失败时先解决原语或采用安全后备路径，不进入完整 kernel。

### 阶段 1：抽取共享 SWA 核心

- 从当前 one-pass 抽取 common、planner、online 边界；
- 先只支持当前连续二维 SWA；
- 保持默认规格精度不回退。

门禁：当前 one-pass golden 全量通过。

### 阶段 2：mask、sinks、LSE

- NoMask、Causal、Band；
- 两侧 window `-1`；
- sinks 初始化；
- 真正 LSE 和空行语义。

门禁：小规格逐元素 reference 与 D=512 回归通过。

### 阶段 3：TND 和变长

- B>=2；
- N1/N2/GQA 单核遍历；
- cu_seqlens 和 seqused；
- Q/KV 非整除尾块。

门禁：不同 batch 使用不同长度且互不串读。

### 阶段 4：PA

- 先实现 SWA + ori PA；
- 再实现 cmp PA；
- 覆盖乱序页、尾页、block size 与 Tk 不同。

门禁：相同逻辑 KV 的连续布局和 PA 布局输出等价。

### 阶段 5：五模式

按依赖顺序：

1. HCA：验证双 KV 源连续合并；
2. ORI_SPARSE：验证单侧 sparse；
3. CSA：验证 ori range + cmp sparse；
4. ORI_CMP_SPARSE：验证双侧 sparse。

每增加一种模式，都同时覆盖连续/TND 和 PA 数据路径。

### 阶段 6：组合回归

完成最小组合矩阵、边界、空行、极端 score 和旧版本回归。

## 14. 测试与验收

### 14.1 Golden

从 transformer QSMLA Python reference 提取语义，建立 SuperNPUBench 独立 CPU reference：

1. 随机或确定性 FP32 输入先转换为 FP16；
2. 再转换为 FP32 执行 reference；
3. reference 实现与本设计相同的五模式、TND、PA、sinks、mask 和 LSE；
4. NPU 输出 O 与 FP32 LSE 分别比较。

### 14.2 最小验收矩阵

| 维度 | 必须覆盖 | 核心断言 |
|---|---|---|
| 模式 | 五种模式各有连续与 PA 用例 | ori/cmp 共享归一化 |
| Mask | NoMask、Causal、Band、左右 -1 | 阈值与原 reference 一致 |
| TND/变长 | B>=2，不同 q/ori/cmp 长度 | batch 不串读，尾块正确 |
| PA | 乱序 block table、尾页、不同 batch | 与连续逻辑布局等价 |
| Sparse | 乱序、重复、-1、topk=0 和非整除 | 仅消费有效索引且不越界 |
| Sinks/LSE | 开/关、空 KV、极端 score | O 正确，LSE=`m+ln(l)` |
| Head | N1>N2 的 GQA | head/sink/index 偏移正确 |
| 回归 | 当前默认 SWA 规格 | 精度不低于现有实现门限 |

不要求一开始运行所有维度的笛卡尔积。使用成对组合覆盖独立交互，并为高风险组合增加专项用例：

- CSA + TND + PA + 非整除 topk；
- ORI_CMP_SPARSE + 两侧 PA + B>=2；
- HCA + `cmp_ratio!=1` + residual；
- Band + `win_left/right=-1` + 变长；
- sinks + 空 KV + LSE。

### 14.3 完成标准

本阶段完成必须同时满足：

1. 五种模式均可从统一入口执行；
2. 所有目标布局和变长组合无越界访问；
3. attention output 和 LSE 通过 CPU reference；
4. 当前 SWA 默认规格无精度回退；
5. 现有三个 QSMLA 文件和未跟踪验证产物未被破坏；
6. 文档明确记录尚未实现的 HIFLOAT8、metadata 和性能工作。

## 15. 后续演进

### 15.1 Metadata 和多核

将单核入口的 batch/head/query-tile 循环替换为 metadata 提供的任务序列。Planner、Resolver 和 Online Softmax 的单任务语义保持不变。

### 15.2 HIFLOAT8

新增 HIFLOAT8 ScalePolicy，并替换输入/概率转换路径：

- QK 应用 q/ori/cmp descale；
- PV 应用源侧 V descale；
- 保持 ori/cmp 共用 Online Softmax；
- 单独处理 HIFLOAT8 概率量化带来的 scale；
- 输出改为目标 BF16。

HIFLOAT8 工作必须在 Linx LLVM 当前 i8 legalization 崩溃解决后开始，不能与本阶段 FP16 语义问题混合定位。

### 15.3 性能

语义稳定后再评估：

- Q/KV tile 复用；
- 双缓冲和流水；
- sparse/PA gather 合并；
- split-G；
- metadata 负载均衡；
- 避免完整 mask buffer。

这些优化不得改变本设计定义的 WorkItem、地址解析和 Online Softmax 语义。
