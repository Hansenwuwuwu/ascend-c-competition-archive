# 题目 10：点到线段最小距离——获奖实现设计复盘

> 三份提交源码均已完成连续首轮审读；详细行段见[源码账本](10_TreeLineMinDistance.md)，哈希与范围见[coverage](../coverage/10_TreeLineMinDistance.json)。本文强调哪些设计可复用、哪些近似筛选尚无证明，不表示已在NPU验证。

## 1. 数学语义

点 `P[B,N,3]` 对每条线段 `(A,C)[B,M,2,3]` 求真实欧氏距离和最小线段索引；FP16/FP32输入相同dtype，half中间计算需升FP32。令 `D=C−A, R=P−A, L=D·D`。`L>0` 时 `t=clamp((R·D)/L,0,1)`，距离平方 `||R−tD||²`；`L=0` 时不能除以零，改为点到端点距离。最终sqrt只对最小平方距离做。完全tie按最小原始j。比较平方保持数学单调性，但要分别管有限数溢出、tail mask和索引tie。

## 2. 三份获奖实现的算法结构

| 顺序 | 主线 | 特别之处 | 正确性/适用边界 |
|---|---|---|---|
| 1 | 按shape专门化tiny、M=1、N=1、short、一般pair和M=1024大任务；一般pair按点tile遍历全部有效线段。 | segment不变量驻留/按batch复用，M尾段可切换并行轴；先归约平方距离和索引，再做sqrt，距离和index异步写出。 | 大M=1024专路用half筛选16个候选再FP32重算；重算只保证候选内精确，不能找回被筛掉的全局真最小。源码还存在越界volatile预读风险。 |
| 2 | 加入Morton空间排序、AABB/投影分离下界、按cell裁剪候选并对保留候选用FP32计算；另有packed短行及一般pair。 | 不是固定TopK，而是通过几何下界做保守筛选的思路；若下界可靠，可少算无关线段。 | 安全margin尚非已证明的全FP32误差界，且坐标scale未涵盖所有point幅度；需要验证无漏候选和fallback。 |
| 3 | 更丰富的精确/近似几何证书、candidate compaction、近tie修复、dyadic/整数limb比较及Hilbert空间排序路径。 | 将复杂浮点比较拆成limb表示/误差无损变换；歧义候选进入精化或精确回退，并用排序后原索引映射恢复输出。 | `exact/certificate/safe`函数名不是证明。误差范围、浮点转换、候选完整性、跨核同步及目标API仍待形式化/设备验证。 |

## 3. 最值得带走的套路

- **按退化维度改变并行轴**：M=1时沿点向量化，N=1时沿segments并归约；删除在数学上无意义的argmin或广播工作。只在题目语义确保证明成立时使用。
- **不变量移出内层循环**：固定一个batch的segments时，先算D、L、1/L等再复用于许多点；batch-pinned仅在负载与核分配条件合适时有效。
- **空间排序只用于访问/候选分区，不等于答案**：Morton/Hilbert把邻近点聚到cell；正确性来自每cell的保守下界和所有候选检查。迁移到碰撞/邻近搜索需重新证明下界。
- **保守剪枝 + 精确回算**：用下界丢弃能证明不可能更优的候选，对边界/近tie留下候选并用高精度算。必须证明舍弃集合无遗漏；固定Top2等经验筛选不具此保证。
- **索引与距离分开输出**：索引ready后可先写，sqrt/cast距离同时继续；buffer的GM写完成后才能复用。
- **按线段宽度打包尾部**：正常64段批量，最后1段切成沿point方向处理，避免向量广播同一个尾段。专门的M形状策略可迁移，64是该实现选择。

## 4. 隐蔽技巧、重要反例

| 设计 | 可学之处 | 不应照搬原因 |
|---|---|---|
| float位模式构造index ramp | 用整数bit/倍增生成规则索引，减少Scalar set | 仅对限定范围与表示成立；half位pattern最大有限值不等于Inf |
| 归约距离/index成对维护 | tie时按原索引稳定选择，块间严格规则可证明 | WholeReduceMin的tie API语义仍须核官方版本；只按距离最小不够 |
| half Top2粗筛 | 极低成本减少后续FP32算量 | 无no-miss证书，不应作为精确argmin实现范本 |
| volatile越界预读/预热 | 试图触发cache或规避启动成本 | 与题面禁止越界读不符，分配器padding也不等于张量合法extent |
| max(L,FLT_MIN) | 避免除零 | 对小正L会改结果，不等价于只对L==0分支 |
| 零workspace/固定高UB绝对偏移 | 减少外部临时缓冲 | 编译器保留区、资源版本及别名生存期都需逐项验证 |

## 5. 复用分级

- **P 本题**：投影clamp、零长度段、sqrt后置、最小原索引tie、half转FP32。
- **F 几何搜索族**：退化维度改变并行轴、共享线段不变量、空间cell排序、保守下界剪枝、歧义候选精算。
- **G 编排**：候选compaction/原索引逆映射、短尾专路、分输出事件和片上活跃期规划。
- **R 平台**：native reduction/mask、裸UB、bypass cache、专用float-limb指令及同步API。

## 6. 结论与证据边界

最可复用的创新是**以可证明下界做候选剪枝，再对剩余与歧义部分精算**，而不是“空间排序本身”或“只留Top2”。其次是按 M/N 退化情况改变并行轴，和把线段不变量移出 point 循环。迁移 rank1 的 half 筛选、rank2 的安全 margin 或 rank3 的证书时，关键是保留各自的误差界和候选覆盖条件。本届获奖提交的通过情况与耗时以官方 Judge 结果为准；本地复测原代码不是理解或复用该设计的前置条件。

## 7. 证据入口

- [题面契约](../SPEC_CONTRACTS.md#10-treelinemindistance4页)
- [rank1数学与路由](10_TreeLineMinDistance.md#a-题面和数学基线)
- [rank1筛选风险](10_TreeLineMinDistance.md#f-明确风险与不能从pass推出的结论)
- [rank2空间裁剪与误差边界](10_TreeLineMinDistance.md#h-顺序2-kernel-1801–4038-连续闭合)
- [rank3证书与迁移分类](10_TreeLineMinDistance.md#i3-可迁移性分类)
- [覆盖记录](../coverage/10_TreeLineMinDistance.json)
