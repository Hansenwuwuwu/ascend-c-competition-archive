# 题目 01：安全带几何判定——获奖实现设计复盘

> 本文面向复用与后续比赛准备，不按源码行号复述。源码确已逐段连续阅读；原有行段笔记与 SHA/覆盖证据仍保留在[源码审读账本](01_SafetyBeltViolationGeometry.md)及[覆盖记录](../coverage/01_SafetyBeltViolationGeometry.json)。下文的代码行为结论为静态源码结论，不等于目标设备运行或性能归因。

## 1. 题目本质与实现主线

这是一个批量 AoS 几何谓词，不是通用矩形相交：对每个样本取 `y1,y2,cy`，按题面规定的 FP32 步骤计算

```text
d     = fp32(y2 - y1)
p     = fp32(ratio * d)
edge  = fp32(y1 + p)
bound = fp32(edge + threshold)
out   = (cy > bound) ? 1.0f : 0.0f
```

FP16 输入先提升到 FP32；输出接口是 FP32 `[B,1]`。`>` 是严格比较：相等输出 0，不能用 epsilon，也不能随意把表达式代数化简或融合，因为边界附近舍入可能改变离散结果。合法 batch 为 1–4096，ratio 是属性，threshold 可缺省为 0。

三份实现的共同骨架都是：Host 验证/分类 → tiling 或 key 选路径 → 从 `[x1,y1,x2,y2]` 与 `[cx,cy]` AoS 中抽出需要的列 → 做同一谓词 → 只写有效输出。真正的设计差异在于“把哪些输入条件编进 kernel、如何切分任务、怎样处理窄尾部和标量属性”。榜单顺序只说明作品名次，不能证明某个单独 trick 带来了速度收益。

## 2. 三份获奖实现怎样组织

| 公示顺序 / 提交 | 实现策略 | 做得特别好的部分 | 主要代价或边界 |
|---|---|---|---|
| 1 / `1120490` | Host 按 dtype 与 batch 边界分成多组策略；精确小 batch、整齐 batch、宽 batch、尾部 batch分别用不同每核行数/容量。kernel 入口再按 dtype、区间、ratio 等 key 派发专门类。 | 形状族不仅选 tile，还让固定长度、单位 ratio、常见常量路径在编译期变成不同实现；大 batch 输入两路 DMA 分开设就绪事件，先消费已到达的 `boxes` 算边界，后等 `anchors` 的 `cy`，用真实依赖关系隐藏搬运。见 [Host](../../extracted/SafetyBeltViolationGeometry/rank1_submission_1120490/code/op_host/safety_belt_violation_geometry.cpp#L22)、[kernel 入口](../../extracted/SafetyBeltViolationGeometry/rank1_submission_1120490/code/op_kernel/safety_belt_violation_geometry.cpp#L4037)。 | 路径/模板数量大，Host、key、kernel 三方必须完全一致。部分家族按 32B 向上取整使用普通 GM `DataCopy`；逻辑 shape 本身不能证明额外读写空间已分配。FP16 输入的推导输出类型与题面 FP32 ABI 不一致，是需要认真区分的接口风险，不能把历史兼容签名当题面契约。 |
| 2 / `1175118` | Host 组合 tile class、输入/输出签名、阈值模式、地址计划和 worker 数；严格 native 签名与混合 ABI generic 回退分开。kernel key 包含多维编译期计划；对不整齐尾部可将 tail owner 旋到特定 block，避免尾处理和普通 owner 共用不合适的地址计划。见 [Host](../../extracted/SafetyBeltViolationGeometry/rank2_submission_1175118/code/op_host/safety_belt_violation_geometry.cpp#L1)、[key/入口逻辑](../../extracted/SafetyBeltViolationGeometry/rank2_submission_1175118/code/op_kernel/safety_belt_violation_geometry.cpp#L2582)。 | 最强项是路由表达力和针对地址/广播/metadata 的静态选择；代价是宏/key 维数、消费者布局、workspace 解释与 SDK 生成规则高度耦合。Strict 题面签名若不匹配 native 条件会落 generic，不能拿其他 dtype 组合的快路代表本题执行情况。私有控制寄存器/owner API 和 workspace 包装替换须限定 CANN 版本核验。 |
| 3 / `1177391` | 明确区分 Strict 题面 ABI 与 Legacy 兼容 ABI；Host 固定 8 个 launch block，再按容量/owner 分配有效工作；kernel 使用成组容量、固定小 batch 与尾部专用类。窄 AoS 输入先抽列，half 先 Gather 再转 float；较宽形状中两路输入搬运后先算独立部分，再等待后续数据。见 [Host](../../extracted/SafetyBeltViolationGeometry/rank3_submission_1177391/code/op_host/safety_belt_violation_geometry.cpp#L1)、[strict/overlap kernel](../../extracted/SafetyBeltViolationGeometry/rank3_submission_1177391/code/op_kernel/safety_belt_violation_geometry.cpp#L563)。 | Strict/Legacy 的接口边界比较清楚；许多路径按有效长度做 GM 搬运、在 UB 内补齐向量工作长度，只对 `N` 个有效元素写回。小形状常量特化比“按数值预存答案”安全，因为输入坐标仍来自 GM。 | 固定 8 核是平台选择，不是通用最优结论；Host/key 可达域必须相互覆盖。静态源码可见的 Axpy 路径与逐步 FP32 题面公式可能存在舍入差异，须围绕阈值附近输入验证；未经 SDK 确认的底层指令/系统 workspace 宏不应直接移植。 |

### 三个实现各自的设计主线

- **rank1：按边界形状拆成“固定族 + 泛化族”。** Host 不只用 `ceil(B/cores)` 生成统一 tile，而是把 32、256、4096 等边界和区间内普通 shape 分开，并分别为 float/half 选择容量、owner 和尾部策略；kernel 将这些选择落实到类/key。短 shape 通过常量长度压掉循环/索引开销，长 shape 以多行/多 owner 承接吞吐。数据路径上先启动 `boxes` 与 `anchors` 搬运，独立计算 box 侧边界，直到最终比较前才消费 `cy`。这是“多 specialization 服务于真实不同的 shape 资源问题”的例子，同时也是 Host/key/Kernel 路由漂移风险最大的例子。
- **rank2：把布局/地址决策也提升为编译期计划。** 它把 tile 类、阈值形式、AoS/转换地址计划、worker/owner 数编码到多维 key 中；native ABI 是高性能族，严格题面以外的签名则明确退 generic。kernel 里短长度、整齐批量、接近整块的尾部和大块各走不同消费者；某些尾部通过逻辑 owner 旋转把 irregular work 集中到特定 block，其余 block 保持规整。值得学习的是“先把计划定义完整，再让消费者照计划解释 payload”；真正不可迁移的是其私有 key/CTRL/workspace ABI。
- **rank3：让接口兼容性和实际 GM 字节数变成显式分层。** Strict 题面接口与 Legacy 输出/threshold 类型兼容层分开；kernel 在多个容量之间选择，但只对 `N` 个有效样本读取/写回，额外向量 lane 留在 UB 内。小任务虽然固定 launch blocks，真正参与计算的 owners 仍由 shape/tiling 限定。它展示了“固定 dispatch 外壳 + owner 分配 + exact-tail IO”的组合；固定 8 blocks 本身并非通用优化结论。

## 3. 值得记下来的设计亮点

### A. “按依赖分段消费”比机械地等两路数据更有意义

上界只依赖 `y1,y2,ratio,threshold`，最终比较才需要 `cy`。因此可先搬 `boxes`，把列抽取和边界计算做完，再等待 `anchors` 中的 `cy`。这是一个由公式 DAG 直接推出的流水机会，而不是泛泛的双缓冲口号。可迁移到“输出由早到的输入子集先算出一部分”的算子；前提是事件与缓冲生命周期精确配对，并且另一输入搬运确实能和前半段计算重叠。

### B. AoS 只搬/转换真正消费的列

输入虽是紧凑 AoS，但 kernel 只消费三个坐标列。获奖实现中可见按列抽取、half 先 Gather 再 Cast、以及多列连续时合并转换等设计。可迁移到小字段集合的 AoS 谓词/过滤算子；前提是源 stride、Gather 索引单位（字节还是元素）、padding lane、half 转换顺序均核清。不要机械地把“多次抽列”换成“整结构 Cast”，后者可能做了更多无用工作。

### C. 计算 padding 与 GM 有效范围必须分开设计

向量计算通常希望行数按 16/32/64 对齐；这只说明 UB 内可以多算，不意味着 GM 输入分配也有尾部 padding。迁移到其他题目时，可用精确/带明确 padding 语义的输入搬运、UB 局部补齐、mask/valid 长度控制、按真实 `N` 写回。rank1 的普通 `DataCopy` 向上补齐依赖调用侧相应的分配与题目形状；复用时保留这个前提即可。

### D. 属性特化要由真实 Host 属性守卫

ratio=1、threshold 缺省或严格为 +0 等常量可使算术链缩短；只有 Host 对真实属性做了等值判断并选到匹配 key，才是合法专化。坐标值仍必须从当前输入读取。这个区分适用于很多小算子：静态属性决定代码形状可以特化，输入数值不能被误当成常量或缓存结果。

## 4. 不显眼的 trick：分类、收益证据和迁移边界

| Trick / 实现线索 | 它实际做了什么 | 复用范围与必须条件 | 当前证据 |
|---|---|---|---|
| half AoS 的 Gather→FP32 Cast | 先挑 y1/y2/cy，再升精度，避免无用字段转换 | 小结构、字段少且布局规则；需核 Gather 模式、lane 与 dtype | 源码已读；无单机制配对性能证据 |
| DataCopyPad 单元素搬入 + stride-0/广播式消费 | 利用特定接口的 padding/块语义构造向量重复标量 | 仅在对应 CANN API 文档、目标架构与数据块规则明确时 | 源码机制可见；跨版本移植前必须官方文档/实机验证，不能笼统称作“未初始化 UB 可用” |
| threshold 的 +0 快路 | threshold 确为 +0 时省去加法；非零仍读取并按原序相加 | 有限输入、严格比较、signed-zero 影响被最终比较证明无差异；NaN/异常值另按题约束处理 | 代码路径存在，不等于本次独立测得收益 |
| tail owner rotation / row padding | 把不规则尾段映射到已设计的 owner，并让内部计算对齐 | 只有 Host 的 batch、核数、slice 公式与 key 表逐项证明覆盖时 | 源码可证设计；具体设备性能未验证 |
| `TILING_KEY_VAR` 多字段编码、私有 CTRL/owner 读写、workspace getter/setter 宏替换 | 让运行期选择压成编译期族，或直接绑定生成器/硬件协议 | 只适合作为对应 SDK 版本的研究对象，需看生成代码/汇编与目标机；不能照抄字面 key 或 CTRL 位 | 版本依赖高；源码本身不足以证明稳定 API 或收益 |

## 5. 面向下一场比赛的复用清单

1. 先把公式写成逐步 dtype/舍入 DAG，再标每个输出依赖哪些输入；从图中找早到数据可先消费的部分。
2. 对 AoS 谓词先画出“读取字段—向量布局—有效 lane—输出字节数”，选择列抽取而非默认转换整结构。
3. 把有效 shape、内部 pad、GM 对齐、UB 预算分成四个量；尾部只靠题面或 allocator 契约兜底，不能靠大视图指针兜底。
4. Host 的每一种 route 都列出 `(dtype, shape class, attributes, key, block dim, kernel template, fallback)`，并检查“有声明不等于可达、Host 发出不等于 SEL 存在”。
5. 先保留一个全域正确 generic 路径，再逐一加高收益特化；每个特化的 guard 与反例域要写清。
6. 比较排名代码时把“设计合理”“源码看起来少工作”“某特化被真实派发”“实测有收益”分成四种证据，不把名次归因到某一 trick。

## 6. 复用结论

**最值得复用：**依公式 DAG 拆早晚依赖、AoS 列抽取、真实属性驱动的编译期特化、UB padding 与 GM 有效长度分离、Host/key/kernel 路由表闭环。

**条件复用：**标量广播、SPR/bitmap compare、尾 owner 旋转、手工 event pipeline、固定 core 数量——都依具体 API、shape/资源族和设备测量。

**不要直接照搬：**没有分配契约支持的 GM 向上读写、题面 ABI 不匹配的兼容路由、私有 CTRL/workspace 变更、未经逐 ULP 检查的算术合并。源码静态审读没有目标 CANN 设备验证，也没有分离变量的 paired 性能实验；因此本文不声称“某个 trick 造成榜单名次”。

## 7. 逐行审读证据入口

- [题面契约与公共构建审读](01_SafetyBeltViolationGeometry.md#contract)
- [rank1 主体与补读闭合](01_SafetyBeltViolationGeometry.md#rank1-kernel-continuation-2026-11-09补读)
- [rank2 主体、key 和补读闭合](01_SafetyBeltViolationGeometry.md#rank2-kernel-continuation-2026-11-09补读)
- [rank3 完整 Host/Kernel、同步与数值风险](01_SafetyBeltViolationGeometry.md#rank3-kernel)
- [逐文件 SHA 与连续行覆盖](../coverage/01_SafetyBeltViolationGeometry.json)

“完整”在此指源码连续行审读覆盖完成，不代表代码正确性、API 合规、NPU 运行或性能归因完成。原始公示分数与提交映射见[归档清单](../../README.md#27-份提交)。
