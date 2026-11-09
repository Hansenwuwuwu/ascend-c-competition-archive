# 题目 02：矩阵伪逆——获奖实现设计复盘

> 本文面向算法理解和后续复用，不按源码行号复述。每个获奖提交及公共文件均已完成连续源码审读；细粒度来源、行段、SHA 和边界分析保留在[源码审读账本](02_pinv.md)与[覆盖记录](../coverage/02_pinv.json)。以下是静态代码设计分析，不是数值认证、SDK 兼容性认证或硬件性能归因。

## 1. 题目本质：近似可逆不是“全都可以直接求逆”

题目输入 `A[...,m,n]` 是 FP32，`1≤m,n≤512`、batch 1–64；输出 `A+[...,n,m]`。参考定义对奇异值分解 `A=U diag(σ) Vᵀ` 使用

```text
cut = max(atol, σmax * rtol)
A+  = V diag(1/σ, only where σ > cut) Uᵀ
```

相等阈值的奇异值必须丢弃；矩阵允许秩亏。题面另给谱间隔：保留值至少 `0.01 σmax`，丢弃值不超过 `0.1 cut`。这给“先试一个便宜的满秩逆，再验证是否可合法接受”创造了条件，但**不**等于所有输入可逆，也不允许省略阈值或把普通逆直接当伪逆。逐元素误差门槛为 `1e-4`。

这是一道数值算法与调度的组合题。大矩阵不能只靠某个 tile；真正的共同设计思想是路径组合：小矩阵/可认证满秩候选走快路径，矩形或秩亏/病态/误差证书不通过时走更稳健的谱分解或 Jacobi 路径，最后还要正确输出形状和 dtype。

## 2. 三份实现的整体思路

| 公示顺序 / 提交 | 主要算法路线 | 设计亮点 | 主要复杂度/注意点 |
|---|---|---|---|
| 1 / `1060506` | 大型多路线系统：小矩阵紧凑/驻留逆候选；矩形专路；稳定通用伪逆；Producer/Worker 与窗口化 pipeline；可选 AIC Cube、INT8 digit-plane 近似乘及高低残差修正；由残差/误差证书决定接受还是回退。见 [Host 路由](../../extracted/pinv/rank1_submission_1060506/code/op_host/linalg_pinv.cpp#L1)、[kernel 入口](../../extracted/pinv/rank1_submission_1060506/code/op_kernel/linalg_pinv.cpp#L113)、[Cube 数值服务](../../extracted/pinv/rank1_submission_1060506/code/op_kernel/pinv_rect.h#L1)。 | 最突出的不是单一求逆公式，而是将“候选生成、误差认证、精化、fallback”包装成多层求解器；矩阵状态/行变换、producer 预处理与 worker 消费拆开，Cube 作为可选服务而不是默认数学真理。 | 路由和 ABI 极多，代码尺寸、状态协议、SDK 私有 DSL/Cube 接口成本高。每条快路需确认 Host 可达且证书前提覆盖；候选出错时所有核必须同意 fallback 并完成同步。榜首分数不证明某一支路是其主要贡献。 |
| 2 / `1049211` | 清晰的算法族分层：小尺寸逆候选、并行行/列求逆、单边 Jacobi 通用 SVD、混合 AIC/AIV 行块路线；对逆候选做 FP32 残差认证，必要时 Newton 修正或回到通用伪逆。half Neumann/低精度 Cube 路线是候选种子，后面仍需 float 验证。见 [Host](../../extracted/pinv/rank2_submission_1049211/code/op_host/linalg_pinv.cpp#L1)、[主 kernel](../../extracted/pinv/rank2_submission_1049211/code/op_kernel/linalg_pinv.cpp#L93)、[Cube 与 half seed](../../extracted/pinv/rank2_submission_1049211/code/op_kernel/linalg_pinv.cpp#L1561)。 | 候选与兜底职责区分明显；误差控制不是只看一次 `AX≈I`，而会纳入范数界、残差修正和对阈值的判断。Cube 路径按 tile/事件交接，half 主要当初值而非最终答案。 | 手工 Cube 与跨核状态机复杂；float compensated arithmetic、workspace 大小、同步需要单独证明。题面要求的混合输入/属性组合不一定走 native 快路，必须以 Host guard 核实实际路线。 |
| 3 / `1173359` | 形状特化更直接：小矩阵走专用逆/按列并行；一般矩阵以 Householder 双对角化、隐式移位 QR 产生谱信息，并以 Jacobi polish/重建处理输出；特定大尺寸又有手工 panel/elimination 变体。见 [Host](../../extracted/pinv/rank3_submission_1173359/code/op_host/linalg_pinv.cpp#L1)、[kernel](../../extracted/pinv/rank3_submission_1173359/code/op_kernel/linalg_pinv.cpp#L104)。 | 将若干固定规模的大工作变成静态容量/面板算法，Cube/Vector 两侧各做适合自己的任务；小矩阵有按精确长度/列宽专化的短路。 | 静态模板规模和硬件布局绑定强。源码账本发现 Host 产生的某些 `WIDTH` 值未见对应 key 枚举，以及 16 阶候选中的 `break` 可退出 job 而未写输出的源级控制流反例；这是必须保留的审计发现，未声称已在 NPU 复现。 |

### 每份提交的算法执行链

**rank1（最高公示顺序）**先由 Host 根据 `(m,n,batch,dtype,workspace)` 选择紧凑小矩阵、矩形、通用稳定、窗口/producer-worker 或 Cube 族，而不是让一个 kernel 承担所有尺寸。小而合适的矩阵可用驻留高精度逆候选；矩形路线直接围绕较小维组织计算；通用路线执行 Jacobi/稳定分解。高代价分支可先用 quantized digit planes 或 Cube 产生候选/中间量，再由 Vector 路径解码、估计残差、精化和逐元素 certify。producer 负责一部分 pivot/packet/scale 数据，worker 消费后继续，若证书拒绝则进入精确或更稳健的 fallback。独特价值是这些部分形成了完整 solver portfolio；阅读/移植时必须把 Host 守卫、key、候选构造、证书和 fallback 作为一个整体闭环，而不能截取“INT8 Cube”或“window”单独照搬。

**rank2**按尺寸分派小逆、并行化的逆候选和单边 Jacobi 通用路由。小矩阵函数用受限容量和预打包几何索引/identity payload 减少运行时索引构造；较大的候选由并行行/列 owner 协作。普通伪逆路径在 `B=A` 或 `Aᵀ` 上做单边 Jacobi，并同步变换附加的 V 列；完成后按奇异值平方范数与题面 cut 重建。满秩候选的 refinement 使用高低分量/补偿 dot 算左残差，只有通过残差和谱间隔证书才接纳。Cube 路径以 FP16/Neumann 作为低成本种子，再做 float 级证书/修正；这使低精度成为候选生成器，而不是规范输出路径。

**rank3**把小维度、特殊大维度与一般 SVD 处理拆开：小矩阵使用固定宽度逆或按列 owner 逆；一般矩阵 Householder 化为双对角形式、再用隐式移位 QR 得到谱值/向量信息，随后依据原输入重建 `A·V` 并做 Jacobi polish；特定 199/256 家族用独立的 panel 消元/隐式逆结构来减少一般路径成本。这里的静态专用算法展示了高性能代码可以是“多套真正不同的算法”，但每个 Host 传入的 WIDTH、key 声明、内部容量和输出 store 都必须逐项匹配；账本里的 WIDTH/key 不一致与 16 阶 early-break 反例说明，模板多不自动等于覆盖完整。

## 3. 最值得迁移的“求解器结构”

### A. 候选 → 同一输入的证书 → 精化/回退

廉价候选不需要天然覆盖所有输入，但必须有一个可证明的接受门；拒绝时必须有完整 fallback，而不是只输出失败或继续用坏候选。伪逆尤其适合这样的结构：快试普通逆；若能证明矩阵满秩且谱间隔下没有应丢弃奇异值，才可当作参考伪逆；对已接受的逆候选再用 `R=I-AX` 检查残差，必要时做 Newton 修正。秩亏、阈值附近或残差超界就回到 SVD/Jacobi。

这个模式可迁移到线性系统、det、最小二乘等“快解容易、误差验收困难”的题，但证书必须绑定该题输出误差定义。不能仅复制某份代码的经验常数或接受/拒绝阈值。

### B. 逆 Newton 的精确算术逻辑及其边界

令 `R=I-AX`，使用 `Xnew=X+XR`，则精确算术下

```text
A Xnew = AX + A X R
        = (I-R) + (I-R)R
        = I - R²
I - A Xnew = R²
```

因此新左残差为 `R²`；对应误差满足 `||Xnew-A⁻¹|| ≤ ||X||·ρ²/(1-ρ)`（在 `ρ=||R||<1` 且范数/逆存在等前提下）。代码中的乘法方向要与残差定义一致；浮点版本还必须评估 residual 计算误差，不能把代数恒等式当逐指令舍入证明。

### C. 误差精度本身也是 kernel 的一条数据路径

获奖实现中可见 TwoSum/双浮点高低分量、4097 分裂乘法、Kahan/多累加器，以及低精度 Cube 结果再用高精度残差补偿。共通思路是“快速主项 + 便宜的误差分量”，不是拿 FP32 数字冒充 FP64。迁移前要：写出被省略的误差项、界定幅值/溢出域、保证编译器不任意重结合、对接近截断阈值的输入验证误差是否影响秩决策。

### D. 形状改变时并行轴也要改变

小方阵适合完整驻留或按列分工；大矩阵适合行/块分工；矩形问题可能转置为更窄的单边 Jacobi；多 RHS 则可能要共享分解。这些是算法结构变化，不是单纯调 tile。可借鉴的是“先按 `min(m,n)`、batch、rank/conditioning 选求解路径”；不能把固定 `n=16/199/256` 的地址布局当任意维度实现。

## 4. 不显眼的 trick：真正做什么，何时值得搬

| Trick / 代码线索 | 具体作用 | 复用条件 | 需防止的误读 |
|---|---|---|---|
| 五 digit-plane INT8 Cube 乘 | 将 FP32 值经尺度/高低 digit 分解为多个 int8 plane，Cube 算整数乘积，再按 digit 权重重建主项与残差 | 范围/缩放已界定，所有交叉项、溢出上界、残差消费都闭合，且 Cube 实测优于 float 路径 | 不是“FP32 改 INT8 就无损又更快”；重建项多，误差与带宽/同步成本不可略 |
| Neumann/half 只作逆种子 | 用低成本矩阵乘得到初值，再通过 float residual / Newton 与证书筛选 | 初值满足收敛域，证书可覆盖舍入与病态输入，拒绝会正确 fallback | 不能把低精度候选直接当最终伪逆；秩亏要走截断路径 |
| 静态 metadata/payload 编码 | 把索引、转置偏移、单位阵或计划打包到 tiling/只读常量区，减少运行时 scalar 构造 | 几何模板完全由 shape 决定，边界 index 可验证，不包含输入答案 | 地址元数据不是预存答案；索引宽度、padding 哨兵、消费者 stride 都要配套 |
| producer / worker / peer / window 切分 | 一组核生成 pivot、panel 或校正 packet，其余核消费；窗口缩短 producer 驻留并在边界重装 Schur 尾块 | 每个发布槽有明确 owner、ready/done 事件、cache 可见性、版本/slot、最终 drain；所有 reject/abort 路径同样完成协议 | 文件里出现 flag 或 pipeline 并不代表所有 Host 组合都启用；屏障缺失是全局正确性风险 |
| “候选逆不需要 SVD”的谱间隔证书 | 若精确逆存在，则 `||A⁻¹||F ≥ 1/σmin`；有被丢弃值时 `σmin≤0.1 cut`，故 `cut²||A⁻¹||F²≥100`。若能可靠证明与 `cut²||X||F²<25` 相容的 gate，则可以排除存在应丢弃谱值。 | 题面同样保证谱间隔；输入真的满秩；范数/逆误差已计入 slack；阈值 cut 计算与题面一致 | 只证明“在前提成立时可排除截断”，不证明有限精度候选正确，也不证明任意伪逆题能省 SVD。若 `atol/rtol` 默认细节或边界严格性处理错，证书无效 |
| 低级 Cube/UB API 与宏重绑定 | 直接管 L1/L0、MMA、Fixpipe、事件、系统 workspace 和编译 key | 仅在目标 CANN/芯片版本实编实测，生成代码/反汇编确认；精确遵守私有 ABI | 不是可携带的通用算子技巧；静态源码审计不证明官方支持或收益 |

## 5. 数值与控制流的“反面教材”也有复用价值

- 伪逆不能只检查 `A X≈I` 的单侧残差就断言截断伪逆正确；rank-deficient case 的正确对象不是 ordinary inverse。
- 对奇异值用 `σ>cut`，所以边界相等必须丢弃；浮点门限附近必须覆盖相邻 representable 值。
- 固定轮数 Jacobi/QR、4步 power norm、经验 residual slack、近似 Cube 结果都要分别给出收敛/误差论证，不能相互替代。
- 源码中某个 specialization 的 guard 若说支持 `n≤32`，仍要核对 workspace/pitch/identity/permutation 容量是否真的覆盖 32；宏标签不是证明。
- Host key 与编译 key 集合必须互相覆盖。静态读到的 Host/key 不一致和小矩阵 break 反例都保留为源级发现；没有编译/NPU 重现前，不夸大成已发生线上失败。

## 6. 新比赛里的落地步骤

1. 先从题面写出严格截断条件、默认阈值、矩形方向、秩亏与误差门槛，生成专门的边界测试（`σ=cut`、刚高/刚低、全零、矩形、极端 scale）。
2. 建一个全域可信 Jacobi/SVD 基线，不急着加近似；验证输出转置、阈值、矩形/秩亏路径。
3. 对 shape 家族逐个提出候选：小矩阵逆、行列分工、Cube 或混合路线；每个候选给出代数推导、输入域 guard、误差 certificate、拒绝后的同调用 fallback。
4. 对每条 route 同时审 Host→tiling→key→实际 kernel，核查 key 命中、资源预算、producer/consumer ownership、fallback 与退出同步。
5. 先做数值/边界 correctness，再做目标芯片 profile；成对消融才能将收益归因到 INT8、window、Cube 或 pipeline。

## 7. 结论与证据边界

**最可复用的核心思想：**把高风险快速算法变成“有严格适用域的候选”，用同一输入的残差/谱证书验收，必要时精化，拒绝则回到稳定通用路径。第二层价值是按矩阵几何改变并行轴，并为误差控制直接设计补偿算术。

**条件复用：**INT8 digit-plane、half Neumann seed、手工 Cube、跨核 producer/worker 与窗口流水、静态索引 payload——这些只有数学域、同步协议和 SDK 硬件全部闭环后才值得搬。

**不可据此声称：**任一具体 trick 导致了第一名；源码完整覆盖意味着所有路径正确；精确算术残差平方意味着浮点实现自动满足 `1e-4`；源码出现的宏/key 一定被提交 Host 选中。当前只完成源级审计，没有运行提交包、目标 NPU 数值/性能验证或逐机制消融。

## 8. 逐行审读证据入口

- [题面契约、构建与 rank1 路线](02_pinv.md#contract-题面契约与证据边界)
- [rank1 追加源码闭合](02_pinv.md#r1-window-dsl-window-全文件及最终可达性闭合)
- [rank2 完整 kernel 与数值前提](02_pinv.md#r2-kernel-rank2-单文件全文语义审读)
- [rank3 Host/Kernel、算法边界](02_pinv.md#r3-kernel-rank3-单文件全文语义审读)
- [精度门限推导、矩阵路径对比](02_pinv.md#math-admission-谱间隔浮点与资源结论)
- [逐文件 SHA 与连续行覆盖](../coverage/02_pinv.json)

“完整”指三份提交的源码首轮连续阅读覆盖完成；数学误差、CANN API、同步、生成代码、设备运行与性能归因仍是各自独立的证据等级。公示排名映射见[归档清单](../../README.md#27-份提交)。
