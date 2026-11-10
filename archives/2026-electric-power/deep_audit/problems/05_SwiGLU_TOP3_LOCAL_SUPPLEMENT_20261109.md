# 电力巡检算子 SwiGLU 本地补充：前三名代码设计复盘

日期：2026-11-09
范围：用户本地提供的 `1.zip`、`2.zip`、`3.zip`；按用户给出的映射分别记为第一、第二、第三名。提交 ID、队名、榜单分数和代码是否就是最终提交包，ZIP 内没有可核对证据，本文不补造这些信息。

这份复盘补充 2026-11-08 公开页面快照。快照当时没有 SwiGLU 获奖源码；本地 ZIP 是后来单独提供的材料，不应倒写成 11-08 已下载的公示附件。三包 CRC 均通过，原始 ZIP 已保留，解压只用于只读审阅；本轮没有执行其构建脚本，也没有在 CANN/Judge 上复测。

## 1. 审阅口径与来源

题目契约采用 [`SPEC_CONTRACTS.md` 的 05 条目](../SPEC_CONTRACTS.md)：FP16 输入/权重/偏置，`Y = SiLU(XW1ᵀ+b1) * (XW2ᵀ+b2)`；隐藏点的 `M,d` 采用项目已冻结的 [`hidden-case-shapes.md`](../../../../../../whatcursordone/hidden-case-shapes.md)。代码行号均相对于各 ZIP 的 `code/` 路径。

原包 SHA-256：

| ZIP（用户给定名） | 本文名次 | SHA-256 |
|---|---:|---|
| `1.zip` | 1 | `2c0a346d4d7b5ec7a7711ac1e5bdb45398e29f54e4ece44eca37cb48071a4835` |
| `2.zip` | 2 | `66bd5fdd7acaae17a4d52f519f1a9564add10f0db4ad260fa493f9a5b25fadd6` |
| `3.zip` | 3 | `ffcb8180e1f0cdb12aced2af3585b8a75ee3b9504248bde652c60a9daf5e1ea4` |

解压后分别有 8/12/7 个文件，按 UTF-8 文本文件逐行计数为 2,940/1,664/10,573 行；第三名的 `op_kernel/swi_glu.cpp` 单文件为 9,849 行。这个大文件由多个 `TILING_KEY_VAR` 条件编译区域组成；源码里重复出现的路径分支不代表都进入同一个设备镜像，更不代表某一 `PATH` 的所有文本段都是该 shape 的实际实现。以下先追 Host 选出的参数/key，再限定到对应编译区域读设备实现。代码中注释提及的“实测”“选定”只视为作者注释，不代替本项目自己的可重复性能证据。

## 2. 九个隐藏 shape 的选路对照

| 点 | `M,d` | 第一名：Host 路由族 | 第二名：Host 路由族 | 第三名：Host 路由 |
|---:|---:|---|---|---|
| 1 | `1,4` | tiny vector / 单行专用 | `ROW4` AIV-only | PATH 43：4 维子块 GEMV |
| 2 | `2,64` | 小 M 向量归约 | `ROW64` AIV-only | PATH 44：64 维、2 行列 GEMV |
| 3 | `512,256` | Cube；两列块 full-K split 路径 | `CUBE256` 双路 Cube 流水 | PATH 45：256 维 Cube 流水 |
| 4 | `1,32` | 单行 tiny vector 特化 | `ROW32` AIV-only | PATH 46：32 维单行列 GEMV |
| 5 | `4096,128` | Cube 两波双 C0 路径 | `CUBE128` 双路 Cube 流水 | PATH 47：128 维 Cube 流水 |
| 6 | `1,1` | 标量/单元素 vector 路径 | 标量 AIV-only | PATH 48：单通道标量路径 |
| 7 | `21,13` | 13 维小 M 分组向量路径 | 13 维静态行组 | PATH 49：3 行子块 GEMV |
| 8 | `85,33` | 33 维 `FAST33` 分组向量；此 shape 的实际专用路由为每块 8 行、16 个 launch block（11 个 active block，5 个空 block） | PATH `ROWS33`，17 个块 | PATH 50：固定 5 行打包 GEMV |
| 9 | `8192,256` | 256 维 full-row Cube/Vector 波次流水 | 20 AIC、权重驻留计划命中时走静态 resident key | PATH 51：`TILING_KEY_VAR=13057`，`TM=128,Cores=20,Iterations=7,TileN=128` |

这张表描述源码在题目 shape 下的条件路由，不声称已从比赛运行日志验证每个 key。最清晰的共同点是：三份都没有把九点当成同一 GEMM 的缩放问题；小点按输出规模退到 AIV/vector，`M=512/4096/8192` 则采用 Cube 路径或 Cube/Vector 混合。

## 3. 第一名：最多层次的 Host 调度与 Cube/Vector 组合

### 3.1 Host 是性能策略的一部分

`op_host/swi_glu.cpp:21-95` 先依据 `M,d`、AIV 数、估算向量列宽和任务 wave 选 AIV 小算子还是 Cube；再按行数阈值把 13/33/64/128 等小矩阵划成不同 group。`96-141` 对 33 维尾组、分组数和 launch block 数做进一步选择；`188-266` 才进入 Cube 专用族；`267-389` 按 `d,M` 选 `tileCols/tileRows`、full-K tile 和流水模式。它把 Host 已知 shape 折叠进模板参数，设备端因此不必每个元素都重新判断同一套泛化策略。

具体命中关系：

- 点 1/4/6 由小 vector/标量路径处理，避免启动 Cube 或给 1–4 个输出配置通用矩阵流程。
- 点 2 是小 M、`d=64` 的 AIV 归约；不会为了两行数据启动两个 projection 的大批次 Cube 流水。
- 点 7 将 21 行、13 维变成若干小行组复用权重；Host 可按当前 AIV 数调整 group/key，而不是固定一核一行。
- 点 8 对 33 维使用专门尾部布局和行组路径。Host 对 `M mod 2/mod 3`、AIV 核数和可整除 wave 作多个选择（`100-141`）；这说明这个 case 的关键不是只对齐到 48，而是让 33 个实际元素在 padded tile 中有稳定的读取、偏置和写回映射。
- 点 3/5/9 分别为中行数 full-K、双波、长波流水选计划；点 9 大到足以摊薄 Cube 初始化和 workspace 成本。

### 3.2 设备端值得复用的结构

`op_kernel/swi_glu.cpp:14-34` 的 `CopyProjectionChunk` 把两份权重放入相邻 L1/L0B 区域；`36-291` 的 `ComputeProjections` 把 `PIPELINED`、`PREFETCH`、固定维度/行数、full-row 与 tail 策略变成模板开关。其关键不是“用了很多 bool”，而是将两条投影所需数据放入一个 cube 计算过程，控制 L1/L0A/L0B/L0C 的容量和事件依赖：`58-71` 是片上区域与每块 workspace 视图，`73-90` 初始化双槽事件及首块预取，`108-146` 对 tile 做调度/双缓冲，后续按 key 做Mmad和写回。

点 3 的另一族 `ComputeSplitFullK/GateSplitFullK`（`293-447`）将完整 K 维计算与后续 SiLU/门控分在 AIC/AIV 两类执行单元，靠 workspace 与 CrossCore 事件传递两路投影。点 9 使用 `DIM_TAG=18` 的 full-row waves（Host `188-218`；Kernel `1123-1138`），把 20 AIC、每波 128 行、128 列 tile 和尾波容量放进编译期参数。点 5 的 `DIM_TAG=23` 双 C0 计划（Host `258-266`；Kernel `1158-1161`）与点 9 不同，证明大 M 也要区分权重/行数与可用 C0 的组合。

向量侧 `swi_glu_vector.h:4-117` 提供成对复制、偏置/门控和 tiny 基元；`119-497` 是 single-row/通用 vector；`508-945` 是 batched vector；`946-1046` 是成对 projection 的 vector 变体。33 维路径实际由 `BatchedVectorSwiGlu` 的 `SPLIT33/FAST33` 实例化（kernel `swi_glu.cpp:1069-1075`；vector `508-931`），不是 vector 文件 `203-314`。可迁移的不是本题掩码常量，而是“把非对齐布局静态化，让多个行共享权重 tile，再按有效槽执行归约/写回”的思路。

#### 点 8 的实际命中路由（结合源码再核对）

补充分析 `SWIGLU_TOP3_ANALYSIS.md` 对“8 行/块”方向的描述正确，但“11 核”需精确理解为 **11 个有效块，而不是 launch blockDim=11**。Host 对 `M=85,d=33` 计算 `ceil(85/8)=11` 个 row tile，再按 8 个 AIV 为一 wave 向上取整到 16；命中时写 `CALC_MODE=6,DIM_TAG=14,TAIL_CODE=5,PAD_BLOCKS=5,tileRows=8,tileCols=40,cores=16` 并设置 blockDim 16（Host `96-141`）。Kernel 对 `blockIdx >= 16-5` 直接 return，因而 11 个 active block 覆盖 10×8+5=85 行；最后一个有效块处理 5 行（vector `swi_glu.cpp:1069-1075`; `swi_glu_vector.h:519-535,600-607`）。

该实例的关键编译期开关是 `STATIC_DIM=33,ROWS=8,COL_TILE=40,FAST33=true,SPLIT33=true`。输入/权重的 33 项点积被拆成 32 项主段和 1 项尾段：用常量 64-bit mask 的 `GatherMaskCurrentMode` 选主段/尾段，不再逐行构造 Gather offset；主段做 FP32 multiply/reduction，尾项乘积通过 `MulAddDst` 并入累加；偏置与门控完成后，再把 padded 48 槽布局压回每行 33 个输出（vector `swi_glu_vector.h:609-750,900-931`）。这与“完全删掉转置/打包”的路线不同：它仍然有 GatherMask 和 48 槽输出打包，只是把动态索引构造和普通 Gather 替换成编译期掩码路径。迁移时要连同最后 5 行、两路权重/偏置映射及输出压缩一并证明，不能只抄 mask 常数。

### 3.3 需要隔离的特殊点

Kernel 文件最末 `1222-1229` 在特定 TPL key、非 debug 条件下重定义 `SetSysWorkspaceForce` 等库符号/入口宏，注释解释为绕过未使用 workspace 的初始化与调整入口布局。它与 `#define ASCENDC_CUBE_ONLY`、移除 `__MIX_CORE_MACRO__`（`1-10`）属于编译/运行时集成层的环境技巧，不是一般向量优化。复用前必须确认 CANN 版本、编译器封装、Judge wrapper、workspace ABI 与规则；不能只拷贝宏后假定仍成立。

## 4. 第二名：设计较集中，Cube 方案能沿数据流解释

### 4.1 小算子：避免 Cube 和调度复杂度

Host `op_host/swi_glu.cpp:110-127` 将 `M=85,d=33` 固定到 17 blocks；`M=1,d=1/4/32`、`M=2,d=64` 用静态 key，launch 配置即便固定为 8，kernel 也由编译参数决定有效工作量。`op_kernel/swi_glu_vector.h:10-85` 的静态单行实现一次搬入 X、两矩阵与两 bias，FP16 输入逐元素乘、FP32/向量归约后 cast 回 half，做偏置、SiLU 和门控；`132-201` 对 13 维行组使用 padding 和预定 gather 索引；`203-314` 对 33 维 5 行组做更复杂的共享权重/小矩阵布局。

这类专用核的收益来自固定小尺寸减少通用循环、分支和启动浪费。可迁移的原则是“在正确数值顺序可保持时，为极小 M 建小 kernel”；`d=4/13/33` 的索引、tile、vector mask 是题目/架构专用参数，不可原样泛化。

### 4.2 中大型点：把两投影作为一个流水问题

`swi_glu_cube.h:10-149` 的 `KernelCubePipeline<BK,BM,BN,PANELS>`：

- 用双槽 X/L0A、多个 C0 slot 表达可并行的 panel（`12-16`）。
- 先载入 Gate 权重，执行 gate Mmad；将 Up 权重延迟到 gate 计算/SiLU 时间窗再装 L0B（`61-104`），让两路不必同时占满 L0B。
- 每路 projection 通过 CrossCore flag 通知 AIV（`103-104`），Vector 顺序读 gate/up half 投影，分别加 bias，再做 `NegativeSilu` 与乘法、连续写出（`111-148`）。

这给点 3/5 一个比“把两个 GEMM 分别做完再启动第三个 kernel”更连贯的流水。它的前提是固定 tile 和容量确实适配目标芯片；`BM/BN/BK/PANELS` 不是通用推荐值。

点 9 的 resident 方案（Host `PlanGateResident` `8-48`、key 路由 `136-147`；Kernel `swi_glu_resident.h:9-233`）用 L1/UB/L0 的设备容量去算组数、panel 数和 workspace，之后把权重载入/驻留并跨多批 X panel 重用（`141-167`）；在 AIC 写出若干 projection 后，AIV 读 bias/投影做激活和门控（`169-229`）。Host 只有在资源容量判定和预期 group/panel 同时满足时才走静态常量版本。可复用的是“资源先验→精确容量不等式→静态 resident key / 动态通用回退”的计划方式，而不是 10 group/13 panel 这组数字。

### 4.3 需注意的低层接口

Kernel 文件 `8-23` 提供空的 `matmul::clearWorkspace`，只在 `__NPU_ARCH__ == 2201` 设定必要硬件状态，注释限定为不使用 Matmul 对象、仅基础 Mmad 的版本。它是明确依赖设备/库集成假设的 helper；迁移到使用 Matmul 对象或不同架构时不能原样留空。该包的 CMake 只组织 Ascend C 编译目标，并不能独立证明目标 Judge SDK 能构建通过。

## 5. 第三名：模板族很丰富，但 Host 有关键早返回

### 5.1 九个 shape 对应 PATH 43–51

Host `op_host/swi_glu.cpp:241-301` 按九个精确 `M,d` 依次选择 PATH 43–51，并设 blockDim/workspace。Kernel `op_kernel/swi_glu.cpp:1995-2034` 对应关系是：

- PATH43 `SubBlockGemv4`；PATH44 `AlignedColGemv<64,2>`；PATH46 `AlignedColGemv<32,1>`；PATH48 `ScalarChannelSwiGlu`：面向 tiny/窄行。
- PATH49 `PackedSubBlockGemv`（13 维三行）；PATH50 `PackedMultiBlockGemv`（33 维五行）：面向非对齐小 d 的行组。
- PATH45、47、51 是 Cube/Vector 流水类：PATH45 `<D=256, TM=32, Cores=16, Iterations=4, TileN=64>`，PATH47 `<128,64,16,4,TileN=128>`，而题目点 9 实际选中的 PATH51（key `13057`）为 `<D=256, TM=128, Cores=20, Iterations=7, TileN=128>`（`6618-6620`）。同一源码文件里其它 `TILING_KEY_VAR` 分支也出现 PATH51 的 `if constexpr` 文本，但这些不能不加区分地当作 key `13057` 的实现；要以 Host 选出的 key 对应编译区为准。

`PipelinedCubeSwiGlu` 有显式的 L0/UB 容量静态断言（`1836-1840`），四槽输入/输出、Cube ready / Vector consumed 的 CrossCore flag（`1883-1898`），递归按编译期迭代展开推进数据和计算（`1900-1915`）。AIV epilogue 从两路投影加 bias，按 FP16 执行 sigmoid 相关链并门控，再落盘；`1936-1988` 展示向量 mask、事件复用与写回。此处应按源码的实际 `Exp/Div` 路径理解，不能仅凭类名将其当作近似 sigmoid。

33 维 PATH50 使用 48 槽 padded 布局、5 行并行和适配向量读取的权重块。Host 对点 8 选择 PATH50，即 key `12801`；对应编译区 `4290-6740` 中，`PackedMultiBlockGemv::Run`（`5278-5418`）自行构造 offset，入口 `6614-6617` 不传 tiling。其它 key 区域中也有名字相同的类/`if constexpr(PATH==50)` 文本（例如 `1001-1075`、`2030`），但不应将它们误认为 key `12801` 下同时可运行的替代方案。key `12801` 这一实现针对 33 列真实数据与补零槽的位置编排 gather、累加、bias 和最终写回；可借的是“为固定、很小且非对齐的 D 设计整组数据布局并一起验证”，不能直接复制物理 offset。

点 9 的 PATH51 还存在一个需要特别记录的**静态覆盖量疑点**。key `13057` 的模板参数给出 `D=256, TM=128, Cores=20, Iterations=7, TileN=128`（`6618-6620`）；类内 `TN=D/TileN=2`，行 tile 起点由 `row=(core_/TN + I*(Cores/TN))*TM` 生成（`1831-1833`、`1900-1903`）。对 `core_=0..19`、`I=0..6`，每个 N 分片覆盖 `10*7=70` 个 128 行 tile，即模板容量 8960 行；题目实际只有 8192 行，即 64 个 tile。当前所读 epilogue（`1961-1984`）未见依据实际 M 裁剪多出的 tile。仅凭源码可确认“模板覆盖上界大于题目 M 且该处未见显式边界判断”；是否会形成实际越界访问、是否由其它运行时机制约束，尚未在设备上验证。因此这是一项不能直接复用的安全/契约待核项，不应仅因获奖包或排行榜通过就略过。

### 5.2 源码结构审计发现：通用路由在一个 return 后

Host `op_host/swi_glu.cpp:304-312` 对所有未命中前面九个精确 shape 的输入执行通用 `ComputePlan`、设置 key 0、然后 `return GRAPH_SUCCESS`。同一函数的 `313-424` 又出现 PATH42、tiny、Cube 和 fallback 的其他路由代码；这些语句位于无条件 return 之后，C++ 正常控制流不可达。因而源码宣称/保留的那些通用 PATH 路由不能算作当前 Host 可执行的调度能力；九个已知精确 shape 会在更早的 241-301 命中专用 key，其他 shape 实际只会走 key 0 fallback。对未来迁移或扩展形状，这是本包最重要的边界；若获奖测试只覆盖九个固定点，不一定影响这些点，但扩展题面范围时必须先修正/解释。

### 5.3 编译单元专用变体，不要把同一行号误看成同一实现

文件以 `TILING_KEY_VAR=11265`、`12033`、`12801/13057` 等形成条件编译区（`1`、`2167`、`4290`、`6740`、`9835` 附近）。其中重复出现的 PATH dispatch 文本不代表同一条 PATH 在每个 key 区都可达；应先根据 `tiling_key_swi_glu.h` 的 key 选择和 Host 的 `ASCENDC_TPL_SEL_PARAM` 确定真正编译/选择的区段，再读对应 kernel。例如点 8 的 key `12801` 对应 `5278-5418` 与 `6614-6617`，本实现 inline 构造 offsets 且不收 tiling；不能拿其它 key 区域中“从 tiling 读取 offsets”的同名源码段来描述本次实际选中的实现。

第三名的大源文件还有特定 key 下对 `SetSysWorkspaceForce`、`GetUserWorkspace`、`cce_kernel` 和 `set_ffts_base_addr` 的条件重定义（例如 `2153-2165`）。这属于源码与编译器/运行时 wrapper 集成，不是普通 kernel 技巧。该文件规模与重复分支也意味着维护、编译时间和错误定位成本真实存在；只有在明确需要多个独立编译路径并且二进制/实例化可控时，模板拆分才划算。

## 6. 跨名次归纳：可以带走什么

| 设计 | 类型 | 可迁移条件 | 不应照搬的部分 |
|---|---|---|---|
| 先按 `M,d` 分计算族，再做 tile 与 block 计划 | 通用 Host 调度方法 | 题面 shape 分布已知；Host 可读可靠设备核数/内存信息；每个 key 都有明确 Kernel 入口 | 针对本题 9 个 shape 的 if 阈值、DIM_TAG/PATH 数字 |
| 小 M、小 d 改用 vector/GEMV，而不是 Cube GEMM | 条件性通用 | `M*d²` 小到 Cube/混合核固定开销不划算，vector 归约满足精度契约 | D=1/4/13/32/33 的硬编码核和 offset |
| 两个 projection 合并到同一 Cube 任务，X 复用，延后第二组权重，Cube/Vector 重叠 | 有较强迁移价值 | 两路矩阵可共享 X、设备确有可用 AIC/AIV 与 L0/UB 槽位、同步协议经过验证 | 本包的槽位号、BM/BN/BK、事件 id、workspace 大小 |
| 大 shape 权重驻留并跨 panel 复用 | 有条件的通用 | 权重能驻留在目标存储层且预算精确满足，足够多 X panel 摊薄装载成本 | 10 groups、13 panels、目标设备容量常数 |
| 静态固定 shape kernel / 编译期 specialization | 可迁移的工程方法 | shape 极少且固定；编译实例数可控；Host 与 key 严格闭环 | 每个 shape 单独复制一份完整 kernel；无实测理由地堆几十种 path |
| 把非对齐 D 变成 padded tile，再专门映射有效槽 | 条件性方法 | padding mask/地址/输出裁剪可证明正确，额外带宽和索引成本小于收益 | 33→48 的精确 gather offsets；任何未经证明的 padding 读取 |
| FP32 accumulation 后按规定位置回到 FP16 | 数值方法，不是无条件等价 | 误差标准和参考实现允许该舍入顺序；各 reduction 次序误差已测 | 认为“输入半精度，所以所有中间半精度都等价” |
| 低层 workspace/API wrapper 改写 | 特定环境技巧 | CANN 版本、架构、ABI、Judge runtime、构建包装器完全匹配 | 宏重定义和空 `clearWorkspace` 的跨项目复制 |

## 7. 对我们 SwiGLU 代码的实际启发与顺序

1. **最值得先迁移的思想不是排名 1 的宏，而是“kernel 家族+严密路由闭环”。** 我们的 key/Host/入口须逐条对应，确认候选真命中；在 Judge 不暴露 key 时，通过静态条件与编译产物确认，不能从单次时间反推。
2. **Case 3/5/9 按共享 X 的两路投影联合设计。** 对照我们已有的 MTE2/AIC/AIV profiling，分别判断 Cube 是否能在 Gate→Up 之间复用 X、延迟装 Up 权重、把 gate Vector 与 up Cube 重叠。先做数据流/容量证明：L0A/L0B/L0C/UB/workspace 和 CrossCore producer/consumer 一一列明，再做一个孤立变体。
3. **Case 7/8 不是把 d 补齐即可。** 应比较每块重复读权重、active block 数、offset 建立和归约树。Case8 可重点借鉴固定五行的整组布局，但需证明 33→48 每个槽位的来源、尾部补零、第二投影、bias 和最终 33 元素写回。
4. **Case 1/2/4/6 重点核算启动/分核成本与向量指令链。** 拿三份的 specialized tiny 路径作为对照基线，逐项比较是否真正减少 Launch block、GM copy、reduction 层数和 mask，不要只看 source 更短。
5. **不建议直接迁移的内容**：绕过 CANN workspace/runtime 入口的宏；本地 910B/AIC 数假设；rank3 Host `return` 后的 unreachable routes；把注释里的“测得”当成当前 910B4 Judge 的数据。

以上是源码结构给出的候选方向，不是性能承诺。拿我们的目标环境（CANN 9.0.0.0、CANNJudge 910B4）逐项做候选前，仍需用官方文档或该 SDK 构建证明 API/key 支持；最终以全 9 点正确性和重复 Judge 时间为准。

## 8. 本轮验证范围

- ZIP CRC、成员路径和 SHA-256：已检查；原 ZIP 保留。
- 源码静态审阅：Host 完整路由和 shape 守卫；关键 TPL/key/入口、三套小向量和三套 Cube/Vector 结构；第三名的重复编译区与不可达 Host 路由。
- 没有运行 ZIP 中的脚本、编译或设备测试；没有可由本轮代码审阅支持的性能归因或分数结论。
- 后续如果把补充材料放进 GitHub，需在独立 supplementary manifest 中继续使用这三个原始 SHA、用户本地来源与“名次映射由用户提供”的口径；不覆盖 11-08 页面快照的 manifest/source_index。

## 9. 三包完整文件覆盖账本

本轮复核了原始 ZIP 与已归档解压目录：按 ZIP 内路径逐项比较 SHA-256，三包所有成员均与解压文件一致；归档名仍保留“rank1/2/3_local_1109”，名次对应关系来自用户提供。三个压缩包共 27 个文本文件、15,177 行源码/构建配置；以下每个文件均纳入审阅，不只抽看核心 kernel。行数按 UTF-8 解压文本逐行统计。

| 名次 | 文件（均相对该包 `code/`） | 行数 | 本文件审计重点 |
|---:|---|---:|---|
| 1 | `CMakeLists.txt` / `op_host/CMakeLists.txt` / `op_kernel/CMakeLists.txt` | 20 / 24 / 15 | ASC 包、Host codegen/tiling/opapi、kernel library 依赖与编译入口；三份实现的这三份 CMake 文件逐字节相同。默认目标写为 `ascend910b`，未钉死 CANN patch 版本。 |
| 1 | `op_host/swi_glu.cpp` | 441 | 输入 shape/dtype 检查、M 展平、核数/UB 查询、九种 shape 路由、workspace/blockDim/key 的设置。 |
| 1 | `op_kernel/swi_glu.cpp` | 1,229 | Cube full-K、split-K、单行波次、双 C0 双波、投影/激活队列、TPL kernel 入口和 workspace wrapper。 |
| 1 | `op_kernel/swi_glu_tiling.h` | 14 | tiling ABI 字段、vector tiling alias、固定 gather 容量及结构体尺寸断言。 |
| 1 | `op_kernel/swi_glu_vector.h` | 1,062 | scalar/tiny/static-dimension/batched/pair vector 函数族及 gather/尾部映射。 |
| 1 | `op_kernel/tiling_key_swi_glu.h` | 135 | TPL 参数取值域、允许的组合和 AIV-only/Mix kernel type 约束。 |
| 2 | `CMakeLists.txt` / `op_host/CMakeLists.txt` / `op_kernel/CMakeLists.txt` | 20 / 24 / 15 | 与名次 1/3 的构建入口逐字节相同。 |
| 2 | `op_host/swi_glu.cpp` | 200 | 静态 tiny key、13/33 维 row group、split-K、资源门控 Cube 路径、resident plan 和 workspace 预算。 |
| 2 | `op_kernel/swi_glu.cpp` | 116 | 数字 tiling key 到 AIV/Mix kernel type 与实现类的完整分派。 |
| 2 | `op_kernel/swi_glu_activation.h` | 25 | 半精度负输入 SiLU 的 Exp/Add/Div 顺序、mask 状态恢复和操作数别名。 |
| 2 | `op_kernel/swi_glu_cube.h` | 154 | Cube 双投影固定 K 流水、权重延后装入、C0 槽复用、AIV 激活及跨核 flag 生命周期。 |
| 2 | `op_kernel/swi_glu_generic.h` | 477 | Full-K 权重驻留 fallback、Split-K 的 NZ/LoadData/Mmad/Fixpipe/Vector 数据布局、尾块/空子块和 GM 槽复用。 |
| 2 | `op_kernel/swi_glu_resident.h` | 247 | gate 权重跨 panel 驻留、容量核算对应的数据路径、C0 双槽与 AIV panel 消费。 |
| 2 | `op_kernel/swi_glu_tiling.h` | 40 | enum key、tiling ABI、resident/tile 常量；`GATE_RESIDENT_MAX_PANELS` 等是此实现容量契约。 |
| 2 | `op_kernel/swi_glu_vector.h` | 327 | 单行、动态行、D=13 分组、D=33 五行组实现和 workspace wrapper。 |
| 2 | `op_kernel/tiling_key_swi_glu.h` | 19 | 当前文件中的 TPL 声明全被注释，**不参与编译分派**；实际分派依赖 enum 数值与 `TILING_KEY_IS`。 |
| 3 | `CMakeLists.txt` / `op_host/CMakeLists.txt` / `op_kernel/CMakeLists.txt` | 20 / 24 / 15 | 同一构建模板；没有随包附带 test harness 或 SDK 版本锁定。 |
| 3 | `op_host/swi_glu.cpp` | 500 | shape 合法性、资源测算、九个精确 shape 的早期特化路由、通用 fallback 和早返回后的残留路由。 |
| 3 | `op_kernel/swi_glu.cpp` | 9,849 | 三个 key 条件编译大区、PATH43–51 相关实现、Cube pipeline、窄维 GEMV、packed gather、编译器 wrapper 适配。 |
| 3 | `op_kernel/swi_glu_tiling.h` | 112 | 按 `TILING_KEY_VAR` 分支定义不同 tiling payload；Direct/Tiny payload 用 `static_assert` 保护与完整结构体的公共前缀 ABI。 |
| 3 | `op_kernel/tiling_key_swi_glu.h` | 53 | 按 TPL 编译 key 限制 PATH 值域和编译实例；需要与 Host key/path 一起读，不能单独把某 PATH 文本当实际执行路径。 |

**覆盖核对：**名次 1 为 8 文件 / 2,940 行；名次 2 为 12 文件 / 1,664 行；名次 3 为 7 文件 / 10,573 行；合计 27 文件 / 15,177 行。三包解压文件均与 ZIP 成员哈希一致。此前章节的结论仍限于静态源码证据；文件覆盖完整不等于逐个 key 已在 SDK 编译，也不等于设备数值/性能已复现。

## 10. 补充的关键审计结论

### 10.1 名次 2 的“负形式 SiLU”是有意的代数变换

这条路径不能只看 helper 名称，必须连调用端的 `Sub` 参数顺序一起算。`KernelCubePipeline::Vector` 先把 bias 乘以 `-1`（`swi_glu_cube.h:122`），之后调用 `Sub(input, bias, input, ...)`（`131`）。AscendC 的 `Sub(dst, src0, src1, ...)` 是逐元素 `src0-src1`，所以这里得到

```text
input = (-b) - projection = -(projection + b) = -z
```

`NegativeSilu` 做 `exp(input)`、`1+exp(input)`、`input/(1+exp(input))`（`swi_glu_activation.h:15-19`），输入 `-z` 时结果为 `-z/(1+exp(-z))=-SiLU(z)`。第二路同样由 `Sub(input[...], bias[...], input[...])` 变成 `-z2`（`swi_glu_cube.h:137`）；最后 `Mul` 的两个负号相消，得到 `SiLU(z1)*z2`。因此这不是把标准 SiLU 写错，而是成对把两路都变成负形式以复用同一个 Exp/Div 链。

**可迁移点：**当后续门控是乘积时，可评估“分别计算 `-SiLU(z1)` 与 `-z2`，在最终乘法中抵消符号”是否减少显式取负或改善可用指令链。必须满足两路符号成对变换、`Sub` 实参顺序不变，并且 half 的 bias/投影舍入顺序仍符合目标容差；不能只复制 helper，也不能把这一实数代数等式当作浮点逐位等价。此结论来自源码表达式及算术推导，未在本项目设备上重跑。

### 10.2 名次 2 的 TPL 头文件是“遗留草稿”，不是有效路线图

名次 2 的 `tiling_key_swi_glu.h` 内容均处于注释中；真正 key 是 `swi_glu_tiling.h` 中的 `SwiGluKey` 数值，Host `SetTilingKey` 与 kernel `TILING_KEY_IS` 一一配对。迁移时要以生效的 Host/Kernel 配对为准，不能因头文件里留有 AIV/Mix 模板选择注释，就误以为那是编译器当前约束或已经实例化的 template specialization。

### 10.3 名次 3 的关键不是“一个 9,849 行 kernel”，而是多个不同编译 ABI

`swi_glu.cpp` 在 `TILING_KEY_VAR=11265`、`12033`、`12801/13057` 处切成互斥大区；tiling-data 头也按编译 key 重定义结构。`SwiGluDirectTilingData` 与 `SwiGluTinyTilingData` 的前缀尺寸断言，是让特化入口可以读取公共字段而不携带完整 `TCubeTiling` / gather 数组的 ABI 手段。其可复用思想是“编译期去掉该路径不用的 tiling payload”；复用前须确保 Host 写入布局、kernel 编译 key 和 entry 读取布局完全一致，任何结构字段/对齐变化都应同步核对。

### 10.4 rank3 PATH51 的静态行覆盖仍列为待验证项

此项保留为源码审计风险，而非已确认越界：路径模板可枚举到 70 个 128 行 tile（8960 行），题目点 9 只有 8192 行；目前查看的该路径 epilogue 未见清晰的 `row < M` 裁剪。获奖来源、Host 的 shape 守卫和排行榜身份都不能替代地址边界证明。后续如考虑迁移，需先证明最后 6 个多余 tile 不会发起 GM 读写，或找出被漏看的有效裁剪。

### 10.5 构建入口相同，不代表版本/设备兼容性相同

三份 CMake 构建文件逐字节相同，默认 `ASCEND_COMPUTE_UNIT=ascend910b`，只描述 ASC package、Host tiling/opapi 和 AscendC kernel 的组装关系。它们没有在这三包内固定 CANN 9.0.0.0 的 patch/build，也没有携带可独立运行的编译测试工程。因此这里把 910B4 Judge 上的获奖身份按用户给定来源记录，但不声称由 CMake 或静态代码确认了 CANN API 兼容与目标设备复现。

## 11. 与 `SWIGLU_TOP3_ANALYSIS.md` 的交叉核对

用户提供的对照文档补入了自研 T197 与前三名的榜单时间、case8 profiler 摘要、候选解释和后续实验状态。它有助于把源码特征放回实际差距中，但这些时间/Profiler 数字没有随本轮提供原始 Judge 导出或 msprof 数据文件；本节将其作为“对照文档报告值”，只独立复算算术和源码路由，不把它们伪装成本轮重新采集的测量。

### 11.1 可确认的时间差距算术

按该文档表格：T197 合计 56.86 µs，第一名公开成绩合计 48.87 µs，差 7.99 µs；四个大项 c3/c5/c8/c9 的逐项差值为 1.14/1.18/1.86/2.31 µs，合计 6.49 µs，占 7.99 µs 的约 81.2%。其余五项差值合计 1.50 µs。加总与占比算术一致；“公开成绩”“最好成绩”的真实性、计时口径和所对应源码提交，仍以原 Judge 页面/提交包为准，本次并未独立复核。

### 11.2 第一名 c9 的 `PACKED_N` 与双 GM 槽：源码支持，但需限定描述

第一名 Host 对 `M=8192,d=256` 选择 `CALC_MODE=3,DIM_TAG=18`、20 blocks、`tileRows=128,tileCols=128`（Host `188-218`）。Kernel 实例化 `ComputeProjections<T,true,false,false,64,256,128,...>`；类内由 `STATIC_DIM==256 && STATIC_ROWS==128` 推导 `FULL_ROW_TILE=true`，继而令 `PACKED_N=true`、`DOUBLE_GM=true`、`ROW_PREFETCH=true`（kernel `36-47`）。因此这里确实把两路权重布局进同一个 N=256 投影面，由一次 `Mmad` 指令覆盖两路 N 列（每个 K 子块仍按 64 深度循环，不是整条 K 只发一条 Mmad），Fixpipe 以 `ndNum=1` 写入投影结果；两个 GM 输出槽按迭代奇偶交替并通过 CrossCore flag 管理消费者释放（kernel `121-123,190-261`）。

可迁移的是**把可共享输入的两路投影合并成单一 Cube 输出面、再用槽位化 GM 交接形成生产消费重叠**。必要前提是打包后的 N、L0B/L0C footprint、Fixpipe 布局、消费者拆分和 flag 生命周期同时成立；文档中的 `Mmad(128,256,64)` 是每个 K chunk 的指令形状，不应误读成一个指令完成完整 K=256。

### 11.3 c8 profiler 归因只作线索，不等同因果证明

对照文档报告 T197 c8 `dur=5.26µs`，并列出 v_scl/vec/v_m2/v_m3 的 profiler 数值和 195 次 `SetValue`、两次 `Gather`。这些项目可以作为继续定位的线索，但管线各单元时间可能重叠，不能简单把四个耗时相加；也不能只凭指令计数就断言恰有 4 µs 可由去掉索引/Gather 全部收回。源码静态审计确认第一名 c8 的 fast path 有常量掩码、GatherMask、32+1 分段归约和输出压缩；“相比 T197 减少了多少实际关键路径时间”仍需同设备同口径的配对测量来归因。

### 11.4 其余性能解释的证据等级

- “c3 碎指令/flag 等待”“c5 小事务”“c9 Fixpipe/Vector 串行等待”等属于对照文档的 profiling/性能解释；本轮只审代码和文档，没有原始 trace 可重算其关键路径，所以保留为待验证解释。
- “第二名特化较浅所以分数略低”“第一名的某项设计造成名次差异”是因果假设，单次整体分数或源码差异不足以证明。宜逐项对齐测试 shape、源码版本与时间分布后再写成结论。
- 对照文档明确列出 T215 c8 为“已写入、未编译、未测”；它是候选，不是正向/负向实验结果。c9 `PACKED_N`、c3 1:1、c5 双 L0C、c7 padding 也均不能仅按方案描述记作已获得收益。

### 11.5 合并后的结论

对后续比赛最值得保存的不是前三名总分，而是三个层级：一是**源码已确认的结构套路**（小型 workload 走 AIV、双投影共享输入/片上权重、针对非对齐小维度的静态掩码与尾部布局、双槽跨核交接）；二是**明确前提**（资源尺寸、launch wave、有效行数、数值舍入和消费者布局）；三是**仍需实验才能建立的性能归因**（某条指令/某个 pipe 究竟贡献多少微秒）。本次结合文档补强了 c8 和 c9 的可复用结构说明，并没有把 T197 profiler 摘要或获奖成绩提升成未经核实的因果证明。
