# 04 det：逐行审读记录（first_pass_text_complete / proof_open）

题面：[原始PDF](../../problems/04_det.pdf)；[题目契约](../SPEC_CONTRACTS.md)。

设计的题目专属/问题族/通用编排分类见 [适用范围总表](../DESIGN_CLASSIFICATION.md) 第04节；SDK技巧不自动等于通用优秀设计。

已阅读全文：三rank所有27个文件，共9262行，包括rank1全部1687行kernel、rank2全部3548行kernel、rank3全部源码。真实逐文件hash/行数见 `../coverage/04_det.json`。无编译、无NPU执行，未验证公示分数与这些源码的二进制同一性。前面S3/S4/S6等保留分阶段审读时的上下文，新增闭环以S10–S18为准。

## S1 题面数学/shape/精度契约

已读题面04.txt两页，并视觉核对04-1/04-2.png。FP32 A=`[...,n,n]`，n1–512、batch维大小1–256，输出FP32 `[...]`，无尾部单例；无属性。不考n0空矩阵。det(A)=∏λ_i；PA=LU时det(A)=sign(P)∏U_ii，L对角为1。交换一对行使符号翻转，因此符号不是可丢弃信息。主要误差 `|Δ|/|det_golden|≤1e-4`。禁止完整det实现、CPU回退、以logdet输出代替det。

PDF表格将“输入数据保证 `det(A)∈[1e-3,1e3]”拆散在单元格，且残留Markdown反引号；视觉上没有明确写绝对值，不能擅自修成 `|det|` 范围。第二页另有负号/换行校验，存在题面表述边界。不能从最终det有限范围推导各pivot/部分乘积有限，也没有κ(A)上界。例diag(2^100,2^100,2^-100,2^-100)最终det1，而先乘前两项会溢出，指数分离有实际数学必要；但指数分离不修复因消去误差导致的错误pivot。

## S2 构建、ABI和key

三类CMake的SHA在本题各rank及另两线代题完全一致：根1–20要求CMake3.16、ASC包、未指定时ascend910b、SHARED custom包；Host1–24对*.cpp代码生成+tiling、aclnn生成库和包注册；kernel1–15扫描本目录并绑定tiling库。按hash复用代表文件审读，不跳过不同kernel头。生成wrapper、优化参数与实际SDK仍需编译产物。

rank1 tiling1–6含8个u32（n/rows/batches/blocks/panel/team/groups/reserved），metadata在后面存rowOffsets[64]/diagonalOffsets[64]，DET_CUBE_BLOCK64。key1–39：path1/4 ORDER0纯AIV；3 ORDER0–16纯AIV；5 ORDER33–48与7 ORDER49–64纯AIV；6 ORDER0/1/33–56混合1:2。ORDER是实现绑定值：有时n、有时每team宽度、有时1标记，不能统一解释成矩阵阶数。

rank2 tiling1–3含n/cores/resident u32和batches u64（自然对齐有padding，不能按4个u32直接复制）。key1–18以8bit TASK_MODE范围0–17、32–39、48–79、96–159、160–183编译；0及tiny/full/compact纯AIV，1及96–159混合1:2。UI_RANGE/UI_MIX可变参数编码本轮未查实SDK声明，Host/入口的解码对应见S4。

rank3 tiling1–33：Config9个u32前缀；TilingData相同前缀后TCubeTiling；Payload再32B align的gatherIndices[384]。ReadDetConfig通过sizeof(Config)/4复制前缀，要求这两个结构前缀布局一致。key1–16编USE_CUBE(bool)、PADDED_N0或67–128、ONE_ROUND(bool)、STATIC_BATCH0–6；n3–64编码PADDED_N=n+64，字面不是pad8(n)。非Cube纯AIV；一般Cube tag0混合1:1；tag1/2/3/5/6混合1:2；静态小矩阵另有ONE_ROUND且batch≤4组合。

## S3 rank1 Host 1–84与已读kernel尾

Host1–16引入共享头、ParseShape：rank≥2、方阵n1–512、每batch维1–256、乘积u32上界；输出删除尾两轴，rows起8倍增。17–34检查FP32/AIV>0，blocks=min(batch,AIV)，取UB字节、padded=pad8(n)、pitch=padded+8（pitch局部未用）；按 `panel=floor8((UB/4−(14*padded+128))/(n+1))` 估panel，截至padded，若不足则向下取2次幂（最少8），先按blocks*n*padded*4为workspace。UB减法为u64，极小UB可能下溢，但910b真实UB容量需平台值，不把此视为题面内实测问题。

35–44小矩阵：n≤16 path3、rows16；17–32 path4、rows32并写rowOffset=j*(rows+8)*4、diagOffset=j*(rows+9)*4；33–48 path5、49–64 path7。中小blocks向上8核分组，tiny至少min(AIV,8)。45–65大矩阵team=ceil(n/64)，当AIC≥team，groups=min(AIC/team,batches)并向下找到整除batch的组数；当n不是64倍数或所有batch可驻留才path6。workspace=groups*(G²+128G)*4，G=team*64，schedule1；ORDER0默认，当全部batch驻留且n%64=0设1，否则n整除team且width≤56则ORDER=width。66–84发key/blockdim，非path4只提交base大小，加入系统workspace，shape/type/910b注册。

kernel1603–1635是一般pivot消去末段：bestBits0/selected越界标singular；行交换翻det符号；先提取diagonal指数（夹±126）并乘mantissa，再重归一product累计指数；列乘−1/diagonal、换pivot行、非最后panel把factor列写GM。singular输出0，否则分段±126恢复指数，最后单float DataCopyPad。1636–1647两种noinline recovery：直接DetKernel.Run<true>或从tiling前7u32读配置，再对指定batch重算，真实DetKernel主体未读，不能断言其全部数值稳定性。

入口1648–1687：InitSocState、metadata注册；ORDER编译固定的path3/5/7只从tiling读batch，空闲block提前return（无crosscore消费者）；否则复制配置前缀。3→TinyResident；4→ResidentDet<32>；5根据ORDER%8选ReverseResidentDet<ORDER,72>或ResidentDet<64,ORDER>；7相应104pitch变体。6→BlockDet<64,ORDER>，retryAny时只AIV按group=BlockIdx/(2team)，将其G²工作区作为CorrectnessRecovery输入；其余DetKernel。retryAny的跨核一致性、fallback是否所有参与核同时进入、workspace覆盖都在未读主体，不能用这个尾部证明无死锁。

## S4 rank2 Host 1–57与kernel3464–3548

Host1–12依赖、平台、StorageShape取n和batch乘积，未本地校验rank/方阵/维度范围，batches u64；owners初始化0后未改变。13–19 n>128用4AIC/slot、最多4slots、schedule1；n≤64取至少8AIV但不超过平台核数；65–128按batch减核。resident=n<16，其他0。20–36使用32B dmaFloats8、vectorFloats64、tinyRows16、panelTile128、slots4：默认mode=n>128?1:0；8核单batch tiny设n+1；8核batch≤8时，n58/60/62→160+((n−58)/2)*8+(batch−1)、n64→32+batch−1，n50/52/54/56→48+((n−50)/2)*8+(batch−1)；大矩阵16核、n136..256且8对齐、batch≤4时mode96+(n/8−17)*4+(batch−1)。只绑定shape/核数，不根据输入数值。

37–57发key/blockdim；大矩阵workspace每slot `pad128(n)²+64*pad128(n)+4*8192` float，加系统区；InferShape删除两尾轴，InferType固定FP32，注册910b。无小矩阵workspace但仍系统区。

3464–3474收尾按uint32/float UB物理地址Scalar写product，S_MTE3后DataCopyPad；类声明pitch72和flatIndicesReady，主体未读。3475–3510入口完整：tiny2–17→n=mode−1/batch1/cores8；full32–39→n64/batchmode−31，160–183→n58+2*floor((mode−160)/8)；compact48–79→n50+2*floor((mode−48)/8)；alignedLarge96–159→n8*(17+floor((mode−96)/4))，batches余4+1、cores16。非绑定才GET_TILING；小矩阵无任务核提前退出。mode1/alignedLarge→FastBlockProcess；tiny17→BoundPrefetch16，16→Terminal15，10–15→PrefetchPadded16，其余Fullwidth；full→MediumBoundStaticSuffix64；compact→MediumBoundCompact56；一般n16 Explicit16、n≤16 TinyFullwidth/TinyDet、33–64 MediumDet，其余ColumnDet。

3511–3548是特别重要的**生成代码后缀重写**：namespace AscendC自定义DetSetSysWorkspaceForMode，预处理宏将后续wrapper的SetSysWorkspaceForce重定向；同法将set_ffts_base_addr和icache_preload重定向。仅tiny/full/compact这些无crosscore的模式跳过原初始化，四个DEBUG/DUMP/TIMESTAMP/OOM宏时保留；preload适配器忽略参数并固定1（作者注释声称CANN9生成调用恒1，本轮未核验生成文件）。宏定义放在原函数体之后，原体不被递归替换；后续SDK wrapper是否正好被替换、namespace查找、TEMPLATE_PARAMS存在与其生命周期均绑定具体生成器。机制可能省启动指令，但没同环境paired测量，不能量化或归因名次；绝不可当跨CANN稳定API照抄。

## S5 rank3 Host 1–121

1–20校验OriginShape/rank/n/方阵/batch乘积（无每维≤256检查），21–34以AIC数至少1规划：USE_CUBE=n>128、block=n>256?32:64；Cube paddedN=pad32(n)，否则pad8(n)。capacity=min(128KiB,UB>49152?UB−49152:65536)，maxColumns=max(1,capacity/(pn*8))对应双float列存储，但另有辅助缓冲，最终峰值需S6逐项合计。`else if(n>128)` 位于`if(USE_CUBE)`的else后，按USE_CUBE定义不可达，只是遗留代码不是执行路径。

35–48 n256且4核/matrix、至少16AIC并batch%4=0时只4组compact wave；n3–16固定groups1，否则min(batch,groupCapacity)。columns=Cube?block:ceil(n/perMatrix)，sharedStride=columns*pn+2pn+128（float元素）；payload全零后写前缀。49–61 n3–64构造gatherIndices前q项r*pitch*4、接q项r*4；q=nextpow2(pn)≤64，pitch小n为pn、pn56为56、其余2pn+8；小n非8对齐额外128起重排表，长度ceil64(n*pitch)≤256，无表越界。每offset以字节计，表非矩阵数值。

62–81 dtype从输入，cubeMode/robustOffset给Cube布局和fallback预留；n256额外2pn*64；n200/256且4core时A预载A1/NZ/transpose，其余GM/ND，B/C GM/ND FP32；Mshape≤256时256，否则pn；K n256为64否则32；n≤256关闭双缓冲；FixSplit初M256或128，GetTiling失败只重试M128，再失败返回。82–91 ONE_ROUND小矩阵且batch=groups；STATIC_BATCH compact单波5、多波6、其余n256→3、n200→2、小矩阵一次驻留≤4→batch、默认0。小核至少8blocks，但active guard另审；n3–16 workspace0，其他groups*(sharedStride+robustOffset)*4+系统。92–121 InferShape仅rank≥2并复制batch轴，dtype继承，AutoContiguous，支持910b与910_93；Origin/Storage不一致时需要框架contiguous契约。

## S6 rank3主kernel 1–600：连续行段语义

| 行段 | 人工实读说明 |
|---|---|
|1–18|依赖、det_small引入；KernelLinalgDet<T,ALLOW_FAST>由外部TPipe供给，按coresPerMatrix映射lane/group/first/count，最后lane count可0。|
|19–57|绑定FP32 GM、共享区按group*sharedStride；panel双分量2Cpn、vector2pn、temp8pn、raw8pn、indicespn、meta8、scale pad8(C) float。总`(2C+19)*pn+pad8(C)+8` float；仅非Cube且n>128,C16再加lower/correction各16pn，此条件Host通常不可达但fallback可能设不同配置须核对。directFast单核n3–128跳初始化indices/meta，Fast失败会补正确索引。|
|58–113|round全组同次数，inactive核仍经过每pivot两次SyncAll；单核Fast安全则continue。常规Load后owner MakePivot，共享两分量v和meta，通过MTE2_V/S消费；每pivot交换符号、补列缩放指数、将pivot和product指数置127保存mantissa。Update后全核同步。仅active lane0写精确4B并等MTE3_S。|
|114–133|Fence从TPipe取对应事件ID后立刻set/wait；PowerOfTwo直接写exponent字段；Rescale按±120分块，不构造超范围指数。正常数分解成立；subnormal/NaN/Inf不在位拆分公式内，不能一概当安全。|
|134–202|Fast将row-major A补齐pitch，先全矩阵abs max（len>8192拆2次reduce），仅1e-20<max<1e20试无主元LU；pivot为k行，不搜索行；小于1e-4*原max则拒绝并回退。rank-one update向量广播；指数分离，safe时精确4B输出，unsafe恢复i*32 Gather索引。原max阈值是启发式，不是误差证明。|
|203–217|MultiplyDet分离pivot/product的float指数，带符号mantissa累乘；0直接置product0；无subnormal专门分支。|
|218–255|PanelRankOne行主16列块，Brcb复制v两份，选列mask写lower，重复sh系数形成外积，correction保存加减舍入残差；每阶段PIPE_V，避免读未完成结果。|
|256–268|SwapPanelRows同时换panel和correction，保持补偿行绑定，不只换high。|
|269–339|BlockFast为16列owner panel；清padding与correction，exact-width strided copy，absmax检查范围；每pivot最大值搜索k后行并行交换，阈值0.0625*local原max标risky但继续完成协议；存row列表/scales与panel/meta到共享区，MTE3_MTE2后发布。|
|340–392|所有核同barrier、读metadata、future lane读lower并执行同置换/更新；末尾每组lane0写risk，所有核扫描每组risk并OR，再barrier，只有全局无risk才写结果；anyRisk统一恢复索引并返回false，避免部分核进入重算而其他退出。Host条件让这段常规不派发，不从存在推断贡献。|
|393–417|Pair{hi,lo}；Sum是volatile TwoSum/renormalize型，Product用4097拆high并加交叉及lo*lo，Divide先q=a.hi/b.hi再算双分量余数修正。volatile意在保留独立Scalar舍入/防收缩，不等价FP64；溢出、subnormal仍需假设。|
|418–455|VectorProduct逐元素拆a.hi，乘b高低、加a.lo*b.hi，未加入a.lo*b.lo；VectorSubtract恢复相减舍入残差再重归一。其与Scalar Product不是完全相同精度。辅助s0/s1/s2/ph/pl/sh/sl相位复用在8pn temp内。|
|456–486|Load将每列n个4B元素放UB每32B槽，Gather整pn，raw先全清零保证padding；统计绝对位最大、scale夹±120、乘精确2幂，panelLo先零；“精确”仅在不发生下溢/溢出舍入时成立。列尺度记录供det补偿指数。|
|487–528|MakePivot用Scalar high分量扫描r≥k，low不参与最大值；交换high/low，双Pair倒数（high0取0），计算v high/low，清前k+1行、发布pivotRow/pivotHigh/scale；输出det仅用pivotHigh而非pivotLow。owner写共享先V/S→MTE3，再MTE3→MTE2。|
|529–558|Update只对未消去的本地列，先换pivot行high/low，以factor pair乘v再双分量相减。声明成员/结束类包含所有TBuf与alias，generic无L1/L0显式分配。|
|559–600|引入五个尚未读Cube协作头；noinline DetRunFallback使用ALLOW_FAST=false，固定4batch/wave fallback改cores8、C32、stride34*256+128，偏移4*164096*4；wave版明确batch4避免整batch重复重算。Fast/Sequential/Static200包装也noinline，具体函数体未读。|

空间例：generic fallback n512,pn512,C32→170144B；n256,pn256,C64→150816B，未含编译器隐式/栈开销。workspace共享当前pivot只需2pn+8 float，但Host多留columns*pn+128；Cube robust区能否与fallback所有group正确隔离须读五头消费布局，当前仅核对Host公式与前缀，未伪称完成。

## S7 rank3入口601–724：真实快路/fallback

601–629注册Payload；仅PADDED_N67–80（n3–16）不用user workspace；tag5运行固定4batch cube+vector direct-diag，不safe则固定fallback；tag6按batch/4多波，A/out指向当前波，非最后波SyncAll<false>后复用workspace。630–677 tag3走K64，2走n200预取与偶/奇AIV主/辅助，1走n256预取；每轮batch映射BlockIdx/(2coresPerMatrix)，fallback将剩余batch截至groups、翻倍vector core/C32并用robust区。678–696一般USE_CUBE在AIC n>256走DetWide否则RunLUCube；AIV按n200/partition窄幅选择Static/Sequential/Fast，unsafe同轮whole-group fallback。所有safe值是否跨AIC/AIV一致、退出是否与service握手，依赖未读头。

697–724非Cube：PADDED_N>0解码staticN=PADDED_N−64；n≤16只Block0，其他active block由STATIC_BATCH或fields[4]限制。ONE_ROUND调用SmallDet<aligned,fixed,staticN,true>，否则按groups轮转batch并AND累计safe，任一失败最终调用完整fallback；无PADDED_N时generic KernelLinalgDet.Process（n1/2、65–128等）。noinline边界避免重路径内联污染快核的机制可推测，但未反汇编验证指令缓存收益。

## S8 det_small.h 1–247：全部特殊写法

1–12定义固定事件0、Min、清符号位取绝对值位、原生mask；13–37模板ALIGNED_N/FIXED/STATIC_N/FINAL_OUTPUT，NORMALIZE仅ALIGNED_N>32，q为8/16/32/64，p=q*(2q+8)，raw LocalTensor直接绑定VECCALC地址0（不用TPipe InitBuffer），总容量 `2p+41q+8+(ALIGNED_N≤16?ALIGNED_N²:0)` float，q64时80160B，q16时8800B；__builtin_assume约束realN仅该pad8桶，依赖Host守卫，不是运行时拒绝。

27行pitch小n为ALIGNED_N，ALIGNED56特选56，否则2*ALIGNED_N+8；period取ALIGNED_N/4的最低有效1位（56特16），bankMask=period−1，实际raw基址右移5按32B bank选偏移。此为UB bank安排，不改变GM矩阵shape。38–74读Host字节Gather索引；n%8时先把紧凑A exact-byte DMA至已死panelLo，再小n用Host额外索引vgather重排、大n逐行Gather；8对齐时直接行DMA；FIXED64本地构造行偏移，分别MTE2_S或V事件等待。75–89把panelLo生命周期复用为两列Gather索引，magnitude常量1并未统计输入量级。

90–155每次消2行：对剩4且ALIGNED>16，用6个左上2x2 minor和6个右下minor的互补组合求4x4det，实数下为拉普拉斯展开；小n剩4用2x2块逆的Schur补，乘det(D)*det(S)，要求左上2x2非奇异，riskMask会检查过小/过大pivot，失败交fallback。156–165非NORMALIZE剩3直接三阶公式，没有检查相消条件。166–207两个列Gather、2x2 pivot `aa*dd-bb*cc`，构造−D⁻¹两组乘子，vbrcb后两次vmla更新trailing行；更新起点按8向下对齐，mask限制ALIGNED_N而非真实n，真有效输出只依赖n范围，padding值不应作为pivot。raw bank shift在其预留区域内，需具体指令repeat/stride的SDK契约确认每路广播最大写址。

208–226 product累乘并每4个2x2（k&7==6）或最终归一指数；riskMask的hex是float绝对位阈值，用无分支OR汇总，接受条件检查正有限近似范围，不是数值误差证书。227–247 safe时写结果：FINAL_OUTPUT且ALIGNED≤16、__NPU_ARCH__==2201且输出地址128B对齐，使用WriteGmByPassDCache直接4B写并return；否则meta Scalar写→S_MTE3→DataCopyPad4B，非FINAL_OUTPUT额外MTE3_S防下一batch复用。没有出现日志det替代输出；指数最终恢复为det。

## S9 数值/退出风险、官方核验与可迁移边界

确定的数学边界：2x2 Schur补快路不适用于左上块奇异而整矩阵可逆的输入，源码用riskMask+fallback补充；不能删guard。generic部分主元、列尺度与双float比无主元更稳，但vector乘积略掉lo*lo，det连乘用pivotHigh，4097拆分和指数field假设需有限正常数范围。n3直接公式的magnitude=1与末det非零有限检查不能证明在题面允许的病态矩阵上达1e-4；当前未用设备/编译器浮点契约产生确定反例，列为风险不是已运行失败。

官方CANN9.0 [SyncAll](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0204.html)明确true只AIV、false混合两侧，并要求硬同步匹配kernel类型及batchmode。此支持对现有barrier意图的解释，不证明未读service端调用次数正确。 [GetReduceRepeatMaxMinSpr](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0226.html)注明A2以WholeReduceMax获取准确index；具体引用参数类型与本地SDK头仍未核验。

WriteGmByPassDCache适用性存在官方版本差异：[9.0页](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00089.html)A2/A3标不支持、Atlas350支持；[9.1页](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/scalar_compute/WriteGmByPassDCache_ISASI.md)列A2/A3支持，且要求考虑目标dirty DCache、必要时clean及DataSyncBarrier。源码仅按arch2201编译，不能不锁SDK就断言合法/非法；目标仅写不读可降低dirty cache风险，但外部caller/生成代码状态仍需证明。

可迁移：按容量桶编译专化、真实GM extent独立于私有pad、跨核统一risk后重算、双float消去与指数分离分别解决误差和动态范围、死buffer复用、exact4B输出。不可照抄：写死wave4/batch/核比例、无全局误差界的阈值、原生UB/指令参数、arch-specific bypass、SDK生成wrapper宏。原公示名次不足以说明任何一个技巧单独提速。

后续已补齐rank2 1–3463（S17）。三份源码首轮文本完整；首轮文本完整不代表SDK协议和所有数值前提已证明。

## S10 rank3五个头：实际路由，不能把遗留实现算作有效优化

重新对照Host62–91及入口581–696：Host令n256非compact使用tag3，compact使用tag5/6；n200使用tag2。因此tag1的`CubeLUVectorHelpers::ProcessOne`+256预取，以及通用分支中n200的非PAIRED静态实现，并非当前Host的实际路由。`DetWide::Factor/Solve`带pivot版本虽存在，当前`ProcessOne`实际只调用`FactorNoPivot/SolveNoPivot`。`DET_FIXED_CUBE_CFG`和generic两个Matmul配置仍按实际n/分支分别解释，不能凭constexpr出现宣称它给已计分路径提速。

| 当前路由 | Vector生产/更新 | Cube消费者 | 关键条件 |
|---|---|---|---|
|n256 compact tag5/6|DirectDiag main + DetK64 streamed helper|DetK64 fixed4 K64|4矩阵/波、4组、每组4AIC/8AIV；多波mixed barrier后才复用工作区|
|n256 tag3|DetK64 main/helper 非streamed|DetK64 K64|64列stage，末stage仍完成成对flag10协议|
|n200 tag2|Static<200,true> main + LU helper200|LU prefetch200 K32|四个prefix stage到96，128处合并剩72列并本地消元|
|129–255的其他n|Sequential或Fast K32|LU Matmul实现|Host资源/coresPerMatrix决定实际路由，不能套用n256布局|
|n>256|Wide 32列panel|Wide Matmul|双float panel/correction；仍是无pivot试算，失败回generic补偿pivot消元|

## S11 det_coop64.h 1–381与direct_diag.h 1–323

### S11.1 连续语义账

| 文件与行段 | 审读结果 |
|---|---|
|coop64 1–46|N256/RP104；WaitUChunk用volatile GM progress+DCCI轮询，不是普通GetValue。logical=BlockIdx/2，q=logical%4，group=logical/4。main UB194560B；helper68864B，二者不同布局，不能共享一个容量估算。|
|47–107|main Factor两列一组；Gather两列、求2×2逆、保存负L并以Brcb+两次Mla更新。8次pivot行归一mantissa/exponent；风险汇总不立即退出协议。写64列负L、U及metadata，清本stage lower-ready。|
|108–130|未来panel Solve读取64×64已分解块，广播每组8列系数；已死raw复用作展开区；Update读Cube乘积并加回panel，不覆盖尚需的原始输入。|
|131–194|helper FactorTail按32/64行处理lower；STREAMED按16个pivot行等待U progress再读取。32/64行×64列重排为columnPitch=pitch+8；输入packed在转置完成后才复用为系数展开区。start0且固定4组可直接读原A省中转，不能对后续已更新stage套用。|
|195–256|ready buffer与meta不同区；ProcessOne每stage发布/等待flag10，未来panel才参与paired8/9。streamed helper在FactorTail完成后发布lower-ready。结尾读所有组的risk，所有AIV得到同一safe，只有main q0输出。|
|257–317|Cube basic API，q0没有trailing更新所以不启动矩乘。paired8只保护本core的B和helper，另轮询四组ready保护整个A；两条64B cacheline覆盖4×32B flag记录。A1/B1 ND→NZ，LoadData到A2/B2，Mmad→Fixpipe→flag9。禁HF32；unitFlag3使用仍绑定目标SDK。|
|318–381|fixed4把配置、轮数、robustOffset编译固定；资源与同一protocol仍保留，不能把省tiling加载理解成可以省ready检查。两个wrapper分别选择main/helper和是否streamed。|
|direct_diag 1–68|只有main使用紧凑layout：此前/此后panel各192×64，当前diag64×104，raw同时复用top/展开区；总126336B，低于coop main194560B。helper仍走DetK64的68864B。初始8个offset Scalar生成，其余倍增向量生成。|
|69–154|PrepareStream保存不可变lower，PublishUBlock每16行发布L/U。2×2 Factor先更新下一对主元行，V_S使Scalar可先读这两行，再执行其余行更新；不是删barrier，而是改变发布边界。进度MTE3_S→S_MTE3后发布，不能先写ready再发数据。|
|155–211|Solve按16行U进度消费；Update按diag之前/diag内/之后三段写回，diag行距104、其他64，区间必须保持不交叠。|
|212–273|备用FactorTail与DetK64相同数学，32/64行转置和lower求解；SignalBuffer明确放col后段，不与当前meta重叠。模板虽有helper变体，本入口main直调，helper通过另一类。|
|274–323|四个K64 stage；main准备→全体flag10→owner Factor，helper消费U并发布lower；未来main Solve/paired8/9/Update。结尾统一扫描所有group risk；固定配置{256,4,256,4,4,64,17024,0,164096}。未提前按单组risk退出，否则可能挂住仍在等flag的Cube。|

### S11.2 数学和地址边界

2×2主元块D=[[a,b],[c,d]]，p=ad−bc，负乘子为`−C D⁻¹`：两列分别`(-d*c0+c*c1)/p`和`(b*c0-a*c1)/p`；trailing执行`A22 + (−C D⁻¹)U`，det乘p。这个重排属于**F：分块消元**，不是det独有指令技巧；但不pivot时D必须可逆且数值良好。整矩阵可逆并不保证D可逆。hex风险阈值筛掉部分危险值，不能证明整个合法域1e-4误差。

N256 K64的group GM区以float计：双shared区`2*(256*128+128)`；每q own=`q*(256*64+8192)`，own前4096为64×64 B，后为prod。最大prod行192×64=12288，加4096=16384，小于每q24576；双shared总65792、四own98304，共164096，吻合Host。fixed4四组占2625536B；fallback起点正是`4*164096*4`，不覆盖仍有消费者的fast区。compact main减少的是UB容量，**没有**把GM布局改成126336B。

协议设计属于**G：依赖驱动的ready发布+死区复用**；K64/N256/四组和2AIV:1AIC为专化资源条件。volatile轮询避免普通load被提出循环是源码作者声称的编译观察，不是本轮新反汇编证据；DCCI、跨核flag模式与unitFlag组合仍需锁SDK和二进制验证。若删某一个ready，不能凭另一个paired flag推断全A就绪。

## S12 det_cube_lu.h 1–1097：K32主/辅/预取完整链

| 行段 | 审读结果 |
|---|---|
|1–89|固定MatmulConfig 256×64×32、禁止bias/quant/defineData；通用Vector按q分64列，panel(Npad+32)×104，raw Npad×64，vec5×ceil64(Npad)，top32²，idx一pitch；原生vgather/muls/axpy/copy及双列版本分别处理full/tail。|
|90–185|Factor<COOPERATIVE>：协作只消32个pivot行并把剩下lower交helper，非协作处理全部lower；n256最后64列可一次本地完成。offset向下8对齐，先更新下次pivot依赖再发布V_S；metadata区分本stage risk与owner累计risk。|
|186–247|Solve每2行更新future U；FactorTail分摊trailing行，按pitch+8转置并用同2×2逆计算lower；Update只改本panel仍live的列，owner首半stage只更新后32列。|
|248–317|单AIV/core ProcessOne：初始化索引/padding和真实extent DMA；协作条件`remaining>=coresPerMatrix*32`，global10发布panel，协作FactorTail以global12汇合，paired8/9只在Cube需要时执行。末尾跨所有groups扫描risk，拒绝后回主kernel fallback。|
|319–400|Helpers类奇AIV只分raw6912 float、vec320、top1024，共33024B，不初始化panel/idx；偶AIV保留完整布局。使用“helper不会触碰不存在tensor”的分支契约，不是两边同构。|
|401–500|与前Factor共享数学，但协作时额外清four lower-ready；helper协议不能只搬用此函数而忽略清flag。|
|501–575|helper Solve；FactorTail对n200/256且rows%16==0使用TransDataTo5HD<float>，否则vgather。16行×32列通过repeat4重排到pitch+8；V barrier后原packed输入已死，复用为最多8行×32列系数的2048float展开，少一次scratch分配。每8行更新展开，最后仅6个仍用行。|
|576–660|主AIV更新、signal分区；n256非协作阶段先paired11允许A预取，cooperative早stage则A仍等所有helper；helper与Solve并行，必要时等lower。奇AIV的signal在col192、meta最多160float；偶AIV在col4pitch。|
|661–716|helper200只做start0/32/64/96四段，128跟随global12/10后break，匹配main合并72列；普通helper跟随同stage枚举，n256非协作同时发布11；结尾与主AIV扫描相同risk。|
|718–951|Sequential类不分摊lower，仍用pair pivot和K32 Cube更新；真实full/tail extent、双shared轮换、global10、paired8/9和风险汇合均保留。不是“无并行”，只是lower归owner。|
|954–968|Matmul库消费者设置OrgShape(Npad,64,32)，仅枚举本q需要的start；等8→SetTail/SetA/B→IterateAll/End→FIX发布9。跳过n256最后64、n200最后72，与producer本地收尾一致。|
|970–1026|prefetch200基本API：等11，再轮询全部4个lower-ready后发A GM→L1；只在B需要时等8，B到达前可以先搬A。rows=200−start−32不16对齐，ma=ceil16(rows)，额外InitConstValue清A末padding；Fixpipe只写真实rows。A1/B1相邻但不重叠。|
|1029–1097|prefetch256 start>=128时lower由owner在global10后完整，所以等11即能读A；早stage A依赖全部helper，先等8读B再轮询四ready。MTE2→MTE1→M→Fixpipe→9，最后PIPE_ALL收尾。RunLUCube按n256/n200/MatmulConfig选择。n256此家族被当前Host tag3/5/6替代，保留为历史实现参考。|

可复用的核心是**F/G：同一Schur更新中，A与B就绪条件不同，分开发信号后才能合法预取A**。不能从“有两个输入”直接推成“两次DMA任意重叠”。硬件TransDataTo5HD取代Gather的形状条件、底层VA地址表与repeat语义必须重新核验；这里只确认源码调用和生命周期，不宣称某代SDK全部支持。

## S13 det_cube_static.h 1–452：静态活跃域、末72列融合

| 行段 | 审读结果 |
|---|---|
|1–40|模板N/PAIRED决定block映射，裸VECCALC地址分配：panel(PN+32)×104、raw PN×72、vec5VP、top32²、idxVP；N200 PN224/VP256，合计181248B。无TPipe InitBuffer不等于无UB使用。|
|41–90|NativeGather/Muls/Axpy/Copy及Pair模板的full/tail；NarrowMla在columns≤8时把8个不同row映射到64lane，一条repeat处理8行，dst blockStride13、repeatStride104；否则每行一个repeat。前提pitch104且物理scratch覆盖那些行，不能只改mask照抄。|
|91–191|FactorCoop200始终COOPERATIVE=true；8/32列分区更新并提前V_S；保存32列负L和U，owner风险乘积累计；global ready清零。|
|192–208|signal位于vec第5个pitch中，meta另区；DMA读4×32B lower-ready，poll需MTE2_S后看Scalar值。此函数定义不代表所有入口调用WaitLowerReady。|
|209–298|Factor仅处理仍live rows，rowBase=floor8(k+2)，减少Gather/Muls/Axpy元素数；按8列分块、独立event1..4提前写完成的lower到GM，不重用未消费事件；部分lane round-up靠初始化panel末32行保障地址。下一pivot行先ready，剩余列随后更新。|
|299–352|Finish72把n200尾72×72放同一panel，前64列来自q2，尾8列由q3提供。前3对pivot跨64+8两mask，后续只保留一个mask；先完成接下来2行后发V_S。指数/risk保留，最后写owner累计meta。|
|353–372|Solve对32行paired更新，Update仍按q/owner偏移改本地64列；Prod GM只读真实rows64。|
|373–452|8个offset Scalar写、其余向量倍增；padding与identity明确初始化。N200到start128时q3先写72×8、global12后q2载到panel+64；q2 Finish72，所有核global10后退出stage循环；读取全组owner meta统一safe，q0乘各owner的m/e后写4B。|

**P：n200尾72列融合**是该尺寸资源子域的优秀候选，不是任意N都能多合8列。**F/G：缩小活跃行域、提前发布已完成的lower tile、窄列时换lane映射**才可推广。`__aicore__`裸UB绑定、mask归一/重置、event ID常量是实现前提；少一个restore可污染后续函数，不能只摘取最短vmla片段。

## S14 det_cube_wide.h 1–249：大矩阵32列与补偿

| 行段 | 审读结果 |
|---|---|
|1–43|q分32列，panel/corr各Npad×32，rawNpad×8，vec8Npad，top32²，idxNpad；总`(81*Npad+1024)*4`B，Npad512时169984B。Swap同时换hi/lo，但当前主路径不用Swap。|
|44–101|带pivot Factor建立logical↔physical映射，WholeReduceMax挑未选行、记录pivot；两分量先加后转置，保存因子和swap列表。**当前ProcessOne不调用它**，不能把此代码当wide已用部分主元的证据。|
|102–137|实际FactorNoPivot：panel+corr后转列，p取对角，低于1e-4*magnitude拒绝，magnitude在ProcessOne固定1；负除数和rank-one更新，m/e分开；写32列lower与risk。|
|138–164|带pivot Solve先读swap并换hi/lo，是未调用版本；实际SolveNoPivot读lower块并更新32×32 U，只发布own前1024float B。|
|165–177|Update从prod分段读取，每段最多Npad/4行；raw先加旧lo、再与hi相加，按`newlo=(oldraw)-((hi+oldraw)-hi)`一类FastTwoSum残差恢复；没有对|hi|≥|raw|显式排序，所以不能称无条件exact TwoSum，更不能称FP64。|
|178–227|初始化真实n和补单位矩阵，DMA每行width≤32 exact bytes；每stageowner FactorNoPivot、AIV SyncAll后读risk。safe future panel才Solve/Update，但**所有核仍发8等9**，防Cube协议错位。失败owner仍发布risky meta；最终读所有组risk，统一返回false重算。|
|228–249|Cube每stage都等待8，invalidate该stage risk；只有safe且q>owner才Matmul，仍无条件发9。动态SetTail rows×32×32，A为转置lower，B为own U，Fixpipe结果放own+1024。|

wide group GM占float：双shared`2*(PN*32+128)`，每q own`PN*32+2048`；prod最多(PN−32)*32，B1024加prod不超过单q预留。补偿更新只改善部分相加舍入，不修复无pivot失败，generic fallback仍必要。**不能把magnitude=1的固定阈值当相对原A尺度检测**。

## S15 本轮新增可迁移结论与未证明项

1. **P/资源子域**：n256四矩阵compact wave、n200末72列合并，均从真实输入计算，不是预制输出；但适用域窄，照搬必须有Host守卫。
2. **F/分块线代**：双列pivot减少Scalar往返；lower/upper不同生产者拆分；特定U只读原A的stage0可省中转。每条都有矩阵依赖前提。
3. **G/编排**：下一关键依赖先更新、分块ready、A/B独立ready、死packed区复用、风险决定全参与者统一。可跨题借鉴的首先是DAG与活跃区证明，不是flag数字。
4. **硬件实现**：RP104、+8列距、16行转置、原生vmla/narrow lane映射、unitFlag3、volatile+DCCI不可脱离CANN/arch/生成二进制说明有效性。本轮没有新增设备耗时或提速归因。
5. 已闭合源码层的stage枚举与消费者计数；未证明极端输入下无pivot阈值、double-float截断、subnormal指数分解和底层硬同步的所有官方条件。`rank3全文首轮完成`仅指139460行账本中的相应区间，不是证明这些代码在当前题面全域正确。

## S16 rank1 kernel 1–1687：完整补齐，包含反直觉的fallback策略

### S16.1 连续行段语义账

| 行段 | 实读与消费情况 |
|---|---|
|1–45|ASCENDC_CUBE_ONLY在Matmul include之前，限制库展开但本文件仍有AIV，不能从宏名字推断整个entry是纯Cube。PivotProduct向量拆diag符号/mantissa/exponent、分64块树乘；全文只有定义，不把它当当前连乘消费路径。DetAbnormal有两种完全不同的接受集合，见下节。|
|46–115|TerminalDet 1/2/3/4阶公式，4阶六个2×2minor相乘；Guard将绝对乘积和scale与结果比较。TerminalSix Scalar顺序无pivot消到最后；只减少最后几步向量往返，不具备稳定性自动保证。|
|117–179|ReplicatedTiny将每元素广播8倍、向量Div消元；当前入口没有调用。TinyScalarPivotProduct递归平衡乘树，仅在现有V_S后读取已完成diagonal；NativeMuls/Mla封装忽略mask参数，要求caller已设硬件mask，不能独立调用。Count版本显式切换counter并恢复。|
|181–310|TinyResident<N>，P选16/24，主arena1408float，raw8192B/packing12288B/索引16384B等分区；pair索引另在24576B。规则offset仅首8项Scalar，其余倍增。一次flat n² DMA，DirectInput按n行UBcopy，否则packing Gather；N>4使用2×2逆批消元+4阶terminal。raw末padding重复取安全地址不读GM外。有限非零结果直接留存，只在0/NaN/Inf回pivot fallback。|
|312–501|ReverseResidentDet把输入**两轴一起反转**，对应JAJ，det(J)²=1，不改变det符号；pair/宽pair按64→32lane收缩，末6阶Scalar。prepared可跨本次调用的batch复用，fallback后失效，因为恢复函数可占同一裸UB。P72/104、pairIx172032B、wideIx173056B、col174080B均为硬件布局条件。|
|504–710|BlockColumnResidentDet有列块反排实验和LayoutOnly分支，但Run未被当前入口调用；其FactorOnly由FinalResidentFactor真正使用。FactorOnly要求ActualN偶数>32、≤N且差<8，按真实ActualN更新，末4阶guard返回condition；不可把未派发LayoutOnly的errors输出当det行为。|
|712–762|ResidentDet按Capacity/Order/Pitch专化；小容量真实inputN不足时补identity，保持det；Host metadata前缀32B后加载offset，Capacity64用本地规则索引，unaligned先flatDMA再Gather。index/table复用只跨当前调用batch。|
|763–872|Capacity64分64lane前缀和32lane后缀，预测下一pivot Scalar舍入用fp contract(off)保持与Vector步骤更一致；小Capacity用expanded索引直接Gather到广播槽、Div与Mla不经Scalar逐pivot。末6或diag树乘；结果先写后按DetAbnormal(false)决定强制重算。|
|876–938|BlockConfiguration固定64×64×K。AugStep把diagonal suffix和RHS放同一row完成缩放/更新；先处理包含下一pivot的首64项并发布V_S，再处理RHS尾。AugLeftStep对转置left系统使用相应乘子，不是左右照搬同公式。|
|939–1059|AugBlockFactor双列inverse，一次Gather两列；检查abs(a00*c00)<1024，否则先drain推测Gather，再两步Scalar/Vector顺序修复，第二pivot读取**实际Vector结果**避免Scalar融合预测漂移。left对lower系数乘逆、right对pivot rows乘逆，side只保存仍需的RHS，退役diag不存；rounded临时row各128float。|
|1061–1151|FinalResidentFactor Width偶且>32使用FactorOnly，否则64/32lane逐pivot到最后4阶；condition只是末4阶相消guard，不能证明之前每次LU都可靠。Width=0是runtime路径，origin公式要求当前实际被派发的最终block宽度，不能任意对n<32复用。|
|1153–1239|BlockDet Factor<Role0/1/2>分别diag/right/left；一份diag与side共享乘子，Role0留4阶terminal，Role1缩放side，Role2用转置coeff；V_S在下一pivot依赖完成后发。没有全矩阵partial pivot。|
|1240–1331|RunStoredCube仅StorageBlock<64模板使用，当前entry默认StorageBlock64，属于未启用实现。NZ staging→tile存储、paddingidentity；按关键十字顺序优先更新下一diag row/column，L0仍在算时L1在MTE1_MTE2后可预取下一tile；仅行/列变化才加载新L0，Fixpipe atomic加回矩阵。|
|1332–1420|实际RunVector两个AIV/subblock分right/left；panelWords静态≤192KiB，Block64总112384B（裸临时表另在160–176KiB范围）。可remap逻辑W到物理64并paddingidentity；direct first stage直接读原A，其他stage读已更新workspace。奇AIV用TransDataTo5HD转置left/diag；VA列表在Scalar栈，不能当0开销。|
|1421–1460|每stage按own/odd选择Factor/AugBlockFactor；先等上一Cube完整flag1再覆写left/right。新panel就绪SyncAll(AIV)后paired0发Cube。下一stage仅先等关键十字flag2，能与非关键尾tile重叠；lane0在全部旧panel消费者已完成后写结果或NaN marker，累计retryAny。|
|1461–1531|AIC按shape搬初始矩阵、paddingidentity，direct阶段发3；否则mixed barrier。每stage等0，按第1行→第1列→内部顺序分配tile，关键十字全部分配完成即各core发2，所有tile完发1。Matmul.IterateAll(...,1)加到旧matrix；HF32关。不是任意提前下一stage，必须critical集合真的包含其所需两个side。|
|1533–1602|可靠路径DetKernel以column panel装A，保存chosen[512]/bitset，重放前panel pivot与L更新。n×width+14*padded+width+128 float准确覆盖matrix/column/weights/pivot/candidates/active/scalar/scratch/offset；宽repeat按255拆。Recovery无force只重算NaN marker输出，force用于明确指定batch。|
|1603–1687|最大绝对pivot、交换符号，指数分离、factor存GM、精确4B写；fallback一个指定batch或group逐batch恢复。入口与Host派发关系见S3，已与数值体闭合。|

### S16.2 DetAbnormal不能按名字理解为普通“异常检测”

令u为float绝对位。源码41–44精确等价于：

- `finiteExpected=true`：只接受0<u<0x7f800000，即有限非零；TinyResident使用它。
- `finiteExpected=false`：只接受u==0x7f800000，即±Inf。有限值、0和NaN全部调用fallback；Resident/Reverse/Block均使用它。Block另要求末4阶condition。

因此对本PDF规定的有限det输入，只要快路也算出有限值，rank1中n>16那些快路**必然重算**，不是罕见分支；先写的结果会被覆盖。若快路误溢出为Inf则其策略反而可能留下Inf，当前代码没有随后验证真实det是否有限。源码38–40注释正面说明了这个行为，故不是推测笔误，也不据此推断作者违规；应优先对齐其原评测题面、数值域和实际编译提交。不能以公示成绩替这个接受条件作全域正确性证明。

Block的fallback并不要求所有AIV统一返回true：只有group lane0持有retryAny并执行**不含集体同步的单owner DetKernel Recovery**；其他AIV退出是允许的。这不同于rank3全参与者的集体fallback。旧AIC consumers已通过flag1完成，factor工作区复用该group G²，容量n*pad8(n)≤G²；Host groups整除batch使所有轮同数。正确判断“统一分支”要看恢复路径是否还参与barrier，不能设成通用教条。

### S16.3 真正可积累的设计与不能推广的部分

- **P→F**：JAJ双轴反转det不变；小4阶minor末端展开减少同步。其他算子若输出不是置换不变，必须对结果反置换；仅反一轴要补符号。
- **F/分块线代**：下一stage依赖的是关键十字，不是整个trailing矩阵。Cube先更新关键行列，AIV可先分解/解side，直到覆写producer缓冲才等所有尾tile，是真正的look-ahead，不是只增加队列深度。
- **G**：first-stage从不可变原A直接装，L1在L0 load结束后即可复用，prepared索引跨batch复用但fallback后失效；都是活跃期/版本状态管理。
- **数值实现**：fp contract(off/on)、读取Vector-rounded pivot、放大率guard都可能影响误差/速度；其阈值不是一般LU误差证书。ReplicatedTiny、PivotProduct、BlockColumn Run/LayoutOnly、StoredCube当前未启用，不能包装成成绩来源。
- **硬件约束**：native mask、原生repeat、UB地址、Matmul atomic语义、pair crosscore flag仍待目标SDK/生成产物核验；没有新增设备耗时。

## S17 rank2 kernel 1–3548：全部正文，不跳过显式展开

### S17.1 连续行段语义账

| 行段 | 实读内容、有效路由与特殊点 |
|---|---|
|1–132|ColumnDet列主存储，行置换只维护active bitset，不物理交换全部行；选未用row绝对最大pivot，符号用active中比best小的bit数量奇偶（0x6996查4bit parity），这是删除一行的排列符号，不是随机bit技巧。8行DMA+Gather转置；索引首8项Scalar、倍增；subnormal乘2^24再取倒数/拆指数，乘积Finish处理正常/次正规区。实际只n≤32或65–128使用，absolute128与active[2]容量依赖此路由。|
|134–255|MediumDet n33–64：row-major pitch72，输入8对齐时flatDMA+UB行copy，否则一次flatDMA与形状索引构造重叠。偶pivot Gather两行系数、预计算下个对角与倒数，奇pivot免一次Scalar往返。无pivot搜索、对角0直接输出singular0，无fallback；product不分离指数。|
|257–403|CooperativeDet把列分4/8worker，每panel8列把factor、pivot、row元数据打包，双GM slot发布；active/bitset逻辑置换，sink算符号/指数。闲核仍参加所有SyncAll。**当前最终入口n>128直接FastBlockProcess，不调用此类**，不能把它作为大矩阵实测机制。|
|405–520|TinyDet pitch16：一次flatDMA+Gather补列主，成对Scalar读取64bit含2个float，预计算奇pivot；coeff偶槽只读/奇槽Axpy更新，volatile强制预测中间乘法舍入。不搜索主元，0即singular。通用n≤16的非resident子域可走它。|
|523–640|TinyFullwidth把16×16做四次64lane完整更新，factor行复制四份，退役行允许变化，只消费live部分；多算换mask/描述符更短。UB最高factorWide末8576B。加载路径和TinyDet相似，但bases布局重复行地址并将第二系数放后半，不可混用。|
|643–701|Explicit16初始化与TinyDet类布局，n固定16所以只执行直接256float DMA分支；owners=min(batch,cores)，无crosscore的闲核可return。|
|702–1487|显式pivot0..15全部实读：0在703–751，1在752–800，2在801–849，3在850–898，4在899–947，5在948–996，6在997–1045，7在1046–1094，8在1095–1143，9在1144–1192，10在1193–1241，11在1242–1290，12在1291–1339，13在1340–1388，14在1389–1437，15在1438–1486。每体只改literal pivot索引；odd的even分支编译死，最后remaining0删除DMA/更新；do/while(false)让zero guard跳整个展开体。显式展开不自动优于rolled，需生成指令/代码体积实证。|
|1488–1519|Explicit16精确4B输出；KernelLinalgDet dispatch与最终entry分开看，末else Cooperative存在但最终entry已截n>128。FastFence对M_MTE1单独用event4，其余0，源码意图避开TPipe三L0 TBuf初始化占用0/1/2事件，不能统一替换0。|
|1520–1557|FastLoadLocal；DetLeftJobs/Tile将当前十字划成32×128行条和64/128×32列条，选择128列宽的标准是**减少ceil(job/core)波数**，不是单tile越大越好；≤64尾合单tile。OwnerRhsChunk限制≤32。|
|1559–1673|尾pair elimination even/odd两体，提前set V_S、Gather只读和Scalar只读重叠，等V_S后读pivot；系数两槽分离。无zero guard，奇pivot用预测；先64lane、后32lane收缩，数学依赖和退役lane范围都要成立。|
|1675–1712|FastBlockVector 8AIV/slot、4AIC/owner，pitch=ceil128(n)。裸UBmat65536B、loaded8192、tmp24576、index40960/45056、coeff49152+、pair81920/load86016；独立owned raw在98304/102400。用CreateVecIndex/shift/mul/add构造行距40、repeat permutation和pair scale表。|
|1714–1752|Initialize只重置product/bad/currentBatch。PrefetchOwnedRaw在上一phase已SyncAll后装原A；SolveChunk装32×32 diagonal与RHS，privateInput等Cube flag10读历史乘积再相减，随后提前预取本core raw十字另一侧。无效wave清0但不提前抛弃消费者。|
|1753–1812|每两列先试当前最小两个active行的2×2det，若小于0.25*maxEntry或非finite，再两阶段WholeReduceMax选两行；第二行最大化|a00*col1−a01*col0|。该guard有量纲/尺度敏感性，不是一般误差界。对pivot行减一个2幂rowScale，使同一rank2更新能同时产生新变换/单位块，最后按permutation与inverseScale重排RHS。first0才累计det符号和乘积，其他RHS chunk不重复计数。|
|1813–1879|RHS结果Gather按每pair保存的行置换与缩放重复到每列；独立rowDelta不与Add正在覆盖的pairIndices首repeat别名。PrepareOwned每一参与AIC对应两AIV发布11；first超rhs仍由Cube消费11后return。ConsumeOwnedRaw等12，用独立rawProduct而非Solve packing占用区，原A减历史乘积后存当前十字。|
|1881–2009|Rebuild遍历core分配的十字jobs，AIV先读原A，再等10减Cube历史因子乘积；多job每次复用scratch前有11/10握手。FinishTail只within0：even尾走无pivot pair链；odd尾可在|diag|<1或非finite时重新搜索pivot。最后bad写NaN，无可靠全局fallback；product仍直接乘FP32。|
|2010–2062|DetLeftCube各A1/B1/A2/B2/CO1分65536B，HF32关。OwnedProduct将diag与RHS堆在同一NZ plane，row32的偏移是32*8，不是线性32*cols。旧history先算，等最新11再加最新32深度；partial diag/RHS结果flag10先发布，另一raw十字乘积随后flag12，AIV分解可与后者重叠。|
|2064–2111|通用Rebuild历史k<start−32先累加到L0，每job4096float；新变成active的owner不能沿用之前未消费的发布关系，用deferredHistory先等11。最新32深度等11后算；Fixpipe输出core私有8192float temp，写后10通知AIV消费；下一job先等AIV11确认scratch可复用。|
|2113–2140|FastBlockProcess AIV从start0 Prepare，以后≤320且尾>64走owned重建，否则通用Rebuild；≤64 localFinish；AIC从start32对应枚举。仅后续wave再SyncAll防GM因子复用，末输出由自己的DMA与TPipe析构drain。此family源码1512–1513明确是speed/error实验，无精度准入/fallback。|
|2142–2293|BoundPrefetchPadded16仅单matrix/owner0：Init第一时间发input DMA，让地址表生成覆盖等待；n9–14 rolled两pivot体（虽然源码保留其他分支），128bit/64bit索引写需正确对齐。MulsImpl/Brcb后靠随后高层消费者自动sync与显式barrier，不能删注释所指的V依赖。odd尾一次Scalar取值。|
|2295–2457|BoundTerminal15相同数据路，但15阶只做前12pivot，剩3×3公式；消元体0/1死分支均实读，结果是实际矩阵值不是shape查答案。仍无pivot/误差guard。|
|2459–2508|BoundPrefetch16固定16，Init即装mat256float，索引构造与DMA重叠；只允许单batch owner0，由最终entry守卫。|
|2509–3084|固定pivot0..11全部实读：0 2509–2556，1 2557–2604，2 2605–2652，3 2653–2700，4 2701–2748，5 2749–2796，6 2797–2844，7 2845–2892，8 2893–2940，9 2941–2988，10 2989–3036，11 3037–3084。与Explicit16主要差别是原生/Impl接口、无zero guard、部分barrier交消费者自动sync，不是同一文本。|
|3085–3128|只读取最后4×4实值，6个2×2minor公式，乘之前product。未加cancellation guard；直接输出。|
|3130–3292|Compact56只n50/52/54/56、batch≤cores，每owner一次；storage虽按cols72分容量，真正pitch56、flat索引按56生成。两phase<0,56>(0,24)、<24,32>(24,n−4)，coeff按两组64排列，每次Gather固定多读dead rows但仍在本core UB分区，不能误指为GM越界；最终4阶公式。|
|3294–3474|StaticSuffix64 n58/60/62/64的绑定路径，pitch72且full64→suffix32。readable别名扩大到matrix+loaded区域，仅额外Gather的无效系数可以落在dead/owned区，活跃系数仍必须正确。最后直接product，无pivot或finite guard。|
|3475–3548|完整entry模式和后缀wrapper宏见S4：macro仅跳无crosscore路径初始化，不能移植到FastBlockProcess或别题需要workspace/flag初始化的路径。|

### S17.2 可符号推导的契约反例，不需要运行归档代码

取n33、batch1，A=`diag(J2,J2,I29)`，J2=[[0,1],[1,0]]。所有元素0/1均精确FP32，A可逆、det(A)=(-1)²=1，符合本题有限正det范围；没有病态误差问题。Host不命中任何n50+绑定，mode0→KernelLinalgDet→MediumDet。加载保持A00=0，Process在213读diagonal0、220立刻singular，246写0；没有pivot/fallback。故此源码路径**不能覆盖当前PDF所有合法输入**，这属于静态控制流与数学反例，不是设备实测失败，也不说明原计分二进制用了完全相同题面/源码。

同一类限制也影响无pivot tiny与bound family，但需分别按entry子域选择反例，不能拿n33直接指控n16模板。FastBlock有局部pivot但不跨32列块选主元；整矩阵可逆不保证每个leading block可逆。均不应把“省pivot”作为无条件通用优化。

### S17.3 workspace、精度与协议的实际边界

Host每slot预留`pitch²+64*pitch+32768` float；消费者leftOffset=slots*pitch²、vOffset再加slots32pitch、temp再加slots32pitch，每AIC8192float、每slot4AIC，所以temp恰是每slot32768，吻合Host。不能因left/v成员未直接被所有函数引用就删除这64pitch，workspace地址还有相应布局假设。

owned读写部分：先重建diag+lowerRHS并发10，再算另一raw十字并发12；AIV分别消费两结果，rawBase/rawProduct独立，不能把它们与loaded/tmpProduct粗暴合成一个buffer。旧history因子不再写，可以早算；最新32因子必须等11。协议正确性的静态顺序有依据，具体paired模式计数、SyncAll默认模式、TPipe seeded events和GatherImpl接口仍属目标SDK待核，而非本轮已运行验证。

本实现Product/Medium/Tiny绝大部分不分离指数，也没有全矩阵可靠fallback；“FP32实现”不等于题面所有输入都可达到1e-4。局部pivot阈值`|det2|>=.25*maxEntry`并非尺度不变：A乘α时左侧乘α²、右侧乘|α|；不能将触发频率或吞吐外推到任意输入分布。

## S18 三份det的可复用设计对照结论

| 方法 | P/F/G | 哪些源码真正体现 | 使用边界 |
|---|---|---|---|
|逻辑行置换+bitset parity，不搬整行|F：消元/带排列归约|r2 ColumnDet/Cooperative；FastBlock局部pair排列|所有消费者必须使用同一逻辑row，最终符号要算，不能只留max pivot。|
|关键十字优先的right-looking look-ahead|F分块线代/G生产消费|r1实际BlockDet|下一stage只读已完成critical集合，producer缓冲等全部旧tail消费者后才能覆盖。|
|原A−历史乘积，仅重建当前十字|F：left-looking分块线代|r2 FastBlock|减少trailing写不保证减少FLOPs；history重读/每核job数/L0partial容量需权衡；应补可靠数值guard。|
|下一主元依赖先发布、其余行后算|G：缩短依赖关键路径|三份多处Factor/AugStep|必须画出发布前真正完成的数据集合；V_S位置不能随意前移。|
|fixed tiny全lane多算、预测奇pivot、末小块闭式|F小矩阵/G指令开销优化，末公式P|r1/r2/r3不同组合|预测舍入与Vector融合差异、scratch容量、pivot和terminal相消精度不可忽略。|
|分离data ready与复用ack|G异步资源协议|r1 flags2/1；r2 10/11/12；r3 8/9+allhelper|数字没有可迁移意义；必须分别证明数据可读与旧读者结束。|
|编译期容量/尺寸族、裸地址bank布局|G专化/平台实现|三份Host/key与多个模板|需要实际路由覆盖；代码存在、显式展开或拆文件均不是速度证据。|

结论：det全题已完成**9262行首轮逐行文本审读**；已记录各提交控制流、数学、layout、ready/ack和未启用分支。下一阶段若要称某实现“可直接复用”，仍应针对新题契约、SDK和设备补证明/验证；本报告不以公示名次替代这些条件。
