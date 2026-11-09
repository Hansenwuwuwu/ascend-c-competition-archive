# 题2 pinv：源码逐行审读证据账本（已覆盖）

> 面向算法理解与复用的主报告已移至[设计复盘](02_pinv_REVIEW.md)。本文件保留源码行段、数学推导片段、路径与风险证据，作为审计追溯账本；不应把这些行段笔记当成面向读者的最终报告。

题面：[原始PDF](../../problems/02_pinv.pdf)；[题目契约](../SPEC_CONTRACTS.md)。

状态：`complete_first_pass`。2026-11-09；三份提交合计41个文件、19,921物理行已连续逐行完成首轮静态源码审读，包含模板、展开主体、常量与构建文件。无编译、原程序执行或NPU数值/性能复现；全文阅读完成不等于所有浮点、同步与SDK边界已证明。覆盖逐项见 `../coverage/02_pinv.json`。后追加节闭合前文保留的历史未读标记。适用范围分类统一见 `../DESIGN_CLASSIFICATION.md`，不可把带谱间隔的专门admission照搬成任意矩阵通用逆判定。

## contract 题面契约与证据边界

完整阅读 `../specs/02.txt` 三页，并视觉核对原 PDF 第1、2页（留图 `../specs/rendered/02-1.png`、`02-2.png`；第3页提取仅页眉页脚）。float32 输入 `[*,m,n]`、输出 `[*,n,m]`，m,n=1..512，单个 batch 维大小1..64。hermitian 固定 False；atol/rtol 标量。令 τ=max(atol,σmax·rtol)，只保留严格 σ>τ；相等必须丢弃。rtol 未提供且 atol=0 时为 max(m,n)·2^-23；atol>0 时默认为0。逐元素输出误差 atol/rtol 各1e-4。

最重要的额外前提：保留奇异值≥0.01σmax，被丢弃奇异值≤0.1τ。不能离开此谱间隔评判 inverse admission：某些看似宽松的阈值门限，实际上利用了它。禁止调用完整pinv、CPU回退、忽略截断。本审计不以源码名次/Pass推断单机制性能收益，也未验证源码等同计分二进制。

本文件路径缩写：R1=`extracted/pinv/rank1_submission_1060506/code`，R2=`.../rank2_submission_1049211/code`，R3=`.../rank3_submission_1173359/code`。以下路径均相对于归档根，行号为 UTF-8 全文物理行。

## build 三份构建文件

三rank的根 CMakeLists.txt(1-20)、op_host/CMakeLists.txt(1-24)、op_kernel/CMakeLists.txt(1-15)各自SHA完全相同，复用对R3全文的阅读。根1-9 CMake3.16、ASC包和默认910B；11-17 shared包安装到binary dir；19-20加入Host/kernel。Host1-8只glob cpp并codegen；10-18建立tiling和自动生成aclnn库；20-24装包。kernel1-15由当前目录收集核源码、绑定cust_optiling、装包。没有设置特殊优化编译flag；也没有读取/执行构建产物。共9文件记录真实hash复用映射，不将相同内容重复计算为独立算法证据。

## r1-host Rank1 Host、ABI与完整入口路由

[R1 Host](../../extracted/pinv/rank1_submission_1060506/code/op_host/linalg_pinv.cpp#L1)（346行全文已读）：

- 1-24检查指针、float32、rank≥2、m/n范围、每个batch维1..64及uint32乘积溢出。
- 25-48拒绝hermitian；设rows=max、cols=min、初始stride=r8+k8+8；矩形分支直接检查RectLocalFloats≤设备UB、全部并行flag清零、按batch独立核、分配2·batch·cols·RectWidth浮点workspace并选RECT=1，提前返回。此分支不是方阵Gram路径；`pinv_rect.h`尚未读，不能声称矩形数值证书已审。
- 49-86依UB估算双矩阵与scratch；inverseTile从8折半；能localInverse即禁parallel。parallel stride候选128/64对齐+8 bank padding，以resident容量/ceil分片次数选择窄stride；再算非驻留14stride和驻留12stride占用，resident记owned列数，不是bool。
- 87-122在batch整除组数约束下先双驻留、再单驻留候选；singleResident减少一份矩阵，但额外检查owned≥2·active及tile/factorRows存储可重用条件。
- 123-165引入vector-core panelLaunch，重新检查双驻留、factorRows及tile容量；可改为AIV数量而非AIC数量launch。
- 166-200 local容量、属性默认、schedule/workspace、COMPACT/TAIL/WAVE/LOCAL/GROUP生成。COMPACT限制stride≤64且batch不超AIV；local多launch空核由入口guard处理；GROUP含16行的复用下界，不只是groupCount非零。
- 201-244 GROUP可调整launch；producer/worker两套UB公式，满足才PIPELINE；普通producer放不下时尝试约3n/8且16对齐WINDOW，用`local`字段复载window宽度，WINDOW=1同时PIPELINE=1。
- 245-268 WINDOW+32整除等条件选CUBE；先INT8×INT8→INT32五plane残差matmul，再FP32 correction matmul，扩展tiling保留base；workspace基于CubeOffset和CubeStorage。API成功仅为源码条件，未执行SDK验证。
- 269-285非WINDOW管线还可用两个paired producer，检查peer metadata容量后设置CUBE但cube tiling清零，此CUBE是编排key，不能一律解读为真正Cube矩阵乘。
- 286-315 DSL rollout重写workspace：COMPACT和≤64LOCAL彻底0 workspace；peer/window status另配；n200、batch·13可用时改stride456、launch=ceil(13batch/2)、WAVE=1，另64..208整8的peer将TAIL复用为half selector。315输出十个模板参数。
- 316-346返回、InferShape交换最后两维，dtype强制float；注册三个默认属性、LaunchWithZeroEleOutputTensors和910B。

`op_kernel/linalg_pinv_tiling.h`1-20全文：13个uint32+2float恰60bytes；Cube扩展首成员base及两份TCubeTiling；static_assert约束standard_layout/offset/不重叠。WINDOW复用local的注释6行是ABI的重要阶段语义。

`tiling_key_linalg_pinv.h`1-26全文：float32+9bool；16-25只选择10组合法组合（基础/WAVE、LOCAL含TAIL、COMPACT含TAIL、GROUP、WINDOW、WINDOW+CUBE、peer+CUBE、RECT、peer+WAVE）。kernel type既有AIV_ONLY、MIX_AIV_1_0、MIX_AIC_1_2。必须联合Host判定，不能假设9bool全部512组合会编译。

## r1-contract Rank1 常量、地址契约与浮点片段

[R1 contract](../../extracted/pinv/rank1_submission_1060506/code/op_kernel/pinv_contract.h#L1)1-187全文：1-17 key位256..32768及host/device inline；18-48 split4097、u=2^-24、wide/triple误差尺度、power缩放上限2^127、direct gate0.81、谱间隔gate81、经验condition limit、Jacobi40/rect120轮及聚合slack。经验常数不是严格浮点误差定理。

50-68所有metadata单位是float而非byte：8192float总区；pivot256、Jacobi512、group flags1024、prepared1032、status2048、peer3072、Cube control512、5plane、radix256。70-90矩阵pair=2batch·cols·stride，metadata按batch块，Cube起点128float对齐；Cube存储=2048+110n² bytes（两份5n² int8与25n² int32）。91-102 RectAxis是8到512的2次幂bucket，RectWidth两轴相加，workspace=2batch·cols·width，UB=20width+32+8rows+3axis(cols)。103-109 static_assert元数据阶段界限；118-123 DSL work取rect/peer较大且128float对齐。134-149 WindowSupported要求方阵n≤256且n%128=0、物理AIV=2blocks、group2..48、window16对齐、cores·n≤16384、cores16≤1024、cores8+32≤512、producer/worker≤49088floats及输出slice≤2048。150-161 window work=基址+30.5n²+cores·n+16cores+64batch，且不少于13batch·n²；status=512batch+128。165-187仅Host编译的BlockGridCompatible/BlockLaunch重平衡tile数且保留容量与WindowSupported条件。

`pinv_wide.h`1-39全文（代码片段而非独立头）：Wide=(h,l)，Plus以TwoSum类误差修复、Minus取负、Times用4097分裂主项并纳入h·l、l·h，略去l·l，Divide以一次高位商+残差修正；Entry/SaveEntry双tensor访问。`pinv_vector_precision.h`1-31全文：以8份calc scratch做双float消元`(x+y)-factor*(u+v)`，先分裂product后双float减法；Normalize先近似商再用消元残差作一次correction。需无溢出/非正规数影响、禁止不受控重结合，volatile仅约束源码求值，不代替目标编译器验证。消费者尚未全文读，不能证明此片段对所有key可达。

## r1-entry Rank1 kernel入口连续语义段

[R1 entry](../../extracted/pinv/rank1_submission_1060506/code/op_kernel/linalg_pinv.cpp#L1)1-252全文：

- 1-19由TILING_KEY_VAR位条件设置K_MAX_SHAPE_DIM=0后引入native诸头，重复include由头guard处理。此宏改变模板编译边界，实际SDK契约未核。
- 20-40 CompactQuickLight静态pitch=2PAD+8、matrix/low/u/prod/reduce/gather/norms单buf连续分配；41-65加载、maxabs范围[0.1,2^127)、向上2次幂缩放、Frobenius²、增广I、行offset与available位掩码。
- 66-100无物理行交换的Gauss-Jordan：masked max挑未用pivot、order恢复逻辑行、Brcb+MulAddDst外积、逆范数与cut²gate0.81；还原power后分行拷贝逻辑顺序，MTE3双向结束事件。101-111包装销毁TPipe。
- 113-145 native路由：RECT独立RectPinv；CUBE+WINDOW AIC运行PinvCubeService；非WINDOW CUBE的AIC提前return；guard负责Cube生命周期；WINDOW返回每batch四低位状态，未全通过清local后继续；普通pipeline或精确key peer full可能提前返回。
- 146-171 native resident guard、16阶QuickLight、COMPACT Pinv<TAIL,16>或通用、LOCAL LocalPinv、其余StablePinv；WINDOW的prepared状态传递给ProcessPrepared而非无条件重新开始。
- 173-191 DSL头与主入口；WINDOW+CUBE声明mixed AIC1:2，其余MIX_AIV1:0；COMPACT先直接读tiling batch字段挡空核，避免无用tilingcopy。
- 192-212 peer+WAVE直接PvStagePeerTile；COMPACT/≤64LOCAL实际由PvStageCompact(TAIL)/Aligned或PvStageLocal承担并return，所以native同名快路在这些Host组合下不是主执行路径。
- 213-233 WindowSupported真时PvStageWindow返回值+逐batch flags[b·512+16]读取，显式DCCI CACHELINE_OUT；全部admitted即return，否则SyncAll<false>作为continuation rendezvous后再走native。需结合未读window头判断AIC是否同等退出，当前不能声称已证死锁/无死锁。
- 234-248 peer管线TAIL决定HalfPipeline/PeerPipeline并return，其余调用native；250-252末尾把后续`cce_kernel`宏加aligned16384。宏在文件尾定义，其对编译器后续生成wrapper是否生效未核，不能据名字声称kernel已16KiB对齐。

## r2-host Rank2 Host/tiling/key

[R2 Host](../../extracted/pinv/rank2_submission_1049211/code/op_host/linalg_pinv.cpp#L1)1-108全文：1-18rank/m/n检查和core/dtype；19-25 Cube要求float方阵65..256、batch≤AIC/4、AIV=2AIC且≤64；tiny为9..16。26-37构建512uint32静态payload：每行32word，前8word打包16个half输入byte offset（越界指向m²零哨兵），接8word打包转置offset，后16word单位阵float位串。offset是4整倍数≤2044，half精确可表示，编码无需舍入近似；不是预存输入数值/答案。38-50设置t，stride=r8+k8，小方阵column owner≤8、owner宽≤8；parallel=2是smallTeam而非普通并行。51-67默认属性、hermitian拒绝、路径0/1/4/5/6优先级、launch与whole-launch schedule；68-77按resident/parallel分配workspace，Cube至少一份RowBlock stride，非零user再附library workspace。79-108正确形状交换与dtype、float/BF16注册（BF16超出题面，不据其宣称题目要求BF16）。

`linalg_pinv_tiling.h`1-35：tiny_minblocks8、capacity16、min9；column cap64/shards8/width8；Cube4core/batch、panel64、cap256。RowBlock workspace=5·256²+5·64²+256·64+512+8192=373248 float=1492992 bytes/batch。tiny padding6word+packed512word配合40byte base恰2112byte；MaxTiling按较大结构/4装字数组。`tiling_key...h`1-34：path0 AIV_ONLY，1 MIX_AIV1:0，4 MIX_AIC1:2，仅float的5 tiny AIV_ONLY和6 column MIX_AIV1:0；PINV_DIRECT_BENCH可在源中禁入口但CMake没定义它。

## r2-kernel Rank2 单文件全文语义审读

[R2 kernel](../../extracted/pinv/rank2_submission_1049211/code/op_kernel/linalg_pinv.cpp#L1)1-1968全文，分段读取无截断：

- 1-34 includes、directbench条件、Pinv模板serialOnly/mixedCube，CoreCount在混合核时2·GetBlockNum；13份UB buf、DP及dotWidth16/4096tile/128tail，普通FP32 dot。
- 35-92Kahan类AccumVector、TwoSum式DoubleVector、AccumulateExactProduct分裂scalar位mask0xfffff000和vector4097乘法残差，BF16左移16解码及输出CastRINT。乘积补偿只是提升精度，不意味着IEEE double。
- 93-139FullRank按现有Jacobi范数判严格截断；两次Newton仅在FullRank/known mask，通过GM残差与SyncAll两阶段，不将rank-deficient截断伪逆强行逆化。
- 140-149UB/GM Column存取；150-278小逆模板K/noPivot/prepareOnly，K15连续225读入并按i-floor(i/16)扩行，K64额外8列padding；pivot mask和logical row map，标准增广消元；逆范数阈值<0.5，noPivot/prepare xF²≤1e6。K50从原始Gather读signed pivot，不能用绝对max当倒数。
- 279-409 xF²>1e6时两次补偿残差rF²<1/16再修正；否则批量残差与一次X+XR。15/16把256 dots分为128+128避免repeat溢出；K15压缩index恒等式floor(i/15)=(i+1+floor(i/16))/16只对0≤i<225使用。最终只存逻辑n²。
- 410-477 RunSmallTeam复制消元、按lane分残差/输出，所有核即使reject也到同一个norm barrier，bad/rF²集体决策，lane0 fallback。478-551 RunBatch通用串行one-sided Jacobi，B=A或Aᵀ，合并列[B;V]，16sweeps/changed早停、逐列严格norm²>τ²、Kahan重建。
- 552-633 PreciseRowLarge以16个256长dot并行：高低分裂、TwoSum二叉reduce到8项、Gather compact后3层xor butterfly；result按residual或correction两种消费。indices生产在638-642，4096主索引+4·128尾索引，匹配smallIndexBuf容量。
- 634-712 RefineFastInverse最大4步activeMask、completedMask；每batch固定数组容量4与调用条件batch≤4一致；先测rF²<1/16，inverseBound=||X||/(1-rho)，检验bound·rho²的Newton误差，200/256允许5e-5，其余1e-5，finish前仍执行修正并写出。713-740FinalizeSvd先判full mask再复用norm区，截断路径直接存，不做逆Newton；full路径修正、剩余两步兜底。
- 741-860 ParallelInverse按batch交错lane分行，每行固定pivotForRow[32]，fast条件至少8lane且k≤256保证≤32；每pivot两次全核barrier交换pivot；先Frobenius门，必要时||AAᵀ||∞谱范数上界+FP32slack，再RefineFastInverse，不用4次幂迭代估计冒充上界。
- 861-1048 PacketDot和block Jacobi，k256固定L512，width4/8/16；block ring、local ring与跨block仅Q循环；当前输入dot每4轮刷新norm，向量mask组合保障不旋转无效pair，compare完整64lane但只消费count≤16；最多24sweeps，全局max correlation²<1e-14早停。没有自动长表未读：这里indices是明确循环生产的状态表。
- 1049-1133按未完成batch数选width；RunParallel初始读输入/identity、round-robin两两无冲突、普通24sweep；norm记录每col8float；全局strict阈值、矩形直接输出、方阵Finalize/Refine。1135-1173按路径实际计算UB buf容量、packet dot附加scratch/indices容量、smallTeam分支；preparedNorm2只在调用方保证≤4时填数组。
- 1175-1290 ResidentTinyInverse，K16/DP16/L32、读取Host packed，half cast为byte gather索引；小n补为diag(A,I)，无pivot rank-one消元，(I-AX)ᵀ残差向量化；三Frobenius及row/column范数联合证书，必要才correction；GatherMask压掉padding。1291-1326DenseProduct/Polish对截断结果先(XA)ᵀX(AX)ᵀ空间修复再X+X(I-AX)，trace≈16则保留full-rankfallback；此修复固定点可推导但有限次误差未实测。
- 1328-1361 arena7920float=31680byte，无TPipe快路，fallback才动态TPipe并Polish；逐batch失败break后通用fallback可重做已写batch。1363-1534 ColumnSmallChecked每核复制左矩阵、只附8列逆，CP使行stride奇数8块；权重区跨512byte边界加DP以改善冲突的编排，不可归因收益；无pivot逆后集体残差证书，一次全局交换候选，必要X+XR；失败返回不输出，然后每batch leader独立fallback。
- 1536-1550 capacity24..64按ceil(n/8)派发，不局限50/64；1552-1560 RowBlock workspace W,T,D,A,X五plane、G/GI两个64²、H256·64、STATS512、PACK12288half字节换算与IDENTITY8192float，STRIDE static_assert锁定373248float。
- 1561-1605 Cube Tile直接管理L1/L0A/L0B/L0C，ND2NZ、LoadData2D/3D transpose、Mmad、Fixpipe、可AtomicAdd，独立MTE2→MTE1→M→FIX事件；L1=262144、L0A/B各65536、L0C131072bytes，HF32与transHF32关闭。AddIdentity仅对角块half精确I累加，不能把此half路径说成所有输入half近似。
- 1606-1692 HalfProduct、HalfStore、LoadPair64、NeumannPair64、Newton64：64² FP16迭代、r平方复用、最后从原G重算纠正漂移、高低half块重构，Product依据实际active tile数量择64/128输出tile。内部K/m/n均16或8对齐，逻辑n由N=ceil(n/16)补diag I后覆盖。
- 1693-1766 Vector的64块无pivotInvert、UB160KiB、8行identity生产；按lane加载CAP256行pitch、maxabs→2幂scale、对逻辑tail补I、W/A/T/X分区，normA为缩放矩阵Frobenius。
- 1767-1802 NewtonPrepare拆Ghi/Glo half，alpha=1.8/||G||∞；NewtonGuard对当前defect逐行abs和，失败atomic decision；精确逆fallback由lane0做16/32/64 capacity。1803-1854Measure/Update/Decide：1.001²和1.01²slack，rho²<1/64，cut²·x²<62.015625=(9·7/8)²，delta·rho/(1-rho)预算，output最后乘-scale恢复符号尺度。
- 1856-1924混合状态机：Vector Load/Normalize，Cube行Gram、Newton种子或exact小逆，逐panel dual行消元与T变换，最后DᵀT得negative inverse；最大4次残差与更新、DCCI读取每batch accepted flag，所有batch共用barrier；无论accepted是否全过都先StoreOutput（未通过随后fallback覆盖）。1926-1940入口user=workspace假定generated wrapper已移到user；与其它rank GetUserWorkspace不同，须SDK核验。1942-1968Tiny ICachePreLoad(5)、aligned4096、注册MaxTiling；path6/5/4/else真实选择。

### r2-math Rank2数学与数值前提

单边Jacobi对B=A(m≥n)或Aᵀ(m<n)进行B←BR、V←VR，最终列bj≈σj uj，伪逆为Σ retained vj bjᵀ/||bj||²；wide再转置。附加identity区同步旋转就是V，不必显式形成U。一次sweep成本O(qp²+p³)，workspace O(batch·p(q+p))；固定16/24sweeps不含最终全精度收敛证明。

逆Newton：R=I-AX，Xnew=X+XR，故I-AXnew=R²（实数）；误差Xnew-A^-1=-A^-1R²，≤||X||·rho²/(1-rho)。源码把row-dot高低位补偿用于计算R，避免FP32残差精度地板；FP16 Neumann仅种子，后验float证书和fallback承担安全边界。行block算法维护negative dual D_i W_iᵀ≈-I，后续W_j+=W_j D_iᵀ W_i消去先前子空间，T累计左变换；末尾negative inverse=-DᵀT，再乘-scale恢复。非满秩/病态panel可产生Inf/NaN，必须由后验判定拒绝；未运行实际浮点反例搜索。

若rho<1/8，则σmin(A)≥(1-rho)/||X||。阈值上界用||A||F；若cut·||X||<9(1-rho)，得到σmin>cut/9≥τ/9，和任何被丢弃σ≤τ/10矛盾，所以允许走完整inverse。这解释62.015625，而不是“把阈值忽略”。证明不覆盖norm误差、溢出、FTZ和近似division，源码slack的充分性未独立证明。

## r3-host Rank3 Host与key

[R3 Host](../../extracted/pinv/rank3_submission_1173359/code/op_host/linalg_pinv.cpp#L1)1-98全文：1-22 p=min、q=max、pp/qp按8pad、cached=p(qp+pp)≤32768floats及默认rtol；没有rank/shape/dtype/hermitian健壮性检查。23-39 n199、n256特化、large minGroup满足cols(pp+8)≤19000、batch整数组和group cap；256强制偶group并按AIC/2 launch。40-46 floatCube tiling，singleM32/48/64、N/K256、split128/64。47-62小方阵4核、WIDTH编码+64、large256 WIDTH2、199/200 WIDTH3、extended tiling和old/fast workspace并集再加Gram/残差空间。67-74 InferShape/InferDataType仅返回success，不填任何输出；76-98注册float/BF16但无实际output推断、atol/hermitian无显式值默认，未独立核SDK如何填缺省。

`linalg_pinv_tiling.h`1-10是13word base+TCubeTiling，未加R1那样ABI断言；`tiling_key...h`1-14列表WIDTH={0,1,2,3,15,16,50,64,114,128}，不同LARGE/CUBE限制见末尾路由。**已证范围缺口**：Host令任意方阵p≤64的WIDTH=p（或p+64），但key表不含题面合法的4、17、32等值；没有默认WIDTH=0重路由。这是Host/key静态不一致，不等同已经复现框架报错。

## r3-kernel Rank3 单文件全文语义审读

[R3 kernel](../../extracted/pinv/rank3_submission_1173359/code/op_kernel/linalg_pinv.cpp#L1)1-1642全文：

- 1-20FP32Cube alias、event0成对LocalSync、8×256/16×256转置到5HD，8行版本dummy destination放scratch避免覆盖；21-76Pinv普通dot、scaled Hyp、最大绝对值分64+tail、矩形按A或Aᵀ读，float单元素DMA每行32byte再Gather（offset=i32），BF16每行16half再Cast/Gather（offset=i64）。Read/WriteCol根据cached选UB或GM，Rotate更新B/V，两种dtype输出。
- 77-103 FastInverse≤64：增广宽2pp+8、部分pivot、pivot²<anorm·1e-10拒绝、invF证书0.25后输出。104-149Householder双对角化与隐式shift QR（每尾最多80步），右向量V保留；左旋转不积累U，135的左取消步骤也不旋转U，因后续重算AV。150-180用原始输入重建B=AV，再一遍Jacobi polish；Assemble的阈值仍取QR d，norm逆权取重建列，意味着d与polish后的norm可能偏离，未证对题面全谱收敛。
- 181-197 UB194560bytes=48640float，cached前32768floats容两矩阵；scratch38400..46080，BF16分块转换mem36864，raw32768起；batch独立、快逆拒绝后SVD。198-238TinyPinv同样大小静态UB，单两级Sum、norm gate从0.25放到25。
- 239-300ColumnTinyPinv每核复制A并附cols=ceil(N/G)逆列；N>16多核ld136，N50额外16padding；N15连续225输入Gather扩行且mask清最后padding；N≤16限制norm范围，较大N按float指数2幂缩放；无物理行换、active bitmask、permutation、四块外积。301-341PrepareOutput用(permutation+offset)Gather重排，N%8用GatherMask去尾；G1独自判断，G>1先发布每核8float norm、SyncAll、读同组证书，所有成员共同拒绝；G1失败可能已写前batch但fallback重做。
- 342-349noinline/aligned128包装；350-406ParallelInverse多核按列分[ A | I ]，ld=pp+8、owner panel交换、mask清已用row，PowerNorm四次固定初始向量交替A/Aᵀ乘得到估计，不是有保证上界。407-437CubeRefine/CubeCorrection用当前workspace的inv/savedA/residual/corr算X+(I-XA)X并转置输出，偶part真正matmul，row32/48/64分段覆盖256。
- 438-576RefineInverse：特化N>0读取savedA，4条独立acc2048间距降低依赖后合并；generic/N199以wp=2次幂补零和batchDots32/16批量dot；residual以8列interleave，out-of-rangecf清零；FP32最终Gather+stridedDMA直接写，BF16先GMcorrected再copy。577-596 Init给交换双slot、inverse每batch pp²、state和scratch指定offset，cols·ld≤19000与panel/rm起20000分离，red47872到48640剩768float。
- 597-724Run：A列block读取转置，savedA写入，求asq、identity、逐panelpivot记录row/inv，双slot用完Sync后覆盖；所有核写asq/isq/bad，再按s32..1聚合。gate0.25/25、256 sensitive阈值、必要PowerNorm估计与Refine；failed跨batch累计，末尾再global any决定fallback，部分early-return依赖所有group共享决策。未实测多group退出/可运行性。
- 725-782Independent199Inverse复制辅助部分，但Update用ld208、最后8lane，低/高广播plane13376/33408以stride80实现交错，不是任意N泛化。783-857有RefineInverse实现但Run实际不调用（1001-1007只PowerNorm），不可把有函数等同启用精化。
- 858-963199布局：FIXP时重写尺寸，singleCols≥16；只保存A列，owner完成panel后用消元向量替换原A并设pivot1，以后来自所有panel的更新构造隐式inverse，receiver先更新下一pivot所在64lane块、提早Gather下一coeff，再完成其余块，两个broadcastping-pong。
- 964-1037 Gather permutation、乘invs、按perm目标列散写inverse；四acc范数，gate25/sensitive、4步PowerNorm；199 FIXP读取40000floats再Gather输出，200 runtime读取p·pp；没有启动refinement。1038-1112FactorOwnerFixed256/ReceiverFixed256直接UB地址0及raw `vand/vcmax/vgather/vbrcb/vmla/set_vector_mask`，固定stride264，手动mask恢复；属平台绑定高风险，需要官方内建契约/编译版本，不因裸指针断言非法。
- 1114-1193DeferredInverse256结构与相同辅助函数；RefineInverse只设置needsRefinement/outputReady，不当场写输出。1194-1297A/savedA、四acc、panel消元、用前述raw256 owner/receiver，消元面板双slot同步。1298-1361inverse映射和范数，gate与global reject一致，needs标记后由外层Cube执行；未need时自己Gather输出。
- 1363-1371SpecializedInverse<199>走FIXP199，否则Parallel；WideInPlaceInverse名虽199但FIXP=false用于p200。1372-1416ParallelJacobi：矩形B，round pair独占列，普通dot与PreciseDot选择，tiny相对阈值与2e-7归一相关阈值，tau>1e8改0.5/tau防平方溢出。
- 1417-1454PreciseDot用4097拆乘+TwoSum式二叉归约+volatile最后8项，Rebuild原A·V以Kahan累加、额外精确dot一sweep、更新最大norm；非FP64，需无split溢出。1455-1490output严格norm>τ²，Refine对当前y算I-yB，再乘y并存work；对截断结果实数精确下仍是固定点，但不是满矩阵inverse误差定理。
- 1491-1520UB显式布局、workspace all/yall/ns/state，FormGram缓存原B后算BᵀB，GramOutput按G^-1Bᵀ重建再Refine；1521-1546最大20sweep round-robin，per-corechanged→block0any→同步早退，Rebuild→输出→Refine。
- 1547-1580p1专路：标量严格abs>τ，向量||A||²>atol²且rtol<1才保留，结果A/||A||²；UB2048float完整承载q≤512。1581-1599fallback和deferred/Cube包装，AIC关闭HF32，AIV先判种子再注册matmul，okay&&need才CubeRefine，失败只AIV Jacobi。
- 1600-1627入口：LARGE WIDTH3→199/200 implicitinverse→Jacobi，WIDTH2→256deferredCube→Jacobi；一般LARGE方阵specialized，矩形先Gram尝试，gt.atol²/rtol²且rtol floor1e-6、失败才Jacobi。Gram把condition数平方，不能在FP32当稳定SVD等价；门限floor不是题面原阈值，仅候选admission需由fallback保障。
- 1628-1642小路：p1 vector；WIDTH>64 order=WIDTH-64四核ColumnTiny，空launch核参与一次SyncAll后return、reject仅每组part0单矩阵cachedfallback；其余float WIDTH≤16 alignedwrapper或smallwrapper，BF16Tiny；最后Pinv缓存SVD。Host/key可达性缺口见r3-host。

## math-admission 谱间隔、浮点与资源结论

对精确满秩X=A^-1，设c=max(atol,||A||F rtol)≥τ。若存在被丢弃奇异值，||X||F≥1/σmin≥10/τ，从而c²||X||F²≥100；因此R3的门限25（R1的81）在**精确inverse和精确范数前提下**可排除弃值，不需要SVD。若X只是近似，则必须引入rho=||I-AX||<1得到σmin≥(1-rho)/||X||；R2显式证书较完整，R3多个路径只检查pivot与范数、甚至PowerNorm只有4步估计，尚不足以证明上述前提。小norm/极大norm、NaN/Inf及FTZ都是待硬件验证边界。

Newton左右方向必须匹配：R2用R=I-AX、X+XR；R3Cube按内存转置视图计算I-XA再左乘X。实数残差平方与浮点kernel逐步等价是不同命题；分裂乘法、Kahan、4acc树形累加、Cube FP32配置均改变舍入顺序。

R3主UB约190KiB，最大显式scratch始47872、范数全核记录group·groups·8，若physical vector cores超过96，则768float尾部可能不足，Host并未像R2限制cores≤64；当前910B常见core配置不能替代通用证明。此处是硬件容量前提风险，不是已复现越界。所有DMA blockLen字节、UB步长32byte块、Gather offset字节必须分别核对；不能见pp=8pad即判所有操作安全。R2的RowBlock直接管理L1/L0大小已记录，R1长头内的资源/退出协议未审完。

## compare 可复用方法与不能照抄的条件

| 机制 | R1 | R2 | R3 | 可迁移前提 |
|---|---|---|---|---|
| Host形状覆盖 |泛化范围检查、UB条件拒绝矩形|泛化capacity dispatch|小WIDTH未列值缺口|必须枚举Host产生key与编译集合|
| 小逆 |DSL驻留/高低位候选，长头未读|16cap packed几何payload、后验残差|exactN列复制/分核逆|谱间隔、rangeguard、fallback和outputownership|
| 大逆 |producer/worker/window/peer+CUBE，头未读|64行dual、half Neumann种子+float证书|199implicit/256rawpanel+Cube精化|先证明数学门限再审同步，计分不能给trick归因|
| 截断 |contract40/120sweep常数，消费者待审|16/24sweep单边Jacobi|bidiagonalQR/20sweepJacobi|谱间隔不等于有限轮已收敛|
| 精度 |Wide/向量双float、5digitINT8残差契约|Dekker/TwoSum、rhosquared证书|补偿dot/Kahan、4acc、敏感拒绝|编译器重结合/FTZ/Cube模式需验证|

可复用的是“便宜候选→当前输入后验cert→同call fallback”、静态几何payload、容量bucket、明确单写者/phasealias；不能照抄的是测试shape常量、裸UB地址、固定core数量、经验阈值、未经官方核验的内部API。没有paired同环境硬件实验，不能说某名次由aligned4096、零workspace或某一内建导致。

## r1-rect 追加：Rank1矩形路径完整审读

[R1 rect](../../extracted/pinv/rank1_submission_1060506/code/op_kernel/pinv_rect.h#L1)1-404已逐行全文读完，修正前文“消费者尚未读”的局部状态：矩形消费者现已读，方阵长头仍未读。

- 1-42 class与Wide片段引入、RootPair用平方残差修正sqrt、Load/Store同时建立V/S与MTE2/MTE3事件，不用跨核barrier，因为一核一batch。43-58Original按A或Aᵀ提列、head=2次幂bucket补0，tall逐元素DMA到gather8stride后scalar写，wide连续读；只head部分乘1/power，basis初始0。
- 59-110PairAdd/Scale双float（Scale纳入factor.l·b低低项，较通用Wide::Times多一项）、Dot 4种high/low交叉积，TwoSum二叉树到8再scalarWide求和；count=head/basis都是2次幂≥8，循环不会丢奇数tail。
- 111-197后验独立三分量certH/L/T：CertTwoSum、三层CertAdd、Dekker product与scalarproduct、scalar三分量累积、CertDot所有4种crossproduct和pairwise归约，最后压成Wide；kTripleRoundoff=2^-72是模型常数，不是证明所有target环境确有72bit有效精度。
- 198-240Solve重新从原A初始化[B;I]高低GMplane；cheap40轮orth1e-6只存FP32旋转结果且low清0，precise120轮orth2e-13保留双floatrotation；两者dot/theta计算都用Wide，large tau走0.5/tau。达到changed0可早停，轮满不当作成功证据，交给Certify。
- 241-289Certify逐列sigma精确dot、严格σ>cut；测V全Gram、保留U Gram、弃值子块Gram∞上界、B-AV和Aᵀu-σv双残差；retained通过uint32tensor，basis=cols bucket，padding不参与logicalrank。
- 290-329按不同op计数生成dot/basis/norm/product/outputgamma，etaU/V<0.25才计算极分解距离d=eta/(1+sqrt(1-eta))；norm上界纠正与残差delta推最大奇异值上下界，再要求保留s>thresholdHigh、弃值d≤thresholdLow、s>d。lipschitz=1/[s(s-d)]，输出误差含delta、du/dv、outputgamma，除power还原。rank0单独要求弃值+delta≤thresholdLow，给零输出误差。
- 330-356Emit用V·Bᵀ/σ²双float累加，最后一次high+low到float；逐元素预算把未知真值改写为当前value的保守预算(1e-4+1e-4|value|)/(1+1e-4)，加入输出舍入u/(1-u)|value|。**Emit先写完整输出再返回valid**，失败不是未写，而是下轮重算覆盖。
- 358-370连续UB分配：8width输入/输出+8widthcalc+4widthcert=20width；red32、gather8rows、sigma/sigmaLow/retained各basis，恰Host RectLocalFloats；GMhigh/low各cols·width。
- 371-404batch独立；peakmax→2次幂power，零矩阵直接输出0；cheapSolve→cert/emit失败则从原A重新preciseSolve→cert/emit；第二次仍不通过时Div(0,0)显式NaN填满输出。这里是**已证拒绝时显式非有限输出机制**，不声称合法输入一定触发，也不能把它描述为无条件正确兜底。

数学证书依赖注释241的tests/README.md，但该文件不在本提交inventory；已读实现给出具体结构，无法据缺少的证明宣布gamma/slack是形式化充分界。极分解距离公式对||QᵀQ-I||<1有意义，保留/弃值谱分离下pinv扰动随1/[s(s-d)]增长；这解释矩形不能照搬满秩inverse residual，而需左右子空间与截断证书。

## r1-cube 追加：Rank1五digit INT8残差服务完整审读

[R1 cube residual](../../extracted/pinv/rank1_submission_1060506/code/op_kernel/pinv_cube_residual.h#L1)1-223全文：

- 1-13仅TILING_KEY_VAR的CUBE位引matmul，CUBE+WINDOW才编译service主体；CubeArea在metadata之后128float对齐。
- 14-60 AIC永久服务循环等跨核flag8，DCCI控制word；cmd0发flag10并退出，cmd1运行5n×5n、K=n的INT8精确乘，其他cmd读取batch并算float X·residual（SetHF32false），每次销毁TPipe后flag10应答。控制word0/1由AIV写，结果矩阵共享相同product区，不同command分阶段复用。
- 61-81PinvCubeCorrect只core0写cmd2+batch并DCCI，所有AIV Sync→send8/wait10→Sync再读自己的stridedrow；PinvCubeStop同样发cmd0，AIC应答退出。202-214RAII guard在CUBE&&WINDOW所有native返回/正常析构时调用Stop，防服务留在wait；但DSL最外层绕过native时是否另有停止协议，仍需未读DSLwindow消费者确认。
- 82-104PeakScale统计写controlarea[36+8core]、读[32]后mask选第4lane，chunks汇总选择max；peak范围(0,1e20)外用fallback，否则scale为≥4peak的2次幂。此函数对count0的reduce行为未在本头防护，调用方owned须正数或明确API允许；消费者未审完。
- 105-142Quantize把value·256/scale逐层round int32、残差×256，5digit临时halfplane容纳整数。低位+128转-128并向上一digit进位，保证signed int8[-128,127]；RIGHT与LEFT分别拼成n×5n与5n×n，dest/step按core交错独占行，返回最终残差平方和而不是量化前后随意epsilon。
- 144-176BuildResidual用固定scale8量化归一化A，再读当前inverse并动态inverseScale，发布两类量化残差统计；cmd1→8/10握手→汇总所有peer的localErrorA/X。
- 177-200逐25个plane积读INT32，castfloat后weight=8·inverseScale·256^-2·256^-(i+j)，负加到high/low残差，用TwoSum保留加法error；不是把FP32输入简单改INT8 matmul。这里high初始I及cache归一条件由调用方提供，尚待消费者全文闭环。
- 201-223条件编译结束、RAII guard与PinvVectorBlockCount：CUBE时AIV逻辑blocks=2GetBlockNum，AIC service始终使用原GetBlockNum；不能把两种corecount混用。

量化推导：x≈scale Σ(l=0..4)digit_l·256^-(l+1)，第五层后的`values`对应误差scale·256^-6·values；这就是contract kCubeResidualUnit=256^-6。25项矩阵乘还原全部digit交叉项，绝非只保留五项。n≤512时单dot≤512·128²=8388608<2^24，INT32不溢出且此范围内int→float精确（仅对digit确实在范围、Mmad无额外模式截断前提）；最终浮点加权仍有舍入。控制区512float和peer统计32+8cores需cores≤60，上层WindowSupported确有限制，但所有native调用者还要继续核。不能把这一静态整数界当整体误差或速度证据。

## r1-native-followup 原生 inverse / producer / refinement 补读（2026-11-09）

本节为四份头文件的连续全文首轮审读，覆盖 local 1–545、pipeline 1–408、window_pipeline 1–321、refinement 1–326；不是编译、同步或数值正确性证明。后文旧 pending 清单中的这四项现已读完，其余未读项仍保留。

### pinv_local.h：局部宽精度候选与 Jacobi 回退

- 1–63：LOCAL 且非 WAVE 的类、UB/GM 行访问、MTE2→V/S 与 V/S→MTE3 事件，以及普通 FP32 点积；行跨度和逻辑长度分开，不用跨核 barrier 代替本核 DMA 依赖。
- 64–127：AccurateFull 从原 A 建 Gram，使用 high/low 计算其逆并重构伪逆；它只在后续 `rows<=32 && kept==cols && scale<0.001` 的满列秩分支被调用，不能推广成秩亏通用公式。对 tall 是 `(AᵀA)⁻¹Aᵀ`，对 wide 是 `Aᵀ(AAᵀ)⁻¹`；Gram 平方条件数仍是数值边界。
- 128–212：BatchEliminate、LocalInverseBody 用 Dekker/TwoSum 风格双 float 消元，保留主要交叉乘积但省略 low×low；partial pivot 同时交换 high/low 行，输出 high+low。范数准入门检查近似 inverse，不能仅由 `inverseTotal*cutoff²` 小于门限推出所有输入都通过契约；需残差与浮点误差前提。
- 213–283：32<n<=96 legacy 候选，按 pivot 做行归一化和向量 rank-one 更新，最后把 inverse 矩阵范数与 cutoff 对比；失败 return false，由调用者继续下一算法而不是留下最终错误输出。
- 284–400：33..64 的 capacity 40/48/56/64 编译期几何、64-bit available mask、逻辑 pivot map 避免物理换行；已消费的 8 列移出 live 更新区。`ALIGNED` 的 n=PAD 仅当 n 恰为对应 8 倍数 bucket 时成立。PACKED Brcb 的 factor stride 与 tail 情况分开；尾 8 列按 groups/rest 更新，不假定全部 64 lane 有效。反号消元后再次负缩放，输出按 pivotOrder/pivotRow Gather 回原序。`GetReduceRepeatMaxMinSpr` 分支及标量 UB 直接访问的 SDK/编译器依赖仍须核验。
- 401–456：先 quick、再宽精度 LocalInverseBody，结束恢复临时 x/y/u/v alias；Init 的 matrix、inverseLow、batchPrecision、precision 按互斥路径分配。`if constexpr(WAVE)` 在本类恒死；不能因保留代码就算实际运行路径。
- 457–545：每核独占 batch。若 inverse 候选拒绝，one-sided Jacobi 从原 A 重建 `[B;I]`，使用 8 对齐 padded 行；tall 输入通过 4-byte 多 burst+scalar 收集，wide 连续读。零矩阵直接输出零。discard=min(容差安全边缘,roundoffFloor)，有限 sweep 与 changed 早停并存；最终严格 norm²>threshold 保留奇异方向，按 inverse norm² 重构。满秩极小 scale 的 Gram 路径和普通 Jacobi 路径是不同舍入序列，不能以数学实数等价宣布误差相同。

分类：P=谱截断/伪逆满秩准入；F=逻辑 pivot、宽精度消元、Jacobi 回退；G=capacity 模板、DMA 依赖、alias 恢复和单写 batch ownership。没有硬件耗时证据可归因其排名。

### pinv_pipeline.h：元数据 producer 和 worker 自发布 pivot 包

- 1–113：producer 在 UB 保留完整归一化 A，二次归约选 pivot、交换逻辑 permutation；每 16 pivot 发布 64-float 元数据。先 drain 上一阶段 ready/ack，再复用缓冲。这里 repeat 参数部分被收窄为 uint8，必须连同 Host UB 容量上限检查可达 n，不能孤立断定支持所有 n<=512。
- 114–216：worker 按 `core+row*cores` 持有行，选中 pivot 的 owner 把当前 pivot 写入 16 槽 packet，worker 集体确认后加载同一 packet。panel 内前向消元得到变换，再作用到自己的行构成 inverse；每消费 8 pivot 缩短 live 区。部分高层算子之间未显式 PipeBarrier，是否有自动依赖不能仅凭源码缺失断言错误。
- 217–239：group 中一个 producer、其余 worker；所有 group 状态集体扫描后才返回统一接受结论。候选可先写 output 后被门限否决，fallback 必须覆盖全部逻辑输出。
- 242–408：PeerWindowProducer 把列分给两个 producer，以 16 列 header 和 flag7/9 的 ready/ack 交接；Drain 延迟等待，使本地更新可先执行。两角色各算 peak/Frobenius 再统一 scale。当前 TryPeerFullPinv 的 local=cols，因此 `first==window` 的中途 GM trailing 交接不会发生；不能把类中保留的 window 分支计为该 wrapper 的实际行为。仅第二角色发元数据；两个角色仍按相同阶段参加 flag0/1。CUBE 宏在这里首先是 orchestration 条件，不等同每一段都用 Cube matmul。

分类：F=一次分解、worker 复用行变换；G=生产者/消费者分离、独立 ready/consumed acknowledgement、双角色 header 协议；P=满秩和容差准入。同步 flag 的 epoch、cache 可见性和实际物理 block 数仍属 SDK/硬件核验项。

### pinv_window_pipeline.h：缩短 producer 驻留，边界重载 Schur trailing

- 1–64：producer 容量为初始 n×window 与后续 trailing 矩阵容量最大值，加 metadata/scratch；原 A 分两遍取得 peak 与归一化 Frobenius 范数，scale 范围不满足即统一协议失败。
- 65–146：16-pivot panel，低于 pivot 阈值只标 bad，仍走完集体协议。到窗口边界先等 worker 写回 trailing 到 GM，再按 permutation 重载剩余矩阵；flag3、2、4 分别处理就绪/交换和确认。发布下段元数据前 drain flag0/1，避免读写重用竞态。
- 149–249：worker 保存 interleaved 行，pivot owner 发布当前行 packet；集体确认后消元，liveBegin 每 8 pivot 前进。窗口边界所有 worker 将自己 trailing 行写到原物理 row 索引，供 producer 重载；失败不能在其余核 barrier 前单独退出。
- 250–321：最终统一 valid、inverse norm 与 cutoff 检查；低于更宽 refinement 条件且容量允许时，输出重新乘 power 作为后验修正的 prepared 数据，仅 physical0 写准备标记。accepted/prepared 按 batch 位打包，必须保留 Host batch 上限，不可推广到任意 batch。

分类：F=分解状态由 producer 元数据和 worker trailing 重建；G=窗口相位内存复用、单写 packet owner、失败路径仍参与所有 barrier；P=谱门限。窗口减少 UB 不自动意味着提速，额外 GM 与 handshake 必须实测。

### pinv_refinement.h：批读 residual、后验逐元素预算与拒绝闭环

- 1–35：Cube correction 路径逐行范数及元素误差估计，包含 correction 范数、原 inverse 范数、量化 residual 误差和舍入项；不是仅凭总残差小就默认每元素满足契约。
- 36–107：先校验 resident/capacity，保存 batchCalc/U/V/H/L/inverseTile alias 再 repack。原 A 归一化读入 cache；sourceSpan=min(8,可用缓存/r8)，每批读至多 8 行，factor 在容量允许时先 Gather/Split/Brcb 缓存，减少每列重复标量/DMA。
- 108–190：Cube residual 仅在 square、pad32、owned<=8、tile 和有限范数满足时启用；显式累计量化 δA、δX 与交叉误差。否则 vector residual 每批原行只 split 一次，使用广播 factor 消元，按 8 列或尾列再归一化。结束恢复临时 alias。
- 191–256：每核写 residual/inverse 列范数部分和，集体归约（奇数行保留上半元素）；rf 与量化/累积误差组成 `1-rf-er` 分母，必须正，否则恢复 scratch 后拒绝。gamma、gd/slack 是实现采用的误差模型，不是本次静态阅读独立证明的充分界。
- 257–326：Cube 或 vector 第二乘法形成 Newton correction；逐行/逐元素预算验证后，各核写 valid/范数再统一接受。cutoff 门另外检查 correctedTotal；失败恢复所有 alias，成功才按 1/power 输出。这里平方矩阵 Newton 结构 `X+X(I-AX)` 只在 residual/误差界前提下闭环，不能扩展到截断秩亏伪逆。

分类：P=谱间隙准入与契约 atol/rtol 预算；F=后验 Newton/residual certification、批处理 factor 复用；G=UB 相位 alias 保存恢复、全核统一拒绝、精确逻辑输出范围。FTZ、4097 split 溢出、Cube 数值模式和 collective/cache 语义仍需环境验证。

## r1-compact-stable Compact 与 Stable 全文补读（2026-11-09）

`pinv_compact.h` 1–706、`pinv_stable.h` 1–2034 均已按连续编号段读完。它们与 local 的大段相似代码此次也实际阅读，并非按文件名推断重复。

Compact：1–146 为本核 DMA、FP32 点积、Gram/双 float 及 BatchEliminate；147–288 legacy n-lanes quick/accurate 分支与 packed output；289–428 runtime 几何可选逻辑 pivot，429–488 宽精度重新从原 A 消元；489–562 PAD 模板的负归一化逻辑 pivot，563–617 快慢候选选择/精确 UB 分区；618–706 是从原 A 重建的 Jacobi 与严格 cutoff 回退。`TAIL` 通过 GatherMask 拼紧 n² 输出，非 tail 按行写精确 n×n；这属于 G（输出编排），不改变 P（截断定义）。模板 bucket 在 Init 中可直接改写 t.m/n，因此 Host key 与 PAD 的等同性是必须保持的前提。n-bit mask 有 `1ULL<<n`、packed mask 有 `1U<<n`，不能脱离 compact 的 Host 小容量可达范围泛化到 n=64/32；静态头本身不全面防御这些边界。`tryQuick` 形参不控制 quick，实际由 scale>=0.1 决定。逆范数 gate 的充分性仍依近似 inverse 残差，逻辑行置换与正/负号还原已逐段核对。

Stable：

- 1–136 本核加载/写回、Gram 精修与宽精度 include；137–159 GlobalPivot 对每核 8-float 摘要作 lane0 masked max，再还原行号；非标准 core 数有标量分支。160–230 InstallRefinePacking 将原 elimination pitch 改为 r8 packing，从 batch 和释放 resident 区找 12 个不重叠槽，必要时加 64-float 间隙并按 UB address group 置换槽序以减 bank 冲突。这是 G 硬件布局 trick，不是算法的普遍常数，`GetPhyAddr`/bank group 模型需 SDK 与设备确认。
- 231–316 广播 compensated update 有 general 与固定 64-mask 两版；fixed 路仅 r8%64==0 且 rows*r8/64<=255，避免 repeat 溢出。prepared factor 可复用，normalize 每 8 列或尾列执行；不能随意删去归一化而认为 double-float 恒等。
- 317–448 QuickResidentSquare 按核驻留行、全局 pivot、owner 单写归一化 row，其余核消费；在同一局部消元中产生下一 pivot 摘要，少一次扫描。所有核共享 best/total/cutoff，零 pivot 一致拒绝；范数/condition 门后可进 RefineResident 或输出。
- 449–654 ResidentSquare/AccurateSquare 分别双 float UB 驻留和 GM high/low；每 pivot 交换行/发布两 plane，其他核批消元。GM 路记录 best 与 index 两片摘要，按 cores-strided DMA 批读写。certify=false 的精修调用绕过谱门，是上游已判 full-rank 的专用路径，不能直接当 rank-deficient fallback。
- 655–925 RestoreJacobi、带 cached split/zeroLow 优化的 BatchEliminate，以及 quick local logical/legacy 和宽精度 local。zeroLow 只在两低部为零时跳乘积再归一化和 low cross 项；这属于 F（补偿乘加的有效前提），不能一律套用。
- 926–1063 GroupedResidentPath 把全部物理核不均匀分到 batch group，pivot candidate 有独立摘要/packet 两协议，双 parity 槽让当前候选与下一候选不互写；组内逻辑核不参与的其他组仍走全局 SyncAll。valid 失败也继续相同次数 collective；组 flag 最后统一转换 accepted bitmask。
- 1064–1253 GroupedResidentPanels 8 行小 panel：先交换 n×panel 缩小列片并在各核重复 pivot 规划，再由 row owner 发布真实 pivot packet，统一更新 owned 行。1254–1538 PartialPanels 的 LOGICAL=true 版把已消费 identity/原列移出 live 窗口，缓存 reciprocals，尾 panel 仍分 rows 与 PANEL。packedInit 连续读 owned*r8 后再重排到 stride；范数亦可紧凑 vector reduce。PAD_MAX 固定 true，GROUP=false 的内部 else 仍读 prod 而前面更新 x，该分支在当前调用 `if constexpr(GROUP)` 下的 PartialPanels 路不可达，属于保留模板死分支风险，不能报告为已复现运行 bug。
- 1539–1612 按 resident/work/batchScratch 容量依次选 16 logical panel、8 panel、packet 或普通 grouped；RebudgetWaveScratch 在 group 阶段结束后把释放行再分 12 槽，并重定位 tileLow。单 resident 与双 resident 的释放区不同，不能把高/低缓冲相邻简化为随意重叠。
- 1613–1741 WaveGroupedResident 按 batch/groups 轮数运行，候选和 ready 写双 parity。依赖 Host groups 整除 batch、group<=物理核、owned 足够；accepted/prepared 只用低/高 4 bit，契约容许的大 batch 并不意味着这条快路任意 batch 可达。
- 1742–1895 Parallel：prepared residual 失败后仍从原 A 重建，square inverse 拒绝后 RestoreJacobi，再 circle-round-robin Jacobi。每轮每列只属于一 pair，round 后所有核 barrier，changed 全局归约再早停；奇数 cols 虚拟一列并跳过 invalid pair。最终严格 norm²>threshold。n>=256 满秩且小奇异值条件触发 AccurateSquare；此调用忽略返回值直接 continue 的可靠性依上游 full-rank/scale 前提，应列数值拒绝闭环核验项，不能只根据一条 gate 宣布总是安全。
- 1896–2034 Init 的各缓冲与 Host 预算相配；Process LOCAL 模板不进入 Parallel，其余按 t.parallel 分派；逐 batch serial Jacobi 和 compact/local 同族但不等同源码。

P：谱间隙门和 Moore–Penrose 截断；F：逻辑 pivot、panel 消元、Newton 证书、round-robin Jacobi；G：元数据/packet 双槽、释放 UB 再预算、bank-aware 槽重排、同次调用 fallback。以上明确实现路径，未进行性能归因。数值范围（NaN/Inf、scale 下溢/过大循环、split 乘法溢出）、GatherMask/SPR API、自动向量依赖和 SyncAll 的真实 cache/epoch 语义仍是待环境核验项。

## r1-dsl-common DSL 运行库和两个专用消费者全文补读

2026-11-09：`pinv_dsl_common.h` 1–2741、`pinv_dsl_block_local.h` 1–221、`pinv_dsl_peer_rows.h` 1–263 连续全文已读。重要修正：common 9/107–109 的“仅单指令 adapter、没有 solver”注释只对相邻小片段成立，整个文件实际包含完整 solver 控制；不能把注释当文件契约或合规结论。

common 分段：1–55 单指令向量/拷贝/mask/SPR 绑定，字节与 32B block 的 stride 单位由 adapter 转换；56–326 uint64 搬运双 FP32（没有 FP64 算术）、volatile TwoSum/Dekker、Dot/Scale/三分量后验 Cert、Normalize/Eliminate 和宏 ABI。Dot 的 size/=2 树要求调用方 power-of-two 容量，尾不能任意换成奇数。327–356 barrier/CrossCoreFlag、SetSyncBaseAddr(fftsAddr)、索引生成；357–602 PipelineRows 16-panel packet 消费、live 窗口、窗口写回、置换输出与每核 status。603–942 双角色 PipelinePeer 与已读 native peer 同结构，但是真实独立 DSL 展开：先交换各自 peak/Frob，再 flag7/9 传 header，panel pivot 和 descriptor 都按 255 repeat 分块，role1 发布 metadata，退出 drain 全部 flag；有明确 pipe.Destroy。

953–1135 Accurate 使用176544B UB、square high/low inverse、严格范数 cutoff 门后输出并写 GM status；1137–1865 Rectangular 使用105536B UB 和 GM 双 plane，从原 A 做40/120轮两次 one-sided Jacobi。读入 tall 用4B多 burst，wide 连续读；cert 包含 V/U Gram、discarded Gram 无穷范数、B-AV 与 Aᵀu-σv，sigma 上下界分离 cutoff，rank0单独处理。certified 后仍逐元素检查 1e-4 预算；两次失败写 0/0 NaN，再 status0。此处继承 native rect 的误差模型与缺失证明文本限制，不是无条件可靠兜底。

1867–2047 ResidentAccurate 用82208B UB，结果经 reference `module_status` 返回，不再依 GM S；2049–2678 ResidentRectangular 用79680B UB，其16384float resident_work 双 plane 替换 GM 拷贝，算法/证书/NaN拒绝与非驻留版逐段相同族。pool 3296float 与 scalar pair 临时固定偏移要求本入口限制小形状，不能直接用于 n=512。2680–2741 绑定宏、Cast round policy、nomask 向量、`vcadd`+SPR sum（恢复 mask mode）、索引/packing adapter；这些低层 SDK API 尚未现场核验。

block_local：1–170 `BlockQuick` 固定64×64、pitch80、23296float pool，以原矩阵位置复用 inverse，不显式存 `[A|I]`；延迟恢复上一 pivot 对角 reciprocal（previous/recovered），用两轴 order 构造4096个 Gather offset，还原 inverse。它只写 valid 和 condition，没有自己检查 atol/rtol，最终准入必须由外层证书决定。171–221 BlockRowGemm 固定64×256×64 float Cube tile，用 rows/64、cols/256 任务枚举，调用条件必须保证整除/inner64；不能因为存在 min tail 表达式就宣称任意矩形支持。

peer_rows：固定 n=200、stride456、workers9（owned=(208-core)/9）、13个16-panel。先 flag4 就绪、0 metadata、1 packet acknowledge；200列尾panel为8列，通过 `200-max(panel*16,184)` 明确长度。元数据 negative reciprocal 与负号输出和 native 正号版本不同，不能混用 metadata。packed转置索引由32位 Add/Brcb 构造再 Gather；写两float status到2048+8core。窗口边界 flag3/2/4 与 role producer 必须同 epoch；固定196352B UB和硬编码33000索引区只能在该形状/core路径核对，不能直接泛化。

P=伪逆谱阈值、弃值子空间证书；F=双 float residual、logical inverse/panel；G=resident/GM 双实现、显式队列 epoch、mask 恢复、vector 索引生成。代码存在“完整 solver 藏于 common”与固定数值布局，不把来源注释、编译通过或排名当合规/正确性证据。

## r1-dsl-pipelines Local / BlockAtomic / PeerTile / HalfPipeline 全文补读

2026-11-09 已连续读完 local 1–507、block_atomic 1–549、peer_tiles 1–572、half_pipeline 1–759，且回看入口146–252确认实际路由。

- `pinv_dsl_local.h`：1–108 静态77344B UB与inactive guard；109–236 的33..56候选用56容量、pitch120、明确 `[A|I]` rolling live64；237–357 n64改为原矩阵 in-place inverse、pitch80、两轴 permutation；358–490 n57..63用pitch72 padded64。power 由浮点指数加0x7fffff再清mantissa得到 ceil2次幂，只在有限正scale>=0.1路径使用，不能推广subnormal/NaN。n64 nomask操作复用 reduction 的硬件mask状态，SDK语义重要。491–507 fast拒绝后resident accurate再resident rectangular，既有rank/truncation continuation。入口只对LOCAL cols<=64调用，本头其它rectangular GM分支并不能据存在就判实际可达；当前小square work/status=null。嵌套TPipe覆盖相同UB必须连同调用后生命周期核验，不能将所有静态offset都累加成同时驻留。
- `pinv_dsl_block_atomic.h`：1–94 float DMA atomic on/off 与64×128/256 Cube panel，95–165链式 `F*Dinv` 后乘 pivot rows，再atomic加到原行；166–255 lookahead先写64列lead并pc_done、再192列tail并flag12。256–549 BlockRaw 归一化原A、保留副本、每64行 stage 为多候选计算condition，选最小代价pivot block并记录row块交换，source/swapped交替，Cube更新与Vector下阶段规划部分重叠。最终按逆向交换映射重排inverse列。原子只用于非pivot块累加，写完立即disable，不是任意多核浮点和的确定性保证；任务枚举 n/256 等要求固定整除，lookahead条件 batch*n==blocks*64，不能泛化成任意tiles。status512固定历史读回也依最大stage/batch。包含功能不等于入口使用：当前 linalg_pinv 没有直接调 BlockRaw，应追踪 window消费者再判断。
- `pinv_dsl_peer_tiles.h`：1–386 fixed n200的4 producer（列64/48/48/40）、9 worker，pitch固定72、196352B UB。以GM mailbox序号6656/6848的ready/consumed确认取代producer间全局barrier；volatile读取+单cacheline DCCI，10000次poll超时只置valid0。pivot仅搜索余行前32个，是“受限候选+拒绝回退”不是全partial pivot。Frob采用元素数×局部peak²上界再换scale合并，明确不是精确Frob。387–459 PeerPipeline把最后两逻辑核作producer，其他作worker，再扫描valid/逆范数与81谱门、3.6e9条件门；失败调用矩形cert solver。460–572 PeerTile把物理偶数lane前置producer，inactive逻辑核仍做4/0/1的13轮协议，汇总13状态后按batch写7040再全局归约，拒绝统一回退。mailbox超时、跨调用epoch清零和DCCI语义仍需硬件验证；不能因有poll上限就宣称无死锁。
- `pinv_dsl_half_pipeline.h`：1–32 half transfer/SPR与float nomask adapter；35–294 float worker（非half），但使用负pivot重建、nomask64主体+显式tail、vector化最终置换index。296–326重复adapter受include guard完全屏蔽。329–674 HalfSelector先FP32统计/归一化，再Cast为half作pivot规划与Schur更新，188416B UB含float32768、half24576、half_work4096；双角色仍7/9header、0/1metadata协议。half只决定pivot次序，真正worker从原FP32行和packet重新归一化，不能称“输出只用half计算”。余行>128时两次half max，并读取uint16 index，最多两段的容量限制需Host路径n范围支持。679–759临时重绑pipeline宏到halfselector/floatrows，外层81谱门和condition门统一reject后rectangular fallback，最后恢复宏，避免后续include被污染。这是G编排正确性的重要细节。

实际入口：GROUP|PIPELINE|CUBE|WAVE且非WINDOW直接走PeerTile；非WAVE非WINDOW按TAIL选HalfPipeline、否则PeerPipeline，AIC直接return；COMPACT/LOCAL小shape走resident DSL并return，native相同功能是备用/死route；WINDOW只在WindowSupported时拦截，失败才native。故native peer full/pipeline、BlockRaw是否使用必须按该key检查，不拿函数数量充当活跃优化数量。

可迁移：G=多角色物理核布局、inactive参与协议、阶段parity、带epoch的mailbox、宏绑定恢复、分段masked/nomask；F=低精度仅作候选规划后FP32重演、block inverse lookahead；P=谱阈值/拒绝后重做SVD。所有硬编码core数、pitch和浮点mode均需重新推导，不直接照搬。

## r1-dsl-compact Compact DSL 全文与明确的控制流缺口

`pinv_dsl_compact.h`1–1374 已连续全文读完（包括两份展开16-pivot主体，不用关键词替代）。1–21为15列pad/整数索引小adapter；23–702为PvStageCompact；704–1374为PvStageCompactAligned。

两版本均先inactive guard，然后InitSocState和固定UB原地址（pool1440float@0，indices64uint32@5760）；外层job循环按blocks分batch。15/16 no-pivot候选先行，失败按预期应进入logical pivot（4bit打包permutation，最多16行）、再resident accurate、resident rectangular。15列输入一次读225float，再Brcb/Add生成二维Gather index并补零第16列，输出GatherMask拼225；16列展开把已消去8列逐段移出live窗口。所有zero-pad与mask都是G布局，不是数学近似。

Aligned16版不缩放原A，而以负identity起步，pivot_floor=ceil2pow(peak)/256，total=256peak²×slack是Frobenius上界，不是实测Frobenius；最终直接输出正inverse。普通16版先power归一化，以正identity和最后负缩放还原。两者代数符号链不同，不能混抄某一段。ns_sum/SPR、sr_max ORDER_ONLY_VALUE、静态UB和位重解释属SDK边界；本次未运行这些源码。

**确定的控制流问题（不是性能猜测）：**普通展开16候选125–127（以及其后每个pivot的break）、Aligned801–803（及后续break）最近的循环是外层job `while`（75/757），不是候选局部循环。遇到小pivot时直接退出job循环，跳过688–693/1360–1365的兜底，且此前未写X。可由合法数学输入构造：n=m=16，A是交换前两行的单位置换矩阵，atol=0、rtol默认16×eps，全部σ=1，满足保留σ≥.01σmax，首对角0触发小pivot。此时原A满秩并应输出Aᵀ，但源码路径不写输出。该反例只证明源级路径缺口，不是已经上NPU复现的失败；不能因公开榜单通过就忽略契约。15列非展开候选的break位于column_tile for内部，会继续fallback，不能与16列混为同一缺陷。

此外字面通用 `n<=32` guard不等于真实支持32：pool/pitch40、identity16、4bit permutation都是16容量设计，实际Host/key可达范围需限制。`InitSocState`并不使无pivot自动安全；真正可迁移的是**带正确作用域拒绝出口的快候选+证书/通用回退**（F/G），不是假设测试矩阵总是对角占优（P输入分布假设）。

## r1-dsl-window Window 全文及最终可达性闭合

`pinv_dsl_window.h` 1–2981 连续全文实读，SHA256 `6afc5fefe71edf995f407ded745c01d956bf78cdf67a7c0f42d1ead76c74fd07`。下列区间连同此前 native、block、common 消费者一起完成调用链闭合，不据函数名推断正文。

- 1–81：直接 ND2NZ / LoadData / Mmad / Fixpipe 适配层。FP32多行按16行分块，B转置要求K16整除；浮点HF32关闭。参数rows/cols=0会按完整tile处理，不是零元素拷贝，调用者必须保证tail枚举合法。K模板只放行128或FP32的16整倍数≤128。MTE2→MTE1、MTE1→M、M→FIX以及L0复用事件显式配对；未注册Matmul库不是“没有矩阵乘”，而是直接操作Cube。
- 84–617：Rows按交错owned行持有逆变换，producer保留全矩阵或n×window窗口；每16步发布64-float元数据，唯一owner写pivot packet，所有worker消费确认后才复用。每8步压缩live区；窗口交接使用3/2/4三相握手，producer按最新permutation从GM重载Schur行。scale有限正值检查、2幂归一化、masked pivot、Frobenius和最终inverse统计均有实际消费。196352B UB并不是任意n通用：入口 `WindowSupported` 限n∈{128,256}、group2..48，producer/worker≤49088float、每核输出≤2048float。
- 621–1266：PreparedRows保留16步系数history、小panel与center；PreparedProducer发布16×16中心块，额外Transformer一次性对全局packet做三角变换，worker在flag5后读取。这样把各worker重复的panel处理集中给一个角色，不是简单多开一个核。其UB容量与角色数另有2834的完整guard；Transformer也参与窗口握手，不能只统计其算术工作。
- 1269–1414：FusedRaw/PreparedRaw按batch划group，最后一个角色为producer，Prepared再抽出倒数第二个为Transformer；所有状态集体扫描，81谱门与3.6e9条件门只接受特定满秩逆候选。失败不是提前离开后续协议，而是交给总控继续精化/回退。
- 1416–1759：FusedQuant把A和Y拆成五个有符号radix256 digit，high/low误差分解后Cast至int8，末尾余量以256^-6记录。Decode用TwoSum式高低分量构造R=I−AY，另生成half残差×4096与Y/sy，供一次Cube修正。这里不是此前native Cube的25个独立交叉项：窗口版只计算前六条卷积对角线，必须把省略对角线的尾项界计入证书，不能引用“25项全部保留”的证明。
- 1763–1918：FusedInteger把同一对角线的digit对累加成INT32平面。n≤256时最大对角线5×256×128²=20,971,520<2^31，但已超过FP32精确整数范围2^24；因此Decode对第4条对角线特设INT32→float→INT32余数修正，其余最多4项、绝对界≤2^24。此界只在每digit确实落在[−128,127]及n≤256前提成立，不能推广512。FusedHalf为YH×RH做32×128纠正。Norms的256列缓冲由WindowSupported的n上限保护，孤立看不能扩为512。
- 1922–2128：Certify从各核残差列范数、量化余量、A/Y范数构造误差和；包含被截去digit对角线的tail项、FP16/FP32舍入slack、`1−rho−error`分母与逐元素`0.00008*(1+abs(result))`预算。源码明确先写修正后的候选Y，再由总控收集所有核valid；拒绝后必须覆盖全部输出。常数0.0021、1.25、3.02等属于实现采用的模型，本轮并未独立证明它们对所有合法极端输入均为充分上界；不能称形式化证书已验证。
- 2131–2402：嵌套DSL调用后用FrRestorePipe恢复按架构/分核模式对应的全局TPipe指针；这是生命周期编排，不是数值算法。BlockGemm当前针对128×128输出、K64分片用L0双槽事件，旧版保留按列缓存B和全PIPE_ALL方案。未满足当前guard才调用Legacy，不能把两版都当同时执行。BlockQuick/BlockRaw的include及宏在此实际被绑定。
- 2404–2745：BlockPack归约原A、候选逆、残差三个Frobenius范数和各stage标记，按81阈值与condition接纳；BlockDecode形成R与I+R；BlockFinish将中间候选与修正相加并恢复scale。BlockProduct的两half-plane四交叉项、BlockAccumulate虽定义完备，但此文件最终PvStageWindow没有调用其宏，不能因存在就宣称当前路径用了它们。4096元素块拷贝和n/128、rows/64枚举依赖受限n，非任意tail实现。
- 2747–2981：总控先判断blocked n=256分支，实际调用此前BlockRaw，再经三次FP32 BlockGemm形成残差/修正；所以BlockRaw并非全局死代码，前文入口“待追踪”的问题现已闭合。blocked guard中workspace比较后接`||0<=batch*n*n`在正维度下恒真，本地比较不构成容量保护；实际Host的WindowWorkFloats取max(窗口量,13·batch·n²)才是分配依据。非blocked按UB guard选择PreparedRaw或Raw。AIC循环等待8、失效缓存读next_job，整数残差完成发10，再等待half修正工作，负job为统一停止；AIV按相同阶段发8、等10，最后写−1/−2并完成最终握手。所有batch未统一接纳时逐batch检查，必要做量化精化，再校验所有核valid与谱门；fallback_enabled时单owner运行矩形求解并广播状态。最终返回统一all_admitted供入口决定是否继续native，不能只让AIV return而留AIC悬等。

分类：P=带题目谱间隔的81门、特定n/批量布局；F=候选分解+后验残差、低精度候选/高精度纠正、radix digit乘法与尾项控制；G=角色拆分、带消费确认的packet窗口、L0双槽、宏绑定恢复、嵌套TPipe恢复、统一失败/停止协议。可借鉴的是这些有前提的结构，不是直接抄固定offset、core数、误差常数或把源码分支存在当作已产生性能收益。

## pending 全文完成后的核验边界

> 本节是源码补读前的阶段性待办快照，后续 `R1 window/block DSL` 补读已闭合其源码阅读缺口；当前阅读状态以文末补读章节、[coverage](../coverage/02_pinv.json)及[设计复盘](02_pinv_REVIEW.md)为准。仍未完成的数学/API/同步/运行/性能验证不因源码闭合而自动通过。

三份pinv源码的首轮全文阅读现已完成，旧节内“未读/待追踪”是分阶段记录，已由各追加节及本节闭合；coverage以41文件、19,921行全覆盖为准。发现的16阶展开break缺口保留为源级确定问题；没有通过修改或运行归档来掩盖它。

官方API仍未核验：GetReduceMaxMinCount/GetReduceRepeatMaxMinSpr、InitSocState与静态LocalTensor UB0、底层vcmax/vmla等、MIX下GetBlockNum定义、SyncAll与cache一致性、generated wrapper user workspace指针、HF32及Fixpipe/LoadData尺寸限制、kernel aligned宏与ICachePreLoad、嵌套TPipe恢复和跨核flag可见性。这里未引二手资料，也未把私有写法直接判合法/非法。阅读metadata与SHA校验只是覆盖证据，不是同步、数值、SDK或性能证明。

<a id="r1-window-dsl-full"></a>
## R1 window/block DSL 1–2981 全文补读（2026-11-09）

此前coverage里唯一未读文件 `pinv_dsl_window.h` 已按连续行号1–2981读完，SHA256 `6afc5fefe71edf995f407ded745c01d956bf78cdf67a7c0f42d1ead76c74fd07`。这是完整静态源码首轮审读；不表示device验证。

- 1–81：Cube copy/MMA/Fixpipe模板和量化/半量化搬运适配器。模板在编译期区分float、half、byte的L1转置路径；float MMA显式关闭HF32。函数支持的矩形/对齐前提是模板参数和Host形状共同约束，不把helper当任意矩阵API。
- 83–333 `PvDslWindowRows`：A按核拥有行加载，按16列panel从workspace读pivot/reciprocal元数据；每个owner更新自身行并将pivot行包写入packet。8行live区用mad-strided批量消元，窗口边界通过全局barrier+cross-core flags把新列读入/交换；末端按行恢复置换、估计inverse/peak并写X和每核status。metadata为固定8192float/job，核心数量、stride、window界限必须跟Host一致。
- 338–619 `PvDslPipelineProducer`：全矩阵读取(非window)或按4096块重复读取(window)估计max与平方和，要求scale幅度在[0.1,~FLT_MAX)；用power-of-two scale，再将消元分块成最多16 pivot、逐pivot max/交换及≤255行mad批，输出每个panel的64float元数据。窗口路径在panel边界发布/等待cross-core flags。该producer的有限范围门属于专用快速路径，失配由上层valid状态处理。
- 621–872 `PvDslPipelinePreparedRows`：布局将small/center/factors输出到不同容量区，主体仍是行分片和panel元数据；每次pivot先在中心小矩阵做小批量消元，再把packet广播到所有owned rows。额外workspace用于prepared窗口和center repair，不等同于普通producer。
- 874–1164 `PvDslPipelinePreparedProducer`：全矩阵/窗口两种读取方式都以4096元素块重算幅值、平方和；全矩阵先缩放驻留，window每块从GM重读。pivot主循环写两槽panel metadata及center correction数据，phase flags管理consumer。固定的workspace布局/8192float元数据区是该实现家族专用ABI。
- 1166–1268 `PvDslPipelineTransformer`：不重新分解，仅消费producer已写的每panel reciprocal和packet，完成局部逆变换、按窗口分段写回与metadata握手。之后1270–1274宏将通用接口重绑定到这几种实现；宏展开依赖外层`pvPipelineBuffers`和`fftsAddr`上下文。
- 1276–1414 `FusedRaw`/`FusedPreparedRaw`：按block角色将部分核设为producer，其余处理rows；等待所有分片status，合并inverse/total、用sqrt与atol/rtol得到admission cutoff，并检查inverse cutoff和condition阈值81、3.6e9。bid0写global status。Prepared与普通版的角色分配和producer变体不同，不能凭近似重复体混为一路。
- 1416–1623 `FusedQuant`：把A/Y按分块加载，归一化并两次拆分成整数高/低digits，以half Cast作为中间表示、再Cast int8并按digit顺序写五plane；同时输出平方范数。比例常数与降位运算仅在其上游admission及scale范围内有意义，不能当作任意输入无损量化。
- 1625–1761 `FusedDecode`：从5个int plane做FP32累加，diagonal==4处显式Recover/roundtrip余项并追加一次补偿；最后分别生成float R与half Y/RH候选。这里只能说源码用了误差补偿结构，是否满足1e-4必须结合FusedCertify及完整输入范围审。
- 1763–1883 `FusedInteger`/`FusedHalf`：AIC端以32×128或32×128 half Cube tile执行int8/half矩阵乘，输入/输出用ND2NZ、L0加载转置与边界rows/cols截断；整数路径按planes和diagonal累加，half路径是一个plane。调用有效性依赖n、planes、blocks的上层分支及边界参数。
- 1885–2128 `FusedNorms`/`FusedCertify`：Norms算候选矩阵行平方和；Certify将分片norm、量化输入误差、残差、half/float舍入slack组合成逐元素界，按`0.00008*(1+abs(result))`检查并记录sum/worst/status。0.0021、1.25、3.02等经验常数在本次静态审读中未被独立形式化推导，故仍是数值证明事项。
- 2131–2548：跨核flag/cache invalidate和`FrRestorePipe`恢复TPipe上下文；内嵌BlockGemm legacy/双缓冲版本及BlockPack后验聚合。Gemm支持128x128或64x128固定tile，K按64/128条件分支并通过事件双槽复用；Legacy仅为满足guard的旧路径。每个status、workspace offset是固定布局，须与Host布局核对。
- 2550–2745：BlockProduct执行half输入的四交叉项投影；BlockDecode生成`I-P`和`I+(I-P)`候选，BlockFinish恢复scale并写最终值，BlockAccumulate把两plane修正相加。函数定义完整不等于都在本入口激活；本节已将宏、调用点与后面的总控做了逐项对照。
- 2747–2981 `PvStageWindow`：对n=256和工作区、batch、核数等条件选择blocked路由；AIC端执行BlockRaw、三次BlockGemm并经cross-core同步发布job工作。AIV端配合decode/pack/finish，先用全局status判断是否快速接纳；不接纳时逐batch重算门限，必要走量化精化、解码、范数与certify，再按`fallback_enabled`进入矩形求解fallback。最后统一同步并返回all_admitted。源码中workspace guard含正维度上恒真的比较子句，不能把它视作真实容量校验；Host计算的实际分配仍需单独覆盖核对。

分类：**P** 谱间隔条件下的81类快速admission及n=256 blocked路线；**F** int8/half Cube候选、分解数字修正、残差后验门、失败后精确矩形回退；**G** producer/row/transformer角色切分、workspace packet交接、宏重绑定与TPipe恢复、double-buffer Cube事件、失败/停止统一状态协议。**R** 精度常数/全输入误差充分性、私有同步/cache API、所有tile边界及具体性能仍未在本轮验证。
