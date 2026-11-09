# 03 SplineConv：公示源码逐文件逐行语义审读

题面：[原始PDF](../../problems/03_SplineConv.pdf)；[题目契约](../SPEC_CONTRACTS.md)。

状态：**全文首轮审读完成，证明义务仍开放**。公示三份的全部40文件、10607物理行已连续全文读过或以SHA256相同构建文件复用审读；三版语义分别见B–G。行号按UTF8 `splitlines()`，不是默认PowerShell编码计数。只读原档，未编译/运行NPU，没有获奖二进制的对应性证明。实际覆盖见 [账本](../coverage/03_SplineConv.json)。全文覆盖不等于所有数值误差、SDK同步契约和性能归因已证明。

## A. 题面与证据等级

完整题面见 [十题契约第03节](../SPEC_CONTRACTS.md)。关键是：FP32/BF16、dim2/3、degree1/2/3、闭/开样条、按target聚合、mean按入度、root与显式自环独立、可选bias，FP32误差1e-4、BF16误差1e-2。10月题面E下界64，公示代码含E=1路径；支持扩展shape本身不是错误，不由此断言隐藏用例或违规。

证据分为：S=源码可直接确定，M=带前提的数学推导，R=运行/编译器/API契约待核验。公示Pass不是各路径全域证明。

## B. 公示顺序1：Host与真实路由（submission 1174398）

文件：[Host](../../extracted/SplineConv/rank1_submission_1174398/code/op_host/spline_conv.cpp)。

| 连续行段 | 每段实际含义与边界 |
|---|---|
| 1–23 | includes、namespace、Tile入口；检查四必需shape/desc非空、秩与dtype；使用OriginShape。仅FP32/BF16，edge int64，各浮点输入同dtype。|
| 24–33 | 五属性按固定index读取；aggr空默认mean、sum兼容add；open缺省/-1哨兵；dim允许1..8、degree0..3，比PDF宽。`GetAttrs()`未先判空为健壮性风险，合法schema通常提供。|
| 34–44 | 识别edge `[E,2]`、weight `[K,Co,Ci]`、pseudo `[dim,E]`转置扩展形式；这些送streaming。ci允许0，co正、n/e范围比题面宽；不是对输入数据值做Host计算。|
| 45–51 | 混合基数展开，ks不足dim时重复末值；kernel乘积上限65535；scale用unsigned存k-degree*open，再在device转int32。合法题面开样条scale>=1，负scale路径仅扩展域。通过除法避免`E*S*Ci*Co`乘积溢出。|
| 52–57 | `full=true`才访问可选root/bias及其desc，shape/dtype检查后设置hasRoot/hasBias。|
| 58–85 | 先按min(AIV,n)分nodes；窄稀疏与宽Ci128分别控制block数；single/tail需无转置、无root/bias、Ci8/Co16、E1、dim2/degree1/ks2。FP32/BF16的tail对齐条件不同，反映每行64B/32B写出。|
| 86–108 | 查询实际UB；baseBytes包括head、agg/comp、matrix/cast、临时x/y；保留16KiB安全余量。按容量选streaming、weights/graph cache、vectorDot/fusedDot和smallScalar。fusedDot额外要求Ci已对齐，K*Ci<=2040，使Brcb repeat<=255。|
| 109–122 | single=1..4与sparse=1..4；sparse有Ci32/128、Co16、E<=64、输出稀疏等强前提；FP32单边chunk/tail把launch block数除2。|
| 123–139 | NODEPAD编码静态N；single6表示N=31-2*pad；**126行先把sparse4改8，127行再检查sparse4，因此127–132分支不可达**。四边稀疏转换5/6，区分部分/整批输出区域。|
| 140–145 | CLOSED由scale==2编码；六维模板选择、blockDim；小FP32 single路径可设ScheduleMode1；workspace包含库预留与每block的next链。|
| 146–158 | base/full两个tiling入口、输出shape推导、weight转置推断、输出dtype跟x。|
| 159–176 | `SplineConv`只注册四必需输入，不含root/bias；可选属性给默认值，910b配置。|
| 177–195 | `SplineConvFull`另注册两个optional输入，通过opFile/opInterface指定同文件的full入口。**接口双名与题面单接口可选输入的对应关系必须在集成层确认**，不是kernel支持就自动等于原接口支持。|

### B1. Key枚举与入口并非“所有头都在跑”

[key头](../../extracted/SplineConv/rank1_submission_1174398/code/op_kernel/tiling_key_spline_conv.h) 1–11定义单组合宏；13–108逐项选择dtype/SINGLE/SPARSE/CLOSED/NODEPAD，所有行已读；109–117声明字段位宽和合法值。NodePad至30用5bit、single0..6用3bit、sparse0..9用4bit，不能仅看字段DECL就认为任意笛卡尔积都编译，必须同时看SEL。

[kernel入口](../../extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv.cpp)：

| 行段 | 路由语义 |
|---|---|
| 1–24 | 头文件包含；RunBoundedSparse先检查Ci32等前提，强制FixedCi32 resident实例；所有头的物理UB地址不会同时占用，按选中路径分析。|
| 25–38 | SINGLE优先编译期分派；6→ClearAhead(encoded N)，5FP32→TailBulk，4FP32→Staged，3FP32→BulkStaged；BF16 2/3/4→Raw/One/Chunk/Tail模板。静态路径不一定读完整tiling。|
| 39–78 | SPARSE 5/6/9/8/7/4各专用实现；其余1→Compact32、2→Compact128、3→EdgeOwner32。只读最小元数据；SPARSE9没有本Host赋值，4已改8，7被前述死分支阻断：这些代码存在但不可当本Host实际派发。|
| 79–91 | 通用key0依次尝试SingleStatic、LocalWeights、Compact、BoundedSparse；否则smallScalar/streaming/大K/resident，不是直接总走一个generic。|
| 93–159 | Full重复相同专路；专路前提无root/bias，最后general分支才传真实root/bias。不能看Full函数签名就认为每个专路都使用optional项。|
| 161–170 | 定义空的system-workspace处理与`workspace+RESERVED_WORKSPACE`偏移，并宏替换`SetSysWorkspaceForce/GetUserWorkspace`。它位于源码末尾，可能影响生成wrapper后续文本；不是普通算术优化，依赖代码生成顺序和SDK内部约定（R），不可无证迁移。|

[tiling记录](../../extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_tiling.h) 1–11全部为uint32字段，数组scale/ks各8；此版本不使用序列化宏，Host/Kernel依赖同一C++布局。

## C. 顺序1真正的算法变化：先聚合、后乘权重

[resident](../../extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_resident.h) 全260行。

对于固定输出节点i，定义 `a[k,c]=sum_{e:dst=i} sum_{s:wi(e,s)=k} basis(e,s)*x[src(e),c]`。则实数算术中：

`sum_e sum_s basis(e,s)*(x_src @ W_wi) = sum_k a[k,:] @ W[k,:,:]`。

这是交换有限求和次序、利用W对边不变的线性性；不是近似B-spline，也不是省略真实输入。直接方式主项约`deg(i)*S*Ci*Co`，聚合后约`deg(i)*S*Ci + activeK*Ci*Co`。若activeK多而度低不一定划算，需与occupancy/驻留资源共同判断。**浮点顺序改变，不能由实数恒等推出逐位或1e-4误差保证。**

| 连续行段 | 算法/内存/同步具体含义 |
|---|---|
| 1–46 | 初始化；FixedCi重写尺寸和cache标志；分配完整图缓存、dot/broadcast、weights raw/float、head、agg/comp等。FP32 raw/float直接alias，BF16另升FP32；workspace next按block隔离。|
| 47–67 | 图/next访问统一；LoadRow与LoadMatrix用DataCopyPad补零，PIPE_ALL衔接搬运/转换；Co已对齐时矩阵单burst，否则逐行。|
| 68–98 | generic与模板化degree基函数；BF16直接把16bit放入float高16位为数值扩展，不是half转换；floor对负scale单独纠正。合法pseudo>=0/scale>0时非负分支可用截断整数等价floor。|
| 99–122 | 按slot混合基数算wi/factor；非零factor才累加；activeK用bool64或UB flags。indegree高时Kahan式comp更新，否则Axpy(counter mask)；正常路径恢复mask。|
| 123–138 | 单K矩阵：Brcb扩展输入，分Co<=64/大于64两段Mul；按Ci折半树归约，奇数Ci保留未加的中间行。|
| 139–155 | fused把K*Ci视作归约行；Brcb最多255repeat，Mul每255行切块；树归约后加y。这依赖Ci==ciPad，避免权重/聚合行错位。|
| 156–170 | 先一次搬图/伪坐标；权重按需驻留并BF16 Cast，PIPE_ALL完成后消费。|
| 171–181 | 输出可按cyclic或连续owner；每block全扫边构建本block输出的逆序链，head空间按ceil(N/blocks)足够；输出owner互斥，无需跨核atomic。代价是每block扫描E和元数据副本。|
| 182–189 | sparseZero提前写孤立输出0；raw借作zero缓冲，rows<=Ci、Co==CoPad保证容量；cyclic写用GM dstStride，单位字节。|
| 190–227 | 每节点初始化agg/y，计算入度；indegree>128启补偿；合法src才计count。逆序链改变边累加顺序。PDF未定义非法边，此代码跳非法src的语义是扩展健壮性，不能当原题要求。|
| 228–240 | fused或按activeK相乘；else走逐Ci Axpy；mean最后除count。先乘W后mean在实数上成立，FP32舍入顺序不同。|
| 241–246 | root从x_i独立乘矩阵后加y；bias最后加；BF16最终CAST_RINT。root不是度均值的一部分，与显式自环不互斥。|
| 248–260 | 成员声明全部审读：TPipe和TBuf拥有生命周期，浮点/原始视图分离；没有跨block共享UB假设。|

## D. 顺序1全部辅助文件的连续语义账

以下每一行指一个已全文读过的源文件；范围内空行、pragma、模板、成员也已检查。路径前缀均为 `extracted/SplineConv/rank1_submission_1174398/code/op_kernel/`。

| 文件/全部行 | 细分说明与特别点 |
|---|---|
| `spline_conv_streaming.h` 1–107 | 1–20小固定缓冲与BF16标量扩展；21–36通用floor/多项式；37–63分列搬W，支持转置W标量gather；64–73构建owner链与有效入度；74–99每128输出列遍历边/slot/Ci并补偿累加，mean→root→bias→Cast/store；100–107成员。主元计算在NPU，不是CPU fallback；低内存但大量重复读。|
| `spline_conv_small.h` 1–94 | 1–25完整小图x/weights/root/bias本地化；26–37加载和BF16位扩展；38–50一次搬入/转换；51–71按节点扫描E、4K*8通道的标量Kahan；72–87输出通道标量点积/mean/root/bias，再向量Cast/store；88–94成员。8输出但BF16 Cast16槽先初始化y16，避免读未定义值。|
| `spline_conv_single_interpolate.h` 1–22 | 1–11四个线性basis；闭样条ks2通过q→2q→三角波折回，等价周期索引插值；13–22一次Muls+三次Axpy，ExplicitBarrier编译期开关。q=0/0.5/1端点逐个可推导；允许删barrier不等于API/编译器hazard已证明。|
| `spline_conv_single.h` 1–114 | 1–20Raw/One/Chunk/Tail与通用guard；21–28固定UB字节布局、BF16转换区分开；29–46搬edge，先放Tail的weights/pseudo，N可由blockNum还原；47–60cyclic/连续16节点窗口、zero输出；61–98按target唯一owner插值/搬X/升精度；99–104feature dot与8行树归约；105–114输出事件及对齐DataCopy/DataCopyPad。输出全写0再填真实target，非target仍正确。|
| `spline_conv_single_stage.h` 1–49 | n=2*blocks-Tail，只有少量active块各写16节点；权重事件0，sourceX事件1，插值与X搬运交叠，最后输出。输出slot与blocks缩放必须和Host成对，不可单移植kernel。|
| `spline_conv_single_bulk_stage.h` 1–155 | 1–45单worker先清全输出，whole-x stage，LinearReduce可用同一目的repeat7次实现8输入行累加；46–97更大N时每16节点一owner，仍全x搬入；98–100include；102–155odd-tail路径block0只清非target、block1数值，其余退出，图元数据预载。MTE3 barrier避免清零覆盖刚写结果。|
| `spline_conv_single_tail_sector.h` 1–49 | 一worker先清N节点；结果按8节点sector保留边界，sector满时整块写、末段直接写1行；避免复写不相关空间。此函数只被include，当前入口未直接调用（S），不能认作获奖实际收益来源。|
| `spline_conv_local_weights.h` 1–106 | 1–30单大TBuf按32B Reserve、图head/links/occupied；31–52按某边4个wi载选中矩阵→先插值W再feature dot；53–91cyclic节点、仅active节点链/mean/output；92–106成员和严格guard，仅实例化C128且K>4/E<=16。C32分支虽然类能表达但当前wrapper不实例化；不可把未激活模板问题混入实际路径。|
| `spline_conv_compact.h` 1–292 | 1–43C32/128硬编码UB与后置64*C feature预取；44–94Reserve、SingleDot、全4K融合dot；95–147混合cyclic数值owner/连续clear-owner，按edge预取X；148–182C128单边快速路径和多边active链；183–230多边按K聚合，必要mean；231–252只写nodeMask去掉directMask，否则stride写完整窗口；253–261成员；263–279C32单owner特化；282–292generic guard。分开清零与数值写mask，使互斥输出区域不需跨核等待，但要核对32/128行和mask位界。|
| `spline_conv_sparse_output.h` 1–32 | FirstBit用`mask&(-mask)`+CLZ；CopyRows只写实际row，按GM基址对齐选基础/扩展搬运；ClearInactiveRows用occupied补集找连续零跨度，减少多次小MTE3。当前调用rows<=32，转uint32寻找低bit才安全；不能任意扩成64+rows。|
| `spline_conv_clear_mask.h` 1–45 | 1–20双64bit占用mask跳过真实目标；22–29大zero跨度每128行复用zero区；30–45四mask支持256行。必须有0/非0mask guard，移位不得>=类型宽度。|
| `spline_conv_clear_owner.h` 1–32 | 1–25专门clear worker的连续行界，按8节点对齐、读全部4edge找到occupied，避免清真实输出；27–32四target相等关系位编码。固定Edges4/Blocks8下bound<=16、uint32mask够；泛化参数需重新证明。|
| `spline_conv_edge_owner.h` 1–168 | 1–22固定C/布局；23–48block置换与DedicatedClear分工、整x/每edge feature预取；49–92用更小edge索引作为每target唯一owner，duplicates合并；93–120单edge/多edge与mean输出；121–149等待所有已发事件再退出，C32特殊repeat累加；150–168字段和静态N=64-Pad或16*(blocks-4)-Pad还原。block置换把(0..7)映射(0,4,1,5,2,6,3,7)，意图安排数值/clear空间分散，真实调度收益未测。|
| `spline_conv_four_owner.h` 1–91 | 1–23固定四edge/four-clear；24–44每权重面板独立DMA+事件0..3，features事件4；45–56检测同target就先Wait所有DMA并进fallback；57–84按到齐面板逐次插值、dot/重复加、写target；87–91N=64-Pad。涉及真实数据驱动fallback而非预存答案。|
| `spline_conv_four_group.h` 1–49 | `noinline`重复target fallback；不传W地址，直接复用调用者2304字节UB的已完成weights；features/aggregate/spread/dot用另一组固定地址，caller死缓冲可复用；按最小edge owner聚合，再mean。必须保留caller所有weight事件已Wait这一前置条件，独立拷出函数会失效。|
| `spline_conv_wide_tile.h` 1–41 | Load0..3分别设置MTE2_V事件；Interpolate逐面板Wait并计算，FeatureDot两次Brcb填16输出通道broadcast；GroupDot把4*C=512行按255+255+2重复Mul后树加。按255切的是指令repeat范围，不是数学分块随意取值。|
| `spline_conv_wide_owner.h` 1–207 | 1–24固定UB并在dot前留256B gap（可能影响bank，但源码不能证明收益）；25–46CachedMeta的预载或DMA图；47–66DedicatedClear分256rowmask；67–118Early/Eager清零和先插值，维护prepared与owner，退出仍消费已发事件；119–149单/多边聚合并mean/output；150–194元数据访问、清零、WaitWeights与single；195–207字段与EarlyOwner实例。当前HostSPARSE8使用Early=true/Eager=true/CachedMeta=false；7/9虽实现但本Host不可达。|
| `spline_conv_encoded_owner.h` 1–10 | N=256-8*Pad、E8、CachedMeta=true的窄包装；Host未派发SPARSE9，不能归因其预载减少元数据开销。|

### D1. 特殊布局与资源推导示例

Compact C128 FP32末端：WO2560，W32768B，feature512B，agg2048B，broadcast16384B，dot32768B，out8192B，后置64*C features32768B，最终最高地址128000B；BF16对应转换区分开，最终124160B。不是把所有header中的数组容量相加，因为它们互斥选路。该推导确认静态局部上界，不等于编译器未另用scratch/stack。

Single6把N写进NODEPAD使N=31-2*pad，对N3..31所有奇数一一覆盖；Host launch=(N+1)/2，kernel仅block0计算。Single5大odd N下分清零/数值两worker，mask排除target，因此可不全局同步也不覆盖结果；依赖row独占及真实目标mask。

## E. 顺序1的可证问题与待证风险

1. **S：不可达源码。** Host126 `sparse4→8` 后127仍判断4，4/7相关分支不能作为当前选中路径。SPARSE9也没有本Host赋值。
2. **S：基本/Full ABI分离。** base Op不注册optional root/bias，Full另名；需确认评测/下一题集成实际调用哪个接口。
3. **R：同步与mask契约。** 若干fast helper依赖同PIPE_V数据依赖的调度而省显式barrier，或用物理UB跨noinline函数复用；需目标CANN产物/官方契约，不能在纯静态审读中宣布“安全无问题”。
4. **R：聚合重排数值。** Kahan只在某些路径/度阈值使用，无法据这一点推导全域1e-4；需依照数据上界/错误累积或实测判定。
5. **R：框架宏重写。** SetSysWorkspaceForce/GetUserWorkspace与生成wrapper和库预留强耦合；没有本环境二进制验证。
6. **S：额外域不等于题面域。** E1/degree0/转置输入是扩展支持；不能拿扩展路径存在反推本题普通输入走它。

## F. 顺序2完整首轮审读（submission 1176862）

Host379、key124、tiling37、kernel2253行已连续实读。三份CMake分别与顺序1的根/Host/kernel文件SHA256完全相同，复用语义审读，不把它们算作新增算法证据。顺序3同样仅复用这三份构建文件，其他未读。

### F1. Host与编译期元数据

Host1–98检查storage shape、输出shape、dtype、属性列表和optional参数；uniform允许标量/统一列表广播，不等于忽略dim。99–145计算通用任务、邻接workspace、resident UB上界：只有FP32、无root/bias、dim2/degree1、两个ks2且open1、Ci≤128且8对齐、Co≤32且8对齐、N≤2048、E1..64才选resident。合法题面中E≥64，故诸多E1/4/8专路并非由这份题面一般域触发；必须确认赛事不同时间版本，不能把所有技巧都当当前常规路线。

146–199按资源族而非输入数值编码key：E1的power2 N用log编码，非power2小N用offset+N编码；FULL要求8块/E8/Co16/Ci已满对齐/足够权重字节；HALF要求8块/E4/Ci已满对齐/范围内N。完整N区间与资源约束应和SEL组合逐项对应，不能只检查DECL字段宽度。workspace按resident无需next、general每块edgeStride分区。202–254是base/full入口、shape/dtype推导；256–297注册两个接口。300–379导出tiling回调，并在OP_TILING_LIB下复制内部registry类声明、用dladdr定位自身库、静态初始化注册：这是SDK ABI绑定设计，不是通用NPU优化。

### F2. Kernel连续行段语义账

文件前缀：`extracted/SplineConv/rank2_submission_1176862/code/op_kernel/`。

| 连续行段 | 实读含义与特殊设计 |
|---|---|
|1–22|Brcb包装；仅CCE220且非CPU Debug时可直接`vbrcb`复用64位正常mask，其他编译目标回公共Brcb。内建指令不是跨架构稳定接口；需要保留条件。|
|23–118|预载edges/pseudo、整块或分4面板权重（事件4/5/6/7）；从最小tiling字段或key还原resident/full/half metadata。固定UB视图跨函数别名，依赖相同布局与事件配对。|
|120–341|General single：数值worker与清零worker按physical block/subblock分工；fullFeatures与单行X两种DMA；weights和X各有ready事件。四basis线性插值→feature broadcast→Mul→树归约；fixedTinyTail以VA地址表、逆3bit顺序和转置后BlockReduceSum收尾。单独写target与清零排除target使store互斥。|
|343–540|Aligned single：N≤32且N%4==0等静态资源前提。421–450权重/X预载与mask复用，454–495权重到齐后先插值、X晚到再Brcb，499–517转置/BlockReduceSum，519–534普通折半归约，535–538输出。sourceList与destinationList分别16个VA地址：不是GM transpose，也不是改变矩阵数学含义。|
|542–556|single包装按完整静态条件选Aligned，否则General；不能将Aligned去掉guard用于所有N。|
|558–719|EdgeOwnerLate：固定8块，core对应edge，最低edge索引拥有重复target；扫描构建pending和占用bitmap，连续ZeroRuns只写非target。多edge按真实source/pseudo计算；mean度数可在扫描计数或popcount(pending)延后求。EARLY_WEIGHT决定W是否由caller预装，不能独立调用不发DMA。|
|721–906|NumericResident：前4块计算，整x驻留而非逐source DMA；四个target比较构造matching，较低owner独占，首边提前插值。等待事件4来自调用者，而非本类；实际入口只以EARLY_WEIGHT=true实例化并先SetFlag4，故不能据类模板默认false直接判实际死锁。UNLIKELY_DUPLICATES只给编译器分支倾向，真实重复边仍计算。|
|908–951|HalfOriginalSeparateZero：后4块各清16行窗口；读取四target排除真实输出，rows裁尾。它的正确性依赖caller只在N≤64使用，不是普适N清零器。|
|958–1164|EdgeOwner：权重插值和feature Mul先排队，再做target/zero扫描。即使后发现本块不是最低owner，投机计算可丢弃但DMA/事件仍需消费。STREAM_WEIGHTS的4个独立ready依次等待，使每面板一到就能消费；第二边起不再等待权重事件。最终32元素归约可写到expanded+8而保持64元素mask，避免频繁设小mask；低16个结果才输出。|
|1166–1395|CompactOwner：Index用u64，但NodeIndex对合法int64节点只取低32位（FULL_LANE_TERMINAL直接读取words[2*i]）；题面N≤65536时索引值范围可证明不丢有效位。ALIGNED_ROWS将窗口宽度向8节点对齐并裁尾；FULL_LANE_TERMINAL把accumulated扩成64元素并多留间隔，末端仍只写16元素，不能保留小buffer后抄全mask。zeroCount向64元素取整，避免短Duplicate未覆盖。|
|1397–1662|NativeConsumerReadiness：static_assert锁定Ci128、stream、full-lane。首边快路把标量目标读取塞在两次面板消费间；重复target的AdditionalContributions为noinline冷路径，重新构造同物理UB视图并复用已完成的W/X/accumulated，再按degree mean。行1447–1451清零repeat使用rows/4，实际静态vector-tile路由需保证每块rows是4倍数；后续移植必须保留Host/key条件。末端省barrier/mask改动要以目标编译器hazard规则核验，不从数学推导代替同步证明。|
|1664–1807|Resident通用小图：cyclic owner用位与/位移代替除余（owners必须power2），全输出UB归零，pending最多64边，先发所有source DMA再计算；counts先V_S再Scalar读改，重复target时再S_V给向量。mean有scalar按行倒数乘或Cast→Max1→Brcb→Div两版。输出stride散写cyclic节点。|
|1809–1904|General fallback：TPipe分配每块next、node头和矩阵/向量/三层补偿缓冲；按nodeTile和Co128任务遍历，邻边求和后mean，再root与bias，最后BF16 Cast。与rank1先聚合再乘W不同：本实现一般路径按每条边、每basis先点积，算量/累加顺序不同。|
|1906–1958|每1024目标构建逆序邻接，next按block cacheline对齐；LoadVector精确字节DataCopyPad，LoadMatrix逐行原Co stride进入补齐列距，再BF16→FP32。co未满列padding值可能未定义，但后续只消费outCount列，不要求未用padding是0。|
|1960–2056|Kahan式Accumulate分dot/basis/neighborhood三级；dot逐Ci标量系数乘整Co向量。degree1/2/3多项式、负数floor修正、混合基数slot和模ks wrapping逐项保留。是否全域满足误差容限仍需输入尺度/舍入误差或设备证据；“用了补偿”不是自动证明。|
|2058–2082|全部成员与StaticSplineNodes编码解释；node offset和log两套表示不能混读。|
|2084–2208|入口依次single、FULL、HALF、resident、general。FULL先预载四权重事件；HALF只有core<4才预载numeric事件4，其余clear。Native/Compact/late-owner由PADDED_ROWS和NODE_LOG编译期选择；一般resident特化仍有少量运行时E判断。root/bias仅general可达，因为Host明确排除resident。|
|2210–2224|base与full两个global函数共享RunSplineConv；REGISTER_TILING_DEFAULT与模板参数保持一致。|
|2227–2253|先include kernel_common再宏重写SetSysWorkspaceForce；对single/half私有UB路径跳过系统workspace初始化，instrumented/debug条件保留原调用。收益来源是生成wrapper路径而非数学，依赖CANN9生成TEMPLATE_PARAMS的时机；未编译本环境，不确认可迁移性。|

### F3. 可迁移与不可直接迁移

可借鉴的是：资源条件驱动编译期布局、真实输出分区证明替代全局同步、权重面板就绪即消费、冷重复分支outline、只保留有效输出的全lane终结、均值计数晚做。它们都需要完整前提。裸UB地址、VA转置、vbrcb内建、workspace wrapper宏、注册器类复制属于版本强绑定技巧，单独搬用不保证正确或更快。

这一轮是全文语义审读，不是已完成Host×key全部组合自动枚举验证，也不是NPU数值/同步证明。顺序3续于G节。

## G. 顺序3完整首轮审读（submission 1177413）

Host502行、tiling139行、key987行、kernel3593行均按连续带行号片段全文阅读；三份CMake59行与前两版同SHA复用。不是用下面摘要代替未读正文。

### G1. Host连续行段与真正的输入域

| 行段 | 语义与关键边界 |
|---|---|
|1–32|输入SDK/共享头、8个launch worker的分组函数、32B UB对齐、64KiB BF16存储区。注释声称8核本地更快，本轮没有对应日志，不当实测结论。GroupedLaunch若补齐8超过硬件就保留requested。|
|34–71|RuntimeSparse与GeneralSingle两套峰值公式，包含zero全4096元素、support scratch、独立messageSum、BF16 raw区和8448B余量。不是只用M*4估所有资源。|
|73–152|metadata验证：StorageShape、dim2/3、degree1..3、ks2..4且open时ks≥degree+1、N1..65536、E0..262144、Ci/Co1..128、E*S*Ci*Co≤2³¹；dtype FP32/BF16与int64 edge。比PDF N≥16/CiCo≥8/E≥64更宽。aggr接受mean/add，不接受其他字符串。没有读取输入数值。|
|155–174|通用策略directOutput恒1，precomputeFeatures/contractKnots恒0；权重缓存按真实UB和额外8192/2048元素预算约束，worker=min(AIV,N)。未显式检查AIV=0，仅硬件通常>0的前提。|
|176–202|Dispatch与两套packed metadata：node17bit容N65536；edge16bit、zeroRows12bit、mean/coowned/CI；single复用其他位存ks/scale，不能混读。|
|203–284|小图家族：bilinear dim2 degree1 open2×2；runtimeSparse要求E<AIV、E≤64、Ci/Co power2、M≤4096且UB够。coowned还要求E≤8、每owner≤64row、输出row≥32B。静态E1/2/4/8选MODE6；E8必须coowned。N不做exact shape识别。|
|285–328|smallSingle与boundedSingle都选MODE2；generalSingle要求degree1、E1、power2 Ci/Co、M≤1024、knots*M≤16384、完整输出≤4096。3D open2×2×2进一步选MODE7/8，按N*Ci是否够128元素prefix区分；runtime sparse可整X缓存，但额外字节至多一个Ci×Co矩阵。E3/5/6/7静态资源足够时仍MODE6/ES0、实际E来自metadata。|
|329–375|未对齐BF16单行保持单owner；未改成cached/static的MODE4回退generic，不编译MODE4。小FP32整X32B块且≤512B才MODE12/14，按64B行余数选择coordinate transport；MODE15只FP32、E8、M1024..2048、Co16/32等完整资源条件。|
|378–402|generic按连续输出rowtile算容量；若owner多于rowCapacity且E≥256，构造私有边bucket避免每个小rowtile反复扫描全E。容量不足降row1，不伪造合法输出。|
|404–449|初始化全部tiling；schedule0；私有bucket才分E向8entry对齐×8B×owner workspace。MODE5模板DIM固定2只是去重复entry，真实3D存在packed位，device再解码。NN仅输入DMA完整extent或BF16 hint上界，dtype与input0绑定。|
|452–502|shape推导X行/W最后维、dtype跟X；**仅四输入注册，无root_weight/bias，也无rank1/2那种SplineConvFull接口**。支持910b/910_93，不能由支持两设备推断底层C API在两者所有版本都相同。|

重要域对齐：所给PDF要求E≥64，而这里E<AIV的快图路径在20AIC/40AIV等平台不覆盖该PDF一般域。E1/4/8专路可能来自后续/实际评测版本，不能拿它们解释严格这份PDF大图性能，也不能由域差异直接判作者违规。

### G2. Key987行不是“987个kernel”

1–17定义8个参数及实际枚举顺序：MODE列表并非数值递增（7/13等在末尾），KEY编码不能按模式字面值自行移位。19–40 generic MODE1的DIM2/3×DEG1/2/3×双dtype；41–128 MODE2不同Ci/Co小矩阵；129–238 MODE3稀疏；239–348 MODE5一般single；349–491 MODE6 E静态族（部分FP32 E8条目由MODE15取代，BF16 Co8无独立row路径）；492–601 MODE8三线性；602–799 MODE9/10带整X缓存；800–887 MODE7 bounded三线性（无Ci128，因为prefixRows1时合法N不可能小于1）；888–920 MODE2 BF16 NN1..8 hints；921–953 MODE14/12 FP32 NN偶/奇块；954–987 MODE15四个Ci/Co组合。

MODE0/4/11/13虽然DECL保留，SEL不实例化；Host所有最终赋值须以SEL笛卡尔积为准。尤其boundedSingle在Host实际仍发MODE2，不能因常量名11存在就声称漏编11。每个SEL块都读过参数值、dtype、AIV_ONLY；注释462entries需由展开统计复核，不仅引用注释（见静态表核验产物）。没有推定这些entry都用于公示计分case。

### G3. Kernel完整连续语义账

| 连续行段 | 实读行为、写法与数学/资源/同步边界 |
|---|---|
|1–36|public operator/C API include、事件0定向fence、32B DirectUbAlloc和generic所有成员，裸UB单次分配而非TPipe队列。|
|37–91|LoadWeight逐行补齐；缓存时Co对齐连续搬，否则4095row分批避免blockCount12bit上限；BF16 raw→float。作者注释A2限制未独立对所有SDK核验，R级；只缓存实际矩阵，无Host输入值。|
|92–166|degree1/2/3多项式、BF16高16位Scalar扩展、LoadX与完整mask复用support插值helper。AccumulateKnotSupport在当前文件没有外部调用，是保留方法，不当现路线贡献。|
|167–265|Gemv按Co64分片broadcast/Mul，奇Ci用ceil折半保持孤行；Message按每support先xW再basis累加，不先合W。理由是FP32凸组合近max_float的中间溢出；BF16与其他专路另有顺序，不能称全实现同序。加载W lazy ready可跨message复用。|
|266–290|cyclic owner全扫E，无边0、一边直接product、多边acc，mean最后；每row exact-byte输出，写完MTE3_V才复用。|
|291–358|BuildOwnedEdgeList以256目标批次筛本owner，打包dst<<32|edgeIndex到核私有GM，每256entry32B对齐写，尾精确字节；MTE3_MTE2保证本核后续读。AccumulateOwnedEdge的counts由Scalar改，rows由V改；初始化V_S与消费S/V依赖需分别核。FP32多row输出按UB32B补齐到紧密GM行；BF16逐row转存。|
|359–459|连续owner分域+rowtile；可私有bucket只扫本owner边，或每tile全扫E；counts清零后V_S，最后StoreOwnedRows；闲launch核在任何DMA/UB/workspace访问前退出。AllocateDirectBuffers与Host预算同类型区域，扩展对齐counts最多多7元素，Host额外64B/16KiB余量覆盖意图明确。|
|461–597|FastAlloc每次翻512B内256B phase并向512块上取整；Operand/Out另加32/256B；C API copy/math/Brcb/Scalar UB读写/Cast wrappers；mask-overload明确设normal+hi/lo，count-overload由API处理。FastReduce递归ceil树兼容CI24，不错误丢中间行。|
|598–631|ContractCanonicalBilinear先四W凸组合，再下半Mul/上半FMA，递归归约CI/2；改变舍入且依赖有效range，不能当精确FP32恒等。|
|633–697|四个独立support先各自乘X，前两树层连续layout，第三层同时改为support交织布局并复用已死broadcast，无额外transpose buffer；最后才乘basis。每层四独立store仅一PIPE_V。|
|699–837|PackedFirstPair直接产[chunk,4support,64]，先下/上CI两半Mul/FMA，跨chunk全mask树，再chunk内小mask树；MessageToOutput最后直接写真实output row，省一次Adds。半矩阵恰64元素时还复用最后CO mask。4support*chunkCount容量≤原4M scratch，不能用缩小scratch套同stride。|
|839–1005|EvaluateSupportMessages按M和scratchlimit选1/2/4support同批；Canonical+AllowWeightContraction走先合W；其他按support-index查真实W，Ci归约后按原support序累basis。BuildBilinear/IndexedBilinear/Trilinear区分canonical与实际wrapping索引；3D系数8项，实数下端点零系数可换位，FP32 0*Inf仍需有限range条件。|
|1006–1115|LocalPseudo升BF16；StageFeature按32B对齐选copy/pad；CopyControls读真实2E端点和2E坐标；FastStore的dtype/cast/tail；ZeroSingleEdgeComplement只写真实target前后，无需跨核同步，unaligned BF16不能独立清row。|
|1117–1181|SmallSingleAlloc；actual-row cache preload；ReadActualFeatureWords每8word按2/4/8读取，**若剩6只读4**，但该helper全文无调用，不当当前路径bug。若将来用于CI24 BF16就要补全6word消费证明；当前BF16采用下述全wordCount helper。|
|1183–1377|BlendSmallEffectiveKernel、FinishScalarRowGemv真实wordCount全读并逐channel位解码，不留跨调用缓存；先Ci半Mul/半Axpy→树；SmallSupportFirst以DMA或Scalar输入，四support并排归约、basis最后。prepare/finish拆开，把与坐标无关的点积提前，只有最终basis依赖pseudo。HintBoundedInputSpan是cache hint，不是读回答案或读假feature。|
|1379–1569|MODE2 RuntimeSingle：aligned row worker0数值/worker1清零，其余退出；不aligned BF16单核全输出。FP32小family四stride DMA变[CI,4,CO]，其他整W；BF16 raw在64KiB区。ScalarSupportFamily与packedSmallFeature编译条件不同，冷/旧else即使存在也不实际触发。CI24 BF16最终FinishScalarRowGemv读足6word，不能误报上段未调用helper缺陷。|
|1570–1630|MODE14完整≤512B X由NN确定，单核清wholeOutput，动态source选cache真实行，4W一次DMA，PackedMessageToOutput直接结果到target，全部output一次store。不依赖输入地址跨调用不变。|
|1632–1774|MODE12输入partial64B但完整32B块，Co/16个message owner各写完整64B输出slice，另一个zero owner排除target。partial-cache坐标走8B DMA+MTE2_S，NN偶时可Scalar；4W各多32B UB gap改变bank，actual源矩阵无gap；算所有CI后按16输出片合basis，输出owner严格不重叠。cachedcoordinate薄包装不改数学。|
|1776–1848|运行时Ci递归reduce保留但无外部调用；RuntimeBilinear实际仍由模板CI计算；cached X的float只换本地视图，BF16只Cast选中行；ready只控制W首次Cast/事件，不代表跨调用状态。|
|1849–1958|每support事件1..4，feature事件0；每块ready仅消费一次，非leader也Drain；FinishStreamedGemv合W后Mul/FMA再树。StoreSparseZeroComplement用最高occupied bit逆序写补集，Published等zero专属V_MTE3 ID1，而非等整个计算链。|
|1960–2024|Paired两message×四support交织chunk，一层8support共用reduce指令，两个message仍按自己的u/v合basis。实数等价但FMA/树顺序变；必须保证第二feature所在slot不会被slot0写覆盖。|
|2026–2241|RuntimeSparse分Coowned/Separate、CacheFeatures/staticE；8edge BF16较大矩阵可分4W事件流水；完整controls在独立0/1024/1536B区。独立zero owner排序target并写补集，多出的launch核亦可分zero tiles；数值owner先stage本edge源再扫描去重。|
|2242–2366|最小edge索引leader聚重复target；coowned最多8edge，bitmap只覆盖≤64row区间。非leader消费已发feature/weight token后退出。leader一条直接输出，多条按原序stage+计算+累加，mean最后。cached行切换无新MTE2但V消费顺序仍保留。|
|2367–2664|PairedIncoming是完整8edge/FP32模板，重用与RuntimeSparse相似代码，但常量true/false使若干else不可达；产品/broadcast gap总量保持288B，只调整相对32B phase。不能把注释声称measured当本轮设备证据。|
|2665–2774|重复target成对发源row：第二真实X暂放result+256，slot0结果只写每512chunk的前256，不覆盖第二X；先Brcb消费第二X再slot1覆盖。两条共享channel树后依序加入accumulator，奇数最后走单条；末端mean。后续一般重复循环因前面incoming>1 return及incoming==1 return在合法图不达，代码仍审过。|
|2775–3037|StaticEdgeGraph用于ES2/4或ES0 runtimeE1/3/5/6/7，message owner与单zero owner；全部realcontrols先搬。每条权重事件/首feature早发；重复target非leader先drain再退出；leader多条双slot源输入，最后没有悬空token。E2重复时count恰2，meanScale0.5精确；不是把任意degree都替0.5。|
|3038–3289|zero owner N≤64用bitmap逆序补集，否则取8个target（无效槽n）经2/4/8比较交换网络排序，再逐gap写0；重复target以cursor忽略。ES≤8进入这个完整域，后面更大E插入排序为保留扩展，不由当前MODE6 smallE发。排序网络已按每对读过，另作独立排列核验。|
|3290–3397|UnitCubeSingleEdge装8W和3pseudo；N*CI不足prefix128时只DMA实际row，否则128prefix且source超prefixRows时排空旧DMA后重载实际row。BF16 raw→float，8support点积后basis；无root/bias。|
|3398–3522|GeneralSingle的真实dim/ks/scale从packed解，先4W，若K>4追加其余；3D Z单独token；wrap对合法q∈[0,1]最多减一次ks足够，不能泛化到任意坐标。Co对齐时数值/clear两owner，不对齐则完整输出单owner。|
|3523–3568|真实global分派MODE7/8/5/12/14/15/2/3/4/9/10/6；MODE6 ES8转RuntimeSparse coowned，其余StaticEdgeGraph。default仅编译generic MODE1，REGISTER_TILING_DEFAULT一致。|
|3570–3593|末尾宏把generated wrapper系统workspace设为空、user指针原样；原Host私有bucket按raw user分配，debug/dump/timestamp条件恢复原API可能改变指针前提，不能无编译核验宣称全部模式等价。auto_gen entry仅arch2201宏存在时以cce_kernel/aligned512重新声明，依赖生成器命名/签名，是版本技巧不是通用算法。|

### G4. 与前两版的本质区别与可迁移范围

1. **P/F算法选择不同。** 顺序1先按kernel键聚合feature再乘W；顺序2通用每edge每support点积且三级补偿；顺序3通用每edge点积再basis，但无三级补偿，采用rowtile与私有边bucket省metadata反复扫描。不能把这三者统称“同算法只调tile”。
2. **F/G把计算布局作为输出。** 四support乘积直接形成后续归约需要的交织布局，复用已死broadcast，不再先产生一种布局再转置；可迁移到多分支点积/独立归约，前提是所有support独立、别名活跃期和stride证明完整。
3. **G最晚才等坐标。** feature×W与pseudo无关，prepare阶段提前，finish才乘basis；这是依赖DAG重排，不是去掉必要输入。可用于“共享昂贵主干+晚到小参数”的一类kernel。
4. **G精确输出区域独占。** 整target、target内64B切片、zero补集三种owner粒度；不aligned BF16退回整输出单owner。可借鉴其证明方法，不抄固定8核/16列数值。
5. **R而非通用推荐。** C API wrappers、mask复用、aligned512、workspace宏、cache hint/bank gap都需要同版本官方契约及产物/对照测量。注释中的ABBA、qualified/probe是作者记录，不是本轮独立复现。

### G5. 已确定的契约差异与未闭合证明

- 四输入接口未实现题面optional root/bias；若实际评测ABI是旧四输入，则不能据当前PDF泛称完整支持。此问题与三份源码中E1等扩展域共同提示必须保存版本关系。
- FP32 fast路保留每support点积再basis，BF16某些资源族允许先凸组合W；单靠实数分配律不保证任意FP32幅值/强相消都在容差。FMA第一层也不等于两个单独Mul后Add的舍入。
- ReadActualFeatureWords的6word尾不足属于**未调用helper的迁移陷阱**，不是当前路径已运行错误；实际BF16 CI24使用全wordCount helper。避免把死代码风险混入当前执行结论。
- 当前未对全部合法连续Ci/Co/N/E/ks组合做完整Host×SEL自动枚举；已读精确列表与主要资源族对应，不将目测当完整枚举证明。C API blockCount/mask/stride/alias合法性仍须锁SDK核验。
- row-bucket顺序按E升序，顺序1逆链顺序相反；静态tile也有FMA与树加次序差异。没有设备数值/时延实测，三份谁在何种shape更快不能由名次外推。
