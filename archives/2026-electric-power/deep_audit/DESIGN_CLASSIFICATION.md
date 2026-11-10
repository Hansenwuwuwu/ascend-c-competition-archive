# 十题获奖源码：完整审计与设计适用范围分类

日期2026-11-09。此文是全库源码审计后的统一分类总报告；逐题、逐提交、逐连续行段的语义账见各题报告。9题共有的27份公示快照提交、271个源码文件、139,460物理行均完成首轮连续源码审阅，原档哈希与覆盖范围经 `check_coverage.py` 核验。第5题“电力巡检算子 SwiGLU”不在该公示快照中；后续收到的三份本地补充代码另行审阅，未并入这里的覆盖统计。

**审计覆盖表**

|题目|覆盖行数|文件数|逐题报告|首轮源码审阅状态|
|---|---:|---:|---|---|
|01 安全带几何|12,172/12,172|21/21|[题1设计复盘](problems/01_SafetyBeltViolationGeometry_REVIEW.md) / [逐行证据账本](problems/01_SafetyBeltViolationGeometry.md)|完成|
|02 pinv|19,921/19,921|41/41|[题2设计复盘](problems/02_pinv_REVIEW.md) / [逐行证据账本](problems/02_pinv.md)|完成|
|03 SplineConv|10,607/10,607|40/40|[题3设计复盘](problems/03_SplineConv_REVIEW.md) / [逐行证据账本](problems/03_SplineConv.md)|完成|
|04 det|9,262/9,262|27/27|[题4设计复盘](problems/04_det_REVIEW.md) / [逐行证据账本](problems/04_det.md)|完成|
|05 电力巡检算子 SwiGLU|公示快照不含；另有3份本地补充|—|[前三名设计复盘](problems/05_SwiGLU_REVIEW.md) / [完整静态源码审计](problems/05_SwiGLU_TOP3_LOCAL_SUPPLEMENT_20261109.md)|补充材料独立审计，不计入27份快照覆盖|
|06 lstsq|12,371/12,371|24/24|[题6设计复盘](problems/06_lstsq_REVIEW.md) / [逐行证据账本](problems/06_lstsq.md)|完成|
|07 FireCandidateDecodeFusion|18,301/18,301|44/44|[题7设计复盘](problems/07_FireCandidateDecodeFusion_REVIEW.md) / [逐行证据账本](problems/07_FireCandidateDecodeFusion.md)|完成|
|08 eigh|25,114/25,114|26/26|[题8设计复盘](problems/08_eigh_REVIEW.md) / [逐行证据账本](problems/08_eigh.md)|完成|
|09 solve|8,984/8,984|21/21|[题9设计复盘](problems/09_solve_REVIEW.md) / [逐行证据账本](problems/09_solve.md)|完成|
|10 TreeLineMinDistance|22,728/22,728|27/27|[题10设计复盘](problems/10_TreeLineMinDistance_REVIEW.md) / [逐行证据账本](problems/10_TreeLineMinDistance.md)|完成|
|合计（有源码的9题）|139,460/139,460|271/271|—|27/27份提交完成|

这些提交已在本届官方 Judge 上评测并进入获奖公示；官方通过情况、分数和公开耗时是提交整体表现的记录。此处的源码审阅和设计复盘用于解释实现路线、提炼可借鉴的方法，不要求在本地重跑获奖代码。对单个设计的性能因果关系不从总成绩推断。

## 1. 两条独立分类轴，不能混在一起

**应用范围轴：**

- **题目专属 P**：依赖该题公式、附加数据保证或输出定义。离开此前提，就算代码可编译也不能迁移。
- **问题族 F**：例如广播逐元素、稀疏消息聚合、pairwise归约、小稠密线性系统、谱分解。可迁移的是算法/布局结构，不是所有常量和guard。
- **编排原则 G**：输出独占、只等真正依赖、编译期专化、冷热路径隔离、按活跃期复用缓冲。可用于多种算子，但不是“每个kernel都适合”。

**复用前提：**记录方法迁移时必须保留的数学/数值条件、shape/资源条件或实现特征。它们是面向其他题目和未来比赛的使用说明；本届比赛已经固定的评测环境不是获奖实现的未决风险。每条设计允许多标签：如“本题恒等式P，是线性聚合重排F的实例，使用了Ascend特定实现”。

**复盘标记：**S=源码审读确认的实现事实；M=解释设计所依据的数学关系及其条件；R=将思路用于其他题目或比赛时要满足的复用前提；E=本届官方评测结果，包括 Pass、分数和公开耗时。R 不是对本届获奖实现的疑点，E 不要求本地重测原提交。

官方成绩用于描述整份提交的结果；若要说明某个技巧单独带来多少性能收益，则需要相应的对照实验。报告重点是讲清楚获奖代码“做了什么”和“什么条件下值得借鉴”，不把总排名归因给某个局部写法。

## 2. 按题归类：已读证据中的设计

### 01 安全带几何判定

详见[题1设计复盘](problems/01_SafetyBeltViolationGeometry_REVIEW.md)；逐行来源见[证据账本](problems/01_SafetyBeltViolationGeometry.md)，特别 rank3-kernel 与 math-memory-sync。

| 设计 | 范围 | 可借鉴什么 | 必须保留/不能照搬 |
|---|---|---|---|
|FP32几何中间值后strict GT生成0/1|P：本题二元判定|将“连续值误差”与“离散输出一致”分开审核|即便差1ulp也可翻转边界，不可拿普通allclose接受FMA变化。|
|小N完整静态路径、实际坐标仍来自GM|F小逐元素、G编译期专化|把真实维度/搬运计数/stride进入常量传播|形状专化不等于预存答案；过多模板有代码体积代价；必须有合法域fallback。|
|DataCopyPad补齐块后stride0广播阈值|F标量广播；实现依Ascend API|单次标量载入供向量重复使用|正确性来自该版本padding值与广播规则，不能认定所有“未显式Duplicate”的UB块均已初始化。|
|launch多核但仅一个owner计算|G调度候选，平台相关|区分实际计算资源和launch配置|是否省时需目标机测；不是普适“多开几个核必快”。|

### 02 pinv

详见[题2设计复盘](problems/02_pinv_REVIEW.md)；逐行来源见[证据账本](problems/02_pinv.md)的 r2-math、math-admission、r1-cube。

| 设计 | 范围 | 成立逻辑 | 迁移边界 |
|---|---|---|---|
|满足门限时先尝试普通逆，否则SVD/Jacobi|P附加谱间隔，F快候选+证书+fallback|本题被截断σ≤0.1τ；若有此σ，则τ²||A⁻¹||F²≥100，故小于25之类门限在精确算术且逆可靠时可排除此类值|不能在无谱间隔的一般pinv上抄同一门限；近似逆的误差还需证书。|
|R=I-AX，X←X+XR的逆Newton修正|F满秩逆/线性系统|精确算术Rnew=R²，ρ<1时可用||X||ρ²/(1-ρ)约束误差|不能对截断伪逆强制普通逆修正；左右残差与乘法次序必须对应；浮点残差需足够准。|
|五digit INT8精确plane积再高低残差重建|F高精度矩阵乘/残差；实现依Cube|5×5交叉plane全部重建；n≤512、signed digit≤128时每dot≤8388608，INT32不溢出|不是“所有FP32都转INT8就快”；还要scale/量化残差、25项重建误差、消费端精度与服务同步逐路验证。源码已全读，但以上证明不随覆盖完成自动成立。|
|矩形单边Jacobi同时转[B;I]，最终重建截断伪逆|F谱分解/最小二乘|避免独立积累U，以变换后列和V重建|有限sweep不保证收敛；矩形证书不能简化成方阵inverse residual。|
|Cube永久服务+命令/应答+RAII stop|G异步生产消费；实现依混合核|每条返回路径必须结束服务，不能让AIC滞留等待|只审完service不等于全部caller已闭环；动态fallback也要同一停止协议。|

### 03 SplineConv

详见[题3设计复盘](problems/03_SplineConv_REVIEW.md)；逐行来源见[账本](problems/03_SplineConv.md) C/D/F/G。三份全文首轮已完成，数值/API证明另列待核。

| 设计 | 范围 | 成立逻辑 | 迁移边界 |
|---|---|---|---|
|按target和kernel-index先聚合X，再乘W|P样条消息公式；F线性稀疏聚合|Σ_eΣ_s basis·(xW)=Σ_k(Σ basis·x)W；减少重复点积|只有W在聚合键内不变、操作线性时成立；非线性不能跨过求和。activeK大/度低可能更慢；浮点重排另证。|
|degree1/ks2/open1化为四角双线性插值|P此属性子域|删除通用混合基数和basis循环|degree2/3、其他ks或closed规则不能套；要审Host前提，不只看helper。|
|最低edge索引拥有重复target；其余块只清非target|F稀疏scatter/消息聚合；G输出独占|计算和清零写集合互斥，省全局同步/atomic|重复edge要完整聚合，bitmap位宽与节点范围受限，不能让清零覆盖真实输出。|
|四W面板各有ready，到一个消费一个|F多输入加权组合；G依赖驱动流水|不必等最后面板到齐才启动整个插值链|保留事件配对和缓冲活跃期；带宽瓶颈时不保证有收益。|
|重复target冷路径noinline，复用caller物理UB|G冷热隔离；实现强绑定|常见独立target快路径保持短|fallback布局必须与caller一致，所有必要DMA已完成；裸地址函数不可独立摘取。|
|完整64lane计算，仅写低16有效结果|F小归约；实现依mask/指令|减少mask切换，有时以多算换少指令|必须扩充scratch且证明无越界/无跨lane污染；不能保持原16槽容量。|
|先算每support的XW，再等pseudo完成basis；对照先聚合X再乘W|F线性稀疏消息/G依赖调度|两种数学求和组织适用不同稀疏度，前者能把W/X计算覆盖pseudo搬运|两者浮点重排不同；不能让basis加权前的中间值溢出，或认为文件中存在就已Host派发。|
|按归约消费者布局直接产出support中间值，复用dead广播槽|F加权归约/G存储活跃期|省掉独立transpose和额外buffer|新的layout、full-lane写址、bank偏移和别名最后使用点都须证明；r3对应C API/arch条件。|

### 04 det

详见[题4设计复盘](problems/04_det_REVIEW.md)；逐行来源见[账本](problems/04_det.md) S6–S18。三份9262行全文首轮已完成，实际路由与未启用实现、契约反例、数值待证均已分开。

| 设计 | 范围 | 成立逻辑 | 迁移边界 |
|---|---|---|---|
|mantissa与exponent分离累乘det|F长连乘、行列式|最终有限不保证中间乘积不溢出，分离指数扩大可计算动态范围|不修复pivot错误；subnormal/指数恢复/符号必须处理。|
|2×2消元与最后4×4闭式/Schur补|P行列式小块；F块消元|利用小固定维度减少Scalar pivot往返|Schur左上块可能奇异，整矩阵却可逆；不可删risk/fallback，病态相消误差待证。|
|fast试算失败后全组一致重算|G乐观快路+可靠兜底|所有参与核共享同一个reject决定，避免不同路径barrier不配对|证书必须真检查输出误差相关量，固定pivot启发式不等于误差保证。|
|关键十字优先的right-looking look-ahead|F分块线代/G依赖调度，r1实际BlockDet|下一panel只需特定行列，先更新它们，非关键tail可并行继续|直到复用旧L/R缓冲才等所有tail完成；不能提前覆写仍被Cube读取的panel。|
|原A减历史因子乘积，只重建当前十字|F left-looking分块线代，r2 FastBlock|避免持续写整个trailing矩阵，旧history不可变可提前乘|历史重读/重复算量需权衡；此源码明确没有可靠全局精度fallback，不能照搬数值保护。|
|逻辑行置换+active bitset parity|F带主元消元，r2 ColumnDet等|省整行搬移，同时按剩余行次序保留det符号|不是仅找到最大pivot；后续所有列和最终变量/符号都须遵守同一排列。|
|A/B不同ready、下一pivot先更新、helper分块发布|G异步生产消费，r3多个头|先满足真实关键依赖，而非一律等整个stage|paired本core ready不代表all-helper A ready；需wave/slot版本和复用ack。|
|WriteGmByPassDCache/包装初始化宏|SDK/硬件绑定实现，不单列通用优秀法|可能省tiny输出/启动成本|官方版本支持不同，须锁CANN与生成wrapper；本轮未性能归因。|

### 05 电力巡检算子 SwiGLU

公示快照不含题5获奖代码，但之后收到的三份本地源码已有独立的[设计复盘](problems/05_SwiGLU_REVIEW.md)和[完整静态审计](problems/05_SwiGLU_TOP3_LOCAL_SUPPLEMENT_20261109.md)。不能凭我们历史实验、排名耗时或其他题同作者代码虚构获奖实现。通用对照维度包括：X投影两路的数据复用F（共享左操作数GEMM）、投影与激活逐tile交接F/G、tiny专核与大矩阵Cube分路G、half中间舍入与题面契约P。历史L/T版本的正向结果应保留具体parent、目标硬件、shape、是否重复测量，不能当获奖代码的性能因果证据。题面边界见[原始记录](problems/05_SwiGLU.md)。

### 06 lstsq

详见[题6设计复盘](problems/06_lstsq_REVIEW.md)；逐行证据见[账本](problems/06_lstsq.md) S4–S14。

| 设计 | 范围 | 成立逻辑 | 迁移边界 |
|---|---|---|---|
|m=n方阵改线性求解，避开通用QR|P满秩方阵子域；F退化问题简化|唯一解AX=B与满秩方阵最小二乘一致|m>n不成立；题面方阵residual应为空，不能无条件写若干0。|
|Householder QR后重读原A/B算高精残差并校正|F满秩最小二乘/迭代精化|把快主求解与高精验证分离，保留真实误差源|正规方程κ平方、尺度溢出、残差精度和一次校正充分性需另证。|
|列panel按核驻留，复用分解解窄RHS|F小稠密系统/窄RHS|避免每个RHS重新分解；panel交换小量metadata|核数/列容量/协作barrier守恒；不保证大K或多batch仍更优。|
|按活跃期复用原A、LU预取、residual缓冲|G缓冲分配/流水|减少UB占用，末两步停止无用预取|最后使用点和事件必须逐条核对；RHS溢出GM却被当驻留是已发现反例，不可迁移。|

### 07 FireCandidateDecodeFusion

详见[题7设计复盘](problems/07_FireCandidateDecodeFusion_REVIEW.md)；逐行证据见[账本](problems/07_FireCandidateDecodeFusion.md)。

| 设计 | 范围 | 成立逻辑 | 迁移边界 |
|---|---|---|---|
|先验按candidate分片驻留、跨多个batch复用|P priors跨batch共享；F广播参数算子|相同prior不必每batch重新GM读入|prior必须真共享；UB容量与raw/output复用生命周期要重新算。|
|2/D sigmoid和confidence的1/4补偿/denominator融合|P本题激活几何公式|利用2幂缩放并减少重复除法|所有消费分支要统一缩放契约；cached grouped-tail发现不匹配，不能只看主路径成立。|
|AoS↔SoA用规则Gather/native transpose，VA地址表可缓存|F固定字段解码、特征排布|把重复索引/地址构造从热循环移出|字节单位、dtype、尾packet和硬件VA编码必须对应；属于问题族结构+平台特化实现。|
|raw双槽，prior/output按生命周期共用UB|G流水/存储复用|只在最后使用后释放slot并允许下一DMA|Free/Drain/闲核退出全部配对；不是把DEPTH直接从1改2即可。|

### 08 eigh

详见[题8设计复盘](problems/08_eigh_REVIEW.md)；逐行来源见[账本](problems/08_eigh.md)，三版主kernel现均完成首轮全文审阅，实际路由与待证明项分开记录。

| 设计 | 范围 | 成立逻辑 | 迁移边界 |
|---|---|---|---|
|按UPLO单三角补全，先bitmask再转置相加|P eigh输入契约；F三角存储处理|另一半可以脏，不能用(A+Aᵀ)/2读进无关值|注意对角不能加两次；bitwise清NaN比乘0语义不同。|
|Householder rank2更新后将tridiagonal问题交给分治/ secular merge，并保留反射供Q回变换|F对称谱分解|把对称相似变换、三对角求解、特征向量合并与回变换分层|收敛/deflation阈值、聚类根、重构残差和全局正交性仍须误差证明；三提交使用不同solver/修复路由。|
|constexpr计划表：分核列范围/offset/容量|G静态资源规划|把不会因输入数值变的调度/地址移出device|表必须覆盖合法域；需要逐项核生成规则、Host可达域与实际消费者，不能由静态表本身推断一般n覆盖。|
|输出级Gram/residual证书、polar修正或高精fallback|F候选结果认证|把廉价候选解与输出误差检查及修复分离|R2中的代码路径不等同于所有R1/R3都具同一证书；证书对题面全部shape/极端条件数的充分性待核。|
|PIPE_S padding、kernel对齐、rsqrt魔数初值|平台/编译产物绑定或特定数值域|可作为实验候选，不作为普遍规律|插空指令可能只改变布局/时序；魔数+Newton需定义域误差；源码注释历史耗时不能当本轮实测。|
|RuleB列符号统一|P历史接口习惯，不是当前题面必要计算|说明必须区分题面版本|本PDF允许列变号；它不改善Gram/重构精度，不自动列为应复用优化。|

### 09 solve

详见[题9设计复盘](problems/09_solve_REVIEW.md)；逐行证据见[账本](problems/09_solve.md) S4–S12。

| 设计 | 范围 | 成立逻辑 | 迁移边界 |
|---|---|---|---|
|同A多RHS复用分解、窄K另选列式布局|F稠密线性系统|O(n³)因子不应按每RHS重复，窄K决定最佳并行轴|候选是否真复用以实际route为准；不能只凭类名。|
|原A/B残差+复用LU校正，与重做LU校正比较|F迭代精化|正确残差决定校正；保留因子节省重分解|粗pivot/无主元不是可逆就安全，必须保留数值guard/fallback。|
|Double-float/预拆分乘积减少残差误差|F高精度dot/精化|高低分量、TwoSum/拆积改善FP32残差地板|4097乘法可溢出；Scalar与Vector实现遗漏项不同，不能统称等效FP64。|
|小输出先写核私有cacheline，再exact-byte DMA发布|G多核邻近小输出|避免Scalar cacheline写回互相覆盖|输出划分、staging大小、DCCI和MTE事件是完整机制；只加对齐不够。|
|下个owner先消费关键panel；GM tag带wave并退避poll|G分布式ready协议，r2 Coop完整体|下一步无需等待所有非关键panel更新；tag仍检查数据真实就绪|60/92/115cycle是校准常数不是可靠性条件；退避不等于盲等后直接消费。|
|Coop多RHS共享因子，对照general每RHS重分解|F线性求解算法组织，r2 S10|k增大时重复O(n³)可能压过局部指令优化|必须按实际route判断；general残差普通FP32、n200有条件补偿、n256普通FP32，不能统一称高精。|

### 10 TreeLineMinDistance

详见[题10设计复盘](problems/10_TreeLineMinDistance_REVIEW.md)；逐行来源见[账本](problems/10_TreeLineMinDistance.md)：三个rank版本均已全文审阅。平方距离归约、几何投影、tie保原索引是P；段数据跨point任务驻留、规则offset倍增、分层候选压缩与稳定逆映射属于有前提的F；独立ready链、输出owner和buffer生命周期属于G。rank1 half Top2无不漏解证书，与rank3带歧义修复/精确回退的路径必须区分；volatile宽预读的逻辑张量越界保留为风险示例，不推荐迁移。

## 3. 跨题真正可积累的“方法族”

| 方法族 | 已读实例 | 以后如何使用 |
|---|---|---|
|把不变参数从重复维移出|Fire prior跨batch、Tree segment跨points、Spline W驻留|先画数据复用图，再估UB占用、重复GM字节和任务尾部负载。|
|按语义退化维度改变算法/并行轴|Tree M1/N1、lstsq方阵、pinv满秩候选|先证明子域等价，再配guard/fallback；不是只改tile。|
|数学交换求和减少主项算量|Spline先聚合再乘W|写出索引和交换条件，另做浮点误差分析；大创新通常在这里。|
|近似候选+高精证书+校正/兜底|pinv Newton、solve/lstsq精化、det risk fallback、eigh输出级证书、Tree几何候选剪枝后歧义修复|先定义证书并证明不会漏解，再决定廉价计算；Tree rank1固定Top2展示缺证书的风险，rank3则是另一种带修复/回退结构的实现。|
|把Scalar循环转成规则向量地址生成|Tree offset倍增、Fire ramp/transpose索引|确认索引规则能以CreateVecIndex/Add/shift生成，静态表必须对消费地址。|
|输出owner独占替代原子/全局同步|Spline clear与compute、Tree每point完整归约|证明写集合不相交、覆盖恰好完整，含tail与重复目标。|
|拆开不相关的ready链|Tree index先写、Spline面板就绪即消费、det关键十字优先/独立A与Bready|列事件DAG，分开data-ready和旧reader-done，仅等待真实依赖，最后检查资源复用和退出drain。|
|静态资源族+冷热分离|各题Host/key，Spline noinline冷重复分支|专化必须改变真实指令/寄存器/布局；不要把文件拆分本身当速度来源。|

## 4. 暂不进入推荐库的条目

越界预热读、没有误差证书的固定候选粗筛、按不完整合法shape集合直接返回失败、未启用/Host不可达的“优化”、无SDK版本证据的初始化宏重写。均保留原文证据供学习，但不会包装成通用技巧，也不据这些静态发现直接指控参赛者违规。当前源码、当时题面、实际计分二进制三者还需版本对应。

## 5. 后续每个设计的固定记录字段

`ID / 题目与提交 / 源码连续行段 / 原计算与新计算 / P-F-G范围 / 数学前提 / shape资源前提 / 硬件SDK条件 / UB与事件证明 / 数值风险 / 代价 / E级实测是否存在 / 迁移建议`。

一条机制可跨题出现，但要分别写适用条件；“出现于前三名代码”只算来源，不算优越性证明。首轮源码覆盖现已无空洞；coverage complete仍不等于数学、SDK/API、同步或性能问题关闭。后续工作应从R级待证明项和目标环境验证入手，而不是继续扩展未读源码清单。
