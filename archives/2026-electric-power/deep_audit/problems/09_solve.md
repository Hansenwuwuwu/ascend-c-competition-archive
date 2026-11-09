# 09 solve：逐行审读记录（27份范围内，本题三份首轮源码审读完成）

题面：[原始PDF](../../problems/09_solve.pdf)；[题目契约](../SPEC_CONTRACTS.md)。

设计的题目专属/问题族/通用编排分类见 [适用范围总表](../DESIGN_CLASSIFICATION.md) 第09节；高低位残差、迭代精化与SDK实现条件分别列出。

已连续全文读取三rank构建、Host、tiling/key及全部kernel：rank1 2344行、rank2 1874行、rank3 3988行。前文保留分阶段记录，最新覆盖以S10–S12及 `../coverage/09_solve.json` 为准。无编译/NPU执行，无公示二进制复现；不以名次给某个特化或内联技巧归因。

## S1 题面契约（含全部PDF页视觉核对）

已读09.txt两页并核对09-1/09-2.png。FP32 A=`[...,n,n]`，FP32 B矩阵`[...,n,k]`或向量`[...,n]`，输出与广播后的B同形；n1–512、k1–256、batch维大小1–64。特别允许A带batch而B只有(n,)或(n,k)，共享广播至各batch；A可逆且κ₂(A)≤1000。固定left=True，只解AX=B，不接受稀疏A。逐元素atol/rtol均1e-4，并要求 `||AX-B||_F≤1e-4||B||_F`。禁止完整求解库和CPU回退。B0时残差阈值为0，不能仅用绝对解误差判断。

AX=B实数下X=A⁻¹B，但显式求逆再乘与主元LU/QR的FP32顺序不等价；κ上界也不约束A/B绝对尺度。正规矩阵AᵀA的κ平方≤10⁶，应将初始近似、精确残差和校正路径分开证明。解唯一不等于每个无主元pivot安全；例如[[0,1],[1,0]]可逆、κ1、首对角0，必须有主元或可靠fallback。

## S2 构建、ABI与key全行覆盖

三rank CMake各按hash复用：根1–20为CMake3.16/ASC/默认910b/SHARED包，Host1–24生成注册与tiling/aclnn库，kernel1–15发现kernel并绑定tiling库。没有在这些文件中读到下载/CPU回退调用；这不替代实际生成wrapper与链接产物审查。

rank1 tiling1–8含7个u32：n,k,batches,broadcastB,cores,resident,grouped；key1–49的DT含FP32/BF16、TASK_MODE0–8及64–79、N_VALUE0/1/2/4/8/9–16/32/64/128/256/512。0纯AIV可BF16，2/3/64–79仅FP32纯AIV；1/4/5/6与7/8为MIX_AIV_1_0，绑定组合有限。Host从不发mode4（保留kernel分支，不是线上已用路径）。

rank2 tiling1–14九个u32：n,rows,k,batch,bBatch,cores,stride,parts,groups。SHAPE非0时n字段是packed metadata：bits0–8 n、9–14 batch、15 broadcast、16–21 cores、22–25 k、26起groups。key1–25：FP32 SHAPE1/2纯AIV，0/3/4混合AIV1_0；BF16只0。packed字段的尺寸前提依赖平台核数，见S8。

rank3 tiling1–18含base7u32（n,k,batches,rows,blocks,broadcastB,teams）、cubeWords[128]/halfWords[128]各512B、systemWorkspaceBytes。kOrthogonalWorkspaceFloats=4*256²+2*64²+4*256*32+256+3*64²+4096=319744 float，即1278976B；此处已重新逐项算术核对，所有Host使用前乘4；窄正交538368 float。key1–23：KERNEL_PATH0/1/27/28纯AIV，2/3/5/6/7 MIX_AIV_1_0，9/10/11/13/15/18–26 MIX_AIC_1_2；只有FP32。path4、8、12、14、16、17不在最终key集合，Host中的path4是后续覆盖前的中间状态，不能按变量曾赋4就宣称缺编译体。

## S3 rank1 Host全120行与路由

1–18 VectorRhs：br1直接vector；br2且B首维=n优先当matrix；否则br=ar−1、B尾=n、batch前缀相同才vector。这与rank3在batch=n时的优先级不同，见S8。19–38取StorageShape，n/k/batch乘积不做本地范围/溢出/方阵校验；broadcastB以B总元素==n*k判断；cores=min(batch,AIV)。na=pad8(n)、nk=pad8(k)、width=na+nk，先按n*width*4+32768+窄RHS残差extra与真实UB比较resident。grouped初值1。

39–60仅FP32,n≤256,k≤8走columnSolve；按scratch=((n≥224?36:28)*pad8(n)+8n)*4+4096估剩余UB容量，grouped=max(0,n+k−capacity)，resident=grouped==0。n>128且batch*4≤AIV选paired，根据ceil(n/16)和cores/batch取worker；n241–256,k≤3且每batch≥8核组成256容量family，最多10或8worker；其他n256最多8。paired resident=3、grouped0并schedule1，不供一般行式Kernel使用。

61–81小n按8/16核组上调；N_VALUE仅特定n绑定。task3为n≤16,k1，task2其他n≤64，paired n<224→1否则5，n256恰8worker→6；capacity family→7/8并N_VALUE256；task3 n9–15统一绑定15容量，真实n仍在tiling；task2 n%16==0取32/64，否则0。只有n%16==0或(n+k)%8==0且one8core wave、batch≤8时task=64+2(batch−1)+broadcast，将batch/broadcast绑定进key。

82–120发key/block；workspace按capacityFamily、paired、columnSolve、generic resident分别计算（均float元素乘4再加系统区）。generic非resident每核n*width；column每核pad8(n)*(n+k)；family每batch256*(512+40/41)。InferShape根据VectorRhs拼A batch轴+N/K，InferType继承A；注册FP32/BF16、left属性默认true，但Tiling没有读取left。题面固定true内不构成错误，不据此声称扩展支持false。

入口2245–2310真实route：mode64–79完全从TASK/N_VALUE构造n/k1/batch/core8/broadcast，不读取tiling；task3用KernelSmallFixed，N_VALUE15再区分真实n15与partial容量，其他n≤16静态，小未绑定用SmallOuter；task2→KernelSmallOuterT（n>32用第二bool=true）。task7/8当实际n256,k3调用固定256/K3/8或10worker，否则相同容量但PARTIAL_CAPACITY=true；task1/4/5/6→KernelColumnSolve对应FAMILY/N/worker；其余FP32 n≤256,k≤8仍ColumnSolve，一般才KernelLinalgSolve<DT_A,N_VALUE>。

## S4 rank1 kernel 1–210、2191–2344

| 连续行段 | 实读语义与资源/数值/事件 |
|---|---|
|1–27|依赖和一般行式KernelLinalgSolve；Fence取TPipe事件ID后set/wait。n可由N_VALUE静态绑定；width=na+nk；refine仅FP32 n>64,k≤8。非resident每核GM n*width，UB matrix驻留n*width或32行tile（grouped）或32B；pivot/row/temp各width，column pad8(n)，io=ceil16(max(n,k))+16个DT_A；refine另sol/rhs各n*nk及xt=k*na。|
|28–54|ReadInput exact byte DMA至io，FP32 Adds或BF16 Cast；read/write row按resident直接UB，或GM DMA后MTE2_V/S、写后MTE3_V/MTE3_MTE2，保证同地址下一次读不抢先。S_V在Scalar读改后保证vector使用顺序。|
|55–84|batch按core stride，无crosscorebarrier。每pass重新读A与B/残差，zero-fill宽度padding；column保存当前pivot列，Scalar maxabs部分主元，交换整增广行；forward消去对整width Axpy或32行分块。没有零pivot检测，但合法A可逆仅保证实数过程，浮点underflow/growth需另证。|
|85–110|反向回代先RHS乘倒数，第一次解存sol、第二次校正加sol；每行exact K*字节输出；只更新更前行的RHS，不重复更新A。pass0若refine调用ComputeResidual，再**从原A重做完整分解**求修正，因此省存LU但refinement重复O(n³)而非仅O(n²k)三角解。|
|111–150|ReadTile/WriteTile当resident返回UB偏移、否则按rows≤32搬运；ForwardGrouped遍历后缀32行并更新column，BackGrouped遍历前缀32行只更新nk宽RHS。tail=min(32,remaining)且width32B对齐，无尾行DMA超长。|
|151–164|HighPart按0xfffff000清float低12bits，AddResidualTerm用volatile TwoSum+comp；volatile防重排的意图不等同所有vector/Scalar同舍入已证。|
|165–200|xt转置sol为每RHS列，rhs先清n*nk；重新读原始A/B行，Vector乘p=fl(Ax)，Scalar将high/low四积误差补回，再TwoSum累计B−Σp；每项含al*xl，比只保留high乘法更细，仍非硬件FP64，尺度溢出不受κ界保护。|
|201–210|完整成员声明和类结束；210开始ColumnSolve模板但其体未读，不据此计完整算法。|
|2191–2244|KernelSmallFixed末段：通过正/负候选并联归约读64bit value/index，重建符号与行索引；active位图清已用行，取inv、zero pivot槽、Gather/Brcb扩展，再逐剩余列MulAddDst；最终Gather RHS乘inv、仅owner写真实logicalN字节，跨batch时等MTE3_S。起始/候选阈值和资源布局在未读区，不能证明全小核协议。|
|2245–2310|完整模板入口与route见S3，partial_capacity保留真实extent。|
|2311–2344|生成wrapper后缀适配：FP32 task2/3/64–79在非debug时跳SetSysWorkspaceForce、set_ffts_base_addr、icache_preload；四个调试宏时保留原函数。宏定义放最后，以TEMPLATE_PARAMS拦截**后续**生成调用；原函数体未受宏替换避免递归，preload固定1且忽略mode。绑定CANN9生成器，不是稳态可移植API。|

一般kernel UB例：n512,k256,nonresident grouped时matrix32*768*4=98304B，三行缓存9216B、column2048B、io2112B，共111680B，无显式L1/L0；resident则n*width替代tile。Host32768B预留比这些额外显式buffer通常大，但隐式stack/编译资源未核验。窄RHSrefine最多另2n*nk+k*na个float；Host额外还包含pad8(n)一列，未直接等同完整最终峰值。

## S5 rank2 Host1–146与入口

1–33读取平台AIV/UB（ub_size后未用于容量式）、必需Tensor/attrs，length_A为u32，n来自A尾维、k按B维度判vector；B rank1短路，B与A差一rank且倒数第二轴≠n时当vector。34–53检查n/k/方阵/left，但没有验证各batch维/B前缀/同dtype；batch=length_A/n²，bBatch=Bsize/(nk)，不检查整除余数。

40–46 FP32 n≥8,batch≤AIV内：n≤16,k1→SHAPE1，n≤64,k≤8→2，n200,k4→3，n256,k3→4，否则0。54–77：1/2单核/batch；3/4按8vector/batch、n16对齐、stride=pad8(padn/8)、groups=min(batch,AIV/8)且至少1；一般按parts2倍增拆列，满足padn*stride≤40960float及aux不与matrix冲突，再groups=min(batch*k,AIV/parts)。若真实平台核数不够parts则groups0没有拒绝，910b配置需验证，不编造实际发生。

78–93 SHAPE非0将上述字段打进n，rows恒0；workspace每组共享slot+矩阵因子，加batch*k*padn解，字节乘4并加入系统workspace。94–146 InferShape按vector/matrix拼batch，FP32/BF16继承A输出，left默认true、910b注册。较宽泛注册不等于BF16能达到题面FP32精度，题面本来也只考FP32。

入口1851–1874：REGISTER+COPY_TILING pointer；FP32 SHAPE1→TinySolver，2→BatchSolver，3→CoopGJ<true>（n200补偿残差），4→CoopGJ<false>（n256），其Init直接传原workspace；一般先InitSocState并Solver<DT_A>使用GetUserWorkspace。不能混淆不同workspace基址，Coop是否与系统区域冲突须同时核验其API使用，本轮不下确定错误结论。

## S6 rank2 kernel1–650、1790–1874连续语义

| 行段 | 实读说明 |
|---|---|
|1–34|K_MAX_SHAPE_DIM0、依赖、always_inline Fence固定event0；GatherNoMask/FastBrcb只在g_coreType AIV编译原生vgather/vbrcb，直接UB物理地址、不自设mask，caller负责mask和对齐。|
|35–74|FastSum先按64元素WholeReduceSum，再减尾并二级归约；FastMax产生value/index pairs后用0x5555选偶位值，最终index=(selectedPair/2)*64+局部index，尾用2*full偏移，不能把临时pair当密集标量。count≤64直接归约。|
|75–80|GlobalBarrier=PIPE_ALL+CrossCoreSetFlag<0,PIPE_MTE3>(2)+wait2；全参与核匹配次数必要，不能用它替代Scalar DCache一致性证明。|
|81–107|TinySolver解包n/broadcast、batch=BlockIdx，LocalTensor直接绑UB0共4096float=16KiB；n16精确DMA256A/16B，n8–15按行exact byte DMA到16pitch，B另768；不超真实GMextent。|
|108–142|两个16项地址列表定义16×16转置，rowIdx以u64一次写相邻两offset，TransDataTo5HD转置至列主；设mask16，n<16复制RHS列并清perm；volatile UB指针让Scalar加载与Vector事件显式配对，active低n位、swapped标记。|
|143–184|n16动态候选pivot：优先当前未用行/ctz(active)，先Gather未来列系数；归约并非每步都做，只有abs(candidate)<pThresh时扫描最大abs。pThresh初0x3e800000(0.25)，首列前2项max的指数域减3（约/8，受位阈值分支保护）调整；阈值策略不是全主元每步等价。候选改变则重Gather系数，避免旧pivot系数误用。|
|185–219|标swapped、inv写入pivot的原始行槽，清pivot值/active，按j奇偶双buffer cf；V_S0/1分别跟候选/未来列生产；先Axpy nextCol再MulAddDst后缀，rem1只做一个RHS更新，rem0不再set下一轮事件，入口预置两事件与16次wait相配。|
|220–281|n8–15每步准确abs最大归约而非阈值；读取64bit value/index，先更新nextCol并提前算下一pivot，其他列广播交错；rem1计算最后pivot并更新RHS，末步不再产生unused事件。|
|282–300|swapped时先按原row inv缩放，再Gather最终列序；无swapped可直接Mul。rows非0且batch0会把首解改1e30（诊断/故障注入式分支）；**Host82无条件rows=0，所以真实Host不可达**，不能当正常结果被篡改证据。n8倍数直接DMA，否则exact byte输出。|
|301–353|BatchSolver n≤64,k≤8同解包，UB0声明189KiB（不是实际活跃数组总和），raw从5184、rawB9280、out10944等float偏移；A最多4096floats恰到rawB，B最大512floats恰到row9800；k2读紧凑、k8整行、其余每行byte DMA补至8槽。非k2补padding行，inverse/矩阵必要清零。|
|354–410|构造36对byte偏移（stride72*4=288），16×16 transpose按16行strip循环，每strip源+4096B、目标+64B；k2用GatherMask奇/偶提取，其他transpose8列RHS；ResetMask后active位图，perm最多64*8=512项在预留UB内。|
|411–450|n64分支pThresh0.125，首列2项幅值调整阈值；候选不足再WholeReduceMax，pivot改动时重Gather未来系数，和tiny阈值机制同意图但容量64/stride72不同。|
|451–504|K2用u64打包两个RHS Gather偏移，其他k逐rhs写perm；inv按原pivot行存，双cf轮转；nextCol先Axpy、两事件流水，remaining>3时首后缀列先发再set事件让Scalar与其余列重叠；最后j不发下一轮V_S。|
|505–590|n<64每轮abs归约，直接取最大abs倒数再按原符号bit拼负inv；ctz未用；同样保持下列提前处理/其余列分开、rem≤2手工Axpy收尾。无校正路径，条件数上界并不证明所有FP32相消都≤误差阈值。|
|591–606|RHS先按原row乘inv，再Gather交织输出n*k；rows非0注入1e30分支同样Host不可达；32B整除直接写，其他exact DataCopyPad。|
|607–650|CoopGJ声明与Init开头，COMP_RES选n200,k4,pad208,totalPanels25或n256,k3,pad256,totalPanels32；8核part，rhsSlot=part&3、sub=part>>2，group=BlockIdx>>3；每核按round-robin8列panel存myPanels。原workspace基址shared，factors后移groups*slotSize，sol再移groups*n*padn；直接UB192KiB，fac在41792、后续512/1024float槽。Init后半与算法未读，故尚不能给全部alias峰值证书。|
|1790–1804|一般Solver.Run尾部按job=batch*RHS波，valid才重建indices/LoadB/LoadA，但Solve(valid)所有核调用；parts>1同步，防最后波无效核提前退出。|
|1805–1832|n>64算高精残差（体未读）及其范数，need=非0且不同时满足rn<1e-14和rn<bNorm*1e-10；每组part0写need，barrier后所有核读各组并OR；任一need则所有核Refine(need)，维持集合一致。阈值比题面残差1e-4更严格但不是解元素误差证书。|
|1833–1850|只valid part0读初解加校正；不need先把b置零，防残差当作校正；parts>1末尾barrier，完成所有wave再GlobalBarrier和Pack。Refine/Pack具体搬运缓存/临界区未读。|
|1851–1874|模板入口与InitSocState/workspace区分见S5。|

小核复杂度：列式全后缀Gauss-Jordan为O(n³+n²k)，UB O((n+k)*paddedN)，一核/batch无跨核同步；不是n小时完整调高阶库。阈值pivot减少全归约次数、交错独立两列/coeff链减少Scalar等待，有明确机制但无可归因速度量化。

## S7 rank3 Host1–198、kernel已读片段

Host1–41 Shapes检查n、方阵、k、A batch维与u32乘积，B只接受相同batch或无batch共享，vector先由matchingVector判定，输出严格与期望rank/dim匹配；42–59核对两输入FP32、left=true、AIV>0。60–79早返回：n9–16,k1→27；n33–64,k≤8→28，单核/batch并最少8核组，workspace仅系统区。80–118按n/k/驻留条件依次覆盖KERNEL_PATH：n≤64→1，窄RHS65–128→2、≤256→3；n129–256按10/13/16个AIV驻留→7/5/6，Cube可用时中间4再转9/10/11/13/15/19/20/22/23；n>256先18，≤384再21。n193–208窄k、小batch且≥16AIC→k≤4 path26否则25，blocks=batch*4、workspace每batch538368float。119–125 n256,k≤8,batch≤4且硬件满足→24、blocks16、专用正交workspace。

126–141给matrixPath/splitPath/normalPath分类，cubePath=path≥8；cubeN=128/256/384/512，不同n桶；baseCubeLimit=min(AIC,16)，估normalTiles比较ceil波数以决定是否扩大到所有AIC，不是按名次/输入内容；cubeGroups=min(batch,cubeLimit/2)。t.teams是vector团队容量/16或/32+1，cubePath仍计算但实际blockDim用cubeLimit。142–162 MatmulApiTiling构造FP32 A转置/B非转置，C FP32，M/N tile128、K32；splitPath附加残差matmul：normal路径INT8×INT8→INT32，其他FP16→FP32，OrgK normal为4*cubeN，BufferSpace128KiB；TCubeTiling复制入128word数组前static_assert检查第一份大小，第二ht同类型，ScheduleMode1。163–171按路径分配base workspace与split/normal额外区域，所有float式乘4再加系统区。172–198 shape/type推断/FP32注册/left默认true。

kernel1–25宏SOLVE_DIRECT_BENCH关闭key引入、32个512float scratch region布局、kPanel8与DMA阈值224、多个static_assert约束。26–54 SolvePair和SolveKernel成员、两套独立Vector lane、high/low/panel及GM对象；WriteRows先MTE3_S、整DCache clean/invalidate，再逐行Scalar读columnSolutions到UB ph、S_MTE3 exact k*4 DMA、MTE3_S隔开行。这是row-major发布，不能省去cache一致性步骤。

55–126 Scalar补偿基础：AddScalarPair用TwoSum/两次renormalize（无volatile的加法次序是否被编译器保留仍需产物）；SplitScalar4097中间volatile；Product逐乘volatile防FMA收缩、MulPair保留high/low交叉与lo²；DivPair一次余数修正；Root先Sqrt(high)再用(a−h²)/(2h)校正，h0分支返回0。127–184包装Vector Add/Sub/Mul/Scale后PIPE_V；AddVector与Scalar尾重归一序列不完全相同（见S8）；MulVector/Pre省ae*be，Scalar版本未省，不能把所有Pair运算统称相同精度；预拆a减少同一q的重复拆分。185–240定义NB无PIPE_V包装并双lane交错相同相位，s[0..]和s2分离、qh/ql/q1/q2只读以尝试重叠依赖链；完整调用方/潜在alias在未读中段，不能仅从注释证明安全。

kernel3771–3855为NarrowOrthogonal尾：按Packed和panel是否第二96阶段选择Schur Gram/重新Gram；Vector PadGram与decision guard，Cube按8/14次Newton或refreshed路径；前后VectorToCube/CubeToVector或SyncAll<false>；decision单cacheline clean后非0触发InverseFallback96或Inverse<128>并计算defect修正。之后Forward96/ApplyFactor得到XP，splitXP、INT8 packed4或all乘积、Assemble6、forward/factor再一次残差校正，Finish16/Finish写解。每个fallback/拆位权重体尚未读，不能由调用名推导数值界或L0容量。

入口3857–3988被SOLVE_DIRECT_BENCH保护：27→TinyColumns，28→SmallColumns；25/26→NarrowOrthogonal（26 Packed=true）；24→Orthogonal；23→DirectPanel128；22→NormalInverse256,true；18/21→NormalInverse512/384；20→NormalInverse256；19→DirectPanel208,256,true；9/10/11/13/15按N128/256且10选DirectPanel，其余CubePanel<N,matrixPath,halfResidual>。Cube路径从GetUserWorkspace减systemWorkspaceBytes回到system指针（依赖运行时这两个区域相邻且未变更），copy words重建TCubeTiling只在AIC。按resident=BlockNum/2逐波调整A/B/X起点，广播B不偏移、最后波batch截短，非最后波SyncAll<false>后工作区复用。其余5/6/7→RefinedSquare208/256/160,16；2/3→RefinedSquare128/256,32；1→SmallSquare16/32/56/64；0→SolveKernel。

## S8 可证边界、待证风险与版本契约

1. **Host存在shape语义分歧，不擅自判哪一个合法优先级。** A=[16,16,16],B=[16,16]既可表述16个向量，也可表述共享16×16矩阵。rank1第10行优先matrix→X[16,16,16]；rank3第16–19行matchingVector优先→X[16,16]。题面没有完全消除此维数重合歧义，需固定调用端的官方解释；不应仅用总元素决定含义。rank2也把B倒数第二维=n排除vector，偏向matrix。
2. **位打包有硬上限。** rank2取batch6bit，batch64时(64<<9)写到broadcast bit15；cores只留6bit，64会侵入k bit22。Host有batch≤AIV但无batch<64/cores<64的显式限制；若平台AIV≥64且题面batch64，即可产生重叠。已知运行平台实际AIV数本轮未读，故列为可构造平台条件风险，不宣称910b当前点已错。
3. **注释“bit-identical”不是证明。** rank3 AddScalarPair最后用完整TwoSum重新归一；AddVector141用简化 `(s0−hi)+e`，其准确性依赖FastTwoSum的幅值关系，非任意a.hi/b.hi/低项都相同。MulVectorScalar与MulVectorPre可能在同舍入操作顺序、指令模式与无溢出条件下匹配，但本轮未读全部caller、未比对CANN指令舍入/FTZ，故不认可无条件位等价。Scalar乘含lo²、Vector乘省lo²也必须区分。
4. **尺度风险与guard限度。** 4097*a的拆分可因大幅值溢出；κ≤1000并不限制整体缩放，不能借condition上限替代数据幅度约束。n16/64候选阈值非全最大pivot，只能解释“减少归约”的机制，不能宣称与部分主元每步等价或残差必过。
5. **诊断分支可达性必须回Host。** rank2 rows!=0会写1e30，但Host总rows0，当前route不触发；不能关键词命中就给“作弊/错误”结论。
6. **同步协议尚未审完。** rank2一般尾need通过所有组OR使Refine调用集合一致，这是正面机制；但Refine内部/Coop中段/pack与cache写回未读，不能宣称无退出死锁。rank3正交fallback的Cube service必须与每个Vector decision路径一致，待读完整。

官方CANN9.0 [SyncAll](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0204.html)区分true AIV-only和false混合，硬同步要求合适mix任务类型及batchmode；该条件不是某一wait存在就自动满足。 [GetReduceRepeatMaxMinSpr](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0226.html)指出A2获取准确index应配WholeReduceMax；原生vgather/vbrcb/GatherMask、CrossCoreSetFlag模式0、COPY_TILING、宏拦截wrapper和Scalar DCache策略的本地SDK契约尚未逐项核验。

## S9 rank比较、可迁移性与未读清单

rank1偏容量族+少数固定绑定，广义fallback是行式部分主元LU并可重做一次残差求解；rank2偏n200/256固定8核、n16/64阈值候选pivot与原生指令流水、一般按每RHS作job；rank3有最多路径、双float QR一般fallback与多种Cube近似逆/正交/INT8残差改正。后两者完整主干尚未读完，比较限定到已证Host/入口和已读数值基础，不从函数名虚构完整算法。

可迁移：按真实extent精确DMA、输出前私有cacheline staging/显式clean、所有组统一决定是否校正、尾部不再预取/发事件、独立lane交错需先证明无alias、通用fallback和小核专化分离。不能照抄：假设64核以下的位打包、歧义RHS推断、生成器宏重写、固定threshold作为数值证明、未锁SDK的内建/任务类型编号、只凭公示分数认定速度收益。

未读：rank1 211–2190；rank3 241–3770。rank2 651–1789现已补齐，详见S10；前述“体未读”等为分阶段记录，以下修订优先。当前不能签发“三版本完整审计完成”。

## S10 rank2 651–1789补齐：全文件首轮完成

### S10.1 连续语义账

| 行段 | 实读与核对结论 |
|---|---|
|651–698|fac末2×256float分unused/perm，fac2048起packIdx，colBuf在a35072。只part0构造RHS interleave map，首8行Scalar，后续向量倍增；row浮点索引因Size≤256全整数精确，乘4/12/16后CastRINT形成B Gather byte offset。|
|699–724|StartLoadAB先读完整B，再轮流读本part的8列panel，首panel单独MTE2_V0、其余MTE2_V1；PickPivot把4组64元素归约值/索引以u64读取，比绝对正float位即可选大小，gMask屏蔽不存在的第3/4组。不能对NaN当正常序关系。|
|726–812|FactorTile一次本地消8列，先16行转置到colBuf的8×256列主；unused bitmask清已选pivot。V_S0覆盖nextCol归约，1覆盖下第2列，2覆盖其余列；先发下一主元所需数据，Scalar可与尾列更新重叠。每两个完成factor就发GM，全部8factor的MTE3 drain后才发metadata tag，确保tag表示数据真ready。|
|814–838|WaitTileTag以24float DMA重读meta，等MTE2_S后volatile UB读tag；tag包含wave前缀，防旧wave数据被误认。minTarget和spinTicks用GetSystemCycle延迟poll，**这是仍会循环检查真实tag的退避，不是等待固定时间就假定ready**。具体60/115/92cycle等是校准常量，不能跨频率/设备照搬。|
|840–901|UpdatePanels将8factor广播到a8256起，逐q更新myPanels；RHS更新用Scalar递推b0..b7处理pivot相互依赖，再把8次axpy分到b/col/red0/red256四累加链，最后两级Add。减少单条累加依赖链，但改变FP32求和次序。|
|903–956|Solve初始化unused/padding、Gather本RHS并备份originalB在fac3072；wave0等初始化flag2。每panel双pf/meta槽，nextOwner特殊分支与普通消费者分开，避免把依赖不同的工作合并一次全局barrier。|
|957–990|即将接手下一panel的owner先等tag，前4factor DMA到即更新nextPanel，再读后4factor继续，只优先服务关键panel。perm以u64打包拷贝byte offsets、inv保存在row256。padding/source行仍按真实Size。|
|991–1027|其他part读整个8factor并更新余panel/RHS；当前owner可能还需补前一tile的非关键update。无RHS且所有本地panel已完成的part可break，因为剩下协议基于每panel独立GM发布，不是每步全体barrier；有RHSpart继续消费全部factor。|
|1029–1083|n256 ResidualHalf：sub0发布初解和tag，sub1等tag后算后128行；两次64行A块读取+4个64列向量乘/加/归约，形成originalB−A*x。这里“Half”指**半行域**，不是FP16；计算仍FP32。提前装下一factor块与残差读取复用不同a区。|
|1085–1150|n200补偿残差按52行×pitch208分aa/ah/ph/pl四区，Dekker4097拆A和x，product residual包括hh−roundedProduct、hl、lh，省ll。下一52行A只有在旧aa最后读结束后预取；product输出已在其他buffer中，可与后续归约重叠。|
|1151–1212|fullComp对每行最大product位指数夹16..240加11构造大2幂shift，(product+shift)−shift取高粗分量，差值+拆积残差另归约；高低两次从B相减。此为分量求和而非FP64，溢出/FTZ/相消误差需分析。后续行Scalar u64暂存避免当前a流水区域覆盖B段；最多额外复制1个paddingfloat，在内部UB已分配范围。|
|1214–1304|Coop Refine复用已记录factor/perm，不重新分解。32factor块双a槽8192float，MTE2_S/V与V/S→MTE2分别保护Scalar系数读和Vector读；每4factor解三角依赖后分配4累加链。结束统一Join/Gather/乘inv恢复变量序。|
|1306–1340|Run先清所有panel/辅助flag，一次global2发布初始化，tag按wave变。n256 diagDom只看前8个perm恒等和首inv幅值<.25来决定是否跳校正，**不是数学上的整矩阵对角占优证书**；n200始终校正。两个sub的判定不依赖各自RHS数据，因此兄弟消费者needRef一致。|
|1341–1397|n200根据逆pivot幅值的位差启用fullComp+dekker；每sub算不同行域，sub1把后半residual写GM并发布0x5B tag，sub0等此tag后复用LU求校正。n256需校正时使用普通FP32 ResidualHalf，同样半域交接。|
|1398–1438|无需校正只Gather+inv；rhs>0先把完整列解写sol后发布0x5C tag，rhs0等全部RHS ready，再一次读取并Gather成row-major输出。输出仅part0，其他RHS没有并发相邻Scalar写问题。rows!=0注入值分支仍Host rows0不可达。|
|1439–1488|general Solver按job=batch*RHS分组，不共享同A各RHS分解；layout为8×8列分片、裸189KiB。fac从40968float起，idx47624起512项末48136，expanded/unused/perm根据容量放fac尾或matrix后。索引含Scalar整数/除模，不是通用路径无Scalar。|
|1490–1543|LoadA逐row、每8列块DMA入blocked UB，BF16按row先Cast再UBcopy；LoadB紧凑≤4096时整B+Gather，否则每row exact scalar-size DMA，UB每个burst占32B、Scalar提取。可避免GM越界但对大K/大n描述符数高。|
|1545–1621|General Solve逐pivot最大abs未用row，owner广播原factor和reduction，其他parts通过GlobalBarrier等待；同组每个part都维护完整b/perm。pivot行仅缩放还live列；余列rank-one与RHS Axpy，最后Gather变量序。没有在factor选择前凭可逆就跳pivot。|
|1623–1675|General Residual是**普通FP32**A*x乘积和分64列累加/WholeReduceSum，非double-float。4行chunk重读原A/B；对BF16转FP32后同计算。修正先前S6“高精残差”措辞：仅Coop n200的有条件补偿具有额外分量，这个general体没有。|
|1677–1730|Parts>1只有valid part0消费GM已存factor进行校正，其他part无内部barrier可return；Parts1使用UB已变换列和perm。复用factor而非重做分解，但初解仍按每RHS独立生成。校正内部非collective，不能强制所有part都做同算术。|
|1732–1789|Store列解到sol。Pack若全sol≤40960float，每核整体读并索引重排；否则**每个输出float一次4B DMA**到fac32B槽，再Gather512元素tile连续写。正确extent与效率须分别评价；最大K场景可能有大量请求，不能只按必要GM字节估计吞吐。Run随后按S6已读wave/need/final Pack协议执行。|

### S10.2 新补齐的资源/算法边界

Coop布局：192KiB=49152float，fac41792、b45888、col46400、row46912、red47424、idx48448；idx512至48960，均在声明范围。n200四个52×208残差阵列至43264float，覆盖旧panel/colBuf等已死区，但不覆盖originalB fac3072=44864、unused45376、perm45632和packIdx43840。可迁移的是相位复用分析，不是“每个buffer互不重叠”：有意重叠恰是优化的一部分。

General布局：fac40968、b45064、col45576、row46088、red46600、idx47624，idx512末48136≤48384float（189KiB）；**不能把Coop的idx48448套到general**。Host容量条件同时保护expanded/unused/perm与matrix/因子不冲突；BF16输入暂存按各row的实际pad粒度读取。

Coop n200/n256每batch只因式分解一次，多个RHS复用factor；general SHAPE0每个RHS一个job，会重新LoadA/Solve，主成本可到O(batch*k*n³)，不是“一律同A分解共享”。它有全域结构但可能牺牲大K吞吐。后续新题应先按RHS数量决定算法族，而不是把所有路径都称最优multi-RHS。

`diagDom`/`ill`只是不完全观测的启发式，不是残差或κ(A)证书；κ≤1000仍允许整体缩放使4097拆分溢出。n256半域普通FP32残差、n200省lo*lo的Dekker式残差、general普通FP32残差三者必须分别记录，不统一写“高精校正”。补偿归约的有限范围和误差、tag轮询的缓存/DMA顺序以及poll参数的性能，仍待设备/SDK条件验证。

### S10.3 可迁移分类

- **P/资源子域**：固定n200,k4和n256,k3的8核/batch、52×208或64×256残差面板，只在这些guard内成立。
- **F**：Gauss-Jordan逻辑pivot、因子跨多RHS与校正复用；残差按行域划分并在校正前汇合；多独立累加链缩短依赖，需要另审浮点重排误差。
- **G**：tag代表data-ready并带wave版本；下一个owner只先消费关键panel；只无后继消费者的worker才提前退出；数据ready与buffer复用ack分开；输出汇聚后一次连续DMA。
- **平台/校准**：GetSystemCycle延迟poll、原生Gather/Impl、固定事件ID和裸UB表属于强SDK/设备条件；不能把cycle常数或省初始化宏直接迁移。

rank2所有1874行kernel及其Host/key/tiling现已完成首轮文本审读；rank1后续补齐见S11，rank3仍保留明确未读区间。

## S11 rank1 211–2190补齐：全部2344行kernel完成

记录日期2026-11-09。以下区间已连续实读，不按函数名推断；旧S4的“体未读”现由本节补齐。

| 行段 | 语义、路由与特殊设计 |
|---|---|
|211–256|ColumnSolve模板按STATIC_N/K/WORKERS、FAMILY和PARTIAL_CAPACITY派发。logicalN/K独立于私有容量；cp=na或na+8/16。matrixCols按普通/并行/族选择，UB依次mat、input、transpose、indices、factor、column、order、output、inverse；workspace按每group与family分别分区，publication另有独立区域。裸LocalTensor不是零UB使用。|
|257–353|串行column路径每8行装并Gather转列，前firstResident列可spill到GM，其余UB驻留。active屏蔽已选pivot、64项多组WholeReduceMax后Scalar选全局row，逻辑置换不搬全部行；更新全部后缀列/RHS，Assemble后Refine。GM spill写后事件保证再读不抢先。|
|355–444|并行LoadRange每8列、FAMILY3宽装16列，用16行×8列硬件transpose。LoadCapacityRange真实GM读取只logicalN×validCols，其他私有元素清零并补4I、额外RHS0：实数下diag(A,4I)解为(X,0)，合法容量专化而非改变问题。条件guard与dtype/extent必须保留。|
|445–577|FactorUnpivoted按owner列连续分工，FAMILY1面板4列持久发布、其他8列全同步。owner因子归一后带inverse发布packet；已完成且无RHS职责的owner不用消费未来packet。接收前缀两slot交替用event2/3防复用；owner的最后packet同时发布inverse metadata。退役因子成对组合后写canonical GM，等待原packet DMA完成才覆盖源。|
|578–605|publication先零DMA→SyncAll→invalidate旧cache，再把不可变32B的1保留到最终join。每packet使用不同128B槽；producer先drain数据MTE3再写flag，consumer volatile读0时invalidate。flag在一次调用内单向0→1，但跨调用必须清零/失效，不能把这个机制变成跨调用结果缓存。|
|606–720|FactorWindow每64行窗内选pivot，首未用row幅值<2才在窗内max归约，不是全矩阵partial pivot。FAMILY3固定256持久路径只分布式处理前192列；8/10workers各有不同前缀分工。下一个pivot先完成，S/V events1/2分开等其他coeff；packet带inverse+row元数据，关键window可用events3/4先ready。|
|721–859|retiredMetadata owner结束后break，因为最后tail owner会汇总完整order/inverse；非持久路径仍维持全体barrier。四factor/pair组合只改retired canonical因子，不改已发布packet；前192列meta暂存在n+12/n+13，避免覆盖n+16仍live packet。lane0等所有producer flags，再合并末64列并发布tail+RHS+完整metadata。|
|860–961|CollapsedTail<192>把64×64加RHS留同UB，20736B，不是整256矩阵。窗内pivot，nextCol先更新ready、其余列后更新；末尾pair-compose retired因子、存回GM canonical，inverse/order写retired packet区域。仅该tail producer发布后其余RHS worker可消费。|
|962–1039|非blocked并行：初解广播给各worker，残差按行分工、RHS分配worker校正、最后一个owner打包输出。n8整除用GatherMask去输出padding，非整除每行exact byte。三RHS紧凑输出索引放output[3na:6na]，原解在[0:3na]；logicalK小于容量时重用已死原解区生成新索引。|
|1040–1119|Blocked流程初解Backsolve、全组汇合后残差、Forward/Backsolve校正、最后pack。固定256三RHS先原解存output，再残差可覆盖flag seed，所以校正完成前重新生成不可变flag1；最终用局部publication204/212而不是无条件全体join。只最后owner输出logicalN×logicalK。|
|1120–1218|ForwardBlocked按64factor双槽回放，n≤200全因子驻留；固定256前192以四factor组合，其他成对。Backsolve逆序块，当前已死upper因子原地乘解并树归约，可覆写是因后续不再读该upper；coefficientsReady时复用Forward最后两块，而非重复DMA。|
|1219–1288|NormalizeOwnedPanelEarly先下一column Axpy后S_V1，再其余columns后S_V2。WindowReady又分首column下一64row窗及其余窗，发布两阶段events3/4。FAMILY1换lane映射：每次8列×每列8行，把broadcast重复变为更少指令；必须cp/8符合stride且scratch容量足够。|
|1290–1404|ResidualTileOriginal逐RHS读原A/B，用清低12mantissa位拆A/x、四乘积恢复乘积残差；树归约保留TwoSum误差，末8项再三层packed TwoSum，B−dot仍保留error。不等价FP64，不可忽略overflow/FTZ。family路由：2 Rounded、1 SplitProduct、3 Accurate，其他按n桶。|
|1405–1506|固定256三RHS一次A32×256、三组product形成96row批归约，属于**普通FP32残差**，不是函数名Rounded就有乘积补偿。部分容量只读真实rows/cols，补零。其他Rounded把已舍入product清低15位分高/低独立求和，改善sum误差但不恢复先前product rounding。|
|1507–1578|Accurate先一次读并拆A，跨RHS复用high/low；四积误差含low×low，再把rounded product清低15位分层归约。比Original省重复A拆分/装载，仍是浮点重排而非数学运算被删除。|
|1579–1703|JointFour/TileSplitProduct清低20mantissa位，HH每operand仅4个有效位；用Al*Xh+A*Xl保留其余贡献，后项包含low×low。JointFour跨k≤4一起归约复用A，short finish用已死A区；仅n≤200、n8整除等guard下调用，不能将全局rows*k和repeat计数任意扩展。|
|1704–1832|Assemble按order与RHS pitch Gather、乘inverse、可加原解。CorrectSingleRhs复用canonical factors双槽；ApplyCorrection对所有RHS外积更新；非并行Refine把驻留coeff一次spill再复用mat作残差，**不重新分解**。与一般KernelLinalgSolve重做LU的refine不同，不混淆。|
|1833–1917|小核transpose与OuterT：UB固定布局最高output末75616B。A可硬件transpose，B走bounded flatDMA/Gather；owners=min(cores,batch)，空闲块本体会取模重复计算但最后只有owner写，entry的guard决定能否到这段，不能仅看循环判重复输出。|
|1918–2026|FULL_WIDTH取最大**有符号**pivot，不是max abs；仅最大signed恰为0时再abs搜索，仍可在所有未用值负数时选到最接近0的pivot。BLOCK_TRANSPOSE分支先更新nextCol并提前归约、再更新其余列；归约改变mask后显式模板true恢复64lane。逻辑pivot记录完整，末Gather/scale/GatherMask连续输出。|
|2027–2115|SmallFixed constexpr布局与n/k；PARTIAL容量15/16通过bounded DMA、4I扩展保证原AX=B实数解不变。索引首8项Scalar再倍增；packed n9..16优先flat A DMA，真实读取n²不扩大到capacity²。|
|2116–2190|n16固定只在行p/p+1这对中选较大abs，odd用另一行；n15首pivot全搜，以后低于阈值才搜，最后两行挑大值；其他n后续正/负并行归约体已在S4读。所有条件均按源码guard解释，不能称为全partial pivot。|
|2191–2344|前批已读小核归约尾/入口/生成wrapper宏；连到本次起始后rank1全文无空洞。|

### S11.1 可推导的限制与迁移分类

- **P/资源子域**：n256前192分布式+末64collapsed、n15容量扩展、三RHS compact pack。常数是专化条件，不是一般求解器的最佳分块。
- **F**：同A多RHS共享因子与校正；因子canonical pair/four组合可减少回放依赖；将真实系统扩成diag(A,cI)时需c≠0和新RHS0；残差按“product rounding”和“sum rounding”分别补偿。
- **G**：immutable packet+单调flag替代每panel全局barrier，data-ready和旧reader-done分开；下一critical列/window优先；保存索引跨batch但fallback/重排后失效；只在最后使用结束后复用因子/seed区。
- **SDK条件**：volatile GM轮询+DCCI、event1..4、count-mask/native transpose和生成wrapper宏须锁具体CANN/arch；没有本轮设备计时，不能量化某条机制提速。

数学反例边界：SmallFixed<n16,k1>只允许从当前行对挑pivot。取A为交换行0↔2、1↔3的16阶置换矩阵、B全1，则κ₂(A)=1、精确解全1，而首列在row0/1都为0，源码会取零pivot倒数；没有回全局pivot的路径。对窗64限定的256路径，也可把首列唯一非零放row64而窗内全0，整体仍是κ1置换矩阵。因此这些快族不能仅靠“可逆/κ≤1000”证明当前PDF全域正确。这里是源码+数学可定位边界，不声称已经设备运行失败或原比赛版本违法。
## S12 rank3 中段闭合：241–3770 连续实读（2026-11-09）

本节与先前1–240、3771–3988合并，闭合[rank3 kernel](../../extracted/solve/rank3_submission_1177540/code/op_kernel/linalg_solve.cpp)全部3988行，SHA256=`aa1aabad7ba1cb3b82ebe6555a10cbb1b2f461d609a9539c224ee4b260f1593f`。以下按连续语义段记录，包含被编译期关闭的分支，不把死分支当作实际执行路径。

|行段|逐段语义与不寻常设计|适用范围及边界|
|---|---|---|
|241–297|双列高低位乘法/加法交错发射；DotPre2树规约停在8元素，最后标量尾和；Transform2共享只读Householder向量处理两列|F：独立依赖链交错；不是完整FP64，乘法截断项仍须误差界|
|298–338|8列panel一次装入/写回高低位，树减半到8保持32B对齐，再Tail8；Dot与DotPre区分是否预拆q|G：批量DMA降描述符；源码注释“PIPE_ALL快0.6%”仅作者历史声称|
|340–391|列装载：n≥224时4B/row DMA到8槽pitch再标量抽取，小n直接GM GetValue；回代逐行读上三角并以Pair除法求解|F：阈值必须重新标定；padding行显式清零，不从最终输出读残差|
|393–454|QR以2的幂行容量工作；32×512 float主缓冲+2×32768B staging=131072B；A/B同幂缩放、Householder符号选相反首元、8/2/1列尾覆盖|F：稳定反射与scratch相位复用；指数钳位并不证明所有中间值安全|
|455–569|TinyColumns P16/W24；n16用原生VA transpose，其他n用Gather；逻辑选pivot不交换整行；留下3×3标量部分主元并将解嵌入死weights，最后Gather/Mul统一收口|F/G：按极小尾块换执行单元，输出索引/scale共同定义变量顺序；原生dav_c220地址打包是硬件绑定|
|570–706|SmallColumns P64/W72/L72，96000B二维存储；aligned原生transpose与unaligned Gather两条装载；列尾8槽存广播权重；k≤2留下2×2标量尾|F：padding兼任元数据避免额外阵列，须证明活跃区域不重叠；最终只写n×k|
|708–776|SmallSquare增广矩阵：按逻辑pivot行而非物理交换，更新所有行后恢复pivot，避开每轮scalar置零；k≤8统一扩8列，其他k分行写回|F/G：减少S/V往返；scratchGap注释是经验，不自动迁移|
|778–809|ColumnPanel固定B列活窗，输入后缀逐步替换为transform列；pivot缩放推迟至transform发布，保存inverse按原始row排序|F：消元映射复用，是DirectPanel/RefinedSquare共同代数接口|
|810–908|RefinedSquare以4097拆A/x，四乘积含低低项求残差；按最大乘积指数选择3平面移位规约，极端指数回TwoSum树|F：残差精度投入优于一律高精度；移位常数、舍入与指数范围均为前提|
|909–1099|分组owner发布panel、行标签重排inverse和initial；SyncAll分隔生产/消费；残差b−Ax后乘近似inverse，Steps>1才回写中间initial|F/G：相位复用11个tile；`if constexpr(true)`之外旧row-panel体已读但未运行；k≤8的rhs装载前提须与Host路由一并看|
|1102–1209|CubePanel显式ASCENDC_CUBE_ONLY注册，HF32关闭；P384的树从256开始仅加剩余128，之后正常减半；矩阵容量与实际输入宽度分离|F/G：非2幂规约不是简单丢尾；不能把此宏认作通用提速开关|
|1210–1251|四个base256 signed digits，row scale取max指数；零行单独处理，超指数写1000哨兵；half整数dot最大512×128²=2²³|F：整数可精确表示的低精度Cube残差；仅每个digit平面累加界成立，不等于最终解精确|
|1252–1356|所有lane读取全部exponent以作一致fallback；同权重digit对按3平面stride合并；d≥2以int32求和，保留转FP32余数；Rows版128行任务并支持4/5组|F/G：把舍入损失作为第二平面保留；仅保留前若干权重时仍需截断界，指数门不是误差证明|
|1359–1437|Cube服务按panel同步更新增广矩阵；MatrixRhs借Cube算初值和校正；SplitResidual只算10个i+j≤3 digit对|F：多个向量dot转矩阵乘；GM workspace的16平面不是16平面都有效|
|1438–1639|Vector初始化[A,I]并padding成diag(A,I)，逻辑labels重排得到inverse；小k走vector乘，宽k走Cube；split失败才用补偿vector残差，输出32列Gather转回|F：数学上等价块padding；两域`SyncAll<false>`配对数必须保持，不能局部删barrier|
|1642–1851|DirectPanel不显式形成inverse，存每panel transform和order，原始RHS随消元推进；校正RHS按同映射重放；非owner在panel空档拆A digits|F/G：重用分解/映射，跨任务做独立准备；FactorCols补齐与perm尾显式清零；N208/P256额外保留权重4|
|1853–1889|NormalInverse构CᵀC后32块无主元消元；当前Gram块先减I，使统一累加更新自动完成pivot行修正|F：SPD块下的代数融合，不能用于任意可逆非SPD pivot；正规方程κ平方风险不消失|
|1890–1947|四个base128 int8 digits；A按升序、X前缀倒序，使一条int8 GEMM收同权重；最多4×512×64²=2²³，转float保整数精确|F/G：布局替换多条GEMM，分组G复用指数及B读；只证明整数累加，未证明丢弃高权重尾项|
|1950–2016|Cube先Gram、再factor和inverse更新；pass0普通FP32残差，pass1 int8高精度残差，合计2轮精化；原始A指针与padding Gram复用分开|F：按迭代阶段分配精度，节省完整FP64开销|
|2017–2207|全lane归约最大指数后幂缩放；transpose原A生成初始Cᵀ；Gram死后复用为padding A；Grouped8列装载、Grouped4残差；scaledResidual防止重复乘scale|F/G：workspace生命周期复用有实际同步边界；有限2轮精化及无主元SPD浮点退化仍需测试|
|2210–2309|NativeMM自行管理L1=262144B、L0A/B各65536B、L0C=131072B；FP32 transpose靠Load3D，Nd2Nz与Load2D另一路；int8 K编译期、直接L0整数Mmad|G/H：原生数据布局闭包替代Matmul服务；BM/BN/BK整除、片上容量和SDK intrinsic为硬条件|
|2310–2413|Int8All将六权重任务均摊，Packed4把多个权重/列排成1536长dot；Newton64在L1保X/R，Neumann同时推进X与R²，最后重新算真实Gram缺陷消漂移|F/G：依赖链中有独立Cube工作；原子累加只在Accumulate分支开，Tile完成后关|
|2416–2547|64/96/128块定制Mmad与Fixpipe→L1 half；LoadPair用槽0/1分离活数据；96有3个9216half槽，显式MTE1/M/FIX依赖|G/H：固定布局与resident缓存；非任意shape模板库，不能去掉重用槽前wait|
|2548–2710|RefineResident96 noinline隔离复杂图；Second=false为Newton后再校正，Second=true额外高低Gram/缺陷及两项解更新；Newton96按步数R平方后收口|F/G：noinline控制调用图/寄存器压力是机制，未独立测出收益；half轮次靠guard而非普遍收敛保证|
|2712–2787|owner迭代第二块seed时helper重新计算实际更新panel Gram，并拆成half high/low；SyncCube汇合后owner保留L1/L0状态继续精化|F/G：有依赖证明的跨任务重叠；刷新actual目标避免只对旧Schur近似自洽|
|2789–2956|208/256 active宽度、96/48/32/64块组合；CrossAndSecondGram复用一次A装载算两产物；HWithResidentInverse复用前一Factor96的L0A；尾系数重投影|F/G/H：驻留数据可省搬运，但必须保持task→core映射和相邻调用不覆盖L0A；不是任意函数顺序可换|
|2958–3035|forward从immutable输入分块独立owner，再同owner乘local inverse；32 Schur双owner保留原g21；ComposeForward96把串行尾前代映射预合成|F：三角依赖预合成换并行；in-place只在Tile完整读取并结束后覆盖|
|3039–3185|Orthogonal Vector160KiB；第一行A与全B派生所有lane一致scale，保B供后续pack；Newton seed alpha=1.8/∞范数，缺陷bound>1/64或非有限触发全局原子decision|F/G：同数据重复小归约免一次集体归约；首行不代表全A最大，泛化数值范围需单列；Guard不是最终解证书|
|3186–3247|64块正交闭包，前3块Newton6/7/9轮+guard，末块强制32 Schur精确路径；控制flag读前cache invalidate；对初解再做一次残差校正|F：近似快路+可观测fallback。所谓exact只是作者算法路径名，仍FP32且SPD子块无主元|
|3249–3365|NarrowOrthogonal6个16×96 identity行片；Normalize全A归约版与NormalizeSeed公共seed版；16/32 pitch的B与输出两套pack|F/G：masked Gather仅有效k列，精确写n×k；seed缩放需区别全A缩放的范围保障|
|3367–3450|六base128 digits，逐轮与批量Cast版本；SplitAChunk在非owner空档一次8行；Packed4将6个权重的X前缀排成32行以单次Int8Packed4消费|F/G：减少转换/调用，不是少算算子；指数未有1000哨兵的Split6不能借用另一函数fallback结论|
|3451–3516|Packed4先将相邻base128权重并成base16384，带符号进位，领先int32转float的余数并入中digit；普通版按6组浮点缩放+TwoSum求残差|F：精度补偿须逐层：integer-dot精确≠int32→FP32精确≠最终残差精确；注释中<2³⁰界还依赖digit范围|
|3517–3613|identity复用、死factor行清零、缺陷加I、两遍MGS TailQR后回代产生负逆映射；32块无主元Invert/Negate采用不同GM行pitch|F：小尾使用更稳定QR避免Gram退化；two-pass非全算法精度证明|
|3616–3740|96 Gram补diag(I)到128；NewtonPrepare96 high/low/eye/2I/−I七平面；guard 1/16、递归Schur128 fallback再右侧残差精化；mode2双向跨核通知并配全AIC汇合|F/G/H：数据ready与全局完成分离，必须按目标芯片flag参与者语义复核|
|3741–3988|Packed窄RHS将TailQR与独立Cube forward-map构造重叠；全路径按编译期key分离，宽/窄RHS布局不同，批次分resident波次并保B广播|P/F/G：不能仅复制函数不复制Host/tiling/workspace/核数契约；SOLVE_DIRECT_BENCH宏关闭公开入口，未在此运行|

### S12.1 能迁移的不是“高名次宏”，而是完整依赖关系

1. F：缓存消元/正交变换，校正RHS重放同映射；避免重复分解，也避免为窄RHS显式构造完整inverse。需保持pivot顺序、scale、原始A残差一致。
2. F：低精度整数Cube执行高精度残差平面。base128/256必须先证digit范围、dot整数界、int32转换余数、权重截断尾项、恢复指数范围，再验证解误差。这里只完成前两类可从源码直接推导的界，不宣称整个流程全域满足1e-4。
3. G：利用panel owner独占计算时其他core的空档准备下一阶段；只允许与owner不相交的workspace以及明确发布屏障。resident L0复用更强，必须固定任务/core对应。
4. H：原生VA、Load3D FP32 transpose、Fixpipe→L1、raw ffts标志是硬件/SDK限定实现，不是跨CANN版本可盲贴的trick。

### S12.2 保留的证明义务

κ(A)≤1000不意味着任意无主元Gauss-Jordan可用，也不保证正规方程FP32误差小。QR Pair运算并非IEEE FP64，有限精化轮次及digit截断仍需误差分析/对抗输入；数据依赖guard只检查某个局部逆缺陷，不能替代最终残差契约。源码中的“measured”“exact”“complete”“public safe”等注释均未当成测量或正确性证据。27份的首轮源审读与NPU编译、同步压力、位宽/stride验证、性能归因是不同层级。

