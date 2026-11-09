# 题8 eigh：逐行审读账本（首轮源码审读完成）

题面：[原始PDF](../../problems/08_eigh.pdf)；[题目契约](../SPEC_CONTRACTS.md)。

状态：`first_pass_source_review_complete`，2026-11-09。三版本提交文件及三份kernel正文均已按原始连续行段逐行读完；此状态仅表示全文静态源码审读完成，不表示数学正确性、SDK/API语义、同步安全、运行结果或性能已验证。覆盖见 `../coverage/08_eigh.json`。本报告不以关键词扫描/函数名列表代替全文。无编译、无NPU运行、公示排名或Pass不构成单trick归因。适用范围分类见 `../DESIGN_CLASSIFICATION.md`。

## contract 题面契约

完整阅读 `../specs/08.txt` 两页，并渲染/视觉核对原PDF1、2页，图在`../specs/rendered/08-1.png`和`08-2.png`。float32 `[*,n,n]`→特征值`[*,n]`、列向量Q`[*,n,n]`，n1..512、每个batch维1..64。只使用UPLO选择的三角补全A_eff，另一半允许任意脏值；支持L/U，默认L。λ必须非降序。A_eff谱互异、最小gap≥1e-4||A_eff||2。

特征值逐元素atol/rtol1e-4；向量不逐元素比较，须||QᵀQ-I||F≤1e-4，且||A_eff-Qdiag(λ)Qᵀ||F≤1e-4||A_eff||F。只保证各向量残差不等同保证全局正交；间隔保证使方向稳定，不允许忽略全局Gram。禁止整套eigh调用、CPU回退、只输出部分特征对。

路径缩写R1=`extracted/eigh/rank1_submission_1174206/code`，R2=`.../rank2_submission_1176362/code`，R3=`.../rank3_submission_1177525/code`。

## build 相同构建与README

三rank根CMakeLists.txt1-21、README.md1-12、op_host/CMakeLists.txt1-25、op_kernel/CMakeLists.txt1-15 SHA分别相同，复用R1全文阅读。根设置CMake3.16、compile_jobs默认4、强制ascend910b、ASC、TYPE RUN包装；Host glob cpp（不会把host_plan.h单独编译）、JOBS codegen、tiling/aclnn库；kernel当前目录生成并绑定tiling。README宣称Toolkit9.0或兼容、source set_env与cmake命令，是提交说明，不是本机已验证构建。未执行其中命令。

## r1-host Rank1 Host连续语义段

[R1 Host](../../extracted/eigh/rank1_submission_1174206/code/op_host/linalg_eigh.cpp#L1)1-266全文：

- 1-28 includes、context/shape/desc空指针检查、rank≥2、n1..512、square/float32。29-46按所有batch维正数相乘，没检查每维≤64或uint64溢出；UPLO只看首字符并支持大小写，不校验后缀。
- 48-65读AIVcore和UB，stride8pad，固定UB开销1024+29stride·4后用双plane推maxRows，无法放1行则失败。
- 67-92 n≤32 solve_mode1，groups=min(batch,AIV)，每组lane按小n可一行/两行/四行bucket，矩阵仍每lane完整n行，vecRows是输出归属不是矩阵存储同义字段。
- 93-116 32<n≤256且AIV≥10使用resident mode2(n≤120)/3，最多16lane，>120时最少ceil(n/26)lane，n≤200每lane容量16或8pad，>200容量至少26。
- 117-160其余mode0分布式：minLane来自UB和至少ceil(n/64)向量归属；rows8pad并做tail_rpl=ceil(144²/stride)容量下界，只在可容纳时增加，为后续serial tail留空间，但正文尚未核消费者。
- 162-185填完整tiling；**use_k96=false && ...**永远不激活，相关reserve/workspace offset与schedule代码保留但Host不执行。
- 187-209设置activecores、mode0 whole-launchschedule；实际SOLVE_MODE=4仅mode3/n200/batch3/group13，5被`false&&use_k96`再次禁用。mode0 workspace=groups·(2nstride+7stride+8lane+64)+64active+256float加library；其它1..4零workspace。不能把zero workspace等同不用GM暂存，正文有无复用输出待全文审。
- 213-239 InferShape：Q同input、λ删除最后一维，dtype各float。242-266注册required三张量、UPLO默认L、910B。

`linalg_eigh_tiling.h`1-18：n,stride,uplo,active、uint64batch、num_groups/group_lanes、row/vecrow、solve_mode/wsOffset；按C++ABI可能有uint64对齐pad，未static_assert。`tiling_key...h`1-27：mode0 MIX_AIV1:0，1/2/3/4 AIV_ONLY，5 MIX_AIV1:0。**Host从不选择5**；不得因其存在就把kernel5作为当前计分路径。

## r2-host Rank2 Host连续语义段

[R2 Host](../../extracted/eigh/rank2_submission_1176362/code/op_host/linalg_eigh.cpp#L1)1-218全文：

- 1-29 float/rank/square、每batch维1..64及INT32乘法上界、UPLO完整strcmp只L/U；n>25 stride64pad、否则8pad。30-41 runtimealgorithm小n3..25 ResidentSmall/flag32，n1/2解析专路team1/group按batch、0workspace。
- 42-46 UB主动保留16KiB供spill/library。47-99 n26..64尝试完整fallback容量：owner不超core/batch且尽量8对齐；required包括resident、wide、6nstride，成功选ResidentMedium，flags组合output scratch/certifiedRayleigh/private/twoinverse/vectororiginal。owners>1加shared reduction/boundaryconsensus并去outputscratch；n57..64UBreflectorpipeline，n26..56某些paddingbucket narrowrowpipeline。launch向上8pad，whole-launch schedule；GM只分证书及必要history/classification，与输出workspace复用边界需核正文。
- 102-123 minTeam倍增满足UB、rows≤64：dense、halfwide、colorbank padding、小n17stride²额外arena、n129..256两份96×128tail、n≤64 history都纳入。
- 124-144 groups/targetrows与team，非64整除中小矩阵偏好偶team但保留真实容量；n129..256且n+96≤stride+64才设tail96flag，避免实际rows超64。
- 145-165 USE_CUBE=1仅n>256、stride64整除、paddedRows≥16；stride≥384改2，奇team关闭Cube；整64shape可改alignedRows因子；physical launch有空核8pad，注释明确空组保留所有global barriers。
- 166-184无Cube/n>64/n==stride/无空组且batch整groups，才允许output history复用，之后whole-launch schedule。workspace按flags/USE_CUBE公式，Cube2额外每64panel6·64²float，前缀832KiB固定协议；3..25强制0workspace。协议832KiB并非核本地矩阵存储；实际SDK下限尚未核。
- 187-218 shape inference正确删除最后一维、检查square，dtype检查float，required注册与UPLO L。

`linalg_eigh_tiling.h`1-54全文：base9uint32、flag1 scratch/2twoinverse/4outputhistory/32twisted/256certRayleigh/512private/1024tail96/2048shared/4096consensus/8192originalquotient/16384UBpipeline/32768narrow；每owner16float=64byte证书槽避免不完整DMA共享块；Algorithm enum0distributed/1medium/2small。`tiling_key...h`1-18：USE_CUBE0 MIX_AIV1:0，1/2 MIX_AIC1:2；算法选择是runtime tiling而非所有flag成为模板key。矩阵分类/后验quotient需要读取当前输入，Host没有输入数值分支。

## r3-host Rank3 Host范围与实际分派

[R3 Host](../../extracted/eigh/rank3_submission_1177525/code/op_host/linalg_eigh.cpp#L1)1-100全文：

- 1-7提交作者注释称动态workspace allocation约920..1020us计入耗时，要求全路径0workspace，命名Pure-FP32 Zero-Workspace STEDC-COMPZ-I v11。**这是源码中的作者经验声明，非本审计独立测量，也不能据此推导评测所有环境都如此。**
- 8-23 traits与hostplan引入；MatchesCase对B1只接受rank2、B>1只接受rank3的exact shape（不把多维batch flatten后匹配）。25-38 TilingCase检查预置usedcores≤available，整包constexpr plan复制、CASE_KEY、固定blockdim、0workspace。
- 40-63 float32，GetRequiredInputTensor未判空；UPLO只接受单字符L/l，U直接失败；available=min(32,max(actual,1))，使用OriginShape（与别rankStorageShape不同）；六个MatchesCase之外统一GRAPH_FAILED。
- 66-74InferShape/InferDataType仅success不填任何输出，是独立接口集成缺口，不能断言框架一定自动补。76-100三张量float注册、UPLO默认L、910B。

**已证契约覆盖缺口**：只接受(16,16)、(4,64,64)、(4,256,256)、(15,15)、(3,50,50)、(3,200,200)，且不支持题面U。合法(1,1)、(2,2)、(64,64)、(1,16,16)、任意多维batch等均不匹配；这是明文Host行为，不靠设备复现，也不据此认定违规/计分来源。

`tiling_key...h`1-10：float+CASE_KEY1..6；无显式kerneltype，实际默认依SDK/entry待读。`eigh_case_traits.h`1-21：每case编译常量N/B/U/P/Solver，分别(16,1,0,1,0)、(64,4,0,8,2)、(256,4,0,8,3)、(15,1,0,1,0)、(50,3,0,8,2)、(200,3,0,10,3)。不存在general默认case。

## r3-plan Rank3 constexpr表逐表规则、容量与消费者

[R3 plan](../../extracted/eigh/rank3_submission_1177525/code/op_host/eigh_host_plan.h#L1)1-290全文（包括所有零行和每条core记录）。不是预存矩阵、特征值或Q；内容只有shape几何与arena offsets。`linalg_eigh_tiling.h`1-15定义每core10uint32，plan前导18word含M/M_leaf，bounds9与leaf_bounds33，cores32，总1516byte与284static_assert一致。

共同字段规则：每batch第p核`a_offset=b·n²`、`val_offset=b·n`、p=0..P-1、col_start=floor(pn/P)、col_end=floor((p+1)n/P)、my_cols差；因此每batch列区间连续、互不重叠、并集0..n。小case只有首核，其余31全零；case5启用24留8零，case6启用30留2零。TilingCase整包复制1516bytes；kernel实际按GetBlockIdx索引cores[32]，中/大路径消费列区间、块bounds和leaf bounds。具体写出地址/别名安全另见kernel段，不能由表自身断言。

| case/plan行 | N/B/P/used | stride/maxRows/UBfloat | p_stride/pp_slot/meta_slot/meta_cap | 块结构 |
|---|---|---|---|---|
| 1，8-52 |16/1/1/1|64/16/4096|16/16/528/1056|M=Mleaf=0，bounds全0|
| 2，54-98 |64/4/8/32|64/8/48640|16/128/640/1280|M=0，8列/核|
| 3，100-144 |256/4/8/32|256/32/48640|32/256/768/1536|M4，Mleaf16；64整块、16叶|
| 4，146-190 |15/1/1/1|64/15/4096|16/16/528/1056|与case1容量相同，逻辑尾15|
| 5，192-236 |50/3/8/24|64/7/48640|16/128/640/1280|每核6/7列，尾核终止50|
| 6，238-282 |200/3/10/30|256/20/48640|32/320/832/1664|M4，Mleaf16，块48/48/48/56|

case3 cores(109-140)：每两核一块my_blk=floor(p/2)，plo_b=2my_blk、bp_team=p%2、subP_b=2；a_offset各batch65536递增，values256递增。bounds0/64/128/192/256，leaf0..256每16；max_Lb64、blk_slot4096=64²、leaf_gm_off60416（需正文确认输出复用区域终点）。

case6 cores247-276：p0/1块0、p2/3/4块1、p5/6块2、p7/8/9块3；plo_b为0/2/5/7，subP_b为2/3/2/3，bp_team为本块0..subP-1。bounds0/48/96/144/200；leaf0,8,24,32,48,56,72,80,96,104,120,128,144,160,176,184,200（16片，长度8/16混合），未用bounds/leaf全部0。max_Lb56、blk_slot3584=56·64（不是56²）、leaf_gm_off34880。case5列0/6/12/18/25/31/37/43/50确认floor分配，核心不是统一ceil(n/P)填满，不能换公式照抄。

283-290六caseusedcore断言与1516byteABI断言；它们不验证所有offset、UB容量、leaf/output别名生命周期。需要在读消费者后对每段确切读写区与原始A→输出复用阶段证明。

## comparison 当前已证对比与待审数学

R1是runtime容量分路+多个resident算法模式，mode5明确未激活；R2把资源核类型(USE_CUBE)与numericalalgorithm/flags分离，更多读当前输入的certificate设计待正文核；R3把六shape几何移至compile-time表、0workspace并复用输出，兼容性范围显著收窄。不能把R3只接六shape当性能公平对比，也不能凭注释声称省allocation就是名次原因。

潜在可复用方法：Host/runtime与compile-time计划分离；精确按core ownership生成geometry；64byte单写者证书槽；空核保留全局barrier；容量约束先于策略；零workspace需将输出暂存的生命周期作为显式ABI。不可照抄前提：固定32核、固定P/N、忽略U、写回shape推断stub、无官方文档的同步API、数值证书只看局部而不看全局Gram。

三大kernel现已完成源码全文首轮审读；后文分别记录实际可达算法与未验证边界。全文阅读只确立静态源码数据流，不等于证明Householder H=I-βvvᵀ、三对角特征解、回变换Q、排序、残差/orthogonality证书或FP32高低位实现对题面所有输入成立。接口/数学存在未知不等于已证bug。

<a id="r3-kernel"></a>
## r3-kernel 追加：Rank3 kernel正文审读

[R3 kernel](../../extracted/eigh/rank3_submission_1177525/code/op_kernel/linalg_eigh.cpp#L1)原SHA256=`a2b779a21d87b44e725c80fafd59098cac79252762a05e3098cf5670c3c428c4`，总8977行；本次补读1271-8699，现覆盖1-8977全文。具体solver段、消费者和边界如下；此前1-1270及8700-8977记录仍有效。

- 1-194是长篇作者“踩坑/架构/性能归档”，逐行读过但**不当作实测证据**：5-7声称judge逐元素且依赖RuleB，与本题PDF明确按Gram/重构且允许列变号不一致；说明该评论可能来自别轮/旧版本，不反向修改题面。14-19称细粒度2DDMA拥塞、21-26分支屏障死锁、28-32瓶颈比例、34-36PureFP32、42-56pipeline、58-90各版本Tbest/耗时、93-194历史四tier。历史列512/U路径与现在仅6shape/LHost不一致；不能把历史数据归到当前源码。175注释“久期方程=sum(...)=1+rho”不是通用标准rank1 secular公式（标准为1+rhoΣ z²/(d-λ)=0），正文未读，不能据错注释判实现错。
- 195-204`#pragma clang fp reassociate(off) contract(off)`明确禁止重结合/乘加收缩，再include kernel/traits/Hostplan（device也看到constexpr）。205-251MyRsqrt与Fast实际同一magic0x5f375a86+3次Newton，sqrt=xrsqrt；RsqrtUnit12二次多项式初值+2次Newton限定t∈[1,2]，不是一般sqrt替代。253-295Ldexp分64幂避免构造非法exponent，GetPow2Scale exponent钳27..227对应2^-100..2^100，倒幂254-exp；极端输入scale后仍未证明所有内积无下溢。
- 297-374Shell-sort index以值/原index平局、两类sorted merge；MergeTwoSortedF32固定dl/out[256]容量，需调用者L≤256（当前6caseN最大256满足，但叶子合并具体L仍待核）。376-380SafeSyncAll=PIPE_ALL→SyncAll<true>→PIPE_ALL，不是cacheinvalidate。
- 382-408**已读真实metadata消费者**：gm_ws_实际上是tiling指针，不是workspace；Plan()直接reinterpret全局tiling；Init绑定输入与输出并按编译期plan.ub_floats分配UB。Process按traits::Solver compile-time分派：0→SolveSmallL1，2→读取cores[GetBlockIdx()]后SolveMediumCoop8，3→同样offset后SolveLargeCoopStedc(P,true)。因此cores表地址offset消费已闭环；中/大函数实际消费列范围及块/叶bounds。不存在runtime general fallback分派；SolveMediumPipeV和SolveFallbackGM虽有定义，但此Process不调用它们。
- 411-509一/二/三连续Givens展开。1旋转两行4个向量op；2/3旋转预乘c/s组合减少向量依赖却增加scalar系数，实际小路径直接用Muls/Sub/Add而非这些wrapper；不能按注释认为fusion必快。511-514 EIGH_PAD4=一条PIPE_S屏障、PAD16/64/256各递归×4（数值名指布局近似，不是空注释）。
- 516-695 SolveLeafPipeV16完整：d/e/idx各16，stage320/352临时、z512、ztmp768；初始化m维I、m≤16隐含；隐式QL带Wilkinson位移，norm正常时r²rsqrt、极大/小用比值避免overflow；最多40次/根，deflate阈值2e-6相邻diag或1e-24。n64特化常驻mask16位，under分支插48条PIPE_S布局padding；非64有Adds+barrier，末尾排序事件Wait延迟到scalar sort之后。输出stage0为λ、16/32为首尾行分量、64..319为按16pitch排的叶向量；本次已读消费该布局的D&C合并段。
- 697-850 SolveSmallL1开始：n≤16 fixedstride16，UB4096；z1024、三个256临时、u1792、transposeindex2048、mask2304/2560、rowmask2816、I3072。n1分支源存在但6case不实例化n1。n16先DMA与初始化重叠，n15后DMA以避免作者声称的端口冲突（耗时未核）；构造选三角mask/反射rowmask/I。
- 851-921选择triangle bitmask、transposeindex=(col16+row)·4的doubling生产、rep8index广播同一scalar到8lane，按n15补最后行0，bitwise And清未选三角含任意float位串、转置后用dmask清重复对角并相加，正是A_eff不是(A+Aᵀ)/2；absmax基于所选半。零矩阵直接λ0/QI；否则按2次幂scale。
- 922-1024逆序Householder：u=当前row前i项，g=-sign(f)||x||、h=||x||²-fg，p=-Au/h，q=p-(uᵀp)u/(2h)，A+=uqᵀ+quᵀ实现HᵀAH。u与-1/h副本存UB用于回变换；h≤1e-28跳过反射并依n15/16插不同PIPE_Spadding；Q先I，再正序应用反射，最终读前两diag/offdiag。Gather零offset替scalar broadcast，避免第二次V/S往返。
- 1025-1112小路径QL每根最多45次，deflate2.5e-6·(|d_i|+|d_i+1|)或1e-24，s/c更新的两行16float独立Muls后barrier再Sub/Add；达到轮上限仅break、未见此小路径显式残差/Gram证书，不能证明所有题面谱都达tol。
- 1113-1221 Max/Min4层树找各向量极值，n16与n15不同mask策略；RuleB以maxvs-min判断sign，绝对值平局scalar扫描最前row；λ2次幂还原+选择排序perm，再sign+Gather转置输出，使Q列为特征向量；Dma只存n²，n15每UB行有1padding。所有中间主要矩阵在UB，不需GMworkspace。
- 1223-1270是未被Process选择的SolveMediumPipeV开头布局（64stride、d/e等64、z4096、u16384、rowmask32768、I36864）；后续函数已在本次连续段读完，但它仍不是当前Process可达路径。
- 8700-8760大路径尾部局部：RuleB取反后my_cols连续写gt_sym，**SafeSyncAll在write_out条件外**；随后按p分8行整块读取、片上Gather转置、再全核SafeSyncAll、连续burst写cur_vec，避免边读边覆盖。结合1271-8699现已可定位为大路径输出/临时矩阵阶段；静态生命周期追踪不等于SDK层别名安全证明。
- 8762-8968未被Process分派的SolveFallbackGM完整已读：当前输入三角scalar补对称、零矩阵、2次幂缩放、经典Householdertred2、最多80次/根QL、λ排序联动行、RuleB、in-place转置，最后对gm_val_ ENTIRE_DATA_CACHE的DCCI。此函数没有从当前Process可见调用，不能称当前提交一定有硬件受限兜底。
- 8970-8977真实入口AIC直接return（无Cube算子），注册tiling但未GETcopy，构造op.Init后Process。UB4096/48640来自编译期plan而非运行packet随意指定；workspace参数不使用。

数学小路径闭环：Householder对称相似变换H=I-uuᵀ/h且uᵀu=2h，对称rank2式与源码符号一致；三对角T经Givens正交相似变换QL累积Z，最终Q=H_product·Z。2次幂缩放保持精确普通范围内输入表示、最终λ乘scale，非有限/subnormal和近零hard cutoff未证明。为了满足1e-4全局Gram，单次c²+s²近1还不够，需控制45次×n旋转累计；当前小路径没有后验全局Gram输出检测。RuleB只是选等价列sign，本题定义不要求且不能补救不正交。

<a id="r1-kernel-followup"></a>
## r1-kernel-followup Rank1 kernel连续补读（2026-11-09，滚动记录）

以下从原始SHA256=`ecfd60fd2712f2e583de46d3540cbae8e882df7035a4497b51fc3624da5872ec`逐段读，不经宏替换生成物冒充源码。`Z0..ZJ`是局部包装别名，不是硬件新指令；需先还原含义再看数据依赖。

|原始行|语义、编排及适用边界|
|---|---|
|1–115|EIGH_SOLVER_REPEAT重复类体包装；Z0=u32、Z1=float、Z2=c_first、Z3=PipeBarrier、Z4=LocalTensor、Z5=static_cast、Z6=BinaryRepeatParams、Z7=ScalarAbs、Z8=SetValue、Z9=stride、ZA=u8、ZB=row_count、ZC=PIPE_V、ZD=act_rows、ZE=sc_rep_stride、ZF=GetValue、ZG=i32、ZH=ub_mat、ZI=ScalarRsqrt、ZJ=row_start。每个别名先检测碰撞报错；K96_PASS生成另一类名。G：源码包装/实例隔离，不自动加速。|
|117–210|group/lane划分；n256/10lane为24+8×26+24，其余将额外列置居中lane；mode0按8行block分配。vector列归属与协作矩阵行归属不是同一个区间。|
|212–402|mode1两plane；mode2三plane含reflector；mode3 n≤200保完整矩阵+有限refl、n>200压缩32行zone上三角并固定49152float；mode0矩阵/vec共享phase scratch，GM存reflector/列通信。mask存≥、>、=三族8×8表。G/F：UB驻留与活跃三角压缩，必须保Host容量前提；K96分支复用dead stage及mask是独立生命周期。|
|404–514|mode1每lane复制完整小A，只负责所属eigenvector；mode2/3分别中/大resident；mode0所有核按wave调用协作tridiag，非active仅在局部无全局barrier阶段跳过。K96从checkpoint恢复，不是当前Host启动路径；注释CPU-only不是运行CPU fallback的证据。|
|516–641|动态FetchEventID fence；magic rsqrt两轮Fast/三轮普通，Sqrt=x·rsqrt；ComputePow2Scale构指数位，极端/非有限退1；ClampPivot绝对1e-15；三对角移位分解仅相邻主元交换，回代含第二超对角。F：带pivot O(n)逆迭代，不是无pivot Thomas法。|
|644–700|四个shift并行推进特征多项式递推并数符号变更，避免每步除法；没有此函数内逐步缩放，长n/大跨度下溢溢出需范围前提。F：Sturm批量化，不等价任意长度稳健计数。|
|702–842|小A仅载入所选三角，scalar32pitch Householder，成对行更新并复用下一轮norm；先以A00/A01/A11平方和决定是否扫全A缩放，之后再缩tridiag。n1时probe读未初始化a1/a33。早期probe省归约但不能证明其余元素范围。|
|843–977|n≤18且每核≤2列时4shift分割+插值少轮定位；其他用平方offdiag QL（30n上限）后排序；Gershgorin界带padding。绝对0.0015/0.015间隔阈值是分支策略，不是题面gap直接推导。|
|978–1227|1/2列逆迭代，确定性index seed，邻近特征向量可在同核额外重算并正交化；最多3轮，中间2轮rsqrt、最终3轮。最终Rayleigh回估λ、反射回变换、maxabs分量选正号。F：跨核依赖换本地重复计算；只校邻居不代表全局Gram已证。|
|1228–1430|3/4列AoS16同时分解/回代，2轮inverse iteration，末norm和Rayleigh；按column ownership以strided DMA输出，最后MTE3_S保证scratch复用。F/G：共享三对角数值读取+多依赖链，不把种子当预存解。|
|1432–1540|suffix Dot/Axpy，≤64 WholeReduce、8倍长度两级规约、其他ReduceSum；Brcb与独立乘法并发，mask尾后负w rank2准备；开始pair回变换。G：低级规约按布局选，不是统一少barrier。|

|1541–1851|pair unit Householder回变换：合并两个v的矩阵dot，并补交叉项v₀ᵀv₁，不能当成两项可交换反射；FAST200固定stride25，mode3手动mask缓存并恢复64，其他模式分strip规约。F/G：减少依赖层级而不是删交叉项；重复参数依赖物理pitch。|
|1853–2142|单步/成对backtransform、通用suffix MatVec和对称rank2更新；利用Brcb扩8lane权重、64lane条带和独立指令间交错，读写scratch分阶段，最终统一恢复矩阵行布局。F：H₁H₂复合有序；G：只在已知pitch下走扁平Sub捷径。|
|2144–2163|n256 reflector地址随24/96/112/128/144/160/192/224分界切换；n200以11个zone offset/row常量取压缩反射。P：特定容量的生命周期映射，不是数值查表，改边界需同步所有repack与backtransform。|
|2166–2458|中矩阵完整row load+列Gather补对称；大n32row zone只保selected三角，L用列块/U用行块装载，分块2880/3840双scratch归约max后幂缩放；尾padding显式清零。注意浮点0 mask对NaN脏三角不等于bitwise消除，题目另一半“任意脏值”需另核是否允许NaN。|
|2460–2603|IssueLookaheadNormRed只Set V_S不等，先更新下一反射所需首8行，再发下一norm，随后更新其他行，下一轮Wait；Norm=0不发lookahead。F/G：依赖前沿优先、将一次scalar读取延迟隐藏于独立vector尾更新，而非无证删同步。|
|2606–2640|n256在k79将压缩三角展开为176尾，k95压成160尾；bounce先读后写避覆盖，reflections另转192pitch；硬编码704byte Gather stride与生成矩阵pitch一致。P/F：死区再利用与分段layout改变。|
|2642–2782|n200持续缩活跃尾，k7/23/39/55/71/87/103/111/135/167换zone；先保存本轮归一向量再覆盖旧矩阵；首行完成后发norm，FAST200把两个rank1 phase拆开。P/G：需要所有别名阶段同时成立。|
|2783–2872|其他n≤200按可用scratch算tile_rows并上限64，MatVec分条带、rank2两项覆盖前保证读完，反射存旧row后缀。F/G：容量决定tile，不可只拷贝固定数。|
|2873–3109|n256 k79起与其他大容量zone路径；k79/95/111/127/143/159/191/223连续repack，尾由176→160→144→128→112→96→64→32；norm/lookahead状态跨repack保留，slice起点按8对齐。P/F：动态活跃子矩阵可换空间节省后续指令。|
|3111–3360|n256 k15/k23将前24反射写入当前输出的每lane自有列，然后shift回收6144float；上下三角MatVec分别做row dot与转置累加，后者两组zone交错规约。P/G：output-as-workspace要保当前数据依赖、独占区、重载时机；不可笼统等同“0workspace不访问GM”。|
|3361–3674|转置MatVec规约树32→16→8→4→2→1、通用zone分批容量；rank2第一项全发、第二项先首行、提前发norm再补余zone，最后覆写本轮反射；w临时别名最终恢复。F/G：同一V pipe上的排程仍可隐藏S往返，不是不同V指令真正并行。|
|3677–3861|中大路径成对反射backtransform，tau=0退单步；scratch取已死的后部矩阵；n256在uk95预取先前24反射、uk23才Wait使用，数据来自其owner输出槽；非特定n由可用scratch算pair/single tile。F/G：预取距离由生命周期决定，非固定延迟魔数。|
|3863–3880|开始协作路径：每lane读取自有rows和对应完整columns，两次DMA为UPLO补全输入准备；后续正文继续审读。|

已定位的小路径数值边界：令n=4，对称A的对角全1、A02=A20=10^30、A13=A31=2·10^30、其余offdiag0，谱为1±10^30、1±2·10^30，互异且相对gap充分。probe仅看1/0/1得到2，跳过全A预缩放，而首个Householder计算10^60在FP32溢出；这是可从执行式推导的未覆盖输入尺度，不是已在NPU复现的失败。不能以“榜首代码”替代该范围证明。

<a id="r1-kernel-tail"></a>
## r1-kernel-tail Rank1 kernel末段补读（2026-11-09）

原SHA256=`ecfd60fd2712f2e583de46d3540cbae8e882df7035a4497b51fc3624da5872ec`，总8279行；本次连续补读3881-8279，闭合此前1-3880覆盖。该末段包含协作Householder三对角化尾部、PWK→Sturm特征值路线和Sturm区间/根计数实现；`ComputeEigenvalues`按group_lanes与n选择Sturm或PWK回退。逆迭代有缩放三对角、移位带邻近主元交换的LU、最多3轮、gap条件Gram-Schmidt与Rayleigh更新，并有低列数备用路径。后续按反射子回变换、特征向量输出的符号规范化和数据布局收尾；另含Host禁用的K96专用替代实现。源码存在多处假设和硬阈值，但仅静态推导，尚未证明全部题面输入上的收敛、残差与Gram阈值。

分类：task-specific math/dataflow 是Householder/Sturm/逆迭代、分支门限与向量回变换；transferable design 是迭代主路线带回退、gap触发的局部正交修复、输出规范化；general orchestration 是跨核同步、输出DMA与UB/GM scratch排程。后两者的SDK/API正确性、同步安全及性能均未运行验证。

<a id="r2-kernel"></a>
## r2-kernel Rank2 kernel正文审读（2026-11-09）

[R2 kernel](../../extracted/eigh/rank2_submission_1176362/code/op_kernel/linalg_eigh.cpp#L1)原SHA256=`e3171b5bdf9c1f25c6556b4f548da3d6ecfc0166218b7ad1cf129dd7d0fe4fe1`，总6602行，本次连续逐行覆盖1-6602。代码使用三段高/中/低位`ExtendedFloat`算术、驻留UB arena、分层Householder与多条基于尺寸/聚类检测的特征求解路径；小矩阵QL分支带基于原矩阵回代的Gram/残差验收，失败可转后续路径。中等根路径包含Sturm/二分查询、PCR/逆迭代、Rayleigh更新，并对不稳定向量做扩展精度移位求解；medium cluster还尝试构造变换后子空间并检查commutator与Gram，失败则正交化/细化。输出阶段存在全局Gram/残差证书、可修复时的一阶polar更新`Q'=Q(I-E/2)`，以及对证书失败结果重建/替换的控制流；cube相关分支另有投影与跨核协作。

分类：task-specific math/dataflow 是分层精度算术、Householder/三对角求解、聚类检测、逆迭代与Rayleigh更新；transferable design 是以输出全局Gram和重构残差作验收门、可修复则投影且不满足时重建，以及小路径回代证书；general orchestration 是Cube/AIV资源分派、UB/GM复用、跨核空块参与和硬件事件排序。静态检查尚不能证明扩展算术误差界、各阈值覆盖题面条件、polar一步后必满足证书、矩阵/向量访存布局、API同步或硬件资源容量。没有SDK构建/运行，也没有正确性或性能结果。

<a id="r3-kernel-followup"></a>
## r3-kernel-followup Rank3中/大路径及输出生命周期（2026-11-09）

原SHA及总行数见`r3-kernel`节；新增连续覆盖1271-8699。实际可达中路径为`SolveMediumCoop8`（Solver=2），大路径为`SolveLargeCoopStedc`（Solver=3，plan中的N=200/256 case）。前段的闲置`SolveMediumPipeV`不在`Process`调用链。

- task-specific math/dataflow：中路径实现协作Householder与小块/叶子STEDC合并；大路径从按UPLO读取并补足对称矩阵、幂次缩放、分布式Householder降为三对角，进入分层D&C叶子、块合并和rank-one secular roots/向量更新，排序后再应用反射子回到原空间。输出含特征值缩放恢复和按最大绝对分量符号规范化。静态代码能串联主要公式与数据流，但未证明secular求根误差、聚类/deflation边界、正交性或题面误差阈值。
- transferable design：constexpr plan将core/块/叶子映射数据化；两阶段Householder与D&C分层；输出列作临时矩阵/reflector scratch后重载并转置，降低显式workspace；共享边界元数据、按拥有权分配输出列。能否移植取决于固定尺寸和所有workspace生命周期不变量，不应照搬case常量。
- general orchestration：各stage通过`SafeSyncAll`、CrossCore标志、MTE/V/S事件及多轮GM/UB搬运衔接；空闲core仍可能需参加全局同步。源码静态读过同步点与数据阶段，但未在目标SDK验证事件配对、屏障活性、DMA边界、UB峰值、输出/workspace别名、不同工具链的指令约束或运行性能。注释中历史耗时/优化结论不是审计实测。

已确认计划表消费者会从`GetBlockIdx()`索引core记录；中/大路径消费列区间、块bounds和leaf bounds。代码把工作矩阵、tridiagonal数据、边界向量、合并根和输出矩阵按stage复用`cur_vec/cur_val`区域；虽源码能追踪预期的阶段性读写，完整别名安全仍需SDK/内存布局验证，不作已证明结论。

## pending 数学、SDK与运行证明缺口

源码覆盖：按JSON账本，eigh 25,114/25,114行、26/26文件已完成首轮连续静态阅读。此状态不等于已审明所有精确数学边界或所有SDK调用语义。

仍未完成的证明/验证：题面全部允许输入上的特征值误差、全局Gram与重构残差；数值阈值/溢出下溢及聚类/deflation终止界；AscendC API版本与事件/跨核同步的正确性和活性；UB/L1/L0峰值、DMA对齐/边界及输出和scratch别名安全；构建、NPU运行、极端值/随机与条件数测试、官方performance数据的复现及单项优化归因。源码里的作者历史注释、Pass、coverage complete均不替代上述证明。
