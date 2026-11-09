# 01 SafetyBeltViolationGeometry：源码逐行审读证据账本（已覆盖）

> 面向算法理解与复用的主报告已移至[设计复盘](01_SafetyBeltViolationGeometry_REVIEW.md)。本文件保留源码行段、逐文件观察和风险证据，作为审计追溯账本；不应把这些行段笔记当成面向读者的最终报告。

题面：[原始PDF](../../problems/01_SafetyBeltViolationGeometry.pdf)；[题目契约](../SPEC_CONTRACTS.md)。

设计的题目专属/问题族/通用编排分类见 [适用范围总表](../DESIGN_CLASSIFICATION.md) 第01节；应用范围与性能证据独立记录。

审读日期：2026-11-08起，补读截至2026-11-09。下文保留首轮阶段性笔记，因此个别“未读/待继续”描述是当时快照；当前源码覆盖结论以文末 `first-pass-complete`、[coverage](../coverage/01_SafetyBeltViolationGeometry.json) 与上层[设计复盘](01_SafetyBeltViolationGeometry_REVIEW.md)为准：三份提交源码均已连续阅读。原始源码只读，未编译、未执行下载代码、未使用 NPU。不能由公示名次归因某个 trick 的收益。

## contract

完整读过题面提取 01.txt 六页，并视觉检查原 PDF 第2、3页（输入表、公式、精度和尾部规则）。输入 boxes[B,4]=[x1,y1,x2,y2]，anchors[B,2]=[cx,cy]，连续 ND，坐标有限且 y1<=y2；B=1..4096。坐标 FP16/FP32，waist_ratio 是 FP32 属性且在(0,1]，optional threshold 必须是 FP32[1] 且非负，缺省0。输出固定 FP32[B,1]。

计算是先把 FP16 的 y1/y2/cy 升为 FP32，再计算 d=fl(y2-y1)、p=fl(r*d)、q=fl(y1+p)、bound=fl(q+t)，最终 cy>bound 输出1，否则0；相等不违规、无 epsilon。题面要求最终判定完全一致，所以近边界的一个 ULP 不能用“误差很小”忽略。题面推荐顺序明确为 Sub→乘比例→加y1→加threshold。

## build

三份提交的三个 CMake 文件分别全文件 SHA256 相同，复用 rank1 全文审读：根 [CMakeLists.txt:1](../../extracted/SafetyBeltViolationGeometry/rank1_submission_1120490/code/CMakeLists.txt#L1) 1–20 要求 CMake3.16、ASC 包，默认 ascend910b，生成 custom 共享算子包并加入 host/kernel；[op_host/CMakeLists.txt:1](../../extracted/SafetyBeltViolationGeometry/rank1_submission_1120490/code/op_host/CMakeLists.txt#L1) 1–24 用 GLOB 收集 cpp、code_gen、cust_optiling 与自动生成 ACLNN，加入package；[op_kernel/CMakeLists.txt:1](../../extracted/SafetyBeltViolationGeometry/rank1_submission_1120490/code/op_kernel/CMakeLists.txt#L1) 1–15 绑定 kernel目录和 tiling库。无外部脚本执行、无新增编译参数可证明关闭自动同步。

## rank1-host

定位：[Host](../../extracted/SafetyBeltViolationGeometry/rank1_submission_1120490/code/op_host/safety_belt_violation_geometry.cpp#L1)。以下行段均连续阅读全文。

- 1–21：十二段策略注释；SbgHostBranch 将启动核数、工作核数、每核行数、pad、UB预算分开。紧凑预算 FP32=65p+32，FP16=71p+32。
- 22–67：每 dtype 的六段分别为[1,31]、32、[33,255]、256、[257,4095]、4096。B33/257是单工作核；FP32 B4096用8×512，FP16 B4095/4096用6核×768（尾部255/256）。普通区间按16行对齐、pad按64行对齐。8193/8194/8195是区间标签，不是实际B。
- 68–106：读取 AIV 和 UB 资源，以boxes元素数/4取B；只检查元素字节数2/4，未完整核对末维、两输入shape/dtype。资源不足转key0并重算slice；UB非零时检查needBytes。B=0被容许属于契约外扩展，不影响合法B但须单列。
- 107–129：仅真实属性匹配才选B32 ratio1以及大端点ratio.8的常量模板；threshold只记录存在性，实际值不在Host读，兼容字段恒0。tiling七个32位字段28B，waistRatio偏移12。
- 130–141：启动blocks、ScheduleMode1、用户workspace0；schedule不是双缓冲。
- 142–176：推导[B,1]；**输出dtype在FP16输入时推导FP16**；threshold及输出注册同时允许FP16。与题面固定FP32存在直接接口差异（153–169）。这不是因为0/1不能由half表达，而是 ABI 不同。

[tiling.h:1](../../extracted/SafetyBeltViolationGeometry/rank1_submission_1120490/code/op_kernel/safety_belt_violation_geometry_tiling.h#L1) 1–20说明28B结构，21–55 SbgShapePolicy为constexpr共享核数表，SBG_POLICY_TARGET使Host普通函数/Device aicore共用。注意FP32 B33 policy写working2而Host使用1，须由具体类内部静态策略解释，不能仅相信“共享不会漂移”的注释。

[key.h:1](../../extracted/SafetyBeltViolationGeometry/rank1_submission_1120490/code/op_kernel/tiling_key_safety_belt_violation_geometry.h#L1) 1–20：dtype与15位SBG_BATCH，DECL/SEL同24个标签（0、精确点、ratio标签、区间标签），最大20480小于32768；列表位置编码的最终TILING_KEY_VAR不是B。8196被声明但Host通常不选；有模板不等于可达。

## rank1-kernel-partial

定位：[kernel](../../extracted/SafetyBeltViolationGeometry/rank1_submission_1120490/code/op_kernel/safety_belt_violation_geometry.cpp#L1)。已读1–260与3740–4177；261–3739尚未完成，不把十二个类中未读主体当已审。

- 1–41：前言明确“普通DataCopy向上补齐32B”“真实GM分配必须有padding”“结构审阅候选，来源v528；并非原高分二进制”（第8行）。这只能证明包内声明，不能证明公示包与原高分artifact相同，也不能据此推断违规或分数不实。22–33是16行、64lane、奇偶mask、字节对齐函数；位运算AlignUp只适用于2幂参数，现读调用16/64满足。
- 42–111：FP32[1,31]类，start=blockIdx*rowsPerCore，越界核退出，valid=min(rem,per)，pad64；UB boxes4p、anchors2p、outp、constant2p、scratch7p+mask+32B，合计4192B。
- 112–141：boxes/anchors以有效字节向上补齐32搬入；发出两个DMA后标量读tiling[3]以重叠延迟，InitSocState随后；threshold存在才读取，类型跟T。
- 142–211：Duplicate0/1，MTE2_V event0 set/wait后消费；PairReduceSum配奇偶mask抽AoS列，不是跨样本累加；Sub→Muls→Add→Adds，严格GT及Select。对padding lane会计算，但只应输出有效lane。
- 212–237：V_MTE3 event0后写回，非32B长度**仍普通DataCopy向上补齐**，而非DataCopyPad；setglobalbuffer扩大视图不创造内存。239–260开始精确B32类，literal资源4192B、static_assert启动策略；后续主体未读。
- 3740–3745：前类最终V_MTE3与按16个half/块搬出；该片段前置条件未全文读，不能独立认可coreRows整除。
- 3746–3867：fallback分片、可选threshold cache preload，pad64向上，half新增FP32转换暂存；scratch在bankGap非零时直接放65536B地址。构造CMPMASK地址载体时额外预留32B；输入DMA和标量属性读取交叠。MTE2_V事件后才消费数据。
- 3868–3937：float重解释输入；half以GatherMask抽3列再Cast，而非整个AoS升精度；CONST_B float用vcpadd分两组抽列，明确在lUb产生后PIPE_V fence；fallback用PairReduceSum。
- 3938–4035：float常量族直接vsub/vmuls/vadd/vadds并逐依赖barrier；保持分步FP32边界。小constant族直接CMPMASK_SPR消费单repeat；多repeat用UB位图、vcmpv_gt及vsel，half设counter mask覆盖奇数64lane块，结束恢复normal；V_MTE3后DMA仍补齐32B。
- 4037–4135：完整读总入口。SBG_BATCH→base/ratioKind；float/half各六类实际派发，B33/257/4095复用区间类编译参数，4095 float分尾核模板；key0调用fallback，旧未Host选标签也进入fallback。此处只能建立类级路由，未读主体仍未核对。
- 4137–4177：SBG_NO_SYSTEM_WS_MASK=0xFEF及最终key字面匹配；位4关闭、其余相应精确点可替换SetSysWorkspaceForce为空函数。dump/timestamp/OOM保留标准包装。依赖SDK自动生成entry的宏展开和key位编码，属于高风险版本绑定；不能由Host用户workspace0推出系统workspace必可删。

已确定的额外GM容量假设：B=1 FP32普通区间中boxes逻辑16B→搬32B、anchors8B→32B、输出4B→32B。若调用者仅保证逻辑连续ND内存，输入各额外16/24B读取、输出额外28B写入没有题面授权。未取得运行时分配器保证，因此不宣称已有真实越界崩溃；建议用受保护边界/canary复验。行间起点16对齐使相邻完整owner不互盖，但不能解决张量最后尾部的分配上界。

## rank2-host

定位：[Host](../../extracted/SafetyBeltViolationGeometry/rank2_submission_1175118/code/op_host/safety_belt_violation_geometry.cpp#L1)，全文265行。

- 1–18：验证[B,4]/[B,2]与B1..4096，写batch/perCore/阈值类型和比例；24B tiling，waistRatio偏移20。threshold可缺省、half/float动态标签。
- 19–53：64/256/512 tile；float小批32；ratio1且32整倍数启identityTiling。工作核与8核最小launch分开，核资源不足扩perCore至16对齐；若平台AIV为0会发生cores0除法风险，但实际目标有效平台是否可能为0未核验。
- 54–99：nativeSignature要求in/out相同且threshold与其匹配；其余混合接口统一generic。tileClass描述32/64/512/full-tail/worker数；512 near-full尾、64短尾fold、half128前缀合并，以及小shape静态长度族。
- 100–136：真实ratio.8/.9/.5驱动thresholdKind3，它并非第三种tensor dtype；address/broadcast计划由完整资源族选择，不由坐标数值选择。
- 137–201：float metadata、folded256、raw folded、splitHalf、full early、unit entry等重编号，repeatAddressPlan有68+full64等多级tag。只读Host可确认这些结构标签依赖真实shape/属性，不能确认未读kernel全部正确消费。runtimeFold最后覆盖前述tileClass，8核启动。
- 202–239：非native重置四个计划与实际阈值标签，generic block=ceil(batch/perCore)；half/float batchMerge重选launch和tag；某些floatMetadataPlan用8个float的32B metadata覆写原24B整数header，意味着消费者必须按计划解释格式，而不能所有key套同一个struct。
- 240–265：workspace0、输出InferDataType固定FP32；注册额外列举mixed/native及half输出兼容组合。题面要求的FP16+FP32 threshold+FP32 output不是native，**走generic fallback**；不能把half/half高调优族性能直接推广至题面严格ABI。

[tiling.h:1](../../extracted/SafetyBeltViolationGeometry/rank2_submission_1175118/code/op_kernel/safety_belt_violation_geometry_tiling.h#L1) 1–15：4096、floatwide owners8、fold最小launch4，六字段24B。key长文件1184行未全文读，不能声称所有Host发出的8维组合都被SEL登记。

## rank2-kernel-partial

定位：[kernel:1](../../extracted/SafetyBeltViolationGeometry/rank2_submission_1175118/code/op_kernel/safety_belt_violation_geometry.cpp#L1)。已读1–210、2574–2700、4050–4297，其他明确未读。

1–210不仅是可跳过的重复宏：以TILING_KEY_VAR低16位dtype、16位后的threshold索引、tile8位、address/broadcast/metadata4位及repeat6或7位解析编译族。SBG_PARSE_*链把早期配置包含到新配置；宏名涉及动态ratio、full/batchMerge、fold、literalbits、zero-owner、threshold-block-broadcast、分块DMA、SPR bitmap等。全部主要守卫arch2201且关闭CPU-debug/dump/timestamp/OOM，故与调试构建的实现不同。已读范围内不执行数学body，尚不能把宏标签的意图当实际资源证明。

2574–2580是前一私有half主体的末尾（输入前置未读）；2582–2700读入口最前分派：8个模板维度，rawOwner与subOwner零核守卫、Float/Half batchMerge、fallback Geometry<...,0>、short literal、scalar bitmap、foldedRaw。动态ratio路径直接调用DcciGMImpl再volatile读tiling；GetTaskRation参与logical owner恢复。4050–4103覆盖identity32及32尾族；4104–4129覆盖512族、尾核旋转，完整RawEntry另走特化；4130–4194覆盖64族与256/generic；4196–4220闭合大量条件。**2701–4049未读，故总入口中间大量特化未完成核对。**

4222–4297是WorkspacePolicy：解析key筛选half/native、floatFull、folded、identity等，debug关闭unused；函数式宏同时替换SetSysWorkspaceForce和GetUserWorkspace，非unused仍调用真API，unused getter返回nullptr。这比普通workspace0更深度绑定CANN8.5生成器，需原SDK生成代码/反汇编验证调用位置、key编码、系统工作区消费者。

## rank3-host

定位：[Host](../../extracted/SafetyBeltViolationGeometry/rank3_submission_1177391/code/op_host/safety_belt_violation_geometry.cpp#L1)，全文199行。

1–19：SBG_STATIC_SMALL默认1且static_assert只许0/1，AlignRows为64；20–55：完整shape/dtype/ratio合法性验证，并区分严格FP32阈值/输出与legacy half阈值/输出。56–71：**available硬编码8**，未据真实平台查询；pack=(B-1)|((rowsPerCore/64)<<12)|(hasThreshold<<16)，ratio为第二word。72–128：capacity64/128/192/256/320/512，used<=8，八owner ratio.8/.9/1/.5常量族12..15，near-last64尾class19，float257..320分256+tail（class28/29），half257..319 ratio.5 class30。129–146：class16只保留ratio1常量，其余runtime；key包含dtype/output/grid/tail/threshold/长度偏移/ratio；始终SetBlockDim8，用户workspace0。147–199：InferShape严格[B,1]；默认FP32，仅half阈值legacy推half输出；注册包含题面所需严格签名与兼容签名。

[tiling.h:1](../../extracted/SafetyBeltViolationGeometry/rank3_submission_1177391/code/op_kernel/safety_belt_violation_geometry_tiling.h#L1) 1–16：alignas8的单slot8B，偏移/sizeof静态断言。 [key.h:1](../../extracted/SafetyBeltViolationGeometry/rank3_submission_1177391/code/op_kernel/tiling_key_safety_belt_violation_geometry.h#L1) 全222行：1–13声明位宽；14–59严格float的generic/owner/small/tail组合；60–95 half→float；96–131 half→half；132–185 half320完整/尾及class30；186–222 float class29、两种输入strict/legacy class19。每个重复0..63列表都已读，是长度偏移完整枚举，消费者N=offset+1或256+1+offset，非坐标LUT。

## rank3-kernel

定位：[kernel](../../extracted/SafetyBeltViolationGeometry/rank3_submission_1177391/code/op_kernel/safety_belt_violation_geometry.cpp#L1)，全文1369行已按连续1–300、301–620、621–960、961–1200、1201–1369读完。

- 1–22：K_MAX_SHAPE_DIM=0、分拆低层intf头而非kernel_operator总头，type_traits；影响编译依赖/元数据支持，不改变数学。
- 23–113：Legacy Layout容量64..512、32B隔离/packed半精度平面、<192KiB断言；直接LocalTensor绝对UB偏移，无TPipe分配器。Load/Store对齐用DataCopy、不齐用DataCopyPad精确字节；GatherColumn用uint16/32位别名与vreducev2 PATTERN4/6/2抽y1/y2/cy；unusedPattern指针也被构造但模式不应消费，私有契约待核。threshold stage用DMA+MTE2_S ID1，consume配wait。
- 114–247：Legacy VectorThresholdRunTile：真实rows DMA，round64计算；half先gather再cast，packed三列合并一次Cast；Sub+基础Axpy，阈值DataCopyPad单元素整块广播，GT比较。one-vector直接CMPMASK_SPR，多vector UB位图+set_cmpmask+vsel。Legacy输出T；V_MTE3后仅valid写回。
- 248–387：Legacy Baseline先DataCachePreload阈值，然后输入DMA；阈值直接GM标量读取，非零才Adds。其余与上段一致；RunTile只有half256/320转VectorThreshold，其他Baseline。Cache prefetch不是数值缓存，无跨调用答案复用。
- 391–471：LegacySmall仅raw/subblock零owner；N1..64编译期精确DMA，先两路DMA再runtime ratio标量读取；threshold有无为模板；half packed3列Cast；ratio1用Sub+Add而不是直接y2，其他Axpy；GT/Select/store只写N。
- 473–562：Strict Layout的OUTPUT/MASK按FP32长度计算，THRESHOLD也FP32；DMA辅助与vreducev2复制Legacy规则；StageScalar/Consume保证FP32阈值。
- 563–691：Strict VectorThreshold在half坐标下仍FP32计算/输出/阈值，单标量DMA+32B块广播，其余packedcast/mask事件按上段；692–831 Strict Baseline和RunTile与Legacy只在ABI上分离，公式仍Axpy。
- 835–927：StrictSmall：FP32输入预取阈值cache；half输入stage FP32阈值并MTE2_S消费；仍先DMA后读取ratio。未传threshold的模板路径完全不访问其GM；只写FP32 N个结果。
- 930–1065 OverlapLegacy；1067–1202 OverlapStrict：boxes DMA之后设event0，anchors DMA之后event1，先wait0抽boxes，再wait1抽cy；把独立抽列与后一路搬入交叠，非双缓冲。布局、valid写回、Sub/Axpy/threshold/GT不变。标量threshold读取受hasThreshold守卫。
- 1204–1227 DualIO根据Out路由Strict/Legacy；EarlyInput仅用于512容量，static_assert约束。1229–1292入口class29固定256+1..64、class16 small、class30 half320长度、class28动态256+tail；闲核早退，class28把尾派核0、full派核1。
- 1293–1355：计算CAPACITY/USED/PACKED/CONST_RATIO/NEED_SLOT；只在需要时读8Btiling。tail存在时旋转logical=(block+USED-1)%USED，把尾核放物理0；固定lastvector在448..511恢复尾长。general class0解pack取rowsPerCore，start越界退，valid=min。所有合法普通slice应无重复且覆盖[0,B)；不靠输入数值决定B。
- 1357–1369：注释称CANN9.0 experimental entry specialization，非dump/timestamp/OOM时宏覆盖SetSysWorkspaceForce为空。没有CPU-debug/arch守卫，具体SDK生成器适用范围未获官方承诺；不能把“无用户workspace”当系统包装必可删的证明。

## math-memory-sync

rank3的容量是UB上界而非GM授权：所有GM输入/输出按valid长度DataCopyPad，内部round64产生的padding值只作用padding lanes，Gather是逐lane拣选，不跨有效样本求和。512容量strict float总UB=27552B；strict half packed=18208B；均低于代码192KiB断言。推导依据Layout字段逐次累加，包括threshold32B、threshold-float32B、mask ROWS+32、zero32。该静态断言不替代实际芯片UB查询；Host固定8核也需设备前提。无L1/L0/矩阵资源、无用户workspace，已读rank3无跨核同步，各owner地址独立。

重要非显然机制：单元素DataCopyPad{false,0,0,0}不是“其余lane未初始化”的充分证据。官方 [DataCopyPad(ISASI) 9.0](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0265.html) 明确非32B长度且左右padding为0时，dummy复制搬运块第一个元素；因此单float生成8份、half先生成16份再Cast8份。结合 [高维切分语义](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/SIMD_compute/high_dimension_slicing.md) srcBlkStride0重复首DataBlock，可实现threshold全向量广播。此机制应显式记录其SDK/芯片支持，不应用“false总是随机填充”的概括误判。

同步：MTE2_V set/wait保护搬入到V；V_MTE3保护结果到搬出；MTE2_S ID1专为scalar threshold，配对正确出现在HAS_THRESHOLD分支；Overlap0/1两路事件在相同流水方向按数据分开。直接内建vreducev2/set_cmpmask/vsel与SetMaskCount/Norm依赖全局向量状态，源码没有在每次tile函数末尾完整恢复mask，虽然后续调用大多自己设置，复用到别处须补状态契约。普通高层API隐含同步与低层手动barrier不可混为一谈，最终依赖编译器插桩待实机/反汇编确认。

### Axpy边界风险（未实机确认）

rank3 [StrictSmall:902](../../extracted/SafetyBeltViolationGeometry/rank3_submission_1177391/code/op_kernel/safety_belt_violation_geometry.cpp#L902) 基础Axpy替代Muls+Add；官方[精度最佳实践](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/devguide/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0005.html) 明确复合Axpy可能与分指令结果不同；本轮没有证明目标芯片必执行哪种舍入。

自写NumPy IEEE32模拟的最小输入候选：B1，boxes=[0,641.1708984375,1,1347.549560546875]，anchor=[0,1206.27392578125]，ratio=float32(.8)，threshold缺省。d=706.378662109375；分步bound=1206.27392578125（0x4496c8c4），将乘加合并为单次舍入bound=1206.2738037109375（0x4496c8c3）。参考GT=0，融合GT=1。y1位0x44204af0、y2位0x44a87196、r位0x3f4ccccd，便于精确重建。此输入全有限且y1<=y2；这是**可检验的条件性反例，不是已证明NPU失败**。所有Axpy路径都需等号/邻ULP测试。ratio1仍保留Sub+Add，避免简单替换y2造成极端消减/溢出语义变化。

## comparison-and-remaining

> 历史阶段总结（2026-11-08），其中“剩余”部分已由下方 2026-11-09 补读及 `first-pass-complete` 闭合；当前摘要见[设计复盘](01_SafetyBeltViolationGeometry_REVIEW.md)。

rank1：十二类与属性特化、直接内建、32B GM padding、精确key省系统包装；代价是代码膨胀与超题面ABI/分配前提。rank2：结构族/多维计划与原始key位解析最复杂，native快速族与严格ABIgeneric分开，metadata可变解释；代价是模板/生成器/宏复杂度，主体尚未全审。rank3：Strict/Legacy明确分离、8Btiling、容量族、尾核前置、单标量DMA广播、early输入抽列；有完整已读源码支撑，但Axpy精度与硬编码核数/包装替换仍需验证。

可迁移：先抽所需列再half升float、有效长度DMA与计算pad分离、输入DMA和真实标量属性读交叠、事件分流保护独立数据。迁移前提是复现API的字节/块单位与mask状态、保证strict输出ABI；不能照抄TILING_KEY_VAR字面位、系统workspace宏覆盖、未经契约证明的GM padding。

官方未核验契约：vreducev2 PATTERN/unusedPattern、set_cmpmask地址载体、DcciGMImpl、GetTaskRation低层owner映射、SetSysWorkspaceForce/GetUserWorkspace包装替换、SetScheduleMode1调度保证及所有raw指令自动同步。报告不推定这些非法，也不推定稳定可移植。

剩余：rank1 261–3739；rank2 kernel211–2573及2701–4049、key1–1184；必须继续连续阅读并逐族补资源/索引/事件证明，不能由当前Host分析替代。rank3虽全文读完，尚无编译/NPU/逐边界数值与性能验证。原始文件最终hash与本轮盘点一致，账本记录全部21个文件，未读明确标注。

## rank1-kernel-continuation（2026-11-09补读）

以下是连续实读账，不以函数名检索或概要代替源码。仍仅静态审读。

| 行段 | 连续语义及证明边界 |
|---|---|
|261–420|FP32 B32尾段：单owner，UB箱/锚/输出/常数/工作区分离，`vcpadd`按奇偶mask抽y列；Sub→Muls→Add→Adds逐步FP32，UNIT_RATIO仅跳乘1；`set_cmpmask`载体是地址，不是答案。有效32行，输出128B。|
|421–807|FP32 33..255：固定33实际1owner、rowStep48、pad64、UB4192B；泛型按owner分行，pad64。fixed33 mask33、计算pad64，但GM普通DataCopy仍向32B上取整：33行输出132B却搬160B，不能由逻辑长度推导GM分配安全。|
|808–1019|FP32 B256：2×128行，bankGap64float；scratch字节65536，1192float，最高70304B。输入2048/1024B，输出512B。mask地址放scratch最后32B；native算术和compare/select之间显式PIPE_V，末V_MTE3保证写回读源。|
|1020–1600|FP32 257..4095：固定257单owner、pad320，scratch11520/2296float，最高20704B；固定4095八owner，7×512+511，pad512、gap64，UB33760B。boxes/anchors各独立MTE2_V事件0/1：先抽box并计算边界，再等anchor抽cy；这是输入依赖交叠，不是跨tile双缓冲。尾511仍按512计算且输出2048B而逻辑2044B。泛型用65pad+32布局。|
|1601–1826|FP32 B4096：8×512；scratch65536+3976×4=81440B。两个独立输入就绪事件；box参考线先算，cy稍后才等待anchor。输出每owner2048B，完整shape无需GM尾pad。|
|1827–2046|FP16 1..31通用：pad64、71pad+32=4576B，GatherMask固定模式4/6/2先抽三列half再精确升float；ratio真实GM读取；threshold在算术中途读取half（legacy ABI）。三输入列/常数/compare互不覆盖，但DMA上取整仍需GM padding。|
|2047–2248|FP16 B32：1owner；halfStage896B、常数2432、scratch2688，最高4576B。三次独立native cast共用前后V屏障，正常64lane mask复用；后续Sub/Muls/Add/Adds保留。单repeat compare放SPR，再half select，输出64B。|
|2249–2665|FP16 33..255：固定33为单owner，48行定位步长、pad64；box两列邻接half与float布局允许一次2repeat cast，先计算box边界再等anchor事件1；mask33严格有效lane，GM输出仍66→96B。泛型为pad64倍数、71pad+32B，3Gather+3Cast，GT位图Select。|
|2666–2882|FP16 B256：单owner，pad256/gap64，scratch65536+2088×4=73888B。三个原址native cast各4repeat，未把有bank gap的列错误合并。FP32 compare4repeat，half select count模式256，模式复位；写512B。注释称“baseline三次Cast”而实际为native vconv，应以代码为准。|
|2883–3501|FP16 257..4095：固定257单owner pad320、最高22624B，3cast各5repeat；固定4095**实际5×768+255、6owner**，不是前导注释的6×640+255。tail输入补256多读8/4B，计算256、输出510→512B；全文“只搬有效”注释不构成精确DMA保证。scratch65536+7688×4=96288B。native固定模式vreducev2的src1+512依赖SDK；box先做，anchor后合流。真实threshold±0省Adds对最终GT安全（±0相等、NaN比较均假），正阈值保留。泛型仍完整步骤。|
|3502–3739（接既读3740–4177）|FP16 B4096同6owner/5×768+256，UB96288B；calcRows按256取整，尾正好256。事件与half抽列/native cast/FP32GT一致；output真实整32B。后续入口与fallback已在首轮读过，现rank1 kernel4177行连续覆盖闭合。|

分类：列抽取与严格GT公式是 **P**（本题）；按有效长度与pad分离、独立事件提早消费box是 **F**（窄AoS几何谓词族）；地址载体、native mask和无TPipe固定UB是 **G/SDK条件**。明确的GM尾部向上取整是待补分配契约的风险，不是靠获奖名次可消除的证明缺口。

rank1补读结论：所有12个尺寸族主体已读，固定attribute只在Host真实值守卫下使用，不是结果缓存；没有跨tile队列或双缓冲，不能把CopyIn/Compute/CopyOut函数拆分当作流水证明。half三列预抽取与独立cast共用barrier值得迁移；反之“普通DataCopy加padding”“half输出/threshold历史ABI”“覆写workspace包装”必须单列环境/契约，不能作为通用优秀实现照搬。

## rank2-kernel-continuation（2026-11-09补读）

|行段|连续语义账|
|---|---|
|211–440|宏依赖链继续展开：half/float pad-score、raw-owner、SPR、short/fold/near-full等条件由dtype/TH/tile/address/meta位交叉守卫；只有dav2201非debug/dump/timestamp/OOM路径启用。某些高位是编译变体实验标签，不可把宏名当作所有输入都会运行的优化。RAW_ONLY可令Geometry只声明不定义，减少当前特化编译体。|
|441–810|CopyFirst/metadata/owner等编译策略与UB族条件；GeometryWideSocState只清CTRL原子位6/7与计数mask位56，mask是否初始化由模板决定，需SDK ABI而非普遍替代InitSocState。OwnerIndex rawOwner×GetTaskRation+subOwner需真实task映射。SCALE1/2令工作区8192/16384float，即32/64KiB。|
|811–1175|Geometry.Init/Process与NativeWideTile：64/32/256/512及折尾分片，部分start单位刻意为byte-origin，boxes÷sizeof(T)、anchors÷2sizeof(T)、out÷4sizeof(O)保持同一row-origin。pad只在UB，非整块GM用DataCopyPad精确extent。half先抽三列，field连续时一次多repeat升float，并把真实threshold打包第四字段。compare后死区复用；half直接选精确0/1避免float结果再cast。双输入事件1/0分离，尾期anchor-cast与最后threshold-add独立共用barrier。一般是单tile，跨tile退化为MTE3→MTE2串行复用。|
|1176–1450|普通Tile：n≤64用SPR，wide用bitmap；真实scalar threshold支持+0位快路，其他模式重新读float保留加法。单行只搬box内部12B与anchor-y4B，精确字段抽取。DataCopyPad右padding生成ones/zeros、真实阈值padding广播零，不从GM越界读常数。early box事件/后anchor事件；输入DMA可先于CTRL reset，但第一向量/输出之前仍建state。|
|1451–1849|四字段half-stride Cast在源/目标各1024B pitch，完整64/UNIT32域下可合并。依赖完成后复用raw/anchor/y2/cy区域；threshold broadcast和无关Sub共用屏障。严格Sub→Mul→Add→Add→GT保留；±0区别只影响快路命中，非0fallback仍算。最后按n精确GM输出。某些高层API连续调用不带源码显式PIPE_V，其自动依赖语义留SDK R，不据此指称缺同步。|
|1850–1985|真实header24B或batch4B通过MTE2_S读；raw完整512族与near-full505..512族使用固定byte offset布局、ratio.8。tail旋到owner0，prefix为owner−1，共享算术体；tail用align_b32真实字节数，非整块不授权GM padding。+0 threshold省Adds，非0执行真实Adds。|
|1986–2157|raw折尾float64+1..8：owner0尾、其余prefix，64/128计算repeat，真实有效GM长度；prefix和tail别名生存期不同，尾cy要等y1最后读后才复用。dynamicRatio与固定.5分支均保留分步数学。padding ones、自GT造0-CMPMASK、mode1 select复用，非结果表。PIPE_ALL实验只是本地pipe drain，不是核间同步。|
|2158–2424|half共享padones、scalar half bits duplicate、short33..63与float完整64特化：half编码经u16读零扩成u32，little-endian排成[half阈值,+0]，随后精确cast及Brcb。明确依赖小端与native字节/stride ABI；不适合作为跨架构黑盒宏。SPR直接0/1、bitmap+自GT0两个score家族均读真实坐标/阈值。|
|2425–2573|half/float合并128..512整64倍数为单owner，缩请求数但数学相同；bitmap padding预先清零保证额外128lane select安全且不写GM。half grouped-cast三列连续，threshold另cast。随后half UNIT32 nativeDMA主体前段（2574起此前已读），ratio1仍Sub+Add而非直接y2。|
|2701–3130|entry编译期裁剪和runtime fallback双层路由：RAW_ONLY的complete/compact/fold；native19/20与near-full36..98；half107..121与136..150最终是相同497..511行尾（加390/361）；half155..178折128/256域；float179..241短域>32分32+余量两个owner，其余一owner。所有tail搬运精确字节。|
|3131–3580|短域、fold64/128、unit32、wide512、generic256全部C++路由继续；workers==0仅代表全launch owner，不是可不做owner核对。metadata仅tail需要时才读，其他固定族d虽未初始化但所访问成员被constexpr排除。预处理特化分支重复同一if-constexpr骨架，不能计为独立算法。|
|3581–4049（接既读4050–4297）|其余预处理分支与通用fallback完成闭合：fold尾旋owner0；prefix owner−1；短域goto只跳idle不丢有效tile；支持模式保留完整InitSocState。kernel4297行实读闭合。|

### rank2-key-full：key头1184行闭合

1–28：输入输出dtype可绑定编译资源，TH2bit、tile8bit、address1、broadcast1、metadata2、repeat7bit，repeat声明0..75。31–321覆盖所有四种input/output组合的tile0、TH0/1/2（缺threshold/混合ABI走generic，不把native缺省阈值误路由）；float/float与half/half各增加匹配族。322–543为full/wide/unit/fold静态计划；544–1073为repeat1..58逐族实验资源，1074–1184增加float batch59/63/66、halfshort60、fold61/68..75、unit62/64/65、halfwide67。每条SEL都保留明确dtype/TH/tile/plan交集，不是全笛卡尔积。

Host1–265再核对：InferDataType固定FP32，nativeSignature要求实际in=out且有同dtype阈值，因此严格half输入+FP32阈值/输出确实退回tile0。runtimeFold由真实batch/ratio!=.5选repeat68+full64（1..7→69..75），key头末SEL覆盖；halfBatchMerge50+batch/64（128..512→52..58）已覆盖；floatBatchMerge映射tile64..70/repeat66已覆盖；near-fullhalf最终136..150/repeat46已覆盖；其它当前Host的repeat18/19/45/49/60/64/65/67/68均存在对应SEL。中间版本标签仍在key中，但当前Host不选，不可当作启用收益。

剩余SDK证明：上述DECL/SEL的源码值域与Host组合核对闭合，**原始TILING_KEY_VAR位编码仍是SDK前提**（datatype enum两字节、list索引编码不能仅凭源码DECL推出）。没有编译/反汇编证据，不把这个值域核对冒称工具链兼容证明。

新增迁移分级：**P** 严格GT、跳真实+0阈值、ratio守卫；**F** 窄AoS先抽列、少量scalar metadata并入field cast、按死区生存期重排UB、尾owner前置；**G** 真正编译期裁剪/独立事件/单repeat无效repeat-stride取0；**G/SDK条件** CTRL位、raw owner映射、native copy/compare/SPR ABI。所有“descriptor改两段/四段”仅是等效布局候选，源码没有性能证据，不能称更快。

数值细分：只有half输入、真实ratio=.5且d已FP32舍入的特定族使用Axpy；有限half提升到float后差值范围≤131008，乘.5不下溢且精确缩放，因此合并最后乘.5与加y1不会引入第二个乘法舍入差异。此证明不推广到float输入或ratio.8/.9，亦仍需目标SDK确认Axpy参数/融合方式。

## first-pass-complete（2026-11-09）

本题三份共21文件均已全文连续审读或SHA相同构建文件映射，rank1/rank2先前的“未读”段由本续审闭合。保留前文首轮状态是历史记录，以本节与coverage为当前状态。完成是源码覆盖，不是正确/速度/合规盖章。最高优先证明缺口仍为：rank1与题面FP32输出/阈值ABI不一致及GM pad；rank3 Axpy严格边界；rank2/3私有CTRL/owner/native包装在目标SDK的实际语义。所有原源码未修改、未运行。
