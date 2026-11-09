# 10 TreeLineMinDistance：连续逐行语义审读

题面：[原始PDF](../../problems/10_TreeLineMinDistance.pdf)；[题目契约](../SPEC_CONTRACTS.md)。

状态：first-pass source review complete（2026-11-09）。顺序1/2/3源码均已连续实读或按相同SHA256复用；本轮不编译、不运行下载代码；不以获奖名次替代正确性/性能证据。源码总覆盖以 [账本](../coverage/10_TreeLineMinDistance.json) 为准。行号按UTF8物理行；重复展开的公共头也记录范围和复用依据。

## A. 题面和数学基线

已读全部10.txt四页。点[B,N,3]、线段[B,M,2,3]，同dtype FP16/FP32；B1..8,N1..16384,M1..1024，不广播。有限坐标，建议[-1000,1000]，不是明确数值上下界保证。输出距离同dtype、索引int32精确；FP32容差1e-4，half1e-3；half中间差值、点积、除法、clamp、归约与sqrt均FP32或等效高精度。允许比较平方距离，但最后必须sqrt。相等距离取最小索引；普通测试避免两最小距离过近，但没有可用作数值证明的定量gap。

令u=C-A，v=P-A，L=u·u。L>0时t=clamp(v·u/L,0,1)，d²=||v-tu||²；L=0时直接||v||²，禁止对0做除法。若按j升序且仅在新d²严格更小时替换索引，则相等时保留最小j。多级归约必须在块内和块间都保持此规则；只证明最终距离精度不够。

## B. 顺序1 Host全部538行

前缀 `extracted/TreeLineMinDistance/rank1_submission_1175341/code/`。

| 连续行段 | 每段实读内容与边界 |
|---|---|
|1–41|内联模板头的include guard；Config先定义batch/m/n/k等u64，再dtype等u32。注意聚合初始化第四个位置是k，不是dtype。|
|43–113|十题公共sizes函数逐分支已读：HOST元数据验证和字节容量计算，不计算输入坐标。实际本Op只调用TREE分支68–70；其余题的通用检查不能当本kernel都执行。TREE要求B≤8,m≤16384,n≤1024，两个输出分别B*m*z、B*m*4。|
|115–157|再次展开Config与TemplateTilingData，guard使已定义Config不重复生效；tiling含Config、scratch_bytes、6输入dtype和2输出dtype。无序列化宏，需Host/Device固定ABI一致。|
|159–200|Shape辅助：product控制正维和积上界，exact要求全shape相同，empty识别零shape，Plan包含tiling/输出/workspace。|
|201–273|完整十题make分支；TREE211–216要求[B,N,3]和[B,M,2,3]且batch相等，两个输出[B,N]，第二dtype I32。其他题分支不被本Op调用。末端统一64B workspace，最少64B，后续专用tiling会覆盖为0。|
|277–340|通用dtype/输入数/属性转换，TREE无属性操作；异常用于Host返回失败，不是device数据分支。|
|342–369|infer_shape用代表FP32进行元数据验证、写输出；infer_type距离继承输入、索引INT32。|
|370–431|tile获取输入storage shape与dtype、核输出shape/dtype、构造公共plan、先选1核/系统workspace；try/catch转换GRAPH_FAILED。只检查metadata，不预计算答案。|
|433–484|专用路由：单点/单线段/tiny、短M、一般pair、大规模M1024各族；多个确切shape专化，均仍从GM读实际输入。无证据据此判违规，但要区分同类资源族设计与仅此shape参数。短M才运行UB预算检查；大pair用裸固定UB。|
|485–496|真实key与launch：PATH8..14/17通常launch8，PATH16 launch9，PATH7 launch1；较大多batch可batch-pinned占满AIV，前提cores%batch==0。专用plan最后workspace0，不能把公共plan的64B当实际分配。|
|498–538|infer和Op注册、ND、float/half、int32、910b；OP_TILING_LIB下导出shape注册。|

### B1. Host→入口路径对照

| PATH | Host条件/编排 | Device执行 |
|---|---|---|
|8/9/10/11/12|B1，(N,M)=(1,1)/(1,2)/(1,6)/(3,1)/(3,2)，通常launch8|tiny_fixed，仅block0计算；不能把launch8理解为8核分摊计算。|
|7|其他points≤3且M≤6，或M1且points≤32|只block0，动态tiny；count上界32，符合静态输出区。|
|1/20|M1；20另要求B1,N16384|沿点向量化，不做argmin，索引全0。20入口只前8块。|
|4/14|N1；14要求B1,M1024|沿线段向量化，先局部min再全局min。14只block0。|
|5/6/13/16/17|M≤32、points≤64；13=B1N19M7，16=B1N33M31，17=B8N3M19|TreeShort，width8×tile8或width32×tile4，点×线段在UB内展开，不写完整GM距离矩阵。|
|18/19/21/22|B4N127M129/B2N256M64/B2N1003M513/B2N4097M513|TreePairs tile32或64；固定shape入口省部分tiling读取。|
|15/23|15为两组确切大shape；23为M1024,N≥8192且64整除的其他族|run_1024：half粗筛16候选再FP32精算，见F，不能直接推广为契约已证明的精确argmin。|
|2/3|其余一般/较大shape|TreePairs tile32/64，遍历全部有效线段。|
|0|key有枚举，当前Host不赋值|标量compute_tree_range保留路径，不作为实际获奖执行来源。|

key头1–16的PATH0..23在DECL与SEL均有，dtype两种；5bit能容纳。tiling头1–46含相同guard/Config/TemplateTilingData；short_plan1–19给width>16时tile4，否则tile8及UB容量公式。具体UB公式是否覆盖未来改布局不能由注释保证，实际布局见下一节。

## C. 顺序1全部Device文件的连续语义账

三个CMake为根20/Host24/kernel15行，与已审SplineConv同类文件SHA256完全相同，复用审读。`tree_vector.h`的1–217行与tree_edges/tree_tiny/tree_pairs/tree_short各自1–217行逐行字符串比对完全相同；这四份公共头复用已经阅读全文的portable段。除复用外，下表全部连续读过，无省略的中段。

### C1. 主入口 `tree_line_min_distance.cpp` 全659行

1–220是Config、float/half/BF16位级转换、Tensor Scalar GM访问、device Math的完整portable展开；222–278是tiling ABI和状态Trap；280–284 includes；285–504又展开相同guard公共头（实际编译跳过重复定义）；506–545为参考标量算法，按升序strict<、精确L>0分支、每点一次sqrt，且Scalar GM输出后flush。CPU模式只用于同源算法测试的条件代码；没有证据说NPU入口会回退CPU。

547–618为静态PATH13..23等入口，619–626 tiny或scalar闲核退出，627–659读取必要config字段并派发short/one-point/tiny/pairs/scalar/one-segment。`Config c{B,N,M,dtype表达式}`第四项实际写k，但这些模板消费者依T判断dtype且不读取c.dtype，故当前不能据这一写法声称half被按float读；它是字段易误用/维护风险，移植为Tensor消费者时会出错。tree_vector函数存在于头文件但当前入口没有调用，不能为其归因性能。

### C2. `tree_vector.h` 全313行

1–82实现float bit/half subnormal与ties-even舍入/BF16；84–169 Tensor支持CPU条件访问与NPU GlobalTensor/GetValue/SetValue、offset slice、DCCI；171–217 Math通过8lane Sqrt/Exp加S_V/V_S事件实现单Scalar超越函数。219–247分配tile512、14个FP32平面、raw、offset、output、mask；half raw先升float，Gather以12B字节距拆三轴。249–299每point-tile搬点，一次一segment、L>0才Div、夹t、求d²；Compare以整64lane，但Select仅有效tail count，strictLT更新index，Min更新值。300–313最后sqrt/cast和精确字节双输出，等待MTE3才能复用。该函数未被当前入口调用，仅作为旧/备选实现记录。

### C3. `tree_short.h` 218–377

218–241：width8或32、point tile8或4，pairs=64/128，work_stride=(pairs&64)?pairs:pairs+64，追加64元素使平面起点相对512B变化；固定raw/stage/work/offset/result地址。242–267布局各轴/广播/中间值，实际T决定升精度。268–299循环任务、两次精确字节DMA；只Scalar建8个offset，随后Adds扩展到32，避免每元素SetValue；Gather拆SoA。300–322 width8把3轴合repeat，323–355 width32把点×segment按repeat stride广播并计算。两版用max(L,FLT_MIN)避免除零，见F的数值边界。WholeReduceMin以真实M mask，尾线段不会参与min。357–371 GatherMask拆[value,index]交织结果，先发index的V_MTE3，再sqrt/cast距离，最后单独fence发距离，利用索引与距离不同完成时间。373–377关闭类和单事件fence helper。

### C4. `tree_edges.h` 218–456

220–235前32个Scalar offset后按倍增Adds扩表。237–284 one-segment tile2048、9个带64元素gap的工作平面；length>0先算倒数及u/L，索引全0单独初始化；285–320先写索引，V同时完成点Gather/投影/残差/sqrt，末写距离。单segment不需要全局argmin是题目专属语义消元，而独立输出先发是同类可迁移编排。

322–371 one-point每batch一核，1024线段上限，raw/SoA/offset/result布局，尾部无效lane初始化Inf。372–394 M1024时把三轴拼接为3072连续vector指令；395–420一般M分轴。这里使用A-C与A-P，两个符号同反，dot和残差平方数学相同；不是几何方向错误。421–456两级WholeReduceMin，仅偶数value槽参与二级归约，再以slot/2*64+局部index还原全局index；sqrt先发，Scalar索引恢复与V重叠。WholeReduceMin平局行为仍需目标版本API/产物确认，不能仅从函数名证明最小索引。

### C5. `tree_tiny.h` 218–570

220–241裸UB视图与packed Half4/Half2/Float2 union。243–266按dot≤0返回起点距离、dot≥L返回终点距离，仅内部投影做除法：对零u，dot=0直接走第一分支，不除0。该分支简化依赖几何公式，是题目专属；“先判无需复杂算术的区域”是可迁移原则，不适用于任意向量公式。

268–344动态tiny，Scalar读实际坐标、每点strict<min、UB暂存最多32项，最后一次S_V后批量sqrt/cast并双DMA；没有为每个点往返V_S。285–288 volatile 64bit预读存在边界风险，见F。346–388静态tiny只block0，按P/M决定额外预读及固定输出计数；389–447 P1M1/M2用64bit或32bitpacked标量读取，M2用strict<保持较小index；448–490 P1M6禁止展开循环，避免过度代码展开；491–540 P3M1/M2复用线段信息，每点重新计算。543–568先发sqrt/cast，再由Scalar写index并发S_MTE3，两个输出分别等待依赖，不用PIPE_ALL一把串行化。

### C6. `tree_pairs.h` 218–861

| 连续行段 | 含义 |
|---|---|
|220–289|一般pair kernel固定segment_width64/point_stride512，各工作平面4096float；显式地址不与输入暂存同时活跃区域冲突。230–248注释出现两套历史布局，必须以249–264实际常量为准；不能把注释bank方案当现代码。seg_raw在131072与后续工作平面复用，segments在dx区，先转换/SoA后才覆盖。|
|291–339|任务按point tile拥有完整输出；batch_pinned只在N≥512、batch>1、核数整除batch时成立。固定batch使每核线段预处理只做一次；full waves+最后均分tail，count≤point_tile，不需要跨核合并argmin。|
|340–404|batch变化才整批搬线段和第一块点；offset前8项Scalar、随后倍增Adds；M=64q+1时把末线段offset广播至padding64，尾边单独处理；half先Cast。算u和L，再预算u/L三轴供所有点复用，减少每点Div，但改变舍入次序。归一化分母也max FLT_MIN。|
|405–454|点AoS→SoA与Brcb；64条segment块向量算投影/clamp/residual/d²，每点WholeReduceMin输出[value,index]对。只剩1条的尾块跳出主循环，后面沿point向量算一次，不用64lane重复同一线段。|
|455–504|GatherMask分离所有chunk min与index，给index补chunk*64，tail1走broadcast预处理；块间strictLT并SelectCal同mask更新距离和index，升chunk顺序保持块间tie较小索引。块内tie仍依API契约。|
|505–517|先允许index写出，sqrt/cast与之交叠，再距离；重用result前等待前次MTE3。|
|519–650|run_1024专用：9平面FP32/9平面half、16候选scratch、固定最高UB越过188416的SDK临时区并标注空洞；实际硬件可用UB与编译器scratch仍需验证。核按batch固定，预取第一point块；half位模式0x3800+i构造精确index ramp和候选剔除mask，0x7bff为最大有限half而非Inf。|
|651–713|完整segment FP32预处理，动态power2缩放防half溢出；maxcoord从segment A/u算，不含尚未完整处理的全部point；缩放后A/u/uL转half，uL另夹±65504。缩放不能自动给出投影/距离排序误差界。|
|714–751|按wave获取点，先Gather到独立point，再预取下一raw；仅待当前raw不再被V读才覆盖。point也缩放/cast到half，点幅值若远超segment尺度未在scale估计内。|
|752–787|8个128segment chunk，每chunk用half算几何d²选Top1，以half码ramp/mask排除该index再选Top2，总16候选。这里明确用了低精度筛选，不是全1024 FP32。|
|789–834|每chunk两个index排序、加入全局chunk偏移，用half可精确表达的小编码转float后*8192形成Gather字节offset；读取原始FP32几何量，重新算16候选距离。对保留候选精算不能补回已删除候选。|
|835–861|候选升序合并，strictLT维护index，最后offset>>2恢复segment index，精确输出；下一wave用MTE2_V/MTE3_V事件闭环。|

## D. 适用范围分类：什么能借鉴，什么不能照搬

分类有两条独立轴：**应用范围**与**证据成熟度**。泛化范围大不代表已经证明正确或更快；题目专属也可能非常优秀。

| 设计 | 应用范围 | 可以提炼的原则 | 迁移前提/不能直接照搬 |
|---|---|---|---|
|仅对最终min开方|题目专属数学→单调变换归约这一问题族|减少昂贵单调变换次数|变换在有效域严格单调，tie/NaN/overflow规则要核对；不能对先求和再sqrt同样改。|
|M1删掉argmin、N1改沿M向量化|几何题专属→退化维度问题族|按退化维度改变并行轴并消除无意义步骤|输出定义和索引0由语义保证，不是把答案常量化。|
|segment驻留且batch-pinned|pairwise/广播类问题通用|把不变量预处理移出point循环|数据真跨任务复用、容量足够、batch分配守恒；不承诺总比均衡调度快。|
|64q+1尾边改沿point算|非整齐两轴计算/归约族|特殊尾块选更有效并行轴|每点仍覆盖最后segment，比较和tie规则一致；q/width是实现参数不是普适常数。|
|小表前8/32项Scalar后向量倍增|规则索引/地址生成族通用|减少Scalar SetValue依赖链|索引为规则仿射，字节/元素单位、整数溢出、真实mask均须证明。|
|索引与距离分别ready和写出|多输出kernel通用编排|独立输出不要共等最慢链|不共享正在写的UB，分别满足S/V→MTE3事件；实际收益需测。|
|裸UB间隔/内建SelectCal/mask SPR|Ascend目标架构/SDK专属|控制bank和少包装指令|平台、CANN版本、临时UB区、指令hazard证明；不是算法层通用设计。|
|launch8仅block0数值|平台调度候选，非几何专属|区分launch规模与有效计算核数|需目标机单独比较1/8launch；公示源码存在不证明加速，闲核功耗/调度亦有代价。|
|half粗筛Top2再FP32精算|候选筛选这一问题族，但当前未获正确性证书|用廉价有界近似缩候选|需要误差界+保守候选集合/失败fallback。本实现固定Top2无保留证书，不列为可直接复用优秀方法。|

## F. 明确风险与不能从Pass推出的结论

1. **S，逻辑张量越界预读。** tiny_fixed353行不论dtype先volatile读8B。合法B1,N1,half点只有6B，超2B。M6 half线段72B，但366–368还读偏移64/96/128各8B，后两次超逻辑张量；即便底层分配有padding，也不满足PDF“不得越界读写”的原始张量契约。目的可能是预热cache，不能迁移为安全trick。动态tiny285行同类风险。没有执行，不断言一定崩溃。
2. **S，half筛选和题面要求不一致。** Host可对half大shape选PATH15/23，752–787确实以half算距离/比较并删除候选，和题面half中间FP32要求存在差异。FP32输入也走同一筛选，最终16候选FP32并不证明整体argmin正确。应先核比赛是否有后续题面版本，不能直接给人物/合规性下定论。
3. **M，Top2不是保留证书。** 对任何chunk，若三条不同真实距离在half映射下相等，保留两条可能删真最小。power2缩放保指数动态范围，不保持所有FP32相对间距；没有定量gap/误差界或保守fallback，不能推导不漏解。Stage2正确只证明候选内最优。
4. **M，max(L,FLT_MIN)不是无条件等价。** L=0时u=0，结果可保持退化点；但0<L<FLT_MIN时改了非零分母。half输入可表示的最小非零坐标差约2^-24，其平方2^-48仍为正常FP32，该边界主要影响FP32输入；FTZ行为和极小值测试范围仍需设备/契约确认。先取u/L再点积的溢出和舍入也要另证。
5. **R，API/编译期与性能。** WholeReduceMin tie、SelectCal内建、无显式PIPE_V间隔的依赖、裸UB最高区和SDK临时buffer均未在本环境编译/验证。不能用“源码全读”替代这些证明；也不从名次为某一处gap/launch归因收益。

以下顺序3补审完成后，本报告覆盖rank1/2/3三份源码路线；它仍是首轮源码语义审读，不等同于数学证明或目标设备验证。
## G. 顺序2 首轮连续补审（2026-11-09，当前kernel至1800）

Host412/key23/tiling58已全文连续阅读。Host验证B/N/M和同dtype，AIV预算上限64、UB描述符18bit；先通用资源族路由，再短table packed、静态scalar和dense scratch替换。最后Host将B−1/N−1/M−1/workers−1/UB编码为51bit私有描述符，写前两个u32；仅在明确新layout消费，不能再按旧struct字段读取。KEY存的是DECL列表下标而非ALGORITHM数值，dtype占低8bit；SEL有意删除旧族，必须对完整Host可达域枚举验证。launch最少8，有productive worker guard；空间排序族另workspace16B/point，不是Host读取坐标。正常TABLE数据始终设备处理。

| kernel连续行段 | 语义、布局、数学/同步 |
|---|---|
|1–45|规则offset前8 Scalar种子+向量倍增，fresh表的Seed/Expand分离允许合并一次S_V，旧活跃表仍用完整V_S/S_V保护。|
|47–206|SpatialTreeLine按每worker点切片做Morton30bit排序（3轴10bit），代码bit转float正数实现排序而非数值float索引；填充极小哨兵。保留排序点于UB或GM并保留order，按Morton前缀分至≤64 cell，每cell再≤64点几何块。这里只改变访问/候选顺序，不假设空间排序就是答案。|
|207–309|全部segments FP32 SoA、不变量方向/逆长度、AABB、三个垂直方向长度倒数。零L选1保留非零分母；SAT分母floor1e-20仅弱化下界，不改变最后几何计算。22mp状态平面；candidate coefficient gather额外896索引scratch由预算选择，不够仍可原7次Gather。|
|310–438|每cell真实点包围盒；AABB平方间距和垂直D×axis分离平面下界取max。较小M用点盒中心到线段投影方向构造分离平面，(norm−support)²/norm是实数下界，norm0选1。key=负下界，先compaction最小下界/零下界，再以当前最差point上界阈值保留剩候选；负无穷padding不进入正常候选。|
|439–608|64或8/16/32打包候选计算真实FP32距离，WholeReduceMin后Gather恢复原segment int32索引；跨chunk distance相等显式min原索引，不依候选排序。初始化best只由真实非空chunk；短候选padding索引0但归约mask只count。half输入仍FP32中间（其平方用FMA），FP32输入分乘加。|
|609–773|恢复原点顺序，再sqrt/cast，exact-byte双输出。保留point/state/查询scratch有互斥预算，输出异步pending在下一Rank覆盖前等待。当前best最大值+3.2e-5*(segmentScale²+bound+1)作为浮点安全margin；实数下界推导成立，但这个margin不是已证明的全有限FP32误差界（点幅度也未单独纳入scale）。异常空候选fallback全部segment只覆盖seed异常，非每次剪枝认证。|
|775–1298|保留旧generic Scalar/PointVector/Vector实现，需结合最终入口确定是否被SEL取代。Scalar按strictLT，PointVector缓存短表、M1删argmin，Div一轮残差/Newton修正维持投影精度；Vector点×段矩阵按UB适配rows，尾M多层归约并恢复全局索引。L零用整数bit方法只把精确0改1，不把全部小正数夹FLT_MIN。|
|1299–1534|轴对齐段直接区间clamp，点段同时成立；其他segment只有内部投影除法、端点直接用实际end而非a+u避免重构舍入。ScalarDmaTiny/FixedPoint精确搬真实输入extent，无rank1式volatile超范围预热。固定单点FP32 M≤2用WriteGmByPassDCache+DDR fence发布单producer输出；SDK语义仍待核。|
|1536–1697|OptimizedLayout独立入口保留同offset工具；SinglePointTreeLine按16..1024模板宽度、实际m mask，half先升FP32，二层min恢复原索引，最终bypass精确dtype输出。与legacy不同上下文独立审读，未按相似名字跳读。|
|1699–1800|TiledTreeLine新族按StaticSegmentPitch 128..1024专化；segment和首point DMA一齐预发，复用段SoA；残差scratch在4096上半平面，固定短span才可使用。读取half后Cast、所有几何仍float，零长选择安全1。后文消费者与完整输出/预算尚待闭合。|

可迁移性：空间排序+保守几何下界属于 **F（几何搜索）**；Morton指令、SIMD短候选packing、索引种子扩张、独立输出和别名活跃期属于 **G/SDK**；指定ALGORITHM编码与51bit描述符是此实现 **P/SDK ABI**。需要保留的是保守下界+tie规则，不是固定TopK的经验筛选。

## H. 顺序2 kernel 1801–4038 连续闭合

| 行段 | 全文补审结论 |
|---|---|
|1801–2096|Tiled全64候选主块，count≤3且点≥32时沿point算scalar tail，其他8/16/32候选打包tail；三者相同residual公式/平方顺序保持跨块tie可比。按segment升序strictLT更新，块内tie依WholeReduceMin。首段与首点DMA并发索引生成，下一点预取只覆盖独立pointInput；尾batch不跨表。固定8256平面距、33024后状态与37504点暂存，Host预留37888元素+8mp+8KiB闭合。|
|2098–2538|DenseScratch独立类全文另读。Host只对完整worker span32..64、M pitch256..1024选此族，所以usesResidualGeometry恒真；将productScratch移dot上半、residualY移x上半、平方移y，减少多源同bank。若删Host资格条件，代码仍无条件min(residualSquared)会消费未定义值，不能把该类当任意point span模板。tail>8先保证broadcast最后读结束再复用上半scratch；scalar tail仍用相同平方序。|
|2540–2644|≤3点×≤21段FP32的三slot交织，只在同batch/segmentCount*3≤64选择。每3lane一个segment，三套mask分别归约；恢复index使用floor(43*lane/128)，对0..63严格等于floor(lane/3)。合法数据仍全部对计算；不是TopK剪枝。|
|2646–2991|PackedRows W2/4/8/16/32，point cap224/240/216确保uint8 repeat和UB。W<8用GatherMask裁掉Brcb多余复制、按subblock slot mask独立min，商做残差修正；W≥8保留倒数乘与residual/FMA。FP32和half均FP32中间。首DMA与新索引Seed/Expand并发；半精转存后再Gather，原pointInput不与下一DMA活跃读冲突。全有效pair覆盖，无静态shape答案。|
|2992–3137|Optimized generic另有64半精offset种子uint64配对写、一份lane ramp派生12/8/24B索引、完整64段不重复清零；partial仍清padding。零长只改精确0。属于减少初始化/Scalar指令，不能把不需初始化推广到尚未被DMA覆盖lane。|
|3138–3395|PointVector的M1 FP32长tile缓存A/u/L和Newton校正倒数至Scalar，w≥1024平面另加64float gap并复用无argmin平面；首点DMA先发，元数据与invariant计算在其间。输出indices为0由M1数学保证。多segment保完整strictLT。|
|3396–3689|generic Vector所有分支独立连续读；rows≥8向量化恢复min/index，否则Scalar收集，STATIC_WIDTH64半精将dist放已死dot平面，避免tmp仍参与平方的银行冲突。offset消费者读完后才能把ramp改segment字节距。|
|3690–3792|Scalar metadata只读必要当前words，layout descriptor支持8B对齐一读或4B两读；逆位域与Host相符。TryStorePairedFloatResults检查地址差≥4且dstStride可u32表示，把distance/index按GM地址排序后两burst一条DMA，只写每处4B，不填中间gap。FP16保持不等宽双请求。|
|3794–3918|Coalesced M1小点集launch8仅block0，两个输入DMA、共享段不变量，全部点和索引连续DMA。StaticScalarDomain每productive block一点，编译期N/M守卫，B1 N≥8 M2用PointsPerBatch0不读tiling（Host精确launchN）；多batch仍读worker数量排除padding。几何与actualGM输入保持。|
|3920–4038|真实路由：849..854 coalesced，864..876 dense，784..790 wide单点，397..400/408..415优化generic，768和512..608 scalar，320..327多batchM1，64..304二维scalar标签，37..51 tiled，52..56 packed，57M1 pointvector。≤36旧族仅SEL仍留1/3/4/5..12；spatial用GetUserWorkspace，其余无workspace。不能把legacy0/2/13..36代码当启用路径归因。|

顺序2有价值的差异：相较顺序1的half Top2粗筛，它选择**设备空间排序+保守几何下界、真实FP32候选几何**；可推广到范围搜索/碰撞检测，但浮点下界安全margin仍待误差证明。新微编排涵盖packed短行、多输出合并DMA、元数据压缩、fresh索引共享handoff、死平面复用。均须目标SDK确认裸UB/WriteGmByPassDCache/DDR completion与指令hazard，不从作者注释“measured gains”判定当前硬件收益。

## I. 顺序3 rank3_submission_1175418 首轮连续补审（2026-11-09）

顺序3剩余4个源码文件均已通过 `read_source.py` 按连续编号范围读完；各范围SHA256由读取器校验。无代码执行、无SDK/数学/性能验证。

| 文件 | 连续实读范围 | SHA256 |
|---|---:|---|
| `op_kernel/tiling_key_tree_line_min_distance.h` | 1–126 | `71049b9894c0314b06ee7fd1cf4b7511f93bfdb72dbdf8ac4b11da75abb94f68` |
| `op_kernel/tree_line_min_distance_tiling.h` | 1–6230 | `c632673efc0bfc5a31b8fcb74594d9f2cfd42f9822a7f6e0df6e302a5ba4b904` |
| `op_host/tree_line_min_distance.cpp` | 1–247 | `008401c08ffd0bf390c046ea306da1017383a2b114e0c90c2bf7969d9b6f7e42` |
| `op_kernel/tree_line_min_distance.cpp` | 1–7562 | `a9d29748754e035577ed3b302590eb305d4f67e0298c85950776f3acef4aec13` |

### I1. Host与模板选择

Host 1–247验证rank3路由支持的B/N/M、输入秩/末维和同dtype FP32/FP16；以AIV核心数、UB容量、padded segment数和point tile生成schedule、blocks、模板参数及workspace。Schedule多处被后续条件重写，包含single-point、short-tail、resident-owner和空间排序分支；仅依据Host表达式不能证明SEL展开后每个配置均可编译或覆盖预期。Schedule与resident-owner是实现ABI/资源选择，不是算法贡献本身。`InferShape`仅检查形状兼容并输出[B,N]，shape inference中不重复Host的范围限制。

Tiling-key头 1–126声明FP32/FP16、schedule与resident-owner模板域，并以宏/条件SEL裁剪类型、schedule和kernel-type组合；注释中的schedule说明不构成实际可达性或设备能力证明。

### I2. Device实现与机制边界

`tree_line_min_distance_tiling.h` 1–6230包含多代搜索资源与大量按dtype/capacity/tail编译期分支。整体仍是点到线段投影clamp的几何距离：常规候选先用FP32近似投影/平方分数块归约；边界、tie或误差证书不能唯一确认时进入补偿几何、候选精化或精确回退。文件含prefix/chunk下界、两词位图tie证书、短候选复算、零方向/轴向与共线修正、空间裁剪稳定压缩与原索引映射等变体。若证书不满足条件，路线保留更完整的候选或exact fallback；但浮点margin、所有边界覆盖和fallback等价性仍需独立证明。

`tree_line_min_distance.cpp` 1–7562还包含radix-limb形式的浮点/整数有理分数比较、误差无损变换与端点规范化、分组归约后的候选gap/唯一性检查、FP16 dyadic/int-lattice精确路径及近tie修复，以及面向不同schedule的packet/pair/tiny/通用入口。一个单独空间路径将坐标量化成Hilbert key并排序/分桶，调用既有batch几何路径后再逆映射输出；从源码可见其排序键用于重排与分区，不是pair几何答案。以上描述只陈述源代码结构，不确认算术范围、比较正确性、证明完备性或性能收益。

### I3. 可迁移性分类

- **题目特定数学/布局**：线段投影距离、端点/零长段处理、FP32/FP16误差界与tie/近tie证书、dyadic精确分数；固定UB平面、packet宽度、特定shape/schedule资源门槛。数学公式和阈值不能脱离相应坐标/舍入前提复用。
- **可迁移算法/数据设计**：radix limb向量算术、误差无损变换、保守候选裁剪后对歧义子集精化、Hilbert空间排序与稳定compact/逆映射、保持原序tie语义的候选压缩。迁移时仍须为目标问题重新建立下界/误差/无漏候选证明。
- **通用编排/平台集成**：schedule/模板选择、workspace布局、GM↔UB双缓冲、事件flag与cache控制、异步输出/回退入口。它们是Host/Device ABI及平台API约定，不是纯算法层可直接移植的证明。

### I4. 不能由源码审读推出的结论

本topic现为27/27文件、22,728/22,728行的首轮源码审读覆盖。rank3源码中的“exact”“certificate”“safe”命名和注释不是数学证明；未核算所有有限输入/舍入域误差，也未证明阈值/归约API的实际语义。未在目标CANN/硬件编译或运行，因此模板可达性、Gather/Reduce/Select掩码、UB地址界、DMA/cache/同步flag顺序、异常形状行为均待验证；没有可归因的性能/速度提升证据。完整文件与submission计数需中央账本校验后再报告为全archive总量。

