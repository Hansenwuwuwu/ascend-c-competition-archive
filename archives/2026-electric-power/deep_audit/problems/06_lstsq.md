# 06 lstsq：逐行审读证据账本（首轮源码覆盖完成；证明义务仍开放）

题面：[原始PDF](../../problems/06_lstsq.pdf)；[题目契约](../SPEC_CONTRACTS.md)。

设计的题目专属/问题族/通用编排分类见 [适用范围总表](../DESIGN_CLASSIFICATION.md) 第06节；方阵消元与一般最小二乘的边界分开。

本报告不声称三份实现已经逐行审完。三份构建文件、Host、tiling/key 已阅读全文；rank2 kernel 1–959 全文已读；rank1 kernel 仅 2678–2762、rank3 kernel 仅 3091–3355 已读，其余与 `lstsq_fast_square.h` 尚未读。逐文件 hash、已读/未读行段见 `../coverage/06_lstsq.json`。未编译、未运行下载代码、未做 NPU 数值或性能验证。名次不是具体技巧的性能因果证据。

## S1 题面契约与视觉核对

已读 `specs/06.txt` 全部三页，并核对 `specs/rendered/06-{1,2,3}.png`；第3页没有新增任务内容。A 为 FP32 `[...,m,n]`，B 为 FP32 `[...,m,k]`，相同 batch，m∈[1,1024]、1≤n≤min(m,512)、k∈[1,64]，batch 维大小[1,32]。A 列满秩，κ₂(A)≤1000。固定 driver='gels'，rcond 不参与计算。不覆盖欠定/秩亏。

求解 `argmin_X ||AX-B||_F`，输出 `[...,n,k]`；m>n 时 residuals 为逐 RHS 的 `Σ_i (AX-B)²`、shape `[...,k]`；m=n 时为空张量，不是零值非空张量。两项逐元素 atol/rtol 均1e-4。不允许完整最小二乘库、CPU回退、残差开方。源码额外支持 BF16、向量/共享 B、方阵请求非空残差，均属扩展，不能反过来改写题面。

数学：满秩 QR 有 A=Q₁R，X=R⁻¹Q₁ᵀB；这是实数恒等，不代表任何两种 FP32 运算次序位等价。对方阵可改用求解 AX=B；对高瘦矩阵不能一般地用 A⁻¹B。正规方程 AᵀA 的 κ₂=κ₂(A)²≤10⁶，不能仅凭 κ(A) 上限就保证 FP32 1e-4。

## S2 构建与ABI（各rank按hash复用）

三种 CMake 内容全体相同：根1–20要求 CMake3.16、find_package(ASC)、未指定时默认910b、SHARED custom包并加入 host/kernel；Host1–24 glob `.cpp` 参与自动代码生成/tiling库、将生成 aclnn_*.cpp 标记 GENERATED、注册两个库；kernel1–15扫描当前目录源码、绑定 cust_optiling 并打包。没有在这些文件中看到设置 fast-math、额外链接完整 lstsq 或执行脚本；glob 使实际生成源码和 CANN 工具链仍是构建证据缺口。

rank1 tiling1–12：m/n/k/batches/rows/blocks/teams 为u32，residualElements为u64；两个常量是 **float元素数**438800与587520，Host乘4为字节。key1–28只编FP32：0/5/8为AIC: AIV=1:2混合，1/2/6/7为AIV，3/4为MIX_AIV_1_0。

rank2 tiling1–5：九个u32依次 m,n,k,batches,mp,groupSize,groups,localCols,rounds。key1–10允许FP32/BF16与CASE_ID0–7的组合；是否实际派发仍由Host决定。kernel933–959默认AIV，并对数值key768/1536声明MIX_AIV_1_0；这些编码和模板参数打包的官方契约本轮未核验，不能仅凭十进制常量猜测其他key类别。

rank3 tiling1–46和Host内嵌 Config 共享 include guard；Config 包含六个u64形状字段、rhs_batch_stride、dtype/flags/degree、两个3项数组、max_sweeps和浮点属性。TemplateTilingData在其后存scratch_bytes、input_types[6]/output_types[2]；LinalgLstsqTilingData是别名，不是另一个布局。options1–7的bit5表示额外请求方阵残差。key1–20允许USE_LU0–3的FP32/BF16、4–13仅FP32，未显式声明kernel类型；实际入口3147–3151另有默认AIV及512/539/2048/2304的混合声明。ABI大小、编译器对齐及生成key布局仍需工具链产物核验。

## S3 rank1 Host 全行语义与入口

`rank1.../op_host/linalg_lstsq.cpp`：1–6引入API/注册/平台与共享头；7–26核对 rank、m/n/k、batch一致与u32乘积上界、两输入FP32、driver='gels'；rcond有意不用。27–34获取AIV核数，记录真实残差存储元素数，rows从8倍增至≥m的2次幂。35–55默认path7，方阵且k≤8时n≤16→1，17–32→2，33–64→6，65–128→3，129–256→4（需足够AIV），teams=capacity/32+1；blocks按整组分配，单核小矩阵至少一组8核但kernel必须屏蔽空闲核。56–66再覆盖：满足AIC/AIV=1:2且每batch可驻留4AIC则path5；193–256,k≤4转path0，其中225–256转path8。67–74按窄正交/块正交/多核/一般路径分配words，乘4加系统workspace；75–114结束函数、构造solution与空/非空residual shape、两输出FP32、optional residual、gels/rcond属性及910b注册。

入口2736–2762由 `#ifndef LSTSQ_DIRECT_BENCH` 控制，函数有aligned(4096)，先InitSocState再读tiling。路由：0→NarrowOrthogonal(默认96 panel)，8→NarrowOrthogonal<112>，5→Orthogonal，三者分别在AIC/AIV调用对应模板；1→ResidentElimination<16>，2→ResidentElimination<64>，6→其第五bool=true变体；3/4→RefinedSquare<128/256,32>；7→Lstsq。以上被调用体尚未完整审读，不把函数名当算法证明。

已读2678–2735是窄正交尾部：2678–2696按panel/阶段选择8/9/10/12/14次Newton，Cube生成future cross Gram，Vector拆A块；SyncAll后Vector检查guard，清理/失效decision单cacheline，非零时改走Inverse<128>并重算factor/H，再更新。2697–2715按112/96 panel分别补尾32行、重投影系数并调用Tail，real=n−2Panel要求Host panel范围成立。2716–2733完成因子、应用RHS、拆XP、int8 packed4乘积、Assemble6再一次校正并Finish；Finish的残差写入同时要求 rr非空且residualElements≥batch*k。SyncCube与SyncAll交替是生产/消费协议的一部分，但产生端所在未读区，不能据此宣称全协议已证明。int8拆分权重、误差界、L1/L0/UB峰值和fallback退出未完成。

## S4 rank2 Host 全行语义与完整路由

Host1–6依赖；7–19核对shape与范围、相同batch，AIV至少按1取值，dtype只取A。batch乘积u32无溢出检查，也未显式核对B dtype/driver；注册并不能替代所有运行时校验。20–43仅FP32方阵且batch可驻留时绑定：CASE1=(16,1)，2=(64,2)，3=(256,3)且batch*8≤cores，4=(15,1)，5=(50,2)，6=(200,4)且batch*10≤cores。CASE3/6 workspace分别每batch(32768+256*264)/(32768+200²) float，再加系统区；1/2/4/5零workspace。这是shape专化，不据此判定违规。

44–57一般路径：mp=max(8,nextpow2(m))；`total=n+k`；total*mp>20480时PAR=1，localCols=min(total,max(1,20480/mp))，否则全驻留；groupSize始终1、groups=min(batch,cores)、rounds=ceil(batch/groups)。CASE7分块但并非跨核列并行：每核独立拥有一个batch，GM仅作该核溢出列。Host分配2*groups*(total+1)*mp float，kernel每组只按(total+1)*mp stride用一半。58–78做shape推断、同输入dtype输出、FP32/BF16 AutoContiguous注册以及gels/rcond属性，未计算rcond。

kernel933–959真实路由：CASE1→RowGj<16,16,1>，2→SmallLu<64,64,2>，3→BlockLu256<256,3,8>，4→RowGj<15,16,1>，5→SmallLu<50,64,2>，6→BlockLu200<200,4,10>；其余CASE7→Solver<DT_A,true>，0→Solver<DT_A,false>。只有一般路径读取完整tiling；专化将N/K写进模板。sizeof(DT_A)==4保护绑定FP32，BF16即使编出特殊CASE组合也落一般体，但Host不发这种组合。

## S5 rank2 kernel 1–699：逐连续语义行段

以下全部区间包含注释、空行、括号、声明，不把未提及的宏略掉。

| 行段 | 实读语义、容量/生产消费 |
|---|---|
|1–18|K_MAX_SHAPE_DIM=0、AscendC依赖；Fence在EVENT_ID0立刻set/wait，跨pipe定向依赖而非全局barrier。BlockLu256的P=8、C=32、MP=264、META=8、LU_BASE=32768，静态断言分区/Panel整除；TPipe/TBuf、局部别名、GM对象、lane/batch；Barrier=PIPE_ALL+SyncAll<true>。|
|19–43|每lane DMA N行C列（源stride (N-C)/8、目的额外1个32B槽），转置Gather至列主mat，RHS lane<K各取一列；构造行序列、MP字节列偏移、active位掩码0x7fffffff；orig保存原矩阵列，bOrig保存原B，Brcb广播列偏移。|
|44–57|UpdatePanel依次Gather每个pivot row系数，首q等待panel的MTE2_V，MulAddDst按count列广播更新全部N行；记录pivot/倒数，仅未来owner清active。|
|58–105|Factor按8列panel选owner，每个active数值按float位And清符号得到abs，WholeReduceMax和V_S事件交给Scalar读SPR索引；不物理换行，把pivot槽清零、active清零，mult=-col/d，更新后续列和RHS，panel末向GM写rows+inverse和mult列；最后q不再产生下一个V_S。|
|106–137|所有核barrier后非owner读交替metadata槽，MTE2_S读行号、MTE2_V读panel；count0时仅记录pivot/失活，不等待未产生MTE2_V。结束将piv*4用作Gather，RHS按pivot重排并乘倒数形成解。|
|138–158|Correct等待已由Residual预取的ID1，预取下一C列LU；orig/mat两缓冲以step奇偶切换，后续step等待对应ID，消费后才允许覆写，最后两step不再预取，故没有尾部悬空load；用同样消去变换解校正量。|
|159–180|解写w[1024+lane*N]、barrier、读全K解；负4096掩码即0xfffff000清掉12个低mantissa位，拆原A及前两列系数高低part，广播coeff/coeffHi/coeffLo；这不是FP64。|
|181–230|四路乘积累加，前两路显式高低分解补积误差，后续四列组通过差分保留加法误差；sA/sB交替（C/4=8，末组在sA，随后读取sA成立），合并4→2→1并写每lane两个K*N的sum/error到w[2048+lane*2*K*N]。|
|231–254|lane<K从8核收sum/error并预取LU，链式前缀和、TwoSum型误差重构、二叉合并后rhs=bOrig−A*x；为Correct的首ID1预取提供生产端。|
|255–275|Run以block%8、/8定lane/batch，workspace按WS隔离；5个TBuf分别8712、10240、4096、7936、2128 float，共132448B；alias给v/active等，factors起7168，后面使用到至少7520且在7936内。Factor时rhs借mat尾列，Residual后切aux；仅lane<K校正与输出。Brcb复制每个sol至8槽，DataCopyPad写N个4B带目标stride(K−1)*4，非空rr写单个0。|
|276–313|BlockLu200为P10、C20、CP24、MP200、META16；偶lane起点20lane，奇lane向前4float，24列DMA起点/宽度32B对齐且最右结束N；Gather跳shift恢复真20列。active真200行0x7fffffff、后56行0，屏蔽256宽max的padding。orig保存20*200，余CP空间用于过宽向量尾。|
|314–328|200版UpdatePanel同256但最后off192 mask8，避免把正常列更新错误写到下一列；首次q的panel DMA等待同上。|
|329–378|200版Factor panel10列、owner=col/20；256宽And/归约+256宽mult读取/写入，末尾56行虽触及下一物理列/尾部scratch，但active归约屏蔽；panel增加64float尾空间。真正更新每64块用mask min(64,N−off)。panel输出metadata按2*META=32float。|
|379–410|barrier后读取panel/metadata、更新count、维护pivot，逻辑与256版，pivot输出恰200。|
|411–431|Correct全256宽，数据缓冲有尾部空间，rhs后56元素会进入已死bOrig前部；前200逐元素不混行。两个DMA事件轮转，末两步停止预取。该别名活跃期依赖Residual先读完bOrig。|
|432–480|200版残差没有256版前两项高低拆积，4路MulAddDst+二叉求和，每lane写K*N，10核reduce按8+2再4/2/1合并；需要最终数值实验验证一次校正充分性，不能把两版当同精度实现。预取orig(LU首块)产生Correct所需ID1。|
|481–502|Run的mat8000、orig4800、raw4096、aux7808、panel2096 float，总107200B；factors起6200，alias总在7808内；最终N*4 scatter与可选rr零写，DataCopyPad并不由rr张量长度约束。|
|503–528|SmallLu为N64或50、K2、S72；base A/B向下对齐8float，N50依batch奇偶shift16B；mat(N+K+8)*72、perm(N+K)*72、rawMP*8*K、aux2304 float，分别N64合计48544B、N50为40480B。N64 A用8块行宽+1块stride；B读取ceil((NK+4)/8)*8，详见S8问题。|
|529–568|用16初始行索引扩展至64，N50尾row索引零；cols至72保证向量对齐，Brcb后建立8列Gather偏移。只batch0写整个固定长度rr零：N64写8float、N50写6float，并未读实际batch或rr形状。|
|569–604|转置真N列/RHS；remaining位图屏蔽无效与已消去行；绝对值最大pivot、将pivot值清零、每后续列做消去。使用64位SetValue把两个相同inv与两个RHS偏移打包（低32=row*4、高32=row*4+S*4），减少Scalar写指令；原始位cast保留float位，不是数值转换。|
|605–615|Gather产生交织双RHS乘复制倒数，N64写128float，N50按2*N*4精确输出；无再一次残差校正。|
|616–659|RowGj为N15/16、K1，S16；mat(MP+8)*16+aux2048 float=9728B；对齐A，N15读232float覆盖225+最多7个对齐/尾数，B用exact DataCopyPad补零；构建240/256项转置索引；仅batch0按固定3或4个float写rr零。|
|660–699|转置至列主，active位图选绝对最大pivot；16宽外积消去，不交换物理行，记录pivot/倒数后Gather RHS并缩放，N15输出60B、N16输出64B。每次归约set/wait V_S，末次不set下轮事件。|

## S6 rank2 kernel 700–959：一般QR/补偿、空间与事件

| 行段 | 实读语义 |
|---|---|
|700–726|Solver对象含8个TBuf、FP32矩阵/补偿别名、DT_A GM和PAR工作区。Dot=Mul+ReduceSum→V_S。ReadColumn先清mp，再按每GM标量1个32B UB槽搬运；FP32 Gather offsets=i*32，BF16以half位Gather再Cast FP32，m后面的mp−m保持零。|
|727–743|Split用4097=2¹²+1的Dekker型拆分，ProductScalar/Vec计算p=fl(ab)及近似低误差，依赖无溢出/下溢、binary32舍入行为；不是任意范围精确的double替代。|
|744–760|AccumulateResidual的TwoSum型运算将ph/pl加进rh/rl；AccurateSum每次二分到8，再Scalar逐项补偿。mp为2次幂保证二分布局，无奇数尾；补偿仍float，低项继续相加会舍入。|
|761–779|LoadCol/StoreCol按c<splitCol走GM或UB，GM读写分别MTE2_V/S、V_MTE3/MTE3_V/S，确保复用v前依赖；LoadR只是LoadCol(v,j)，无需独立R buffer。|
|780–798|ComputeResidual重新读原B、原A，逐列构造r=B−Ax的双float rh/rl；StoreResidual只写单个4/2B并等MTE3，NormalizeResidual再规范high+low。|
|799–832|一次refinement：高精残差→原A每列与r做高低积并AccurateSum形成g=Aᵀr；先forward解Rᵀz=g，再backward解Rδ=z，最后sol+=δ。若要高瘦残差，再将−Aδ累入rh/rl，normalize后平方求和；忽略low²及乘high*low的显式项，但normalize后low相对high小，仍需误差界；方阵wantResidual则写0而非实际残差。|
|833–846|group=BlockIdx、lane恒0，PAR只溢出列非跨核。8缓冲合计 `4*localCols*mp + 88*mp + 32` 字节：matrix、raw=32mpB、v/tmp/red/indices各4mpB、extra40mpB；mp1024、localCols20时172064B。indices全部Scalar写i*32后S_V。无L1/L0显式分配/无Cube主计算。|
|847–855|轮转batch=round*groups+group，越界break发生在单核私有流程内；清matrix后读所有A/B列，c<splitCol写GM。 |
|856–892|逐列Householder：清v前j+1，tail2，alpha=−copysign(norm,lead)、tau=(alpha−lead)/alpha，v_j=1。tail2=0时tau=0不反射，保留lead。每后续列z-=tau*(vᵀz)*v。无safe norm缩放，tail²+lead²可溢出；below-diagonal原值保存但回代只使用上三角。|
|893–901|n1时直接用变换B的m−n尾部平方输出，高瘦无需后续refinement；m=n且wantResidual仍零写。|
|902–912|R回代对每RHS直接访问UB列（与S8缺陷有关），j前缀Axpy后写sol_j；n>1逐RHS作一次refinement。|
|913–931|每RHS浮点位复制到raw的8float间隔，或BF16 Cast后16half间隔，exact scatter写solution；S_MTE3保证Scalar staging提交，MTE3_V/S结束确保下一RHS/下一batch复用安全。|
|932–959|模板入口、task宏、注册tiling与全部CASE路由，见S2/S4。|

算法代价：一般QR约O(mn²+mnk)，回代O(n²k)，补偿refinement再读原A O(mnk)并多种乘/加；每核空间O(mp*min(n+k,20480/mp))加O(mp)，溢出GM O((n+k)mp)。BlockLu做全后缀消去约O(n³/P+n²k)、global barrier约n/PANEL次，再一次O(n²k)原A残差和同因子校正；仅8/10核驻留条件下可用。小核额外输出/归约元数据固定化减少调度，但泛化能力更窄。

## S7 rank3 Host 全行语义与入口已读部分

主Host1–41定义内嵌Config；42–113是跨十算子sizes/异常检查。LSTSQ分支95–99要求batch≤32,m≤1024,n≤m≤范围,k≤64并定义(mn+2mk+m)*4scratch，但**实际TilingFunc调用lstsq_host::tile，不调用这段通用tile**，因此不能把这些范围检查算到真实路由。114–157重复config因guard关闭并声明TemplateTilingData；158–275通用shape contract检查/拼输出/64B workspace对齐，LSTSQ允许向量/共享B，与题面扩展区分。276–336 dtype/属性（driver检查、rcond忽略）、337–369 shape/type inference、370–431通用tile均为模板工具定义。本Host已定义guard，所以`lstsq_plan.h`1–431同样的内嵌大段在此翻译单元预处理时关闭，并未重复定义。

主Host432–464引入共享头与专用plan，先专用tile后按m=n,k≤8选择USE_LU：精确形状/批数且匹配blockdim/rhs stride/FP32/flags时4–13，其他n≤64→1、≤256→2、≤512→3，否则0。465–507包装专用infer_shape、通用infer_type，注册FP32/BF16、可选residual、属性与910b；OP_TILING_LIB额外注册同一推断器。没有断言这些专化对应原计分二进制。

`lstsq_plan.h`1–431与主Host同组include guard的内嵌通用工具已连续读完；其宏屏蔽如上。432–456的日志包装逐个检测DlogRecord/ReportInnerErrMsg并try/catch，诊断不参与数值计算，私有/版本API未核验。457–480引入options/string，定义1GiB **每次组合分配检查**和乘法/加法溢出守卫、正维elements；481–496允许含0输出计0、64B对齐及Plan成员。497–538实际make：m≥n，兼容vector/shared B，内算FP32而输入输出dtype保留；方阵非空残差flag作为显式扩展。539–560按小方阵scratch0、中方阵(n*pad8(n)+4096)*4、通用(m*pad8(n)+2mk+m)*4及私有64B output staging给per_core_bytes。**此专用make没有调用sizes，因此没有重做m≤1024,n≤512,k≤64,batch≤32上限检查**；题面内不受影响，题面外不能视为安全支持。

561–604检查concrete hint、拼接原始/存储shape的详细报错。605–649 shape inference优先保留已有输出hint，未知/歧义先返回success不编造形状，无歧义才补shape；异常日志返回failed。650–693专用tile验证同dtype，按非空方阵rr置flag；初按vector解释，若solution元素数不符再按matrix解释；只核对输出元素数而不要求原shape完全相同，故支持密集flatten/view但不证明任意stride view安全。694–720 AIV>0，n200/256方阵窄RHS优先8核/batch，次选4核，其他按batch，user_ws=16384+blocks*per_core_bytes（零scratch时0），加系统workspace并检查总计≤1GiB，选择key/blocks；多核ScheduleMode未在这段设置，是否由生成任务类型隐含保证需官方契约核验。

kernel已读3091–3124为square_residual_vector尾：每行每RHS用FP32 dot算B−AX并补偿平方和，先写每AIV私有staging、flush、S_MTE3，再exact-byte DMA至最终输出，避免相邻batch标量cacheline回写。3125–3143 PublishOutput以uint16原位搬运≤4096B chunk，每次PIPE_ALL隔开GM→UB→GM，保存FP32/BF16位，尾字节长度精确；不作数值转换。3144–3200 USE_LU4–7/10–13调用固定batch/N/K的小方阵体，10–13带rr、4–7传nullptr；8/9调用3×200×4或4×256×3的p8体。3201–3229一般1→small，2按block数p8/p4/large。3230–3288 path3→LinearVector，方阵请求残差时FP32专用Vector、BF16补偿fallback，按batch stride发布；CANN9 BF16 Cast不支持的注释是作者声明，未独立核验。3289–3355一般path0绑定Tensor，配置batch置1、每核轮转batch，方阵compute_solve否则compute_lstsq；64B私有staging后4096B exact-byte PublishOutput，Math与copy的TPipe生命周期分离。这些调用体的误差、缓存写回协议尚未实读，不可宣称已验证正确。

## S8 已能静态证明的问题与风险（非违规判定）

1. **确定的地址契约缺陷：CASE7的RHS不一定驻留UB。** 合法shape A=[1024,1]、B=[1024,64]，令A首元素1其余0，则列满秩且κ₂=1。Host mp1024,total65,PAR1,localCols20,splitCol45；装载850–854将列0–44写GM，只有45–64留UB。n=1求尾残差894–901尚经LoadCol正确，但回代c=1时906直接取 `matrix[(1u-45u)*1024u]`，u32下溢，乘法再取模得巨大偏移，而UB只有20*1024float。915还重复该错误。问题对m512,k>40等也存在。这里是源码寻址反例，不声称已经在设备测得崩溃或某个数值。修复方向需让所有RHS至少驻留、或统一Load/StoreCol，尚未修改原码。
2. **确定读超逻辑长度，物理安全待运行时契约：** SmallLu<64,64,2>的522行读取 `(128+11)/8*8=136` float；batch=q基址128q，最后一batch读取终点128batch+8，比真实B多8float。输入总长本来已32B对齐，这8个float不能用“只补齐32B尾”解释；是否GM分配有额外guard未核验。多读数据不被有效Gather消费，但仍是访问安全问题。
3. **空residual安全依赖不充分：** Host方阵输出shape{0}，kernel的wantResidual仅rr!=nullptr；RowGj写固定12/16B，SmallLu写固定24/32B，BlockLu每RHS写0，一般Solver也写0。空张量地址非null时逻辑容量为0，写操作超逻辑范围。若框架保证空输出传null才避开，当前源码未证明这种保证。SmallLu/RowGj还将固定3/4个batch写进长度，而Host接受任意batches≤cores，不仅写空张量风险；扩展非空rr场景对不同batch会过写/少写。
4. **条件数不保证尺度安全：** 例如A=[1e30,1e30]ᵀ（m2,n1），κ₂=1，tail2和lead²溢出，Householder norm/alpha/tau可能NaN。κ上限没有限定元素幅度。4097拆分也会放大输入而溢出；不把Dekker的实数恒等或常见范围结论当任意FP32正确性保证。
5. **数值边界待测：** blockLu200的普通4路残差与blockLu256的补偿并不等价；一次refinement未提供κ≤1000范围统一误差证明。按位清12低位与4097拆分的舍入条件亦不相同。输入若有NaN/Inf未见这里逐值拒绝，但题面未明确要求该范围。

## S9 比较、可迁移性与未完成项

rank1按容量范围选择resident/cooperative/正交Cube路线，rank2一般Householder QR+双float残差修正且6个离散方阵专化，rank3把共享模板框架、输出view兼容、每核cacheline私有staging和精确形状LU专化结合。对小方阵避免完整QR、按列消去共享因子、以byte精确DMA发布邻近小输出、按生命周期复用UB、尾部停止预取，属于可迁移机制；前提是保持shape守卫、真实A/B extent、事件生产消费及数值测试。不能照抄固定batch rr写零、把RHS全部当驻留、生成wrapper宏重写或默认跨版本key编码。

下一轮必须完整读取rank1 kernel1–2677、rank3 kernel1–3090与fast_square1–3571；补齐Cube高低分解/基128权重、L1/L0/UB/workspace全部alias、Newton收敛guard与fallback握手、官方SyncAll/SPR/Gather/缓存API契约。当前报告不是全题审计完成证明。
## S10 rank1 kernel 1–1350 连续语义补审（2026-11-09）

以下为正文连续阅读，不是按函数名抽样；1351–2677 及 rank3 未读主体尚不在本节覆盖。

| 行段 | 语义、布局与边界 |
|---|---|
|1–196|通用 `Lstsq` 为双 float Householder QR。UB 两块合计106496B；18×1024 float 分平面、另8192 float pack。输入按列逐标量读真实 m 行，补零至幂二行距；A/B 同缩放，QR 同时变换 B，先由变换后尾部求 residual 再回代。AddPair/ProductScalar 用向量单元素操作固定舍入，非 FP64；4097 拆分及指数缩放仍有有限范围条件。B 幅度不受 A 的条件数约束。回代按散布4B读取 R，用32B UB槽再压紧。输出每行 exact-byte DMA，复用前等待；空 residual 同时检查地址和 residElements，优于只判断非空地址。|
|198–307|`AccurateRows<P,T>` 在9PT+14T float内按生命周期复用 raw、拆分高低积与归约暂存。包含乘积误差四项及两级/三级魔数格点求和；仅指数10..240走格点分支，其他值高低对二分归约。P64特化两路归约，P>64按64宽部分和。所有分支实读，不能称无条件精确求和。|
|309–632|`ResidentElimination` 使用P+8列距、逻辑pivot位图、按列消去不物理换行。P16向量除法及宽度对齐，P64标量倒数、部分模式先阈值选首个合格pivot否则最大绝对值。`InitialOnly` P64用5HD装入并把广播系数放死padding列，完全跳过校正；普通P64含一次原A/B高精残差校正，P16没有。Parallel/InverseRows/InitialOwner 还有共享GM+SyncAll实现，但当前入口只使用对应非parallel模板，不能把未启用的机制归于实际性能。输出k/n精确字节；rr零写有容量守卫。|
|634–665|`ColumnPanel` 以N×2B列主布局消去B列并追加变换列，pivot行仅打标签，不搬动整行；最终一次缩放变换列。暂存3N+112+8B float，依赖B=16与调用方补齐N。|
|666–960|`RefinedSquare<N,32>` 按列分组A/增广单位阵，最后lane拥有RHS；固定P256、16行精残差块、workspace包含变换/逆/初解/残差。每16列owner factor后全核同步，其他lane应用变换，最终按label重排并发布完整逆矩阵。一次高精残差加普通精度逆乘校正。`if constexpr(true)`选择ColumnPanel，旧行主factor整段只作为未启用代码阅读。平方残差空张量有容量守卫；数学精度并非由κ≤1000自动证明。|
|963–1196|`NativeMM` 显式L1 256KiB、L0A/B各64KiB、L0C128KiB，关闭HF32；TA/TB影响Nd2Nz布局，事件覆盖MTE2→MTE1→M→FIX及复用。包含分块Gram/重投影、64宽Newton-Neumann、半精度高低重建、1536长int8六位乘积、前向逆应用与32宽Schur并行。固定tile边界必须由调用方保证，不能视作任意尾块通用MM。所有原子Fixpipe分支恢复原子状态。|
|1198–1350|`OrthogonalVector` 固定workspace STRIDE438800 float；EA/EX仅16项且上界静态断言。TailGuard16/32检查近单位矩阵行和并原子汇总拒绝；TailQR小尾用两遍投影再回代。Split6将归一化FP32拆为六个base128 int8数字（魔数舍入，半精中转），RHS逆序；单个尾数残差范围需保持在int8。SplitAChunk按真实n补单位阵，SplitXPacked4构造六位Toeplitz权重，跨lane覆盖和消费者待后文闭合。|

分类：双float残差校正、生命周期复用、按列消去和精确字节发布属于 **F/G** 可迁移机制；固定64/128/256形状、尾块guard及六位int8布局是 **P/F** 且依赖规模/数值条件；SPR、HF32、原始ffts同步和向量魔数舍入均为 **SDK/数值边界待核**，不是可直接移植的正确性承诺。

## S11 rank1 kernel 1351–2762 连续语义闭合（2026-11-09）

| 行段 | 实读语义与适用性 |
|---|---|
|1351–1427|六个int32卷积系数以base16384进位、主limb的FP32舍入残差并入次limb，再用幂二尺度与TwoSum累加到B；128向量处理32行×4RHS，Gather/掩码把有效四列散布到32列距，并保留四份相同初残差供后续消费。k>4不用该int8路径，改AccurateRows256×16直接原A乘X；k≤8的缓存RHS装入前保证列padding。|
|1428–1550|Normalize各lane只保存自己16行块，但独立读取**A首行**和完整B推导公共幂二缩放，防止B上溢的限制有实现；不是全A最大值证书。n外补单位阵且不缩放该单位阵。Finish读取X与校正，Gather紧致k列，尾行按真实n精确写回，rr容量守卫。小Gram无主元消去，Newton初值α=1.8/最大绝对行和、半精高低分裂；guard比较最终缺陷行和，拒绝NaN/负total并原子汇总。|
|1551–1731|Orthogonal主流程每64列Gram→Newton/Neumann→guard→必要时Schur32递归精解→更新后续面板，真实最后块再重投影并按16/32尾块guard分解或QR fallback。Cube与Vector所有分支共同经过相同SyncAll；首次decision保持不变，复检用另一槽。构建完整逆作用初解，再精残差→两轮前向/逆乘校正；R0/R1各保留不可变输入避免消费者覆盖。k≤4走int8六位残差，否则向量精残差。这里的guard只认证中间近逆，不直接证明最终统一1e-4。|
|1733–1953|NarrowOrthogonal另有587520-float工作区，Panel96/112，L1/L0资源同前且增加静态容量断言。五个base128操作数digit仍保留六个卷积阶。int8 MM把64行分48+16，复用完整L0B。Neumann前缀四个Cube复制完整小Gram但各持有32输出行，Panel96的第4核无行仍参与rendezvous；最后仅owner做高低缺陷校正。Gram/factor的96/112不规则形状用48/64细分而非越过L0容量。|
|1954–2135|更新未来行按Panel专化，ComposeFactor按K≤128分段原子加并恢复状态。先合成不依赖尾部的D0/H02，尾部完成后补D0/D1；crossGram由每batch除owner外3核做。FactorPanel与HFromCross任务所有权严格一致，后者读取前者留在L0A的逆行块；中间不能插入其他MMA，是值得迁移但脆弱的F/G活跃期编排。ApplyPackedTail读取每batch ready并清cache，按不相交列拥有者发布。|
|2136–2256|Vector160KiB、Panel行单位阵持久区；16列距RHS和最终输出精确Gather。首行A/全B共尺度同上。SplitAChunk每非owner8行，五位digit写1280列距；ZeroUnusedFactorRows覆盖256−2Panel尾区。重投影系数按W32实现，固定512 float写入仅在实际W32调用成立。|
|2257–2438|TailQRDirect对≤4行一次投影，否则两遍重正交；TailFour/LocalTail32分别以4、16/24前缀Gram消去，缺陷超过阈值则整个真实尾块QR，不能直接使用未经guard的近逆。scratch别名特意避开持久外层因子。Panel112/Prefix24把32×32变换交Cube：Vector先写变换+已解尾再ready，外层SyncAll后Cube应用，失败发布ready0。|
|2439–2603|32宽无主元逆、128补单位阵、α及半精7面打包、按Panel计算缺陷。SplitXPacked4五digit的第六阶只保留A1X4…A4X1，明确**缺少A0X5、A5X0**；不是精确FP32全乘积。重建先把AX形成high/low再与B抵消，避免过早相消；需要结合scale与舍弃digit误差给界，原码未给全域证明。|
|2604–2734|128阶Schur递归到32，Cube乘和Vector取逆间同步；Panel96用于n193..224、Panel112用于n225..256的Host守卫保证真实尾1..32。两轮面板后尾重投影、分解、合成D，最后int8残差与一次D校正。EarlyPacking与后段chunk覆盖合计start8行分片，teams由Host≥8AIV/batch约束；条件不满足应走通用QR而非照搬此核。|
|2736–2762|入口aligned4096、InitSocState、默认MIX任务；key0/8/5分别96窄/112窄/64正交的AIC+AIV模板，1/2/6为resident，3/4为并行refined，7通用Lstsq。代码中的ParallelResident、SharedRhs、Steps>1等非入口实例不能算已启用设计。|

**数值结论边界：** 这是一套以平方问题特化、Gram近逆认证、QR尾部fallback和高精残差校正组合的实现，不是“用half/int8替代FP32且无损”的一般定理。κ₂≤1000不控制绝对量级；仅A首行导出尺度以及五位六阶截断仍需独立数值边界验证。可借鉴的是认证后快路+正确的fallback协议、先构造不变输入再多消费者、L0驻留所有权和任务兼容编排；固定工作区偏移、SPR读取和ffts/SyncAll语义必须随SDK重新核。

## S12 rank3 主kernel 1–3090 连续补审（与既有3091–3355闭合）

| 行段 | 实读语义与风险 |
|---|---|
|1–279|Config固定宽ABI、Tensor按原dtype标量get/set、FP16/BF16位转换（含ties-even/subnormal/NaN）；切片offset为元素。Math sqrt/exp在设备8lane Vector执行、S_V/V_S屏障。CPU分支仅条件编译测试，不是Host回退。非成功状态Trap而非仅可移除assert。|
|281–655|内联portable/config整段重复且include guard关闭，随后共享补偿LU：位清低12bits拆分避免乘4097的拆分上溢，TwoSum和乘积误差，带partial pivot的left-looking LU、重放交换及双三角回代、精残差。浮点乘积本身仍可能溢出；不是无条件double。|
|658–1078|LinearVector列主right-looking LU。n²pad≤40000时UB驻留，否则16列面板/31744float工作块；det特化n256面板32，lstsq solve路不走该det分支。RHS≤8列，每列512槽；n256,k3保留标量RHS更新以保持特定舍入。小矩阵幂二pad一次UB转置，其余16行DMA→Gather→GM列主；half输入每行先转FP32。归约最大pivot并检查finite；行交换同步应用RHS，后续列panel保持已归一因子，先保存未除R上三角供回代。输出紧致排列后exact-byte DMA。所有k≤8/512容量依赖Host真实路由，不是类本身一般支持。|
|1082–1457|compute头再次内联同portable/LU，但guard使之不重复编译；全文已实读。|
|1460–1687|本题专属volatile存储舍入TwoSum，每一项累加都重归一high/low。QR列主GM、缩放norm避免直接平方极大值，先除norm再构造reflector避免x0−α上溢。三个stationary-equation校正：高低B−AX→Aᵀr→Rᵀ/R求解，不显式构建Gram但误差敏感性仍有κ²。最终残差含high²+2highlow、省略low²；这是受误差界约束的近似而非实数恒等。平方空rr保持不写，显式扩展rr算真实范数。|
|1689–2816|solve/device_lu/vector_dot又多次内联guard关闭的portable/LU正文，不能把它们当成多次算法运行。|
|2819–2957|VectorDot长≥16幂二补零（只补a）后4097拆分、四项积误差、pairwise TwoSum至8再标量归约；非finite回退位拆分标量dot，但a已补零部分不影响真实length。cachedLU保留列UB、逐行prefix一条DMA，读取前flush GM标量写，n≤512容量。|
|2959–3024|compute_solve每batch复制A/B，n≥128 cachedLU否则标量LU，再最多2次残差校正。接受新X不要求残差单调；可迁移的数值原则是不能用残差范数单调作为forward error证据。n≤512,k≤256在函数检查，但调用方workspace和固定缓存仍需各自核。|
|3026–3090|引入fast_square与Host/key结构，square_residual_vector另开TPipe，原X一次DMA、每行A/B各一条，FP32 dot及补偿平方；后续发布见S7。应继续核对x_float容量与k>1别名，不能仅从分配总字节判定各view互不重叠。|

本节 **G** 是原位换行序列、精确字节发布和cacheline私有staging；**F** 是平方LU与高瘦QR分流、补偿残差迭代；**P** 是n256,k3舍入分支和固定shape入口。三者须分别迁移，禁止把“存在可移除CPU宏”理解为设备依赖CPU。

## S13 rank3 fast_square.h 1–3571 全文语义账（2026-11-09）

| 行段 | 实读语义、布局与消费者 |
|---|---|
|1–289|同主cpp的内联Config/Tensor/Math被相同guard关闭；本头仅使用既有定义、options和fence。|
|291–644|小方阵增广Gauss-Jordan，31KiB单UB分配；真实offset与注释不完全一致，以代码4672/5760/6560/7168为准。行距采用奇数32B块，n15把RHS放15、n50放50，避免多一块padding。常量B/N/K入口先发A/B DMA，建索引/初始化与DMA重叠；整64位读Reduce(value,index)，逻辑pivot位图不交换物理行，最后剩一位SFF代替归约；负倒数符号用bit XOR。n64尾RHS用跨行repeat，n≤50尾8lane改按8行排布，精确收集n*k输出。常量路径去掉generic singular/finite检查，不可把fullrank当作无上溢证明。|
|645–910|动态FP32和BF16备用支路：多batch网格步进、B单/双列独立搬运，多列按行8槽，BF16逐行Cast、标量填增广RHS。动态pivot检查零/finite，输出按实际n*k；平方扩展rr写0，不是实际平方残差。常量ZERO_R固定batch/k受Host精确shape门限限制。|
|912–1148|load_panel/load_rhs按byte精确，半精每8行转FP32。前后代的系数预先除对角，使循环只做unit-upper更新，末尾统一乘inv_diag；Brcb ping-pong提前构造下一列系数，一次PIPE_V屏障完成当前Mla和下步广播。p8版本64B前置padding允许对齐向量写入死前缀，不能删掉padding后照搬。|
|1150–1609|两种补偿残差4wave/3wave：位mask拆A/X高低，逐列保存product与TwoSum误差，下一列Gather/拆分和前列低误差归并交错。四RHS分块，长k分支独立实现。归约可一次读两个GM部分和并流水TwoSum；SMALL_TOTAL使用每核2048槽，较大4096槽，max1024 chunk和1056padding避免相邻暂存重叠。|
|1611–1868|p8 apply/factor 8列wavefront：partial pivot、每行40槽、最后核RHS增广。其他核提前发前7列LU数据，最后列另发，pivot和inv_diag与payload不同引擎事件各自约束。负逆系数与pivot row交换顺序不是一般FMA替换；不满足pivot值时当前fast路置0而非Trap，精度依赖原题fullrank及舍入安全。|
|1870–2496|8核/batch、n200/256、k≤8。UB48896float=195584B，保留A_orig、末panel、初解；n200前4核残差改26列，后4核24列，恰覆盖200列。factor按8列全核rendezvous，最后owner可独立完成末panel与RHS；其他核并行准备A_orig。初解后8核残差，两级树TwoSum最终归至核7；A_orig死后复用为三/四缓冲LU前后代暂存，少发已驻留panel。归约与初解拥有明确单writer。最终按行k有效字节DataCopyPad。|
|2498–2973|4核/batch用56/64列奇数块距72，复读原A并行残差；n>200最多两次校正，由最大delta>3.5e-3绝对阈值决定第二次，所有batch先合并need_more，跳过batch仍join相同屏障。这个阈值是经验预算而非relative error证书。全局flag读取固定32float但batch*8索引：支持>4batch时必须另核范围；不能从当前榜单输入外推任意batch。|
|2975–3571|单核大方阵fallback：n≤200全panel驻留，否则128+余列，LU因子每8列写GM；回代两panel。残差128列距136重新装原A，4RHS补偿，奇宽panel交换high暂存避免丢最后和；一次/两次校正，maxdelta绝对门限，最后精确输出。全部路径实读，未将p4/p8相似body直接视为相同实现。|

### S13a 新发现的确定布局问题与证据级别

主cpp `square_residual_vector` 分配work仅6×512=3072 float，`x_float=work[512]`，却在3077/3078写入n*k个float。合法平方n512,k8时末端为512+4096=4608，超过该TBuf 3072容量；即使n512,k2只写至1536，`x_col=work[1024]`/`product=work[1536]`也会破坏仍待后续行/目标读取的X缓存（3098–3100）。Host的USE_LU3为n≤512、k≤8，而显式平方rr会调用此辅助体；**原题平方rr为空时不触发**，因此归类为作者扩展非空平方rr路径的确定view容量/活跃期缺陷，不声称原Judge已失败。修复应给X单独n*k容量或按列缓存，不能只增加整个work后保留重叠view。

分类：奇数块距、value+index整64位标量读取、SFF末pivot、首DMA隐藏索引构建属 **G/SDK**；unit-upper归一后代、补偿残差wavefront和原A死后多缓冲属于 **F/G**；n15/50特殊RHS槽、n200残差26列再平衡及固定batch常量入口属于 **P**。平方rr填零是扩展语义弱化，不应作为可复制的通用技巧。

## S14 全题首轮审读完成边界

三份提交所有归档文件正文已连续审读，覆盖账本应为12371/12371。此处“完成”仅指首轮源码语义审读：不等于CANN版本API契约认证、全部同步的形式化证明、全合法域数值精度证明或NPU性能归因。S9旧“下一轮未读”清单为历史断点，已由S10–S13闭合，不再表示当前未读范围。

