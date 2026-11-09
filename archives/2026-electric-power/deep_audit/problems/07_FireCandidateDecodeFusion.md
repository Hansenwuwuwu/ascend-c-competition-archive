# 07 FireCandidateDecodeFusion：逐行审读记录（in_progress）

题面：[原始PDF](../../problems/07_FireCandidateDecodeFusion.pdf)；[题目契约](../SPEC_CONTRACTS.md)。

设计的题目专属/问题族/通用编排分类见 [适用范围总表](../DESIGN_CLASSIFICATION.md) 第07节；公式缩放与广播复用不能混作同一种通用优化。

日期2026-11-08。原源码只读；不编译、不运行下载代码、不调用NPU。rank3全部文件已连续阅读全文；rank1/rank2已完成Host、ABI、入口和部分依赖，巨大kernel/长表尚有未读区。真实覆盖见 [coverage](../coverage/07_FireCandidateDecodeFusion.json)。本报告不把公示Pass/分数当本机验证，不把名次差归因某一个机制。

## contract

完整读过07.txt四页并视觉核对原PDF第2/3页输入表、公式、精度。raw[B,N,6]为FP16/FP32，priors[N,5]固定FP32且跨batch共享，输出同raw dtype的[B,N,5]；B1..8，N1..32768。候选保持顺序，不NMS/过滤/排序。输入有限，gx/gy>=0，stride/anchors>0；没有约束logits或先验的有限幅度。

所有中间计算FP32。s(z)=1/(1+exp(-z))；x=(2sx-.5+gx)*stride，y同理；w=(2sw)^2*aw*stride，h同理；score=sobj*sfire。half只在最终五元素转回half。float容差atol/rtol各1e-5，half各3e-3。不能把实数恒等变形直接视作FP32位等价；但本题不同于题1，允许给定数值容差。

## build

三份三个CMake均与题1 rank1同hash，已全文复用：根20行要求ASC、默认910b、custom shared package；Host24行GLOB cpp/codegen/tiling/ACLNN；Kernel15行kernel目录绑定tiling库。参见 [根构建:1](../../extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/CMakeLists.txt#L1)、[Host构建:1](../../extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/op_host/CMakeLists.txt#L1)、[Kernel构建:1](../../extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/op_kernel/CMakeLists.txt#L1)。没有本轮编译结果，头文件模板/pragma是否成功降低到预期指令未验证。

## rank1-host

[Host:1](../../extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/op_host/fire_candidate_decode_fusion.cpp#L1) 全文226行：

- 1–20：CompleteDevelopmentReuse只保留0/2和列出的特化REUSE，其他恢复Generic832 header{total,n,ceil(total/cores)}及12B size。开发保留代码并非都实际派发。
- 21–39：shape3D/2D、6/5、N一致、priorsFP32和正B/N检查，uint32/6控制元素偏移；未严格限制题面B<=8/N<=32768属于更宽支持。actualAvailable=min(40,max(1,AIV))。
- 40–73：B4N4097且>=20核选b280、20launch；B1N32选7b36(8launch仅一工作)或7b2e；B2N33选7b38(8launch稀疏3工作)或7b34(3launch)。B8N32768/32769 FP32的>=43分支**恒不可达**，因为actualAvailable<=40。
- 74–99：deviceAvailable也min40；half N32769的>=47/43两分支恒不可达。B8N32768/32769且40核启旧pair40策略；grains64/256先估核数。
- 100–181：groups从min(batch,workers)降到1，必须整除batch，lanes、64对齐span，要求256<span<=1024，frame128对齐；geometry=activeLanes|(batch/groups<<16)。e/c frame标签、whole512 a-tag、b240/b280/b500/b501等重编号；available==48两个修订也恒不可达。只knownFrameFamily保留专用，其余CompleteDevelopmentReuse转2。frame metadata16B，其余12B。
- 182–226：未找到frame则按flat/row路由再development filter；shape推导最后维5，输出dtype跟raw，注册ND两签名/workspace0。

[tiling:1](../../extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/op_kernel/fire_candidate_decode_fusion_tiling.h#L1) 全44行：base12B，frame16B；legacy row4128B、flat4640B、single2080B；表从32B开始；replicated slot96B×48并32B prefix=4640B。sizeof/offsetof静态断言检查二进制布局，不能说明当前Host实际填了所有legacy表。

[key:1](../../extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/op_kernel/tiling_key_fire_candidate_decode_fusion.h#L1) 全20行：dtype和16位REUSE，DECL宽列表，SEL只保留13个双dtype标签以及half46593/46595。声明≠被编译选择≠Host可达；不能用DECL全集报告实际执行路径。

## rank1-kernel-partial

[kernel:1](../../extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/op_kernel/fire_candidate_decode_fusion.cpp#L1) 已读1–230与8530–8774；231–8529尚未全文读。

1–41 FireSmallIndexGM548为alignas256的GM常量512u32/2048B：6×32输入抽列字节offset=24*i+4*f，5×32 priors offset=20*i+4*f，5×32输出pack offset=128*f+4*i，共192+160+160=512。43–129 FirePacketIndexGM556是1152u32/4608B：两份512表加128尾；第二表priors列首放640+4f并将原候选0..30右移，用于跨row边界的第32候选；输出offset受FIRE_P8O_INDEX在half图像除2；尾6×8 alternating0/24、5×8 alternating0/20、5×8 pack32*f+4*i，合128。131–167 GM678为544u32/2176B，在512模式中把第2个priors列复制一次，6×32+6×32+5×32=544。均为**索引不是答案表**；消费者在未读中段，尚不能完成其GM→UB使用/容量证明，不把生成规则分析当消费者已审。

169–230 FireStaticStorage无TPipe、DEPTH=tile>256?2:1，raw/prior/out双slot、work512对齐+256、one/rawFloat/packed/index依dtype共用；最高index+7tile*4<188416，预留SDK UB区。已读AllocRaw/Prior/Out分别在复用slot前wait对应V_MTE2/MTE3_V；生产、Free、Drain在未读段，事件闭环未完成。

8530–8536是前类收尾FreePrior/Drain，前置未读；8537–8555总入口：7b36只block0；7b38只0/2/4，其他早退；InitSocState，默认注册frame并字面key对应结构。8556–8565 b240将B4N4096恢复为512frame rowLanes，float硬32launch、half取实际GetBlockNum。8566–8646 b500/b501：float43launch备选不可由现Host达到；当前40launch前39核走pair，尾核320/321；half后续根据FIRE_PAIR_*编译宏选别名路径，中段宏定义未全文审不能确认实际一项。8647–8688 47/43half span族在Host cap40下不可达。8689–8700 b280恢复B4N4097/20核schedule，float/half各RowF4/RowH10。8701–8724 b300/c/e frame读16B；8725–8748小32与双33 flat索引族；8749–8764 REUSE0/1/2对应64/256/832 Generic，其他KernelFireRow，末尾PIPE_ALL；8766–8774 undef宏。未读主体不能判数学、搬运安全或分数来源。

## rank2-host-and-entry

[Host:1](../../extracted/FireCandidateDecodeFusion/rank2_submission_1176342/code/op_host/fire_candidate_decode_fusion.cpp#L1) 全102行：1–20只检查input存在、B1..8/N>0、core>0，从storage shape直接取维度，未核完整维数/N上界/priors描述；21–40由tile、batchOne/dense、owners决定MODE，1640float非对齐batch输出可固定Dual；42–66清32B payload，仅前8B动态有效，large把batch-1塞control bits16..18，small/mid压units13b、extra6b、owners6b、batch3b及高4flags；67–102模板选择(dtype,TILE,MODE)、blockdim、workspace0，shape最后维5/dtype跟输入。

[fire_host_plan.h:1](../../extracted/FireCandidateDecodeFusion/rank2_submission_1176342/code/op_host/fire_host_plan.h#L1) 全253行：1–100真实planner，total<=8192走32/64/128；total>=cores*1664*3走1640float或1664；中段按batch对齐8float/16half分均衡512/528/640/896/1024容量，logical blocks与8对齐launch分开。units/extra商余分工、dense判定、batchOne及power-of-two shift高bits。1640 uniform count先用uint64验证除法、uint32乘积界再塞16bit。所有题面合法B/N的total<=262144，无uint32溢出；core数必须适合6bit owners，这里无显式<=63保护。

102–110小Gather表是2*capacity个u32，r[i]=24i、p[i]=20i；114–169 address_context：G8,Q=ceil(tile/128)*16,P=8Q+gap，raw/prior/column/output固定UB基址；每16个地址>>5取13bit，并四个16bit lane打包到u64，context包含dst4words+src4words，共16contexts/128u64；halfraw3context+3空context，floatraw6，priors5，pack5。171–179 mid在每2word后补2零成为256u64；181–205 large额外5output-to-V0 context，共168compact再扩336word；207–252 half大族P gap16、S0额外448，3raw+5prior+2×5pack=18context，每context16word（分半、补零），288u64。这是Host参考地址生成器，不是计算输入/输出；实际生产使用只读metadata/device builder，尚未读完其对账。

[tiling:1](../../extracted/FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/fire_candidate_decode_fusion_tiling.h#L1) 全53行：expanded32B、packed16B、历史地址1536/3200/2816/2560B且表512B对齐；当前header-only32B。 [key:1](../../extracted/FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/tiling_key_fire_candidate_decode_fusion.h#L1) 全12行：TILE11b十容量，MODE2b三值，dtype两值；DECL/SEL列表一致。

[entry:1](../../extracted/FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/fire_candidate_decode_fusion.cpp#L1) 全227行：1–9 FIRE_COMPILED_SMALL_SINGLE从原始key位置选MODE2/TILE资源前三项，且arch2201/非debug/dump/timestamp/OOM；13–55无需动态tiling的完整small-single，float32 firstDMA在InitSocState前、仅raw/subowner0；其余闲核PIPE_ALL收束。57–74 noinline DualFallback隔离output-pointer动态分支。76–130预取small索引/mid-large地址表，读首uint64恢复control；large重算extra，小mid解13/6/6/3bit；闲核在预取后先barrier再退出。131–167 small RunSmallMany/PadSmall、midDense/BlockedTranspose full/partial guard；168–192 largeFloat根据output指针或batch字节非32对齐启dual，MODE2静态、MODE1动态fallback；half RunDense。196–218按key替换SetSysWorkspaceForce/GetUserWorkspace，unused返回nullptr，其余真API；220–227给自动生成导出entry添加cce_kernel/aiv/aligned128属性。此处依赖生成器key编码、arch及私有包装，不是普通用户workspace0。

device_fire.h1728行、blocked_transpose.h1933行、fire_static_metadata.h1100行尚未全文读；因此所有callee实际UB最高边界、地址表字面项与生成器的完全一致性、raw内建同步尚不可宣布通过。

## rank3-host-route

[Host:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_host/fire_candidate_decode_fusion.cpp#L1) 全211行。1–23验证格式/dtype/B，N容许到1048576（比题面宽）；24–72由总量small/mid/large、vector span初选0/3/5/2/4/6/7及paired8/19；73–104小候选4/8pack、batch2+fixed、四batch medium31/32→33，13→21、8→18明确重编号；105–149 B4 N%256==0/1启34/35、large四/八batch的power2或power2+1启36..43、B1 contiguous52/53，以及448-row65/66；150–164 float B1小16候选启54，launch=cores*4；51 launch=cores*2，其他normal；155–158 segments存log2core（否则32表示除法）。168–211 InferShape/Type严格6→5及priorsFP32，ND注册。

[tiling:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/fire_candidate_decode_fusion_tiling.h#L1) 全8行四u32=16B；[key:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/tiling_key_fire_candidate_decode_fusion.h#L1) 全17行，CandidateMediumRows448、small启动factor4float/2half，PATH8bit46标签DECL/SEL一致。未被Host最终选择的旧2/4/8/9/12/13等分支仍会编译，但不能全部叫当前热点路径。

[entry:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/fire_candidate_decode_fusion.cpp#L1) 全197行：1–21 include真实依赖图；23–44 PATH51/54以launch/factor还原active，15 native4；45–69 PATH65由floorpow2(GetBlockNum*448)恢复N，34/35由blocknum*256(+1)恢复N，35最后owner257行；70–98大36..43由floorpow2(core*Tile/2)(+1)恢复N/B，44..50由launch与batch反解8pack短尾；99–144读tiling的52/53、66、33、31/32、26/24、21及18/19；145–197旧small/paired/generic fallback完整派发。使用ScalarCountLeadingZero反解shape依赖Host选路范围，不能独立拿kernel处理任意shape。

## rank3-leaf-files

以下文件均已连续阅读全文，source链接给起点，行段解释覆盖文件全文。它们无答案LUT，只有容量/布局/地址公式。

- [candidate_index_ramp.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_index_ramp.h#L1) 1–23 scalar seed0..7经S_V→Brcb生成0..63、再生成0/64/128步；25–36双层Brcb构造重复64的全序列；38–59 pragma-unroll倍增拼接，64lane起；61–70从320项模式扩展，filled/5*4修正行偏移。cache布局seed0..7、lane8..71、steps72..263、spread264起；使用者须分配全spread。
- [candidate_transfer.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_transfer.h#L1) 1–36精确搬入搬出，Aligned普通块单位32B；非齐GM stride字节/UB stride块，组数单组时忽略pitch。38–85 groupedTail scalar构造r/pi/pack byte-offset，S_V后half转float，sigmoid逐步+Div，Gather六raw五prior，标准DecodeGeometry，再pack/Cast。调用者的numerator必须1，而非2，见下方确定问题。
- [candidate_denominator_geometry.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_denominator_geometry.h#L1) 1–35标准sigmoid几何，count小则逐列、否则pitch连续大向量；37–96 denominator重排：先Dobj*Dfire，再1/product，坐标直接2/D，保留可选Reciprocal与ReplicatedStride/TransformedPriors；98–111清prior tail；113–146检查prior>1e18/stride>32的mask→int→abs→双层max归约后V_S标量结果；148–155预变换(g-.5)*stride及anchor*stride；157–211 Historical版显式PIPE_V fences。当前paired文件TransformPriors=false，所以这些预变换函数**没有被当前该路径启用**；不能把代码存在当优化已生效。
- [candidate_native_five_pack.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_native_five_pack.h#L1) 1–47 native6索引，13108乘右移16得到floor(i/5)，再quarter重排；49–75三phase输入转置；79–107 native8 prior index支持BatchGroup2/4，短除法为常数商；109–139六phase八路输入；141–170五phase输出反转置；172–211 CandidateAddressWord将16个UB地址/32按16bit lane四合一u64；213–238 scalar写目标4+源4word每phase，并S_V；240–261 TransDataTo5HDVldVaRegImpl直接加载VA寄存器的私有路径，half3phase/float6phase，pack5phase。地址缓存half(3+5)*64=512B，float(6+5)*64=704B。自写静态枚举验证floor(i*13108/65536)==floor(i/5)对i<8320成立，首个反例16384；当前最大Tile1664*5=8320在界内，不可不加界地迁移到更大tile。
- [candidate_native_half_input.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_native_half_input.h#L1) 1–26三phase以field=(16phase+lane)%6、stripe=/6，把half AoS拣列；28–42升float后counter mask作Exp/1+、可保留denominator；44–89 Historical同数学但逐依赖barrier。最终精度不是half sigmoid。
- [candidate_native_prior.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_native_prior.h#L1) 1–23八源复制成16VA地址，将5字段phase映射入columns6..10，BatchGroup和FullStripe控制复制跨度；25–34五phase完整覆盖。priors仍按candidate共享而非按batch取错先验。
- [candidate_native_half_output.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_native_half_output.h#L1) 1–27固定UB4352 halfcast、4608 uint16 pattern、5120 transpose；0x1f保留每row前5字段，GatherMask pack40half，V_MTE3后DataCopyPad只写count*10字节。4352cast80half到4512，4608pattern互不覆盖，输出packed复用4352在transpose后安全。
- [candidate_fixed_small_batch.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_fixed_small_batch.h#L1) 1–15 task/groups反解batch、begin8pack、count尾；raw/out添加batch*n偏移，prior不加batch偏移。
- [native_small_candidate_decode.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/native_small_candidate_decode.h#L1) 1–16固定4row转置；18–112 native4完整pipeline，scalar index20项，ones1，DMA rowpadding后转置六raw与五prior，GT无关，float几何/half最后cast，末MTE3_S；114–126合并raw/prior sources一次转置；128–237 ConfigNativeSmall Rows4/8，ScaledSigmoid默认2/D，score一列先*.25补偿、坐标不再*2；float用0x1f1f1f1f位图压五列，half走前述output helper。raw/prior分别MTE2_V0/1重叠，固定UB最高pattern/ones约5120B。
- [small_candidate_decode.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/small_candidate_decode.h#L1) 1–195 SingleTask旧类：half预缩放2/D、ScoreScale.25、scalar解码，Pack4/8可展开，固定六slot Bytes=align(Pack*24,32)，V_S/S_V/S_MTE3等保护scalar roundtrip；196–365当前SmallCandidateDecode task循环按B→candidate，prior索引i%n，统一sigmoid1/D、公式标量逐步，尾DataCopyPad，MTE3_V回收。Pack8且!SafeCopy用普通DataCopy，Host PATH5保证half N%8==0；PATH24显式SafeCopy=true保护尾。
- [medium_candidate_decode.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/medium_candidate_decode.h#L1) 1–59 TPipe队列depth1与11列，raw/prior byte indices24i/20i，float0.2-floor生pack列地址；只Tile512/576/1024对应pitch常数；61–132按task循环或balanced segments切分，DataCopyPad有效length，FP32sigmoid1/D，Gather标准公式，cast/store/free；134–143资源成员。无double buffer。
- [generic_candidate_decode.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/generic_candidate_decode.h#L1) 1–66 raw/prior按1024或CachePrior启depth2，float可原位省rawFloat/packed；整数13108生成pack索引，**ones初始化2.0**。68–89 Tile1024预发下一task；91–151 Compute按2/D直接几何，confidence*.25补偿，准确写count；152–193 CachePrior按core固定candidate区间、prior加载一次跨batch复用，raw下一batch双缓冲；195–228小余量batch*count<=32走groupedTail，重建动态索引并按batch-strideDMA；230–256加载/队列成员。该特殊尾存在缩放契约错误，见确定问题。
- [contiguous_candidate_decode.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/contiguous_candidate_decode.h#L1) 1–23连续两半priors转置；25–107 Tile128/512固定UB，raw/prior有效count搬入但全容量计算，2/D及score*.25，native输入/输出转置，最后只写count*5并MTE3_S收束。512 float UB最高OneAddr+256=45664B；half48736B（含floatstage）。
- [single_group_medium_candidate_decode.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/single_group_medium_candidate_decode.h#L1) 1–23四batch容量布局；25–83先priorDMA+event0，再四组rawDMA+event1，nativeprior复制给batch、half先native转置升float保留denominator，float也保留denominator，DecodeCandidateDenominators合并confidence，pack后按四batch/validcount写回，MTE3_V结束。Tile1792(float)总OneAddr+512=158560B；half tile1792总169648B，均<192KiB，但未Host动态检查UB。
- [historical_medium_candidate_decode.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/historical_medium_candidate_decode.h#L1) 1–121独立Historical namespace，四组DataCopyPad尾gap按32B，float先2/D再几何/score*.25，half denominator融合，显式PIPE_V更多；PATH35最后257行用Tile1280（BatchRows320）而普通256用1024。不能因“historical”认为不可达，entry明确选它。

## rank3-paired-and-sync

[paired_batch_candidate_decode.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/paired_batch_candidate_decode.h#L1) 全418行：1–34的18个模板参数控制2/4batch组、native六/八lane、地址cache、dtype复用、stride skew/replicate。float大native插32B RawGap；CachedAddresses只在Tile>=1408且nativeprior/pack，UB addresscache+704<=196608；TransformPriors恒false。35–131 Init选择static IDs0/1或TPipe alloc，先PrefetchTile再准备索引，Pattern320/倍增/fastRamp/通用四族生成，ones按Tile<1024选1否则2、FuseConfidence另配1，addresscache S_V发布。132–230 Process按core的16对齐candidate区间，prior只搬一次并在UB复制给组内batch，raw按BatchGroup循环；PoolFree防止前一输出覆盖下一prior，结束drain out/pool/raw并释放动态event。

231–285 PrefetchTile/BuildRamp/FillRamp/LoadRaw的GM stride=n*6、本地HalfTile*6+RawGap；rawFree V_MTE2在上一消费者完成后才覆盖，下一组raw可在当前几何期间预取。286–418 Compute保留全Tile计算（valid只用于搬运/写回），half先转置提取再CastFloat，raw最后使用点释放；floatRawGap额外清8float并用counter mask；native/gather抽列；FuseConfidence走1/(Dobj*Dfire)，非融合按ones2补*.25。output在SharedAddr复用prior，先等待前一MTE3_V；写后set outFree，最后batch额外set MTE3_MTE2 poolFree。对事件的“set→wait”均有路径条件守卫，但自动插入同步仍未反汇编确认。

[historical_paired_candidate_decode.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/historical_paired_candidate_decode.h#L1) 全395行并非hash相同复用：1–29无RawGap/CachedAddresses/TransformPriors；31–128 Init类似但显式barrier更多；129–215 Process计算相同prior复用、drain/release；217–278 Ramp/Load/Prefetch；279–395 Compute两batchhalf native走Historical激活/denominator，raw不加gap，无VA cache，用TransDataTo5HD，显式PIPE_V保护。large half在grouped entry明确路由此类，不能省审。

[grouped_candidate_entry.h:1](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/grouped_candidate_entry.h#L1) 全83行：1–28四batch medium固定512或adaptive512/640；31–59 large half Tile>=1408→Historical paired，float→新paired，并按N%16选择AlignedTransfers；63–83 coarse用segments shift或除cores得16对齐区间，再选1024/1280、half FuseConfidence。固定tile、priors复用、nativeVA表能减少每次动态索引/重复先验搬运，但增加UB、模板数、编译体积；不能据排名断言这些收益。

UB/搬运：rank3无GM中间张量或用户workspace，无L1/L0矩阵计算。largest current largefloat paired Tile1664，Raw39968+Shared33280+Column73568+One928+Stride26752+cache704最高195200B，贴近192KiB196608；half historical1664约181312B。必须保持32B gap与Pitch单位，稍增tile/预留即可超界。half输入只在Cast前以half储存，全部sigmoid/平方/融合float。DataCopyExtParams GM stride单位字节、UB stride单位32B，普通DataCopy对齐分支只在Host保证N/valid长度整16/64时可用；尾写实际valid，不能用UB全容量授权GM超读。所有N/batch索引最大题面范围下<2^32。

## confirmed-and-risk

### 确定的源路径问题：Generic cached grouped-tail缩放不一致

完整证据链：[generic:64](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/generic_candidate_decode.h#L64) ones=2；[generic:168](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/generic_candidate_decode.h#L168) batch*count<=32进GroupedTail；[generic:216](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/generic_candidate_decode.h#L216) 把ones传给helper；[transfer:70](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_transfer.h#L70) Div得到2sigmoid；[transfer:79](../../extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_transfer.h#L79) 调标准DecodeCandidateGeometry，再对前四列乘2、score无*.25补偿。

合法触发例：报告平台AIV=16，B=8,N=16385、FP32或half。Host total131080>65536，groups2049、span=ceil(2049/16)*8=1032→PATH7；pair span=ceil(1025/16)*16=1040>832，不转paired；不是power family。last owner15的区间start=floor(2049*15/16)*8=15360，end=16385，长1025，先1024再count1，batch*1=8<=32触发GroupedTail。全部raw=0，全部priors=[0,0,32,2,2]，该最后候选每个batch将按源码得到[48,48,256,256,1]，参考[16,16,64,64,.25]。此数值差不是舍入误差。已用独立算式推导，未跑原代码/NPU；40AIV平台这个具体shape可能不触发，不能泛称所有平台失败。补复验应覆盖8/16/24/32/40核、多余量1..floor(32/B)，尤其generic CachePrior尾分支。

### 已确定的路由/契约差异（不是NPU数值失败）

rank1 Host cap40导致>=43/47及available48分支死码，前述行号可静态证明。rank2 Host缺少完整输入格式检查依赖题面合法输入；extra字位/owner字段暗含core<=63，未作硬件泛化。rank3 N上界宽于题面不等于在题面内有错，须明确审查范围。

### 尚未证明的数值与可移植风险

denominator融合1/(Dobj*Dfire)与(1/Dobj)*(1/Dfire)舍入不同；两个有限D乘积可溢出为inf，但这时理论score通常极小、可能仍在atol内，不能仅凭乘积溢出判定不合格。Exp(-large)的overflow/flush行为、Div/Reciprocal精度需要官方目标版本和设备验证。scaled sigmoid2/D再*.25对normal区间通常只改2幂缩放，但subnormal/overflow不可简单视作全域位等价。

rank3预变换priors函数可改变结合顺序，但当前TransformPriors=false，不能列为已启用风险路径；保留它们作为迁移陷阱。13108除5只在已验证长度域正确。TransDataTo5HDVldVaRegImpl、16bit地址/32压缩、ICachePreLoad、ScalarCountLeadingZero/逻辑owner、raw key bit解析、系统workspace宏、cce_kernel/aiv/aligned128导出覆盖均未获得当前版本完整官方稳定契约；只能称高风险版本绑定，不推定非法。

## remaining

rank3所有文件全文读完，但未运行编译/NPU/精度测试，也未证明自动同步、寄存器VA内建稳定；rank1中段231–8529、rank2三个大依赖全文仍待继续。后续必须读长表每一连续行段，对字面metadata与Host生成规则、消费者地址边界逐一对账；不能以当前公式摘要充当未读覆盖。原源码最终hash保持本轮盘点值，账本列所有文件与未读区间。公共API仅在必要处参考题1报告内官方DataCopyPad/高维切分文档；其余明确未核验。

## rank2-static-metadata-full（2026-11-09补读）

`fire_static_metadata.h`连续实读1–280、281–560、561–840、841–1100；全部17张表4032个u64，不是仅看表名。1–77为Kind0 tile32/64/128两组Gather byte offsets，low/high u32依次24i/24(i+1)与20i/20(i+1)。78–767为Kind1/2各tile512/528/640/896/1024，256u64；float/half采用不同UB布局，528与640因Q=ceil(tile/128)×16相同而表相同。768–856、934–1022为Kind3 tile1640/1664，336u64；857–933、1023–1100为Kind4对应288u64。1640与1664同Q因而表相同。

在全文阅读后用**自写独立Python算术转录**Host `fire_host_plan.h:104–251`的布局生成规则并逐word比对17张字面表，全部4032/4032完全相等；未执行归档源码/编译器。规则：中型Q=ceil(tile/128)×16，P=8Q+24（float）或+16（native half），S0=20P+128，V0=S0+24P+128，O0=V0+24P+128，F0=O0+24P+128；大half S0改+448。每u64打包4个`(UBbyteAddress>>5)&0x1fff`，各16bit lane，低VA资源插入zero高半寄存器。Kind3额外输出sink V0，Kind4同时F0/V0输出。small表也完整相等。

因此这里可证是 **F**（固定宽AoS转置）的地址描述符缓存，和 **G**（Host预生成元数据、设备减少scalar SetValue）的编排，并非数值输出LUT。这个相等性证明**不等于**地址消费者、VA寄存器指令、padding/同步或数值正确性的证明；后续blocked_transpose/device_fire必须继续逐行对账。4032word最大13bit压缩仍需消费者实际UBextent与目标SDK契约。

## rank2-blocked-transpose-full（2026-11-09）

连续实读1–400、401–800、801–1200、1201–1600、1601–1933，全1933行。

|行段|语义与证明账|
|---|---|
|1–46|cold `RepairBlockedCenters`：只修xy两列、|2sigmoid−.5+grid|<1/16且真实stride>32(float)/4096(half)的候选，raw∈(-20,0)才用FireSoft53软件exp/denominator/reciprocal位；V_S读guard、S_V回写值。noinline/aligned128仅float1664 image，其他aligned4，是代码摆放候选不是数学优化。|
|47–150|布局：WidePhase(float或large-half) G8、普通half G16；Q=ceil(K/(16G))16，P=GQ+16/24；priors5P、values/work6P、rawPacked6P或5P、priorPacked5P、常数及metadata分开。large float1664 P1688、MetadataBase190208+2688=192896B；largehalf1664 P1680、189376+2304=191680B，均<196352。StagePrior在补UB尾后V_MTE2才读GM，MTE2_V事件1；raw事件0。|
|151–283|BatchOne float先在连续raw流算2sigmoid再转置，逐元素函数与排列可交换；NativeHalfPlanes先b16转置后int16加0x8000翻符号再升float/Exp/2Div。固定符号翻转需要SDK int16模加语义。b16 raw三context、b32 raw六context；priors五context，所有字段同一phase置换，不能只把一张量转置。VA0..3/4..7成对context预载，repeat1的无效stride置0。|
|284–443|输出逆转置五context；large稳定prior用F0，不稳定用V0，largehalfcontext8/13、float11/16对应已对齐字面表。BatchOne按part×batch交织owner，power2用位移否则除/模；非BatchOne按tile ordinal分配，ordinal=nTile×B+b。UniformTiles仅largeFloat。首raw/prior提前搬入，初始化不覆盖DMA有效前缀。|
|444–581|完整owner只清plane尾，partial只清与MTE2前缀不相交的suffix/plane；padding补到转置实际消耗16G组，不当作GM额外读。SkipFullPadding仅静态完整owner分支允许。normal/count mask复位显式。非metadata路径建立raw/prior/output全部VA数组，source/destination的G/Q置换与Host规则一致。|
|582–628|JointRangeBitmapAllLE：比较stride/aw/ah三plane≤32，Not失败bitmap→int16→half仅作0/非0载体（最大32768仍有限），Abs+WholeReduceMax归约；不是把真实数值降half算。scratch重用priorPacked；V_S与S_V成对。多比较补齐lane来自先前零初始化。|
|629–883|ProcessKnownOwner hot/cold：相同n跨batch缓存prior；范围测试可预变换grid−.5与anchor×stride；raw解包释放packed后预载下一tile，V_MTE2保证读完再覆盖。cold center-repair后重新恢复VA，不假设函数保寄存器。hot可通过mask切换join替代显式V fence，**需反汇编/SDK证明**而非源码注释保证。score合并denominator；largefloat单独1/product，其他2/product再*.5。|
|884–1079|geometry依赖四stage：xy减.5、wh平方并行；xy+grid与wh×anchor并行；再乘stride（预变换wh可省），score最后缩放。DualOutput hot把scratch挪S、输出F/V双槽，ordinal≥2才等对应MTE3_V，只有真实将来复用才发回收event，避免terminal欠债。largehalf把O前3Pfloat给下一raw、后3Pfloat给half输出；精确count×5 GM写，最终PIPE_ALL drain。|
|1080–1564|ProcessOwnerSelected将真实prior guard结果选hot/cold；Process另一完整重复骨架均读完，不以“与前文相同”跳读。DualOutput在入口Process直接选专门骨架，因此后者里面`if constexpr(DualOutput)`在当前Process调用下实际不可达；保留未启用代码不算当前收益。|
|1565–1748|完整BatchOne guard：stride≤阈值bitmap经过GatherMask统计retained，retained==GQ省标量max，否则DirectStrideMax及cold-repair；只统计真实完整GQ，不错误用容量K。复用前述数值/布局，源码整段已读。|
|1749–1933|partial guard同统计，但padding stride先定义0，因此retained比较GQ仍作充分条件；不足再真max，不丢修复。真实count GM边界保持，排布逆变换后才half最终cast。|

数学边界与分类：**P** decode参数2sigmoid/xy/wh/score及大stride近零中心修复；**F** 多输入AoS共同phase置换、共享prior跨batch缓存、literal address contexts、typed sign-bit翻转、稳定prior与输出双槽的生存期规划；**G** 生命周期event、尾槽仅未来复用时发flag、稀疏cold函数与hot编译体隔离、alignment摆放。只在实数代数上有`(2/D)^2*aw*stride`与`(2/D)^2*(aw*stride)`、`2/D-.5+g`与`2/D+(g-.5)`等价；FP32重新结合和预乘的误差仍须对1e-5契约证明。范围≤32只限制绝对量，不自动证明所有near-zero中心相对误差；本轮不把hot guard称全域精度证书。

数值pending **R**：软件FireSoft53/CenterReciprocalBits尚待device_fire全文推导；阈值32/4096及raw(-20,0)修复域之外未给全域误差证明。`SetVectorMask`被用作隐式V完成join的特化依赖目标SDK生成代码；metadata字面表与Host相等只完成地址计划一环，vld_va_reg/scatter的13bit压缩和高低VA槽解释需SDK。这里确有异步下一raw预载和输出双槽，比“函数拆阶段”更强，但无性能日志不能断言净收益。

## rank2-device-fire-full（2026-11-09）

连续实读1–400、401–800、801–1200、1201–1600、1601–1728，全1728行；连同上面metadata1100、blocked1933，rank2所有先前未读依赖4761行已闭合。

|行段|语义及边界账|
|---|---|
|1–36|StaticUBBuffer只构造视图；CopyDenseInput/Output同时检验真实地址与bytes的32B对齐，整块普通copy，否则DataCopyPad精确extent，input只在UB补0。不是“长度对齐就假设源地址对齐”。|
|37–86|FireSoft53软件binary53：16×u32 Big(512bit)加减乘、shift、tie-even round53、显式double样式D=m×2^e；53bit多项式FMA与32项2^(j/32)常量，x×(32/ln2)四舍五入整数k，r=z−k，三系数近似后按q=k/32缩放。它不是通用IEEE double：unpack把subnormal丢0、narrow32未处理任意overflow/subnormal、Big.shift未对任意大移位封顶。只可在当前bounded cold修复域使用，不能抽成完整软件浮点库。|
|88–142|CenterReciprocalBits对正normal denom作24步整数long-division加tie-even，power2单独返回。若mantissa M∈(2^23,2^24)，quotient=RN(2^47/M)，输出指数253−ef，代回值q×2^(103−ef)正是RN32 reciprocal；正常指数域由denom≈2..exp20+1给出，不能推广到subnormal/Inf。scaled修复加0x800000是正normal指数加1即×2，不是随意改精度。RepairSensitiveCenters先真stride-max、再两个中心候选逐值读raw，S_V回写同步。|
|144–198|GatherChannelPlanes使用count Counter而非读取无效capacity；静态K128 guard拆64+tail两段WholeReduceMax，V_S读max、S_V释放；普通K<=64单段。guard结果是输入值决定的cold分支，不是公开shape认答案。|
|199–366|PadSmallDecode布局P=K+8、Q=K+16，half raw临时区独立；small index可预取已核验Kind0表或8scalar seed后向量扩展。raw连续Sigmoid→Gather6planes，priorGather5planes，owner=b小case；stride归约与raw Gather交叠。输出8lane转置后0x1f mask保每组前5lane，再half最终cast。CompactGeometry把xy与wh无依赖算术共用barrier。|
|367–624|InitWithInputPrefetch/AfterInputPrefetch以及half K32 raw-ready变体全文：raw事件0先等待并算sigmoid，prior事件2延到需要时才等；VA数组只scalar初始化、cold返回后显式重设；不存在省掉真正prior依赖的猜测。PriorMaskForward仅改变允许保留counter状态的K64特化。|
|625–818|多tile small与KernelFireCandidateDecodeFusion历史泛型均审：ordinal按nTile×B+b，prior同n缓存，raw/plane奇偶轮转，转置后释放nextRaw；out源在MTE3完成前不改，两个event阻止跨迭代提前重用。RunSmallMany K128带静态guard、另有预取索引noinline入口；不是主blocked大tile架构。|
|819–946|DirectStrideMax分full64+tail，whole归约后V_S读，两种SDK scalar读取都S_V释放；Begin/Finish把真实guard等待延后但归约槽独占。unscaled/scaled cold helper按相同原始raw地址重新计算exp/denom/reciprocal，scaled用真实pitch policy，调用者不传opaque Tensor对象越cold边界。|
|947–1177|MaskDecode dense：P与large/half dtype区别，6P≤7K保护prefetch suffix；大float raw staging可存全6K，较小float分5K+K两段。大half先odd/even分离再3相位Gather，三个plane通道仍原候选顺序；typed cast合并。raw周期3 mask 0x49249249等、prior周期5 mask 0x42108421等向量扩展；smallprior的每列独立mask head可恢复，large连续prior无prefix故immutable。|
|1178–1292|MaskDecode process：prior guard缓存；2sigmoid后phase解包（largefloat把obj/fire先逐点乘再Gather，排列交换不混样本），nextRaw在当前geometry前发起、目标O+suffix不碰活跃S；完成前不PrepareNextInput。count*2>=P可用一条Counter(P+count)跨两个plane，余量只在UB；稀尾改逐plane以免越到其它真实字段。输出精确count×5；最后PIPE_ALL drain。|
|1294–1521|FlatMaskDecode flat aligned ownership可跨batch：beginRow/endRow按dtype8/16行单位，n=row%N；StagePrior仅wrap时取prefix（≤7行）+上一batch尾+priors头。每列独立mask head清prefix，后续需恢复，One不再消费可省恢复。period24/40word sparse Duplicate直接生成原周期mask；不逐标量SetValue整个表。足够UBsuffix/额外160B避免prior wrap覆盖index。|
|1522–1728|One与many路径完整：One guard槽在G0+256独立，延迟Finish；flatmany prior缓存同时含n与count；signhalf早于升float，2sigmoid/score*.25对应本族；raw下一tile预取、最后按linear row回写，不因batch wrap乱序。RunDenseDecode K<=1024按denseOne/shortN选flat或Mask；K>1024选Blocked，历史Kernel类不因存在就当当前启用。|

### 数学/同步重点与迁移等级

- **P** 数值敏感中心修复：需要`|2σ-.5+grid|`小、stride大、原logit(-20,0)三个真实值条件；不改其它字段。常数TAB/COEFF是函数近似系数，不是样例/结果答案。但当前源码未给多项式最大误差、跳修复域的全域容差证明，继续列R；不把“软件53”直接称正确舍入exp。
- **F** 先按half位图抽列再精确FP32、odd/even+周期3/5mask替代大索引、softmax/sigmoid等逐点函数与一致排列交换、只统计predicate用half零/非零载体、共享prior缓存在相同n域；需以具体AoS字段/生命周期重新推导，不是任意shape都快。
- **G** 同时检查地址/长度的DMA快慢路、raw/prior独立事件、immutable index预取、8seed向量扩张、cold noinline避免hot函数复杂、phase或输出槽复用的精确生存期；只有源码结构证据，无本轮性能验证。
- **G/SDK R** native scatter/vgather、计数mask、int16模加翻sign、noinline/ICache/alignment和私有entry包装。特别是“mask切换即完成join”不能仅凭注释与源码推导成立。

rank2完整覆盖当前只消除“没读依赖”的缺口，不消除数值/SDK/NPU证明缺口。half/float raw与prior contract仍以题面真实dtype及输出order为准；未发现用输入内容预测答案的结果表，所有静态表可回溯到地址或exp系数。

<a id="rank1-kernel-231-1800"></a>
## 2026-11-09 rank1 kernel 连续补读 231–1800

本节仅记录本轮实际连续阅读范围；不将下文尚未读到的类或历史变体当作已核验当前路径。文件SHA256保持 `4693b60ab79d19e3d73fcb245698ef9d1568435ab80e97eb92570cded8cef310`。

|行段|连续语义账与边界|
|---|---|
|231–288|raw/prior/out使用独立ready/free事件与已使用slot位图；Drain只等待真正发布的free。输出索引先写8×5个seed，TILE>256复用尚未活跃的Work作为seed/ramp/Brcb区域；由rawOffsets−priorOffsets=4i构造组偏移，TILE≤256逐倍Adds扩展，明确S_V与V barrier。不是把每个地址都Scalar写一遍。|
|290–398|泛型KernelFire扁平拥有区间，再用pos%n切开batch边界避免prior环绕越界；每块精确6/5字段DataCopyPad。half先精确Cast到FP32，连续Sigmoid，再Gather为6raw+5prior平面；xy、wh、score按标准顺序算，Gather回5字段并最终half Cast。每轮才分配下轮raw/prior，DEPTH=2不代表已显式跨轮预取，实际释放与DMA依赖仍完整。|
|401–477|total==1 Tiny固定UB路径；Gather4个anchor索引[0,4,12,16]，width bitmap0xC只平方/乘anchor第三四项，Brcb复制stride和最终obj/fire，输出Gather第五项从rf[8]读。half raw在128单独转float，ONE_BASE再偏64，避免与raw float视图重叠。全部GM只精确6/5/5字段。|
|479–594|Row版本以block/batches求coresPerBatch并约束rowEnd，不跨batch；余下算术、索引和事件与泛型同构。coresPerBatch非零需Host的启动契约，不能独立任意调用。|
|596–700|FireFrameStorage静态布局RAW双槽(仅TILE>256)、prior单槽、OUT双槽、11个STRIDE平面、ones、half RF/packed、indices、172个u64 VA；TILE128..1024且128倍数，PAD8仅允许half512，全部区间32B对齐且高水位<188416。priorCursor明确恒0；不能误称prior双缓冲。|
|703–1058|Frame通过编译期开关ROW/RAW3/DIRECT_STRIDE/PAIR256/VA_PHASE_ADDS/HALF_ONE8/QUAD128等生成独立布局。b32原生转置VA地址压缩为四个13bit地址/64bit；raw3三相位以半群排列降低指令数，普通raw六相位；prior5相位，pair/quad会在地址层复制相同prior，不额外读GM。CENTER_AXPY交换raw xy与prior xy平面并在输出映射双重还原。重复次数==1时把repstride清0。VA_PHASE_ADDS仅受限定wholeFloat512使用，以int16向量Add生成相位地址；8×phase的32B单位偏移对应加phase。私有vld_va_reg/scatter ABI仍SDK R。|
|1060–1314|Run的static_assert完整限制合法组合。whole固定512；pair把2batch×256压成512帧，quad把4batch×128压成512帧，N=4096时DataCopy blockLen/stride常量恰与GM间距一致。先DMA再VA初始化，half rawFirst让Sigmoid早于prior等待；ONE8用repeat src stride0只广播8个1，而非填满6FRAME。prior转置后cache跨batch；不用DIRECT_STRIDE时消耗原prior AoS区存4份stride。whole xy使用2×(σ−.25)+grid的Axpy，实数等价但FP32并非逐指令bit等价，后续需容差/边界证明；本段未含cold修复。输出5相位恢复AoS，跨batch步长精确，slot复用等MTE3_V，终结Drain。|
|1318–1377|Pair576覆盖两个832候选的pair帧1664，raw/out共享同一slot，另有prior和11平面Work；FP32高水位182912B，half183136B，SDK保留区未占用。复用slot必须同时等V_MTE2及MTE3_MTE2，后者保护仍在DMA输出的旧raw别名，不能只等Vector结束。|
|1379–1711|DTYPE_RAW_PRED宏改变half/float编译镜像及score融合资格，不是数据内容检测。模板仅N32768/32769、B8/C40固定族；block0..38各832候选，block39尾另路。prior832仅加载一次，原生转置复制给两batch。4次iteration涵盖8batch；ROTATE_PAIRS仅把pair访问序列按cohort=(block&3)循环置换，输出地址使用真实pair，未改样本顺序。SCORE_FUSED先乘两个denom再对5平面求倒数，实数正确但分母乘积溢出与舍入需数值审查；本身不是逐元素数值等价证明。halfPackedFloat与RawFloat共区，Cast输出后显式V barrier保护下一轮WAR。|
|1712–1800|宏解绑前固定下游QUAL镜像标志，防止跨TU错误泄漏。Row64静态资源11264/12928B，ones与Div源相反512B bank group相位；索引以4个u64写8个ramp seed、20个u64写40个output seed，S_V后才允许向量扩展。1800处函数尚未读完，不标完整。|

迁移分类：**P** 6raw/5prior/5output字段与decode数学；**F** AoS→平面→AoS的相位置换、跨batch共享prior、pair/quad将小行合帧；**G** 精确生命周期别名复用、初始化与首DMA重叠、seed向量扩展、编译期组合static_assert与银行相位布局。各项均只有源结构证据，无本轮设备性能证据；尤其CENTER_AXPY、score倒数融合及私有VA builtin仍需对应数值/SDK验证。

<a id="rank1-kernel-1801-3000"></a>
## rank1 kernel 连续补读 1801–3000

|行段|连续语义账与边界|
|---|---|
|1801–1881|RowTight64完整余段：8→16→32→64倍增索引，total字段私有解释为rowLanes；batch/lane除法只负责真实拥有区间，count≤64。field repeat stride9对应72float平面，strideBinary源2 repeat stride0复用同一个stride；一次Muls覆盖4字段、一次Add覆盖2中心、两条Mul覆盖2宽高，仍保持原FP32节点。GM精确extent，最终MTE3_V等输出完成。|
|1887–2023|HostComposite32静态资源7168/7872B，index VECIN/VECCALC两视图同物理地址，非重复存储。先raw/prior的event0，再index DMA的event1；Sigmoid只等event0，Gather才等event1。Host生成的16×STRIDE索引把11输入平面与5输出平面一次Gather化，tail STRIDE按8/16/24/32取，valid count输出限制防止UB padding泄入GM。|
|2025–2190|FlatComposite把prior staging扩到288words以涵盖小N的batch wrap，高水位7680/8384B；静态PACKET_GM只允许TOTAL66/N33且CAPI_PIPE非Single。完整32块使用普通DataCopy，其他精确Pad；half EARLY_RAW_C1先发raw-ready，prior由后续index-ready fence一并保护。Packet tail STRIDE8硬编码raw12/输出10，合法性依赖COUNT2的固定route，不是一般8尾实现。C API依赖边有PIPE_V，mask变换是否完成隐式join仍需SDK确认。|
|2191–2380|完整32 CAPI链依次192Sigmoid、192rawGather、160priorGather、128坐标double、64中心/宽高操作，score32；normal repeat stride4跨32float平面，stride源repeat0共享。half镜像可先将5平面Cast后按half索引Gather，排列与逐元素Cast可交换，但索引必须为half字节单位。其他路径Gatherfloat后Cast。Packet可跨batch，prior若wrap整N staging且表按真实先后取，不需重复GM读同priors。SPARSE_EVEN把logicalBlock/2归属同一块：该函数本身未过滤奇block，必须核对上层entry是否做筛选，否则会多写同一输出；此处不提前判正确。|
|2382–2627|Inventory684另建544索引容量，高水位7296/8000B。完整Single多Gather一列stride（6prior平面），anchor基址由288改320，输出索引基址352改384：把stride复制进表以使用128计数一次乘四坐标，避免4次repeat setup；不是修改stride数值。非Single保留老5prior平面及stride repeat0。所有fallback与packet分支逐行审过，和上一类重复也不跳读。|
|2629–2862|EarlyOne684变体把Duplicate(192×1)提前到GM设置和DMA之前，随后完整Single不重复建ones；其余与旧548表布局一致，仍是5prior平面，不误用Inventory684的6prior。输出精确，CAPI/泛型都保留完整分支。它改变启动编排，不改变decode数学；不能由代码存在就认定生效或提速。|
|2864–2946|FourByEight用128项地址表拆4×8，表前48为6field×8 offset=24i+4f，中40为5field×8 offset=20i+4f，后40为5×8输出逆排32f+4i。pos=(block/2)×8同样需上层奇偶筛选；本函数自身对奇block不return。CAPI算术与8平面一致，输出40元素。|
|2948–3000|Even8Half32Index全608字面值已逐项审读：raw字段0..4映射到RF字节基4416+24i+4f；prior字段排列0,1,3,4,2,2,2,2从384+20i+4f形成重复stride；raw scorefire字段5另放在前448项末尾；输出160项为half字节64f+2i。后续Storage索引从5824起到8256B，本段类未结束，不提前计完整。该表由布局产生，与输入数值无关。|

归类：**P** 6/5字段的方程与专属tiny路由；**F** batch wrap的prior缓存、输入/输出Cast和纯排列交换、索引复制stride换取合并指令；**G** raw-ready和index-ready解耦、ones早发与首DMA并行、静态容量/地址单位证明。**R** SPARSE_EVEN及block/2归属要结合entry守卫，私有C API mask/fence行为与普通DataCopy非32B源起始地址仍需SDK/生成物确认。

<a id="rank1-kernel-3001-8529"></a>
## rank1 kernel 连续补读 3001–8529

本节补齐rank1主kernel此前未审阅区间。源文件SHA256仍为 `4693b60ab79d19e3d73fcb245698ef9d1568435ab80e97eb92570cded8cef310`，总行数8774。3001–8529按连续编号段读完；另复读并核对本轮末尾入口分派8701–8774。下面把相邻重复模板按其共同资源/公式归纳，但区间均逐段读过，不以模式匹配代替审阅。

|行段|连续语义账与边界|
|---|---|
|3001–3186|结束Even8Half32专属小核：raw读取192项、prior160项，half转置并构造重复prior stride，再对sigmoid和几何量做逐点计算，索引表Gather出对应输出；同步后half写回。随后Storage显式给出RAW/PRIOR/OUT与工作区的UB区间，TILE>256才启用双buffer，slot复用通过HardEvent等待；Drain只收尾已发布的事件。高水位与参数化tile有关，依赖本文件static_assert和Host route，不外推为任意tile安全。|
|3188–3681|FrameStock1112初始化转置/VA地址表，按geometry将工作分配到frame；首轮raw/prior DMA与地址准备交错，raw3/direct-stride等编译期组合改变数据平面和相位映射。主算式保留FP32 sigmoid、prior/score和坐标变换，输出经转置恢复AoS后转写。VA来自预计算向量地址并借用工作区种子扩展，减少逐地址scalar构造；事件分别保护DMA读、向量消费、输出写和slot回收。私有转置/VA ABI与硬件支持仍需SDK确认。|
|3682–4018|N=32768/32769的half PI8输入路径独立展开：固定UB布局，读取half后做有符号偏移和原生转置，再执行专用分母/score计算。两个N分别采用编译期尺寸与边界参数，涉及输入矩阵末端的对齐、有效长度和尾行地址；它不是对任意N成立的通用分支。公式中的倒数与除法选项由编译期开关控制，数学近似误差未由源码单独界定。|
|4020–4356|`KernelFirePairSpan1112`和`PairSpan47`将相邻候选合成pair处理，span/count/stride显式作为模板约束，输入与输出使用不同的DataCopy参数处理padding/行间距；phase表对应pair内两组地址。需同时依赖Host给出的块数、pair边界和合法尾长，避免跨batch取prior或写入无效输出。|
|4358–5699|`FireA2RowF4`与`FireA2RowH10`分别覆盖float与half镜像，重复的Storage/VA helper按数据类型、row数量、缓存步长和资格宏产生不同布局。VA_GROUP28用于减少地址表/字段搬运；`ROW_CACHED_STRIDE`只对限定row N=4097启用。路径先发布raw DMA，和prior/索引准备重叠，再执行向量数学、重排并回写；slot重用等待对应MTE/V事件。静态assert和UB reserve表达编译期容量约束，不证明目标SDK实际编译后寄存器/UB分配或无竞态。|
|5700–5807|类型/资格宏切换与下一组pair storage之间的过渡段：宏影响编译镜像、HALF/FLOAT路径和score融合许可，不读取样本内容。审阅边界处只确认其宏作用域及随后模板参数，不据宏名推断性能收益。|
|5808–6330|F5 pair storage/kernel为N=32768/32769专用pair批处理，支持score融合与pair轮换的受限模板组合。VA_GROUP38把地址表按组复用；raw/prior/out与Work存在别名，输出Cast和DMA完成前禁止覆写源数据。score融合在实数代数上压缩分母/倒数路径，但分母乘积的溢出、下溢及舍入需以输入域做数值证明。half score融合的qualification静态断言含恒假条件，故该组合当前实际禁用；不能把其实现分支当作可达路线。|
|6331–6900|pair span/尾段变体继续区分N、dtype、pair位置和valid count；小尾块采用精确DataCopy长度/二维stride，地址单位分别按元素或字节解释。ROTATE_PAIRS只重排block访问顺序，真实输出地址仍由pair身份决定，故不应改变逻辑输出排列；仍需运行时与任务契约验证。异步读写别名区要求MTE2/MTE3和Vector相应事件共同完成后才释放。|
|6901–7520|后续F6 span/add pair路径按输入型别和尾长拆开，部分路径把Add或score阶段融合到已有向量区，其他路径保留独立临时平面。模板开关将N、半精度/单精度、批数与span约束编译期固定；固定DataCopy行数/stride只适用于对应的tile族。同步沿用ready/free及双缓冲slot语义，不能把PIPE barrier视为跨核全局同步。|
|7521–8100|F6余段继续处理成对候选与输出恢复，核对pair最后一组的有效候选数、奇数尾、输出半精度打包及batch偏移。对齐扩展区只允许作为DMA padding被消费，写回长度受count限制；尾部缺少有效元素时不得读取padding作为有效值。这里的静态约束是源码级边界证据，编译器生成指令与DMA API实际语义未验证。|
|8101–8529|固定尾核按有限block index分派：T5的320/512元素尾段和T6的513/321元素尾段分别采用不同起始位置、源/目标stride及有效长度。静态assert把block39/42、N和尾长绑定，避免尾块公式被误用于通用分块；输出尾部仅写有效count。T6 513/321路径的ExtParams含非零源/目标gap，必须与Host传入shape/tiling以及DataCopyExtParams的单位逐项一致，不能仅凭变量名判定正确。|
|8530–8700|既有入口分派前段与主入口衔接：按REUSE/tiling key选择专门kernel及启动块数，入口层负责筛掉不拥有工作的block；内核内部`block/2`或稀疏block映射本身并不一定再次筛选。总tile和dtype需与Host tiling相符，否则可能漏算/重复写；源代码只证明分派逻辑，不证明所有key在Judge中都被触发。|
|8701–8774|入口剩余分支完整读毕：0xb300走FRAME=768的Frame kernel并限制block<48；0xc000族从key提取FRAME/ROW并限制block<48；0x7b36在half走Even8Half32、float走Inventory684；0x7b2e在float/half分别路由Inventory/EarlyOne；0x7b34/0x7b38进入packet composite；其余REUSE≤2选TILE64/256/832泛型，5256选Row256，其余Row832。末尾PIPE_ALL barrier位于主dispatch后，不能代替各路径内部DMA/slot同步。宏清理只影响预处理作用域。|

### 源码审计归纳与证据边界

- **P（题目专属数学/边界）**：6raw、5prior、5output的decode关系在tiny、frame、pair与尾核中复用；half/float及坐标公式的细节需依题目契约判定。分母融合/倒数、half近似与中心AXPY的FP32舍入不是源码层面的逐元素等价证明。
- **F（可迁移的数据组织）**：AoS到平面再恢复AoS；跨候选/批次复用相同prior；pair/quad合并短行；预生成并压缩VA地址表；以精确valid count配合DMA padding；这些做法的适用条件是shape、布局与batch边界被Host静态约束。
- **G（编排/资源技巧）**：初始化与首DMA重叠；raw/prior/index不同ready事件；slot双缓冲与Drain；编译期模板特化；原生转置/VA寄存器路径；尾段专用kernel。它们提供可复用的设计候选，不代表当前环境已验证收益。
- **R（待证明事项）**：私有C API/builtin及DataCopyExtParams具体单位和支持版本；多事件/别名生命周期的SDK语义；所有Host key到Device kernel的完整可达映射；分母融合及近似函数的全域误差/异常值边界；实际UB/寄存器、编译、数值正确性和性能均未在本轮运行验证。

本次状态：rank1 kernel 1–8774连续源码覆盖完成，文件SHA256未变；这不是整道题所有rank提交的“全部实现等价”，也不是数学、SDK、同步或性能验收。
