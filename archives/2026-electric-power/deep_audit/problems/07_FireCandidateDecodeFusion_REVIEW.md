# 题目 07：FireCandidateDecodeFusion——获奖实现设计复盘

> 三份源码已完成连续审读；详见[逐行账本](07_FireCandidateDecodeFusion.md)与[覆盖记录](../coverage/07_FireCandidateDecodeFusion.json)。本报告描述静态设计，不声称目标硬件正确性或性能已复现。

## 1. 契约与数据依赖

输入 `raw[B,N,6]`（FP16/FP32）及跨batch共享的 `priors[N,5]`（FP32），输出保持候选顺序的 `[B,N,5]`。每行计算两个中心坐标、两个宽高和 `sigmoid(obj)*sigmoid(fire)`；half输入也必须先用FP32做sigmoid、平方及坐标中间，再按题目约定最终转half。禁止NMS、TopK、过滤或重排。允许N不是32/64倍数，尾部必须不越界。

依赖图有天然可优化点：priors在batch间复用；输入是6字段AoS，输出5字段AoS；逐候选彼此独立。于是可将小段候选搬入后转为字段平面，完成向量运算，再按原序写回AoS。

## 2. 三份实现的设计主线

| 顺序 | 主线 | 特别优秀的部分 | 需要谨慎处 |
|---|---|---|---|
| 1 | 多shape family：tiny、row、flat/frame、不同packet与融合路径；Host按N/B/dtype等选专用key。 | 将priors转置/缓存并跨候选复用；原始输入、prior、索引和输出使用分离ready事件及缓冲slot，启动搬运与初始化/算术交叠。 | 迁移时按目标shape核对Host容量与实际key路由；generic grouped-tail 等备用路径只在其对应输入族下参考。 |
| 2 | 以静态metadata、blocked transpose与device-fire专族进行选择；短行和pair/quad组合帧，压缩地址/index计划。 | 通过生成/压缩VA地址让Gather后直接进入字段布局；小batch合成大frame减少每行固定管理。 | 地址单位、tail stride、batch wrap与静态表消费者需一致；部分C API mask/同步规则依赖SDK。 |
| 3 | 不同frame/row/packet计划和同步配对，按限定shape选择布局变体；保持exact valid count并分开发布输出字段。 | 将“输入DMA、地址表准备、sigmoid、输出”按真实依赖流水，而不是全段`PIPE_ALL`串行。 | 模板组合大量固定化；改变`CENTER_AXPY`或分母倒数次序实数等价但FP32舍入/溢出不同。 |

## 3. 可复用的方法

- **AoS→SoA→AoS融合**：6输入字段变成寄存器/UB平面，做5输出字段向量化，再精确恢复原候选次序；适合小字段逐元素decode。需要算清转置成本是否低于逐字段Gather。
- **跨batch共享不变量**：prior只读一次后供多个batch帧使用，必要时在片上转置。适用条件是prior确实共享且缓存容量够；不要因输出batch维而重复读取同一数据。
- **短行合并成frame**：把2或4个batch的短候选行拼成向量长度固定的frame，降低每行索引/启动开销。前提是batch边界不跨读、输出地址仍对应原batch。
- **地址计划编译期生成**：seed少量规则offset，再以倍增Add或压缩VA表扩展；每个元素不再Scalar建索引。地址表必须由shape推导而非输入内容，索引单位和mask长度要严谨。
- **异步生命周期设计**：raw/prior/index/output用独立事件，输入ready后先执行不依赖index的sigmoid；复用双slot必须等所有Vector和DMA读写均释放。

## 4. 不显眼的 trick 和失效条件

| 技巧 | 用途 | 风险 |
|---|---|---|
| 8个offset seed后向量倍增 | 少量Scalar初始化整段索引 | tail有效数、溢出与seed初值要覆盖全范围 |
| VA表按phase压缩 | 直接表达AoS/SoA gather/scatter映射 | 私有builtin、byte/element offset和SDK版本耦合 |
| pair/quad frame | 小batch合并减少重复prior/管理 | batch wrap必须通过精确切片或合法staging处理 |
| score两个sigmoid分母合并取倒数 | 少做除法/向量操作 | FP32乘积可能溢出/下溢且舍入次序变更；题面要求FP32中间 |
| 坐标公式用Axpy或FMA重写 | 减少向量指令 | 需要验证逐节点舍入容差，不可只用实数恒等式 |
| raw/out同UB slot别名 | 节省片上空间 | 等待MTE3仍完成旧DMA，不只是等Vector读完 |

账本中特别指出：Host/SEL可达性、奇偶block是否在上层筛选、generic tail比例及私有`vld_va_reg/scatter`契约均需按实际执行组合核实。近似/融合写法不能因获奖排名自动推广。

## 5. 复用分级与结论

- **P**：decode公式、5输出字段顺序、FP32中间和严格valid tail。
- **F**：字段平面化、prior跨batch复用、短行合帧、按shape生成地址计划。
- **G**：多队列依赖流水、seed扩展、buffer双槽生命周期管理、精确输出分区。
- **R**：原生VA寄存器、C API Gather/Scatter、cache/事件和宏生成器。

最值得借鉴的是把解码看成“跨batch共享先验 + 独立候选的小型AoS向量变换”，并使地址生成、布局转换和运算融合成一条可证明的pipeline。不能直接抄具体key、帧宽、事件号或score倒数融合。源文件静态审读完成，但全域数值、SDK协议、NPU与性能仍未独立验证。

## 6. 证据入口

- [题面公式与dtype约束](../SPEC_CONTRACTS.md#07-firecandidatedecodefusion4页)
- [rank1长kernel最终归纳](07_FireCandidateDecodeFusion.md#源码审计归纳与证据边界)
- [rank2 static metadata / transpose / device fire](07_FireCandidateDecodeFusion.md#rank2-static-metadata-full2026-11-09补读)
- [完整覆盖](../coverage/07_FireCandidateDecodeFusion.json)
