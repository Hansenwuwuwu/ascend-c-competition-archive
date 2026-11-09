# 题目 09：线性方程求解 solve——获奖实现设计复盘

> 三份源码已完成连续首轮审读；详见[逐行账本](09_solve.md)和[覆盖记录](../coverage/09_solve.json)。本文提炼结构与迁移条件，不以获奖名次替代运行证明。

## 1. 契约和关键问题

求 `A X = B`，`A[...,n,n]` FP32可逆且 `cond₂(A)≤1e3`，`n≤512`，RHS宽度 `k≤256`，A batch≤64；无batch的B可在A batch上广播。B既可能向量`[...,n]`也可能矩阵`[...,n,k]`，输出必须保留正确rank，不能给向量添加单例k维。需满足元素误差与 `||AX-B||F≤1e-4||B||F`。核心风险是pivot/置换、rhs重复读取、多rhs重用分解和残差必须基于原始A/B。

## 2. 三份实现主线

| 顺序 | 设计路线 | 亮点 | 边界 |
|---|---|---|---|
| 1 | 按n/k/batch分为小resident、窄正交/特殊矩阵、Cube或通用消元路线；快路后可基于原输入残差作refine。 | 对shape族选择并行轴/资源族；部分特殊路径用列标签替代物理换行，针对多RHS分组共享A相关数据。 | normal/Gram近似路线可能平方条件数或依赖额外结构；必须看guard及精修条件。 |
| 2 | 小矩阵专用求解、并行按列/批次分工与通用解法；同时保留高低分量残差修正路径。 | 聚合多个rhs并复用因子，窄k避免通用大面板开销；在小shape上用受限模板省动态管理。 | workspace/列距/静态容量依赖Host条件，不能将扩展path假设为题目实际可达。 |
| 3 | Gauss-Jordan/协作式panel与特定small/medium路径，多个核共享因子/校正状态；有精确输出pack和valid wave。 | 从A分解到多个RHS应用分开安排，多核在无效job/尾wave时仍参与集合式同步；输出重排可批量或小tile pack。 | SyncAll/GlobalBarrier/exit协议复杂；通用残差段有普通FP32乘积，不能一概称高精度残差。 |

## 3. 可迁移的设计

- **同一A分解一次，多RHS批量应用**：对k>1最直接的复用机会；判断tile是否能驻留、RHS列并行度是否足够，避免每列重复factorization。
- **按形状变换并行轴**：小n时核间开销可能不值，按RHS或矩阵列分工；大n可按panel/row协作。选择依据是实际独立任务量和同步成本。
- **逻辑pivot/置换标签**：降低全行搬运，但所有factor、RHS和最终输出映射必须同一索引定义。
- **残差校正作为验收链**：对初值用原始A/B计算r=B−AX，再解修正方程；只有明确校正迭代的误差/收敛条件时才有保证。一次小残差并不自动给X的逐元素误差证书。
- **精确shape与广播ABI分离**：向量/矩阵B、batch广播、空/尾输出分别做shape guard与store路径，避免数学计算正确但返回张量rank错。

## 4. 隐蔽技巧与踩坑点

1. **Gauss-Jordan无物理换行**：逻辑pivot行号/位图替代整行拷贝；只适用于所有后续消费者正确解码标签，并用稳定pivot。
2. **列panel共用workspace**：UB里分区、scratch复用降低容量，但要逐阶段画活跃区间，检查late consumer和下一轮写入。
3. **低精度/Neumann逆种子**：只是候选，不是无条件最终解；原A/B residual、数值条件和fallback必须覆盖所有shape。
4. **tail时每float小DMA**：语义精确但可能产生大量请求；是否值得要从DMA请求数和对齐实际测量，而不是只看字节数。
5. **空核也参与barrier**：只要其他核做集体等待，valid=false也不能随意return；需要逐轮证明最后wave无人提前退出。
6. **补偿精度分支差异**：rank3不同算法体的pair arithmetic并不完全同构；报告记录某些一般残差路径为普通FP32，不可笼统归类高精度。

## 5. 复用等级与结论

- **P**：左解、batch广播、vector/matrix B的输出rank、原输入残差。
- **F**：分解复用多个RHS、逻辑pivot、窄/宽RHS并行轴切换、候选解残差校正。
- **G**：面板owner、精确tail pack、valid-wave集体同步与workspace活跃期。
- **R**：Cube/FFT/SPR、私有barrier、裸UB布局和CANN宏。

最大的可复用观念是把solve拆成“矩阵分解、多个右端应用、原输入残差验证/修正、按原B形状写回”，并针对k选择并行布局。切勿把有条件的特殊矩阵或低精度候选推广成全域方案。本轮只静态读源，SDK/API、数值误差和性能仍待实测。

## 6. 证据入口

- [契约](../SPEC_CONTRACTS.md#09-solve2页)
- [题目方案对比与rank2/3补读](09_solve.md#s9-rank比较可迁移性与未读清单)
- [rank3完整kernel同步/残差边界](09_solve.md#s12-rank3-中段闭合241–3770-连续实读2026-11-09)
- [覆盖](../coverage/09_solve.json)
