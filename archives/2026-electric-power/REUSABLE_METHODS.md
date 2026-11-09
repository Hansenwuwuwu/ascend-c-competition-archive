# 可复用方法：首轮源码证据与迁移边界

2026-11-08。对应 [归档清单](README.md) 中的 27 份公开公示代码。本文件是**首轮结构阅读**，不是 13.9 万行全部完成深读、已验证的性能秘籍或合规结论。以下“看到的实现”有源码定位；“可借鉴”是工程推断。没有执行任何获奖源码，也没有独立验证其速度、误差界或分支命中。

## 1. 真正的编译期路径，而不只是拆文件

例：[FireCandidateDecodeFusion 公示顺序1 Host](extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/op_host/fire_candidate_decode_fusion.cpp)，第 21–100 行；[solve 公示顺序1 kernel](extracted/solve/rank1_submission_1171359/code/op_kernel/linalg_solve.cpp)，第 210 行和 2027 行。

- **看到的实现：** Host 根据 shape、dtype、可用 AIV 数选择 template key；kernel 有 `STATIC_N/STATIC_WORKERS/STATIC_K` 或 `SN/SK` 等模板参数。Fire 的第 10–19 行还保留了通用路径回退逻辑。
- **可借鉴：** 一开始按算法和规模分族；让编译器看到维度、步长和资源上限，减少运行期地址运算，并控制每条路径的临时空间。
- **边界：** 拆成多个头文件本身不产生性能收益；必须追踪 Host→key→实例化→真实 kernel 的完整调用。不能复制赛事 shape 表后当作下一题通用实现。

## 2. 小 case 的“启动规模”和“实际 worker 数”分离

例：[TreeLineMinDistance 公示顺序1 tiny](extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_tiny.h)，第 346–352 行；[其 kernel 入口](extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_line_min_distance.cpp)，第 555–596 行；[SplineConv 公示顺序1单边处理](extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_single_bulk_stage.h)，第 6–9、47–53、102–109 行。

- **看到的实现：** `GetBlockIdx()!=0`、`block>=active` 等 guard，显示实际 worker 可以少于启动 block。Tree 还有不同路径使用 1/3/8/9/16 等有效 worker 的入口判断。
- **可借鉴：** 对极小任务，把 launch plan 与有效任务分工分开测，不预设“更多核一定更快”。
- **边界：** 这里仅证明源码包含该组织方式，不证明某个 Host key 必然以 8 核启动只让 1 核计算，也不证明它在下一代设备更快。必须成对核对 Host 和 kernel，不能照搬空转核数。

## 3. 批量 stage、权重/图数据驻留与生命周期复用

例：[SplineConv 公示顺序1 resident](extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_resident.h)，第 16–45 行；[single_bulk_stage](extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_single_bulk_stage.h)，第 18–44、63–96 行。

- **看到的实现：** `cacheGraph/cacheWeights` 控制本地保存图和权重；先搬入整批 features，再从 UB 取实际 source；清零输出与权重/feature 搬运用不同事件组织。
- **可借鉴：** 把重复小搬运转成一次 stage，并按真实复用次数决定驻留哪些数据。将输出清零、地址准备等独立工作放进搬运窗口。
- **边界：** 需计算 UB 容量、读写生命周期和 backward synchronization。合并搬运可能引入无用字节，驻留也可能降低并行度；不能仅按 GM 逻辑字节减少量推算提速。

## 4. 直接 UB 布局与省初始化的静态 kernel

例：[TreeLineMinDistance tiny](extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_tiny.h)，第 381–384 行；[SplineConv single_bulk_stage](extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_single_bulk_stage.h)，第 18–22、63–67 行。

- **看到的实现：** 以固定 `TPosition::VECCALC` 地址构造 LocalTensor，或通过 `local_tensor_at` 定位小缓冲，不一定每条小路径都初始化完整 TPipe/TQue 图。
- **可借鉴：** 对真正固定且很小的算子，显式设计 UB 内存布局和事件表，减少通用运行期开销。
- **边界：** 地址对齐、容量、类型重解释和重叠生命周期都必须证明；不是把所有 TPipe 删掉，也不是删除必要同步。

## 5. Packed VA 表与编译器内建转置：有价值，但属于高风险接口

例：[FireCandidateDecodeFusion 公示顺序1](extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/op_kernel/fire_candidate_decode_fusion.cpp)，第 720–742 行。

- **看到的实现：** `StorePacked` 将地址右移后打包，`__cce_scalar::vld_va_reg` 装入 VA0–VA3，随后调用 `scatter_vnchwconv_b32`；单次 repeat 对步长有单独处理。
- **可借鉴：** 关注“地址表生成/装载”的开销，而不只关注转置算术；固定布局的地址表可以预计算、打包并复用。这是比反复 SetValue 更深的一层优化线索。
- **边界：** `__cce_scalar` 和 VA 寄存器名称不等于稳定的公开 Ascend C API。源码注释标注 CANN9/C220，不代表别的版本/架构支持。迁移前查当前官方接口、编译产物及地址可表示范围；优先找公开 API 的等价表达，不盲抄内建指令。

## 6. GM 标量预读/缓存预载，用真实工作掩盖冷访问

例：[Tree tiny](extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_tiny.h)，第 352–371 行；[Spline single_bulk_stage](extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_single_bulk_stage.h)，第 14–17、58–62 行。

- **看到的实现：** Tree 用 `volatile __gm__ uint64_t` 预读输入首几个缓存线；Spline 用 `DataCachePreload` 提前处理 edges/pseudo 的读。
- **可借鉴：** 小 case 的冷读、标量元数据访问和向量搬运不一定有同一种代价；可把真实必要数据的预读与后续独立工作重排。
- **边界：** 需要证明每个预读地址在输入分配范围内、满足对齐与 dtype 约束，并确认 compiler 没删掉；多余预读也可能增加争用。不能把这当成通用“加 volatile 就快”的技巧。

## 7. Mask/Counter 模式和融合连续平面，减少重复配置

例：[Fire 公示顺序3 denominator geometry](extracted/FireCandidateDecodeFusion/rank3_submission_1173427/code/op_kernel/candidate_denominator_geometry.h)，第 37–81 行；[Spline resident](extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_resident.h)，第 100–118 行。

- **看到的实现：** 先 `SetMaskCount/SetVectorMask`，再调用 `isSetMask=false` 的底层算术；多个连续字段有时用一次长度更大的 Muls/Adds/Mul 覆盖。
- **可借鉴：** 若数据布局允许，用批量向量指令和可复用 mask 代替逐字段/逐小段调用。
- **边界：** API 参数单位、repeat 上限、padding 有效性、别名约束和退出时的 mask 恢复都要审计；“调用数少”并不自动证明生成指令更少。

## 8. 精度分层：快速路径 + 残差/误差检查，而非一律降精度

例：[pinv 公示顺序1 refinement](extracted/pinv/rank1_submission_1060506/code/op_kernel/pinv_refinement.h)，第 235–258 行；[lstsq 公示顺序1](extracted/lstsq/rank1_submission_1172572/code/op_kernel/linalg_lstsq.cpp)，第 10–85 行；[Spline resident](extracted/SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_resident.h)，第 111–118 行。

- **看到的实现：** pinv 显式构造 `gamma`、残差范数和 `denominator`，条件不成立时返回失败以放弃当前分支；lstsq 使用 hi+lo 两个 FP32 分量和补偿算术；Spline 有稳定累加分支。
- **可借鉴：** 给容易出数值问题的矩阵算子建立误差预算，将较便宜的候选解和更精确的修正分开。
- **边界：** 本轮没有证明这些误差界或所有退化输入；必须基于下一题的条件数、rank、容差、NaN/Inf 契约重新推导，不能复制常量就声称数学保证。

## 9. Cube/Vector 分工包含控制协议，不只是加一层双缓冲

例：[pinv 公示顺序1 Cube residual](extracted/pinv/rank1_submission_1060506/code/op_kernel/pinv_cube_residual.h)，第 27–58、61–80 行。

- **看到的实现：** Cube 有 int8→int32 和 float 路径；通过 workspace control 字段、cache clean/invalidate、`SyncAll` 和 `CrossCoreSet/WaitFlag` 处理 correction/stop 协议。
- **可借鉴：** 为大矩阵阶段划分真正的生产者和消费者，并明确状态机、缓冲归属与退出条件。
- **边界：** 该代码含特定模板宏和预处理条件，不能仅凭文件存在认定获奖分支启用了 Cube residual；必须先跟踪 Host/key/编译条件。数值量化和 cache 可见性要独立证明。

## 10. 昂贵函数后移：距离先比较平方，最后统一开方

例：[Tree tiny](extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_tiny.h)，第 320–342 行及前面的 `best/best_index` 更新循环。

- **看到的实现：** 最小值选择完成后统一 `Sqrt(distance, distance, 32)`，再按输出类型 Cast 和存储索引/距离。
- **可借鉴：** 先找不改变排序的单调变换，将 sqrt/归一化等操作从候选内层循环移到最终结果。
- **数学边界：** 对非负有限数，平方距离的 argmin 与距离相同；但仍需处理溢出、NaN、退化线段和并列候选的索引规则。别把“单调”等同于浮点实现全输入严格等价。

## 下一次使用时的优先级

1. **优先复用方法论：** shape 分族、实际路径追踪、驻留/批量搬运、生命周期和精度预算。
2. **有相似数据流时再提取：** packed transpose、标量预读、mask 配置、单 active-worker 路径。
3. **先查版本/协议再碰：** `__cce_scalar`、手写 UB 地址、cache clean、跨核 flag、低精度残差修正。
4. 先完成原题算法/契约与分支命中复核，再提取最小可测试单元；保留来源、作者、提交 ID、原始 hash。
5. 每次移植报告分开写“数学成立”“编译/数值通过”“目标设备实测快了多少”；本公示分数不能直接作为单个 trick 的收益。

## 尚未完成

- 27 份提交逐个建立 Host→key→kernel 完整路由表。
- 各路径的算法复杂度、误差界、越界/同步安全与可移植性深审。
- 官方 API 文档逐条交叉核验、原设备复现、单变量消融。
- 原作者授权/许可证逐份确认，以及最终公示结果更新。

因此当前交付是可追溯的源码资料库与有证据的阅读入口，不是已经提炼并验证完成的通用 kernel 库。
