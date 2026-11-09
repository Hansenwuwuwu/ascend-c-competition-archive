# 十题题面契约：逐行源码审读的判据

来源：用户 `problems/1.pdf` 至 `10.pdf`，各文件为 2026-10-12 的网页打印快照。主审已读完十份提取文本；公式/表格图片的视觉核对状态另列。原 PDF 未改，SHA-256 见 [specs/manifest.json](specs/manifest.json)。这些题面早于 2026-11-08 公示，**契约差异必须先考虑题面/ABI版本变化，不凭差异推断参赛违规。**

## 01 SafetyBeltViolationGeometry（6页）

- 输入连续 ND：`boxes[B,4]`、`anchors[B,2]`，二者 FP16/FP32；属性 `waist_ratio` 为 FP32，范围 `(0,1]`；可选 `threshold[1]` 固定 FP32、非负，缺省 Host 传 tiling 值 0，不访问不存在的 GM。
- `Yref = y1 + ratio*(y2-y1)`；严格 `cy > Yref + threshold` 输出 1，否则 0；等号不可违规，不引入 epsilon。输出 `[B,1]` 固定 FP32、只能 0/1。
- B 1..4096；合法框 `y1<=y2`；坐标有限。FP16 的 y1/y2/cy **先升 FP32**后做公式和比较。FP32和FP16最终判定都要与参考完全一致。
- 必审：输出/threshold ABI、RATIO_ONE 是否真正等价于按题面顺序计算、Axpy/FMA/重排导致边界翻转、尾部越界、可选输入GM访问、数据配对顺序。
- 页2接口表、页3公式和严格边界由分题审读者视觉核对；文本已主审完整读。

## 02 pinv（3页）

- A `[...,m,n]` FP32，输出 `[...,n,m]` FP32；m,n 1..512，batch维大小1..64；固定 `hermitian=False`。
- SVD `A=U diag(sigma) V^T`；`cut=max(atol,sigma_max*rtol)`；**仅 sigma>cut 保留**，等号及以下置0；`A+=V diag(1/sigma_keep) U^T`。
- 默认 `rtol=None`：atol=0时 `max(m,n)*epsilon`；显式atol>0时rtol=0；指定rtol按给定值。atol/rtol仅标量。
- 可秩亏；保留sigma至少 `1e-2*sigma_max`，弃值不超过 `0.1*cut`，有谱间隔。不得因此简单等同所有输入可逆，也不能漏阈值。
- 逐元素 atol/rtol均1e-4；禁止完整伪逆库/CPU回退。
- 必审：阈值默认、截断严格性、矩形/秩亏/零矩阵、逆候选残差验收是否数学排除弃值、SVD排序方向、FloatPair/残差修正的真实误差边界。

## 03 SplineConv（4页）

- x `[N,Ci]` FP32/BF16，edge_index `[2,E]` int64（source行在前、target行在后）；pseudo `[E,dim]` 与x同dtype；W `[K,Ci,Co]`；可选root `[Ci,Co]`、bias `[Co]`；输出 `[N,Co]` 与x同dtype。
- `dim∈{2,3}`、degree1/2/3、`kernel_size[d]∈{2,3,4}`，K为各维乘积、S=(degree+1)^dim；开样条 `kernel_size[d]>=degree+1`。伪坐标[0,1]，`v=pseudo*(ks-degree*open)`，使用floor、frac和周期索引。
- degree1基 `[1-v,v]`；degree2 `[0.5v²-v+0.5,-v²+v+0.5,0.5v²]`；degree3 `[(1-v)^3,3v^3-6v²+4,-3v^3+3v²+3v+1,v^3]/6`。
- 每边 `message=sum_s basis*(x_src @ W_index)`；按target聚合；mean按入边数除、0入度除数按1；然后另加 `x_i @ root + bias`。显式自环与root两条路径不能相互代替。
- N16..65536，E64..262144，Ci/Co8..128；`E*S*Ci*Co<=2^31`；只forward，aggr add/mean。
- FP32 atol/rtol1e-4，BF16 1e-2；禁止关闭或近似基函数。
- 页2 BF16表格和degree表已主审视觉核对。公示代码含E=1等扩展路径，不把支持额外shape本身作为问题。
- 必审：聚合方向、mean/root顺序、样条endpoint和闭样条映射、按kernel先聚合的实数重排与浮点影响、非原子owner写覆盖、稀疏清零和输出共享缓存线、基础/Full两接口与题面对应。

## 04 det（2页）

- A `[...,n,n]` FP32，输出 `[...]` FP32每batch标量、**无尾维**；n1..512，batch维大小1..256。
- 可用部分主元LU：`P*A=L*U`，det=`sign(P)*product(diag(U))`；行交换必须计符号；n=0不考核。
- 页1范围表有破损的Markdown/跨单元排版，原文显示 `det(A)∈[1e-3,1e3]`，没有清楚的绝对值符号；**不擅自改写成|det(A)|范围**。页2另明确交换两行符号翻转检验。
- 主要相对误差 `|delta|/|det_golden|<=1e-4`；不能只输出logdet；禁止完整det库/CPU。
- 页1主审已视觉核对；必审：pivot选择、符号、缩放/溢出、乘积高低位误差、小矩阵闭式、Host输出标量形状、近似/无pivot路径的条件。

## 05 SwiGLU（5页；公示源码缺失）

- X `[...,d]`，W1/W2 `[d,d]` 按 `(out_dim,in_dim)` 存储，b1/b2 `[d]`；全部FP16。输出同X shape/dtype。d1..8192，前置乘积M1..1048576（实际组合受内存可行性限制）。
- `z1=X*W1^T+b1`，`z2=X*W2^T+b2`，`y=(z1/(1+exp(-z1)))*z2`；beta=1。
- 混合逐元素误差 `|out-golden|<=1e-3+1e-3*|golden|`，全部元素通过；非对齐维度和32B尾部必须支持。
- 页2/3公式图片、权重方向和误差值已主审视觉核对。
- 题面**未指定线性投影之后必须FP16物化**，也未完整规定中间舍入次序；本项目历史Judge兼容边界与该数学题面应分开记录。不能因题面实数公式推断某种FP32融合一定与Judge相同。
- 九题公示中不含此题，没有获奖源码可以逐行审读；后续只做既往本项目实验的迁移对照，不能虚构三个获奖实现。

## 06 lstsq（3页）

- A `[...,m,n]`、B `[...,m,k]` FP32；输出solution `[...,n,k]` 和residuals `[...,k]`（m>n），m=n时residuals为空张量。
- `min_X ||AX-B||F`；输出每右端平方残差，不开方。A列满秩、m>=n，cond2<=1e3；m1..1024、n1..min(m,512)、k1..64、batch维大小1..32。
- 固定driver=gels；rcond不参与；不考rank亏/欠定；solution和非空residuals逐元素atol/rtol1e-4。
- 必审：QR/消元方法及条件数误差、normal equations是否平方条件数、square专路、空residual不访问、RHS驻留容量、广播/shape检查、残差是否基于原A/B、FP32pair语义。

## 07 FireCandidateDecodeFusion（4页）

- raw `[B,N,6]` FP16/FP32，fields tx/ty/tw/th/obj/fire；priors `[N,5]` 固定FP32（gx,gy,stride,aw,ah），跨batch共享；输出 `[B,N,5]` 同raw dtype。
- B1..8、N1..32768，允许非32/64整倍数；有限实数，stride/aw/ah>0，gx/gy非负。不得重排、过滤、NMS/TopK。
- `s(z)=1/(1+exp(-z))`；xy=`(2*s(txy)-0.5+gxy)*stride`；wh=`(2*s(twh))²*anchor_wh*stride`；score=`s(obj)*s(fire)`。
- 无论输入dtype，全部Sigmoid/平方/乘法/坐标中间必须FP32；FP16只在读后升精度、最终写回降精度。
- FP32 atol/rtol1e-5；FP16与FP32参考结果转half后比较，atol/rtol3e-3。
- 必审：priors模N索引、6列→5列AoS/SoA转置、FP32顺序/融合与分母乘积重排的溢出/舍入、输出tail、非public内建契约、真实核数约束及不可达分支。公示有N32769路径，与10月题面日期差异分开标注。

## 08 eigh（2页）

- A `[...,n,n]` FP32；只按UPLO选择的三角补全Aeff：L用lower及转置，U用upper及转置；另一半可任意，不能取平均或直接信全矩阵对称。
- 输出lambda `[...,n]` 非降序、Q `[...,n,n]` **特征向量按列**；n1..512、batch维大小1..64。
- Aeff特征值互异，最小间隔>=`1e-4*||Aeff||2`。
- lambda逐元素atol/rtol1e-4；Q列可整体变号，不逐元素比较；`||Q^TQ-I||F<=1e-4`，重构相对F范数误差<=1e-4。
- 必审：L/U选边、排序连同列置换、正交性累积、迭代终止、特征值尺度、tridiagonal/QR/Jacobi退化与误差；不可用题面不保证的正定性或固定谱结构。

## 09 solve（2页）

- A `[...,n,n]` FP32、B矩阵 `[...,n,k]` 或向量 `[...,n]`；无batch B的(n,)或(n,k)在A batch下广播；输出同广播后的B形状，向量不能平白新增k维。
- 固定left=True，只AX=B；A可逆、cond2<=1e3；n1..512、k1..256、batch维大小1..64。
- X逐元素atol/rtol1e-4，另需 `||AX-B||F<=1e-4*||B||F`；禁止完整solve库/CPU。
- 必审：B广播步距、向量与矩阵判定、pivot和置换、多个RHS复用、残差原始输入、条件矩阵算法切换、低精度候选与修正收敛条件。

## 10 TreeLineMinDistance（4页）

- P `[B,N,3]`、segments `[B,M,2,3]` 相同dtype FP16/FP32、连续ND，B必须相同无广播；距离 `[B,N]` 同dtype，索引 `[B,N]` int32。
- B1..8、N1..16384、M1..1024，皆正；有限坐标建议[-1000,1000]；N/M可非对齐。
- D=C-A，R=P-A，L=D·D；L>0时t=(R·D)/L，clamp到[0,1]后求P-(A+tD)距离；**L=0严格以实际量化输入判断，不做该除法**，转点距离。
- 返回实际欧氏距离与最小索引。距离完全相同时取最小索引；距离0仍按最小索引。普通测试避免最小和次小过近，同距测试严格对称。
- FP16所有中间（差值/点积/除法/clamp/min/argmin/sqrt）必须FP32或等效高精度，最后距离才转half；索引完全一致。FP32 atol/rtol1e-4；half1e-3。
- 允许先比较平方再sqrt；无效tail不能产生虚假最小值；禁止越界；不建议完整BNM距离矩阵。
- 必审：退化L=0屏蔽在除法之前、strict ties稳定索引、axis-aligned/端点、平方单调与浮点溢出、argmin索引编码、FP16量化后的零段、packed读地址界限、多核输出缓存线共享。

## 与历史经验的使用边界

既往SwiGLU实验说明：静态公式/布局通过、CANN编译通过、Judge Pass、单点时延改变、成对同环境的可归因收益是五种不同证据。本轮只做源码/题面审读；没有重新取得这些公示代码对应的原二进制、编译选项和全量设备结果。历史“无收益”结果也不能外推否定别的算子或硬件路径。
