# 十题逐行审读：进度与方法索引

日期2026-11-09，**27/27份提交的源码首轮审阅完成**。全部在codexpro独立归档内工作；10题PDF已阅读全文并核对关键公式/表格。9题有前三名源码、共27份提交、271文件、139460物理行；第5题SwiGLU没有这批前三名源码，不制造缺失内容。获奖提交的官方评测结果作为本届表现记录；审读报告服务于理解设计和后续复用。

## 先看哪里

- [设计适用范围分类](DESIGN_CLASSIFICATION.md)：按题区分专属数学、同类问题可迁移设计、跨算子编排原则；另列硬件/SDK绑定、正确性前提、反例/失效域和证据状态。
- [十题契约](SPEC_CONTRACTS.md)：原题shape/dtype、数学、精度与输出边界，不能被源码注释中的旧ABI反向替代。
- [完整进度计划](../DEEP_AUDIT_PLAN.md)：审读方法和交付边界。
- [逐文件覆盖与未读空洞](COVERAGE_SUMMARY.json)：真实hash、连续阅读范围；不是函数名/关键词命中统计。

## 当前真实覆盖

校验结果：**139460 / 139460行（100%）**，271/271文件、27/27份提交完成全文首轮覆盖。计数含明确校验的同hash源码复用；不能把重复公共头当作独立算法发现。该统计用于说明源码审读范围；获奖提交的比赛结果以官方 Judge 记录为准。

| 题 | 已审读物理行/总行 | 报告与状态 |
|---|---:|---|
|01 安全带几何|12172/12172|[设计复盘](problems/01_SafetyBeltViolationGeometry_REVIEW.md)；[源码证据账本](problems/01_SafetyBeltViolationGeometry.md)，三份提交首轮源码覆盖完成|
|02 pinv|19921/19921|[设计复盘](problems/02_pinv_REVIEW.md)；[源码证据账本](problems/02_pinv.md)，三份提交源码首轮完成|
|03 SplineConv|10607/10607|[设计复盘](problems/03_SplineConv_REVIEW.md)；[源码账本](problems/03_SplineConv.md)，三份提交源码首轮完成|
|04 det|9262/9262|[设计复盘](problems/04_det_REVIEW.md)；[源码账本](problems/04_det.md)，实际路由/未启用路径已区分|
|05 SwiGLU|公示快照无；另有3份用户提供补充源码|[前三名设计复盘](problems/05_SwiGLU_REVIEW.md)；[完整静态源码审计](problems/05_SwiGLU_TOP3_LOCAL_SUPPLEMENT_20261109.md)；[题面边界](problems/05_SwiGLU.md)。补充材料单独计数，不并入公示快照的27份提交覆盖统计|
|06 lstsq|12371/12371|[设计复盘](problems/06_lstsq_REVIEW.md)；[源码账本](problems/06_lstsq.md)，三份提交首轮源码覆盖完成|
|07 Fire解码|18301/18301|[设计复盘](problems/07_FireCandidateDecodeFusion_REVIEW.md)；[源码账本](problems/07_FireCandidateDecodeFusion.md)，三份提交源码首轮完成|
|08 eigh|25114/25114|[设计复盘](problems/08_eigh_REVIEW.md)；[源码账本](problems/08_eigh.md)，三份提交源码首轮完成|
|09 solve|8984/8984|[设计复盘](problems/09_solve_REVIEW.md)；[源码账本](problems/09_solve.md)，三份提交首轮源码覆盖完成|
|10 树线距离|22728/22728|27/27|[设计复盘](problems/10_TreeLineMinDistance_REVIEW.md)；[源码账本](problems/10_TreeLineMinDistance.md)，三份提交源码首轮完成|

全文首轮覆盖指每物理行已读或有明确等同文本复用依据。官方 Judge 已给出本届获奖提交的通过情况和成绩；此处记录源码实现，不要求在本地重新跑原提交。R 只表示把某个思路用于其他题目或比赛时要保留的复用前提。所有原源码 SHA256 和行数已重新校验，未改原 ZIP/解压源码。

## 完成范围与下一阶段

本轮最后补齐的长缺口为pinv顺序1窗口DSL、Fire顺序1主kernel、eigh三个版本的长kernel，以及TreeLineMinDistance顺序3的Host/tiling/kernel。逐文件SHA-256、连续行段、未读空洞均收录在各题 `coverage/*.json`；最终全局检查为 **139460/139460行、271/271文件、27/27份提交**，report link检查122个本地链接无损坏。

首轮源码审阅之后，不再以继续读源码作为下一目标；后续借鉴时按各题报告中的R项检查目标题目的适用条件。若对获奖思路作了新修改，再针对修改后的候选安排验证：

1. **匹配题目条件**：对照新题的输入范围、dtype、精度和布局，确认想复用的方法所需条件是否成立。
2. **映射实现路径**：按目标环境接入对应的 Host/key/kernel 路径；本届固定的 CANNJudge 设备与测试契约直接以官方配置为准。
3. **验证自己的改动**：新候选提交 Judge 检查通过和耗时即可开始筛选；只有要量化某个局部改动的独立贡献时，再做成对消融。

复用时先核对目标题目的数学条件、shape 和数据布局；不要把原题特有条件误当通用方法。官方总成绩说明提交整体表现，不能单独归因到某一条代码技巧。

## 覆盖校验命令

自写工具仅读取文本/JSON、检查SHA与区间，不运行归档程序：

```powershell
python archives/2026-electric-power/deep_audit/check_coverage.py
```

`record_review_ranges.py`与`update_spline_coverage.py`仅将明确实读范围写入账本；新增覆盖前必须先实际阅读全文并补语义报告，不能改数字冒充阅读。校验器只能证明记录自洽和原档未变，不能证明审读结论正确。
