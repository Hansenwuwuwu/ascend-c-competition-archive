# 十题逐行审读：进度与方法索引

日期2026-11-09，**27/27份提交的源码首轮审阅完成**。全部在codexpro独立归档内工作；10题PDF已阅读全文并核对关键公式/表格。9题有前三名源码、共27份提交、271文件、139460物理行；第5题SwiGLU没有这批前三名源码，不制造缺失内容。后续数学误差、SDK/API、同步与性能验证仍有R级待办，故不是全方位认证。

## 先看哪里

- [设计适用范围分类](DESIGN_CLASSIFICATION.md)：按题区分专属数学、同类问题可迁移设计、跨算子编排原则；另列硬件/SDK绑定、正确性前提、反例/失效域和证据状态。
- [十题契约](SPEC_CONTRACTS.md)：原题shape/dtype、数学、精度与输出边界，不能被源码注释中的旧ABI反向替代。
- [完整进度计划](../DEEP_AUDIT_PLAN.md)：审读方法和交付边界。
- [逐文件覆盖与未读空洞](COVERAGE_SUMMARY.json)：真实hash、连续阅读范围；不是函数名/关键词命中统计。

## 当前真实覆盖

校验结果：**139460 / 139460行（100%）**，271/271文件、27/27份提交完成全文首轮覆盖。计数含明确校验的同hash源码复用；不能把重复公共头当作独立算法发现。该统计只表示连续源码已读且范围/hash账本自洽，不代表数学或运行验证已经通过。

| 题 | 已审读物理行/总行 | 报告与状态 |
|---|---:|---|
|01 安全带几何|12172/12172|[设计复盘](problems/01_SafetyBeltViolationGeometry_REVIEW.md)；[源码证据账本](problems/01_SafetyBeltViolationGeometry.md)，三份提交首轮源码覆盖完成|
|02 pinv|19921/19921|[设计复盘](problems/02_pinv_REVIEW.md)；[源码证据账本](problems/02_pinv.md)，三份提交源码首轮完成；数值/SDK/同步/性能证明仍单列|
|03 SplineConv|10607/10607|[设计复盘](problems/03_SplineConv_REVIEW.md)；[源码账本](problems/03_SplineConv.md)，API/数值待证单列|
|04 det|9262/9262|[设计复盘](problems/04_det_REVIEW.md)；[源码账本](problems/04_det.md)，实际路由/未启用路径已区分|
|05 SwiGLU|无公示源码|[边界与方法索引](problems/05_SwiGLU_REVIEW.md)；[题面边界](problems/05_SwiGLU.md)，不冒充获奖代码审计|
|06 lstsq|12371/12371|[设计复盘](problems/06_lstsq_REVIEW.md)；[源码账本](problems/06_lstsq.md)，三份提交首轮源码覆盖完成|
|07 Fire解码|18301/18301|[设计复盘](problems/07_FireCandidateDecodeFusion_REVIEW.md)；[源码账本](problems/07_FireCandidateDecodeFusion.md)，数值/API/同步/性能证明仍单列|
|08 eigh|25114/25114|[设计复盘](problems/08_eigh_REVIEW.md)；[源码账本](problems/08_eigh.md)，solver正确性与数值证书待证|
|09 solve|8984/8984|[设计复盘](problems/09_solve_REVIEW.md)；[源码账本](problems/09_solve.md)，三份提交首轮源码覆盖完成|
|10 树线距离|22728/22728|[设计复盘](problems/10_TreeLineMinDistance_REVIEW.md)；[源码账本](problems/10_TreeLineMinDistance.md)，剪枝证书边界待证|

全文首轮覆盖指每物理行已读或有明确等同文本复用依据，**不等于**数学所有误差界、API协议、同步、资源和性能归因均已通过。每题保留R级缺口。所有原源码SHA256和行数已重新校验，未改原ZIP/解压源码；没有执行、编译或安装下载代码。

## 完成范围与下一阶段

本轮最后补齐的长缺口为pinv顺序1窗口DSL、Fire顺序1主kernel、eigh三个版本的长kernel，以及TreeLineMinDistance顺序3的Host/tiling/kernel。逐文件SHA-256、连续行段、未读空洞均收录在各题 `coverage/*.json`；最终全局检查为 **139460/139460行、271/271文件、27/27份提交**，report link检查122个本地链接无损坏。

首轮源码审阅之后，不再以继续读源码作为下一目标；后续应按各题报告中的R项制定验证矩阵，优先级是：

1. **数学与数值域**：完整题面输入范围、退化/极端/病态输入；证明近似、剪枝、证书、收敛和误差阈值。
2. **API与同步**：锁定CANN版本核对私有/低级接口、事件配对、跨核发布与回收、异常/空工作退出。
3. **内存与编译产物**：校验UB/GM边界、别名活跃期、实际tiling-key可达性、资源和生成指令。
4. **运行与性能**：目标设备全case正确性；重复同环境测量，控制波动，只把paired可归因收益记为E。

始终优先可推导的结论；涉及目标API/生成器时用对应版本官方材料，无法证明即保留待核。不凭Pass/高分给某个写法归因，不因源码有风险直接评判作者或比赛合规性。

## 覆盖校验命令

自写工具仅读取文本/JSON、检查SHA与区间，不运行归档程序：

```powershell
python archives/2026-electric-power/deep_audit/check_coverage.py
```

`record_review_ranges.py`与`update_spline_coverage.py`仅将明确实读范围写入账本；新增覆盖前必须先实际阅读全文并补语义报告，不能改数字冒充阅读。校验器只能证明记录自洽和原档未变，不能证明审读结论正确。
