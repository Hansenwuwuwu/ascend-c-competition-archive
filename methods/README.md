# 可复用方法检索

完整的逐题条件与限制见[P/F/G分类总报告](../archives/2026-electric-power/deep_audit/DESIGN_CLASSIFICATION.md)。本页是检索入口，不是已实测提速的模板库。

|先遇到什么问题|优先阅读的机制与来源|迁移前必须确认|
|---|---|---|
|中间转置/Gather很多|SplineConv消费者布局、Fire规则转置与索引|消费者访问、dtype/字节单位、tail和SDK|
|多任务重复读同样数据|Fire prior、Tree segment、solve多RHS因子复用|真实不变维、UB容量与负载尾部|
|每个stage都在等整个stage|det关键十字优先、Spline面板ready、solve关键panel发布|data-ready、reader-done、slot回收和退出协议|
|全程高精太贵|pinv Newton、solve/lstsq精化、eigh证书、Tree歧义修复|误差证书是否充分、失败回退是否完整|
|Scalar地址/索引初始化过多|Fire与Tree seed、广播/倍增生成规则索引|索引上界、字节单位、所有消费lane|
|tiny通用逻辑开销大|静态专核、冷热分离、小输出独占|合法shape覆盖、真实常量传播、目标机对照|

新增方法卡建议固定记录：`来源提交与行段 / P-F-G范围 / 原计算与新计算 / 数学前提 / shape与资源 / SDK / 数值风险 / 正确性测试 / 同环境paired实测 / 迁移建议`。未验证时明确写R，不因出现在高名次源码里就写成E。
