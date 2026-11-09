# Ascend C Competition Archive

昇腾算子竞赛公示代码、原始题目与源码审计知识库。用于学习实现、比较设计及为后续比赛建立可追溯的方法索引。

## 先看这里

- [完整审计与 P/F/G 设计分类](archives/2026-electric-power/deep_audit/DESIGN_CLASSIFICATION.md)
- [可复用方法入口](methods/README.md)
- [题目契约、shape/dtype与数学边界](archives/2026-electric-power/deep_audit/SPEC_CONTRACTS.md)
- [27份提交、原作者/队伍与原始ZIP导航](archives/2026-electric-power/README.md)
- [逐文件源码目录](archives/2026-electric-power/SOURCE_CATALOG.md)
- [第三方材料、来源与授权边界](THIRD_PARTY_NOTICES.md)

## 2026 电力行业算子专场

来源为[公开公示页面](https://globalcomputing.feishu.cn/wiki/OpFlwqe4Ji7jLvkRWRDcIrXRn3b)，采集快照日期为2026-11-08。页面记录的公示期为2026-11-08至2026-11-10，因此本仓库不把此快照表述为公示期结束后经重新确认的最终获奖结论；代码目录的rank表示页面的“决策顺序”。

|题号 / 算子|原始题目PDF|公示源码|逐题审计|
|---|---|---|---|
|01 SafetyBeltViolationGeometry|[PDF](archives/2026-electric-power/problems/01_SafetyBeltViolationGeometry.pdf)|[3份](archives/2026-electric-power/extracted/SafetyBeltViolationGeometry/)|[报告](archives/2026-electric-power/deep_audit/problems/01_SafetyBeltViolationGeometry.md)|
|02 pinv|[PDF](archives/2026-electric-power/problems/02_pinv.pdf)|[3份](archives/2026-electric-power/extracted/pinv/)|[报告](archives/2026-electric-power/deep_audit/problems/02_pinv.md)|
|03 SplineConv|[PDF](archives/2026-electric-power/problems/03_SplineConv.pdf)|[3份](archives/2026-electric-power/extracted/SplineConv/)|[报告](archives/2026-electric-power/deep_audit/problems/03_SplineConv.md)|
|04 det|[PDF](archives/2026-electric-power/problems/04_det.pdf)|[3份](archives/2026-electric-power/extracted/det/)|[报告](archives/2026-electric-power/deep_audit/problems/04_det.md)|
|05 SwiGLU|[PDF](archives/2026-electric-power/problems/05_SwiGLU.pdf)|待后续公示补充|[边界记录](archives/2026-electric-power/deep_audit/problems/05_SwiGLU.md)|
|06 lstsq|[PDF](archives/2026-electric-power/problems/06_lstsq.pdf)|[3份](archives/2026-electric-power/extracted/lstsq/)|[报告](archives/2026-electric-power/deep_audit/problems/06_lstsq.md)|
|07 FireCandidateDecodeFusion|[PDF](archives/2026-electric-power/problems/07_FireCandidateDecodeFusion.pdf)|[3份](archives/2026-electric-power/extracted/FireCandidateDecodeFusion/)|[报告](archives/2026-electric-power/deep_audit/problems/07_FireCandidateDecodeFusion.md)|
|08 eigh|[PDF](archives/2026-electric-power/problems/08_eigh.pdf)|[3份](archives/2026-electric-power/extracted/eigh/)|[报告](archives/2026-electric-power/deep_audit/problems/08_eigh.md)|
|09 solve|[PDF](archives/2026-electric-power/problems/09_solve.pdf)|[3份](archives/2026-electric-power/extracted/solve/)|[报告](archives/2026-electric-power/deep_audit/problems/09_solve.md)|
|10 TreeLineMinDistance|[PDF](archives/2026-electric-power/problems/10_TreeLineMinDistance.pdf)|[3份](archives/2026-electric-power/extracted/TreeLineMinDistance/)|[报告](archives/2026-electric-power/deep_audit/problems/10_TreeLineMinDistance.md)|

原始PDF由仓库维护者提供；SHA-256与审计时的题面清单一致。第5题PDF已归档，而获奖源码尚未包含在本批公示快照，后续取得公开材料后补充，不虚构缺失内容。

## 审计状态与证据口径

- 27份提交、271个源码文件、139,460物理行均有首轮连续源码审阅记录；计数包括明确验证同哈希文本的复用。
- 原始ZIP和解压源码按哈希保留；逐文件范围账本位于 `deep_audit/coverage/`，全局结果见[覆盖清单](archives/2026-electric-power/deep_audit/COVERAGE_SUMMARY.json)。
- **全文源码审阅不等于数学误差、SDK/API、同步、资源、运行正确性或性能验证。** 本次归档没有构建或运行第三方kernel。
- `P`：题目专属公式/契约；`F`：同类问题的算法或布局；`G`：跨算子的编排原则。
- `S`：源码观察；`M`：注明前提的数学推导；`R`：待核事项；`E`：同环境、可归因的实测收益。高名次或Pass不能直接替代E。

## 仓库分层与使用方式

`archives/2026-electric-power/raw/` 保存原始ZIP；`extracted/` 保存不可直接改写的第三方源码；`problems/` 保存原始PDF；`deep_audit/` 保存我们的分析与验证边界。以后自己的移植/实验应放入独立目录，记录上游提交、具体修改、CANN/硬件、shape、正确性和重复性能对照，不在原始归档上直接改。

公开材料不自动带有再分发/改编授权。本仓库没有给第三方代码或题目PDF统一套用MIT等许可证；复用前核对[第三方说明](THIRD_PARTY_NOTICES.md)、原授权及下一场比赛的外部代码规则，并保留引用。

## 本地静态校验

使用Python 3.11或更新版本，无第三方Python依赖：

```bash
python -m unittest discover -s tests -v
python archives/2026-electric-power/deep_audit/check_coverage.py
python archives/2026-electric-power/deep_audit/check_report_links.py
python tools/archive_checks.py
```

这些步骤检查哈希、ZIP CRC、PDF、覆盖记录、文档相对链接与若干敏感凭证模式，不编译或运行归档代码。敏感信息模式扫描不保证检出所有秘密；新增材料仍需人工核对。

[只读自动校验模板](tools/archive-checks.workflow.yml)已提供，但本次未启用GitHub Actions：当前浏览器授权没有workflow权限，初次上传不额外扩展账号权限。以后可在GitHub网页将该模板放到 `.github/workflows/archive-checks.yml` 启用；启用前检查Actions权限和所锁定的官方action版本。
