# Ascend C 算子竞赛档案

这里保存公开的昇腾算子竞赛材料，以及围绕这些材料完成的源码审读和方法整理。目标是让以后遇到相近题目时，能从题目契约出发找到可参考的实现，而不是只留下一个代码压缩包或一串技巧名称。

仓库分三层：原始材料用于追溯来源；逐文件审读账本记录代码实际写了什么；设计复盘与方法索引归纳实现思路及其迁移条件。获奖提交的官方 Judge 结果是本届正确性与性能表现的依据；逐行审读则帮助读者理解代码并找到可借鉴之处，不要求为原提交另做本地复测。

## 从哪里开始

如果只想快速了解某个题，按这个顺序阅读：

1. 打开[题目索引与审计进度](archives/2026-electric-power/deep_audit/README.md)，找到对应题号。
2. 先读该题的“设计复盘”，了解题目实质、各份提交的路线、值得研究的设计和迁移限制。
3. 需要核对源码依据时，再读对应的“逐行证据账本”和覆盖记录。
4. 移植之前回到原题PDF及[题面契约汇总](archives/2026-electric-power/deep_audit/SPEC_CONTRACTS.md)，确认 shape、dtype、精度和输出边界没有变化。

全库导航：

- [十题设计分类与跨题方法](archives/2026-electric-power/deep_audit/DESIGN_CLASSIFICATION.md)：按题目专属、同类问题可迁移、通用编排分类，并注明适用条件和反例。
- [可复用方法目录](methods/README.md)：按优化方法检索，不需要先知道原题编号。
- [2026电力专场材料目录](archives/2026-electric-power/README.md)：题目PDF、公开提交、原始压缩包和来源信息。
- [源码文件目录](archives/2026-electric-power/SOURCE_CATALOG.md)：按题目和提交查找代码文件。
- [第三方材料说明](THIRD_PARTY_NOTICES.md)：来源、许可证和再分发注意事项。

## 2026 电力行业算子专场

本部分根据[公开公示页面](https://globalcomputing.feishu.cn/wiki/OpFlwqe4Ji7jLvkRWRDcIrXRn3b)建立，网页快照采集于2026-11-08。页面记录的公示期为2026-11-08至2026-11-10，因此下表中的提交顺序沿用快照里的“决策顺序”，不把它表述为公示期结束后重新核定的最终名次。

本届 CANNJudge 评测目标按项目记录为 Ascend 910B4、20 AIC。项目使用过的远端 910B2 是开发诊断环境，不替代官方 Judge 的结果或耗时。对本届代码来说，官方评测环境是已知条件；R 只在把思路迁移到别的题目或比赛时提示需要重新满足的前提。

| 题号 | 题目 | 原题PDF | 公开提交源码 | 设计复盘 | 逐行审读账本 |
|---|---|---|---|---|---|
| 01 | SafetyBeltViolationGeometry | [PDF](archives/2026-electric-power/problems/01_SafetyBeltViolationGeometry.pdf) | [3份](archives/2026-electric-power/extracted/SafetyBeltViolationGeometry/) | [复盘](archives/2026-electric-power/deep_audit/problems/01_SafetyBeltViolationGeometry_REVIEW.md) | [账本](archives/2026-electric-power/deep_audit/problems/01_SafetyBeltViolationGeometry.md) |
| 02 | pinv | [PDF](archives/2026-electric-power/problems/02_pinv.pdf) | [3份](archives/2026-electric-power/extracted/pinv/) | [复盘](archives/2026-electric-power/deep_audit/problems/02_pinv_REVIEW.md) | [账本](archives/2026-electric-power/deep_audit/problems/02_pinv.md) |
| 03 | SplineConv | [PDF](archives/2026-electric-power/problems/03_SplineConv.pdf) | [3份](archives/2026-electric-power/extracted/SplineConv/) | [复盘](archives/2026-electric-power/deep_audit/problems/03_SplineConv_REVIEW.md) | [账本](archives/2026-electric-power/deep_audit/problems/03_SplineConv.md) |
| 04 | det | [PDF](archives/2026-electric-power/problems/04_det.pdf) | [3份](archives/2026-electric-power/extracted/det/) | [复盘](archives/2026-electric-power/deep_audit/problems/04_det_REVIEW.md) | [账本](archives/2026-electric-power/deep_audit/problems/04_det.md) |
| 05 | SwiGLU | [PDF](archives/2026-electric-power/problems/05_SwiGLU.pdf) | 本次快照未包含获奖源码 | [边界与方法索引](archives/2026-electric-power/deep_audit/problems/05_SwiGLU_REVIEW.md) | [题面记录](archives/2026-electric-power/deep_audit/problems/05_SwiGLU.md) |
| 06 | lstsq | [PDF](archives/2026-electric-power/problems/06_lstsq.pdf) | [3份](archives/2026-electric-power/extracted/lstsq/) | [复盘](archives/2026-electric-power/deep_audit/problems/06_lstsq_REVIEW.md) | [账本](archives/2026-electric-power/deep_audit/problems/06_lstsq.md) |
| 07 | FireCandidateDecodeFusion | [PDF](archives/2026-electric-power/problems/07_FireCandidateDecodeFusion.pdf) | [3份](archives/2026-electric-power/extracted/FireCandidateDecodeFusion/) | [复盘](archives/2026-electric-power/deep_audit/problems/07_FireCandidateDecodeFusion_REVIEW.md) | [账本](archives/2026-electric-power/deep_audit/problems/07_FireCandidateDecodeFusion.md) |
| 08 | eigh | [PDF](archives/2026-electric-power/problems/08_eigh.pdf) | [3份](archives/2026-electric-power/extracted/eigh/) | [复盘](archives/2026-electric-power/deep_audit/problems/08_eigh_REVIEW.md) | [账本](archives/2026-electric-power/deep_audit/problems/08_eigh.md) |
| 09 | solve | [PDF](archives/2026-electric-power/problems/09_solve.pdf) | [3份](archives/2026-electric-power/extracted/solve/) | [复盘](archives/2026-electric-power/deep_audit/problems/09_solve_REVIEW.md) | [账本](archives/2026-electric-power/deep_audit/problems/09_solve.md) |
| 10 | TreeLineMinDistance | [PDF](archives/2026-electric-power/problems/10_TreeLineMinDistance.pdf) | [3份](archives/2026-electric-power/extracted/TreeLineMinDistance/) | [复盘](archives/2026-electric-power/deep_audit/problems/10_TreeLineMinDistance_REVIEW.md) | [账本](archives/2026-electric-power/deep_audit/problems/10_TreeLineMinDistance.md) |

原始PDF由仓库维护者提供；题面PDF的SHA-256与审计时使用的材料一致。第五题的PDF已经归档，但本批公开公示材料中没有该题获奖提交源码，所以只记录题面和资料边界，不以本项目自己的SwiGLU实现替代获奖代码审计。

## 审计做到了什么

九道有公开提交的题目共27份提交、271个源码文件、139,460行源码，完成了首轮连续源码审读。每题的逐文件范围和哈希记录在 `archives/2026-electric-power/deep_audit/coverage/`；全局统计见[覆盖清单](archives/2026-electric-power/deep_audit/COVERAGE_SUMMARY.json)。重复的相同源码按哈希核对后复用审读记录，不重复计作新的算法实现。

设计复盘是在已有审读账本基础上的二次整理，重点回答：每份实现的大致算法路线是什么、设计的亮点在哪里、哪些结构可能适用于同类问题、迁移时要保留什么前提。它不是又一轮逐行重读。获奖提交的通过情况、分数和公开耗时按官方结果记录；本地没有重跑原始提交。

报告用以下标记区分证据：

- `P`：只对该题公式或契约成立的设计。
- `F`：可能迁移到一类相似问题的算法或数据布局。
- `G`：跨算子的调度、缓冲、owner或异步编排原则。
- `S`：源码审读确认的实现事实；`M`：解释设计所依据的数学关系及条件。
- `R`：迁移到其他题目或比赛时需要满足的复用前提；`E`：本届官方评测结果，包括 Pass、分数和公开耗时。

官方结果记录的是整份获奖提交在本届 Judge 上的表现，不用于推断某个单项技巧的独立贡献。只有我们要判断后续某项改动带来多少收益时，才需要对照实验；不要求为获奖源码逐项消融或本地复测。

## 仓库怎么维护

- `archives/2026-electric-power/raw/` 保存收到的原始压缩包；`extracted/` 保存解压源码；`problems/` 保存题面PDF。审计和本地实验不应改写这些原始材料。
- `deep_audit/` 保存题面契约、设计复盘、逐行审读账本和覆盖记录。新增审计结论应能回到具体源码文件/行段，并注明证据级别。
- `methods/` 汇总跨题可迁移的方法。进入方法索引不代表适用于所有题；每条方法都应保留必要前提和失效边界。
- 自己的移植与实验放在独立目录，记录来源提交、改动内容、CANN版本、芯片、shape、正确性结果和重复测量。不要直接在 `extracted/` 里修改第三方代码。

公开网页可访问不等于允许任意再分发或改编。使用代码或题面前先核对[第三方材料说明](THIRD_PARTY_NOTICES.md)、原始许可和下一场竞赛规则；本仓库没有为所有第三方材料统一添加MIT等许可证。

## 本地校验

需要 Python 3.11 或更新版本；仓库校验脚本不依赖第三方Python包：

```bash
python -m unittest discover -s tests -v
python archives/2026-electric-power/deep_audit/check_coverage.py
python archives/2026-electric-power/deep_audit/check_report_links.py
python tools/archive_checks.py
```

这些脚本检查文档链接、归档哈希、ZIP CRC、PDF、审读范围和部分敏感凭证模式；它们不会编译或运行归档中的算子代码。凭证扫描只能辅助发现，不保证覆盖所有秘密，新增材料仍需人工检查。

仓库提供了只读自动校验模板 `tools/archive-checks.workflow.yml`，当前尚未启用GitHub Actions；启用需要单独确认工作流权限及action版本。
