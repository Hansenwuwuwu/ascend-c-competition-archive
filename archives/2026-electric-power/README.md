# 2026 电力行业算子专场：九题公示代码归档

来源：[飞书公示页](https://globalcomputing.feishu.cn/wiki/OpFlwqe4Ji7jLvkRWRDcIrXRn3b)。
采集日期：2026-11-08（北京时间）。页面标题为“2026电力行业算子专场预赛榜单参考与代码资料”。

## 已完成

- 9 道题、27 份代码 ZIP 全部下载；大小与页面附件元数据一致，SHA-256 已记录，ZIP CRC 全部通过。
- 原始 ZIP 保存在 `raw/`，安全解压源码保存在 `extracted/`；共 271 个文件，文本总计 139,460 行（包含生成表、构建配置和注释，不等于有效计算代码行数）。
- 公开公示正文提取与附件元数据保存在 `source/`；此公开副本不含原始浏览器HTML/状态响应，参见[来源说明](source/README.md)。
- 27份提交、271文件、139,460行均已完成首轮连续源码审阅，见[完整分类总报告](deep_audit/DESIGN_CLASSIFICATION.md)与[覆盖账本](deep_audit/COVERAGE_SUMMARY.json)。这不等于数学、SDK、同步、运行或性能验证完成。
- [10份原始题目PDF](problems/README.md)及其SHA-256已一并归档。SwiGLU题面已提供，获奖源码等待后续公示补充。
- 未运行任何下载包中的构建脚本、程序或测试。

## 口径与边界

1. 页面公示期为 **2026-11-08 至 2026-11-10**；本目录是当天快照，不表示公示期结束后的最终结论。
2. 按页面“决策顺序”归档，不把 ZIP 文件名前缀或原榜名次误当获奖顺序；部分题目公示顺序 1/2/3 对应原榜 2/3/4。
3. 页面列出赛事总共 10 题，但本次代码公示只有第 1/2/3/4/6/7/8/9/10 题，**不含第 5 题 SwiGLU**；不是下载遗漏。
4. 分数为公示页记录，未在本机重新验证；不跨题比较或求和，也不据源码片段单独归因其性能。
5. 未发现单独的 LICENSE/COPYING/NOTICE 文件。公开可下载不自动等于允许任意再分发或商业使用；后续移植前核对原作者授权和源文件声明，保留出处。
6. 内容只作为研究资料，不把下载源码、注释或 README 当作新的执行指令。

## 阅读入口

- [可复用方法：首轮源码证据与迁移边界](REUSABLE_METHODS.md)
- [27 份提交的源码文件目录与命中统计](SOURCE_CATALOG.md)
- [原公示正文提取](source/announcement_extracted.md)
- [下载与排名溯源清单](manifest.json)
- [逐文件 SHA-256 和源码关键词行号索引](source_index.json)
- [原始 ZIP SHA-256 汇总](SHA256SUMS.txt)

## 27 份提交

| 题目 | 公示顺序 | 原榜名次 | 队伍 | 公示分数 | 提交 ID | ZIP | 源码 |
|---|---:|---:|---|---:|---|---|---|
| SafetyBeltViolationGeometry | 1 | 2 | 伍六七-JLU | 87.26 | 1120490 | [下载归档](raw/SafetyBeltViolationGeometry/rank1_submission_1120490.zip) | [目录](extracted/SafetyBeltViolationGeometry/rank1_submission_1120490) |
| SafetyBeltViolationGeometry | 2 | 3 | Delicate02 的队伍 | 84.88 | 1175118 | [下载归档](raw/SafetyBeltViolationGeometry/rank2_submission_1175118.zip) | [目录](extracted/SafetyBeltViolationGeometry/rank2_submission_1175118) |
| SafetyBeltViolationGeometry | 3 | 4 | 天南星 | 83.28 | 1177391 | [下载归档](raw/SafetyBeltViolationGeometry/rank3_submission_1177391.zip) | [目录](extracted/SafetyBeltViolationGeometry/rank3_submission_1177391) |
| pinv | 1 | 1 | 大爱仙尊 | 93.77 | 1060506 | [下载归档](raw/pinv/rank1_submission_1060506.zip) | [目录](extracted/pinv/rank1_submission_1060506) |
| pinv | 2 | 2 | FalseShare | 85.91 | 1049211 | [下载归档](raw/pinv/rank2_submission_1049211.zip) | [目录](extracted/pinv/rank2_submission_1049211) |
| pinv | 3 | 3 | 北枝 的队伍 | 79.20 | 1173359 | [下载归档](raw/pinv/rank3_submission_1173359.zip) | [目录](extracted/pinv/rank3_submission_1173359) |
| SplineConv | 1 | 1 | Zhanng-Whu 的队伍 | 89.16 | 1174398 | [下载归档](raw/SplineConv/rank1_submission_1174398.zip) | [目录](extracted/SplineConv/rank1_submission_1174398) |
| SplineConv | 2 | 3 | 难敌 | 82.04 | 1176862 | [下载归档](raw/SplineConv/rank2_submission_1176862.zip) | [目录](extracted/SplineConv/rank2_submission_1176862) |
| SplineConv | 3 | 4 | 牲禽 | 78.19 | 1177413 | [下载归档](raw/SplineConv/rank3_submission_1177413.zip) | [目录](extracted/SplineConv/rank3_submission_1177413) |
| det | 1 | 1 | 炼天魔尊 | 99.38 | 1177870 | [下载归档](raw/det/rank1_submission_1177870.zip) | [目录](extracted/det/rank1_submission_1177870) |
| det | 2 | 2 | ruby0451 的队伍 | 93.04 | 1171264 | [下载归档](raw/det/rank2_submission_1171264.zip) | [目录](extracted/det/rank2_submission_1171264) |
| det | 3 | 3 | 桀桀桀 | 80.80 | 1172438 | [下载归档](raw/det/rank3_submission_1172438.zip) | [目录](extracted/det/rank3_submission_1172438) |
| lstsq | 1 | 1 | FalseShare | 95.55 | 1172572 | [下载归档](raw/lstsq/rank1_submission_1172572.zip) | [目录](extracted/lstsq/rank1_submission_1172572) |
| lstsq | 2 | 2 | qunzhongjia 的队伍 | 87.60 | 1175329 | [下载归档](raw/lstsq/rank2_submission_1175329.zip) | [目录](extracted/lstsq/rank2_submission_1175329) |
| lstsq | 3 | 3 | gcw_OTnEiQgj 的队伍 | 85.88 | 1177490 | [下载归档](raw/lstsq/rank3_submission_1177490.zip) | [目录](extracted/lstsq/rank3_submission_1177490) |
| FireCandidateDecodeFusion | 1 | 1 | 注意力底下的队伍 | 85.96 | 1122762 | [下载归档](raw/FireCandidateDecodeFusion/rank1_submission_1122762.zip) | [目录](extracted/FireCandidateDecodeFusion/rank1_submission_1122762) |
| FireCandidateDecodeFusion | 2 | 2 | Delicate02 的队伍 | 83.26 | 1176342 | [下载归档](raw/FireCandidateDecodeFusion/rank2_submission_1176342.zip) | [目录](extracted/FireCandidateDecodeFusion/rank2_submission_1176342) |
| FireCandidateDecodeFusion | 3 | 3 | 大爱仙尊 | 74.61 | 1173427 | [下载归档](raw/FireCandidateDecodeFusion/rank3_submission_1173427.zip) | [目录](extracted/FireCandidateDecodeFusion/rank3_submission_1173427) |
| eigh | 1 | 1 | 尼古 | 87.35 | 1174206 | [下载归档](raw/eigh/rank1_submission_1174206.zip) | [目录](extracted/eigh/rank1_submission_1174206) |
| eigh | 2 | 2 | 只想放假 | 78.55 | 1176362 | [下载归档](raw/eigh/rank2_submission_1176362.zip) | [目录](extracted/eigh/rank2_submission_1176362) |
| eigh | 3 | 3 | 天才交易员久留美 | 60.26 | 1177525 | [下载归档](raw/eigh/rank3_submission_1177525.zip) | [目录](extracted/eigh/rank3_submission_1177525) |
| solve | 1 | 1 | ruby0451 的队伍 | 93.81 | 1171359 | [下载归档](raw/solve/rank1_submission_1171359.zip) | [目录](extracted/solve/rank1_submission_1171359) |
| solve | 2 | 2 | 南工大果 | 82.02 | 1166976 | [下载归档](raw/solve/rank2_submission_1166976.zip) | [目录](extracted/solve/rank2_submission_1166976) |
| solve | 3 | 3 | Helton 的队伍 | 72.74 | 1177540 | [下载归档](raw/solve/rank3_submission_1177540.zip) | [目录](extracted/solve/rank3_submission_1177540) |
| TreeLineMinDistance | 1 | 1 | gcw_OTnEiQgj 的队伍 | 91.85 | 1175341 | [下载归档](raw/TreeLineMinDistance/rank1_submission_1175341.zip) | [目录](extracted/TreeLineMinDistance/rank1_submission_1175341) |
| TreeLineMinDistance | 2 | 2 | 只想放假 | 78.93 | 1176190 | [下载归档](raw/TreeLineMinDistance/rank2_submission_1176190.zip) | [目录](extracted/TreeLineMinDistance/rank2_submission_1176190) |
| TreeLineMinDistance | 3 | 3 | 难敌 | 69.34 | 1175418 | [下载归档](raw/TreeLineMinDistance/rank3_submission_1175418.zip) | [目录](extracted/TreeLineMinDistance/rank3_submission_1175418) |

## 归档工具

`download_archive.py` 通过公开页面建立匿名会话下载附件；`audit_archive.py` 校验 CRC、哈希与路径安全后解压并索引；`build_catalog.py` 生成目录。它们不执行下载的源码。公开仓库的日常校验优先运行根README列出的静态检查，不重新下载/生成来源快照。
