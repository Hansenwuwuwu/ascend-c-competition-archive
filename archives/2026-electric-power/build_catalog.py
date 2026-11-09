"""Build a provenance-preserving reading index from validated public sources."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((ROOT / "source_index.json").read_text(encoding="utf-8"))
    for bundle in manifest:
        context = "\n".join(bundle["context"])
        rank, team = re.findall(r"(?:官网原)?第(\d+)名 ([^\n]+)", context)[-1]
        bundle.update(original_rank=int(rank), team=team,
                      announced_score=re.findall(r"官网分数：([\d.]+)", context)[-1],
                      announcement_status="public_notice_2026-11-08_to_2026-11-10")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    operators = list(dict.fromkeys(b["operator"] for b in manifest))
    rows = ["| 题目 | 公示顺序 | 原榜名次 | 队伍 | 公示分数 | 提交 ID | ZIP | 源码 |",
            "|---|---:|---:|---|---:|---|---|---|"]
    for b in manifest:
        rows.append(f"| {b['operator']} | {b['decision_rank']} | {b['original_rank']} | {b['team']} | {b['announced_score']} | {b['submission_id']} | [下载归档]({b['local_path']}) | [目录]({b['extracted_path']}) |")
    readme = f"""# 2026 电力行业算子专场：九题公示代码归档

来源：[飞书公示页](https://globalcomputing.feishu.cn/wiki/OpFlwqe4Ji7jLvkRWRDcIrXRn3b)。
采集日期：2026-11-08（北京时间）。页面标题为“2026电力行业算子专场预赛榜单参考与代码资料”。

## 已完成

- 9 道题、27 份代码 ZIP 全部下载；大小与页面附件元数据一致，SHA-256 已记录，ZIP CRC 全部通过。
- 原始 ZIP 保存在 `raw/`，安全解压源码保存在 `extracted/`；共 {len(inventory)} 个文件，文本总计 {sum(i.get('lines', 0) for i in inventory):,} 行（包含生成表、构建配置和注释，不等于有效计算代码行数）。
- 原始页面、结构化文档和附件元数据保存在 `source/`，便于复查排名、版本和来源。
- 已完成全量文件索引和关键词定位，并制作首轮可复用方法阅读指南；**尚未完成 27 份源码的逐行算法/合规审计，没有重新编译或设备复测。**
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

""" + "\n".join(rows) + "\n\n## 归档工具\n\n`inspect_page.py` 只解析已保存的公开 HTML；`download_archive.py` 通过公开页面建立匿名会话下载附件；`audit_archive.py` 校验 CRC、哈希与路径安全后解压并索引；`build_catalog.py` 生成目录。它们不执行下载的源码。\n"
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    catalog = ["# 源码阅读目录", "", "这是全量文件与静态关键词索引，不是已完成逐行审计的声明。命中不证明该分支被 Host 派发，也不证明该方法的独立收益。", ""]
    for b in manifest:
        files = [i for i in inventory if i["submission_id"] == b["submission_id"]]
        catalog += [f"## {b['operator']} — 公示顺序 {b['decision_rank']} — {b['team']}", "",
                    f"提交 ID：{b['submission_id']}；原榜名次：{b['original_rank']}；公示分数：{b['announced_score']}；{len(files)} 文件。", "",
                    "| 文件 | 行数 | 关键词命中类别 |", "|---|---:|---|"]
        for item in files:
            hits = ", ".join(f"{key}:{len(values)}" for key, values in item.get("evidence", {}).items() if values)
            name = item["path"].split("/code/", 1)[-1]
            catalog.append(f"| [{name}]({item['path']}) | {item.get('lines', '—')} | {hits} |")
        catalog.append("")
    (ROOT / "SOURCE_CATALOG.md").write_text("\n".join(catalog), encoding="utf-8")
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(f"{b['sha256']}  {b['local_path']}" for b in manifest) + "\n", encoding="utf-8")
    # Independent final checks against what exists on disk, not counters alone.
    for item in inventory:
        payload = (ROOT / item["path"]).read_bytes()
        assert len(payload) == item["bytes"]
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    assert len(manifest) == 27 and len(operators) == 9
    assert all(Counter(b["decision_rank"] for b in manifest if b["operator"] == op) == {1: 1, 2: 1, 3: 1} for op in operators)
    assert all(b["zip_crc_verified"] for b in manifest)
    print(f"VERIFIED: {len(manifest)} archives, {len(operators)} operators, {len(inventory)} extracted files")
    print(f"ZIP total: {sum(b['actual_size'] for b in manifest):,} bytes")
    print("Operators: " + ", ".join(operators))


if __name__ == "__main__":
    main()
