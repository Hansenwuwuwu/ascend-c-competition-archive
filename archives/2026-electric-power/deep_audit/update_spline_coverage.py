"""Record only explicitly completed manual reading ranges for SplineConv."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    inventory = json.loads((ROOT / "source_index.json").read_text(encoding="utf-8"))
    records = []
    for item in inventory:
        if item["operator"] != "SplineConv":
            continue
        path = item["path"]
        ranges = []
        if item["rank"] in (1, 3):
            ranges = [{"start": 1, "end": item["lines"], "notes": "连续全文人工实读；按报告B/D表的语义段解释，含pragma/空白/成员/构建配置。", "report_section": "03_SplineConv.md#B/D"}]
        elif item["rank"] == 2:
            if path.endswith("op_host/spline_conv.cpp") or path.endswith(("tiling_key_spline_conv.h", "spline_conv_tiling.h")):
                ranges = [{"start": 1, "end": item["lines"], "notes": "连续全文人工实读，见F；完整kernel仍待续。", "report_section": "03_SplineConv.md#F"}]
            elif path.endswith("op_kernel/spline_conv.cpp"):
                ranges = [{"start": 1, "end": item["lines"], "notes": "连续全文实读；包含single、edge owner、resident、general fallback、入口与wrapper宏，见F。", "report_section": "03_SplineConv.md#F"}]
        if item["rank"] in (2, 3) and path.endswith("CMakeLists.txt"):
            same = [x for x in inventory if x["operator"] == "SplineConv" and x["rank"] == 1 and x["sha256"] == item["sha256"]]
            assert len(same) == 1
            ranges = [{"start": 1, "end": item["lines"], "notes": "SHA256逐字节一致，复用顺序1已全文实读的构建文件：" + same[0]["path"], "report_section": "03_SplineConv.md#F"}]
        record = {"path": path, "sha256": item["sha256"], "total_lines": item["lines"],
                  "status": "reviewed" if ranges and ranges[-1]["end"] == item["lines"] else "partial" if ranges else "not_started",
                  "reviewed_ranges": ranges}
        records.append(record)
    target = ROOT / "deep_audit/coverage/03_SplineConv.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"operator": "SplineConv", "status": "first_pass_text_complete_proof_open", "files": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SplineConv: {sum(sum(r['end']-r['start']+1 for r in f['reviewed_ranges']) for f in records)} / {sum(f['total_lines'] for f in records)} lines manually read")


if __name__ == "__main__":
    main()
