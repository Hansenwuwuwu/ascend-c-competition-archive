"""Append ONLY a range already read and explained; retain all prior ranges."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rank, begin, end, section = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ledger = root / "deep_audit/coverage/08_eigh.json"
data = json.loads(ledger.read_text(encoding="utf-8"))
for item in data["files"]:
    if f"/rank{rank}_" not in item["path"] or not item["path"].endswith("op_kernel/linalg_eigh.cpp"):
        continue
    raw = (root / item["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    assert 1 <= begin <= end <= item["total_lines"] == len(raw.decode("utf-8-sig").splitlines())
    item["reviewed_ranges"].append({"start": begin, "end": end, "notes": "连续逐行实读并在报告记录语义；不是数学全域、SDK同步或性能证明。", "report_section": f"../problems/08_eigh.md#{section}"})
    ranges = sorted((x["start"], x["end"]) for x in item["reviewed_ranges"])
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    gaps, cursor = [], 1
    for a, b in merged:
        if a > cursor:
            gaps.append({"start": cursor, "end": a - 1})
        cursor = b + 1
    if cursor <= item["total_lines"]:
        gaps.append({"start": cursor, "end": item["total_lines"]})
    item["unreviewed_ranges"] = gaps
    item["status"] = "partial" if gaps else "complete"
    break
else:
    raise ValueError(rank)
data["review_date"] = "2026-11-09"
data["status"] = "first_pass_source_review_complete" if all(x["status"] == "complete" for x in data["files"]) else "in_progress"
data.pop("summary", None)  # central checker calculates fresh counts
ledger.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Recorded eigh rank{rank}: {begin}-{end}; existing coverage preserved")
