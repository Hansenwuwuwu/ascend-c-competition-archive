"""Record only explicitly read solve kernels; preserve other problem ledgers."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / "deep_audit/coverage/09_solve.json"
data = json.loads(target.read_text(encoding="utf-8"))
for item in data["files"]:
    if "/rank1_" in item["path"] and item["path"].endswith("op_kernel/linalg_solve.cpp"):
        raw = (root / item["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        assert len(raw.decode("utf-8-sig").splitlines()) == item["total_lines"] == 2344
        item["reviewed_ranges"] = [{"start": 1, "end": 2344, "notes": "连续实读；S4/S11闭合全部行段，不是API/精度/性能证明。", "report_section": "../problems/09_solve.md#s11"}]
        item["status"] = "complete"
    elif "/rank3_" in item["path"] and item["path"].endswith("op_kernel/linalg_solve.cpp"):
        raw = (root / item["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"] == "aa1aabad7ba1cb3b82ebe6555a10cbb1b2f461d609a9539c224ee4b260f1593f"
        assert len(raw.decode("utf-8-sig").splitlines()) == item["total_lines"] == 3988
        item["reviewed_ranges"] = [{"start": 1, "end": 3988, "notes": "连续实读；S10/S12闭合全部行段；非SDK、数值全域或性能证明。", "report_section": "../problems/09_solve.md#s12"}]
        item["status"] = "complete"
data["review_date"] = "2026-11-09"
if all(x["status"] == "complete" for x in data["files"]):
    data["status"] = "first_pass_source_review_complete"
target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("Recorded solve rank1 2344 and rank3 3988 lines; rank2 and other ledgers preserved.")
