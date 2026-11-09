"""Close the one fully read pinv DSL window file, with immutable hash check."""
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REL = "extracted/pinv/rank1_submission_1060506/code/op_kernel/pinv_dsl_window.h"
EXPECTED = "6afc5fefe71edf995f407ded745c01d956bf78cdf67a7c0f42d1ead76c74fd07"

def main():
    source = ROOT / REL
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    total = len(raw.decode("utf-8-sig").splitlines())
    assert total == 2981
    path = ROOT / "deep_audit/coverage/02_pinv.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    item = next(x for x in ledger["files"] if x["path"] == REL)
    assert item["sha256"] == EXPECTED and item["total_lines"] == total
    assert item.get("reviewed_ranges", []) == []
    item["reviewed_ranges"] = [{
        "start": 1, "end": total,
        "notes": "连续逐行阅读完整窗口/blocked DSL文件；只表示首轮静态阅读，不表示数值、同步、SDK或性能证明。",
        "report_section": "../problems/02_pinv.md#r1-window-dsl-full"
    }]
    item["status"] = "complete"
    ledger["review_date"] = date.today().isoformat()
    ledger["summary"]["complete_files"] = sum(x.get("status") == "complete" for x in ledger["files"])
    ledger["summary"]["reviewed_lines"] = sum(sum(r["end"]-r["start"]+1 for r in x.get("reviewed_ranges", [])) for x in ledger["files"])
    ledger["status"] = "complete_first_pass" if all(x.get("status") == "complete" for x in ledger["files"]) else "in_progress"
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{REL}: 1-{total}; pinv {ledger['summary']['reviewed_lines']}/{ledger['summary']['total_lines']} lines, {ledger['summary']['complete_files']}/{ledger['summary']['total_files']} files")

if __name__ == "__main__":
    main()
