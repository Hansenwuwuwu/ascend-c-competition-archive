"""Close the fully read rank1 Fire kernel gap after immutable hash check."""
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REL = "extracted/FireCandidateDecodeFusion/rank1_submission_1122762/code/op_kernel/fire_candidate_decode_fusion.cpp"
EXPECTED = "4693b60ab79d19e3d73fcb245698ef9d1568435ab80e97eb92570cded8cef310"


def main():
    source = ROOT / REL
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    total = len(raw.decode("utf-8-sig").splitlines())
    assert total == 8774

    path = ROOT / "deep_audit/coverage/07_FireCandidateDecodeFusion.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    item = next(x for x in ledger["files"] if x["path"] == REL)
    assert item["sha256"] == EXPECTED and item["total_lines"] == total
    assert item["status"] == "partial"
    assert item["remaining_ranges"] == [{"start": 3001, "end": 8529}]
    item["reviewed_ranges"].append({
        "start": 3001,
        "end": 8529,
        "notes": "2026-11-09 按连续段逐行阅读3001-3100/3101-3200/3201-3300/3301-3400/3401-3600/3601-3800/3801-4000/4001-4200/4201-4400/4401-4600/4601-4800/4801-5000/5001-5200/5201-5400/5401-5600/5601-5800/5801-6000/6001-6300/6301-6600/6601-6900/6901-7200/7201-7500/7501-7800/7801-8100/8101-8400/8401-8700; separately read final dispatch 8701-8774. 首轮源码审阅，不代表数学、SDK、同步或性能证明。",
        "report_section": "../problems/07_FireCandidateDecodeFusion.md#rank1-kernel-3001-8529",
    })
    item["status"] = "complete"
    item["reviewed_lines"] = sum(r["end"] - r["start"] + 1 for r in item["reviewed_ranges"])
    item["remaining_ranges"] = []

    ledger["review_date"] = date.today().isoformat()
    ledger["status"] = "complete_first_pass" if all(x.get("status") == "complete" for x in ledger["files"]) else "in_progress"
    ledger["date"] = date.today().isoformat()
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reviewed = sum(sum(r["end"] - r["start"] + 1 for r in x.get("reviewed_ranges", [])) for x in ledger["files"])
    files_complete = sum(x.get("status") == "complete" for x in ledger["files"])
    print(f"{REL}: 1-{total}; Fire {reviewed} reviewed lines, {files_complete}/{len(ledger['files'])} files; {ledger['status']}")


if __name__ == "__main__":
    main()
