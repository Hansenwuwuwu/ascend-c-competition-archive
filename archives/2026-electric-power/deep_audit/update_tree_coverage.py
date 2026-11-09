"""Record the completed rank3 TreeLineMinDistance source review only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / "deep_audit/coverage/10_TreeLineMinDistance.json"
data = json.loads(target.read_text(encoding="utf-8"))
rank3 = {
    "rank3_submission_1175418/code/op_kernel/tiling_key_tree_line_min_distance.h":
        (126, "71049b9894c0314b06ee7fd1cf4b7511f93bfdb72dbdf8ac4b11da75abb94f68"),
    "rank3_submission_1175418/code/op_kernel/tree_line_min_distance.cpp":
        (7562, "a9d29748754e035577ed3b302590eb305d4f67e0298c85950776f3acef4aec13"),
    "rank3_submission_1175418/code/op_kernel/tree_line_min_distance_tiling.h":
        (6230, "c632673efc0bfc5a31b8fcb74594d9f2cfd42f9822a7f6e0df6e302a5ba4b904"),
    "rank3_submission_1175418/code/op_host/tree_line_min_distance.cpp":
        (247, "008401c08ffd0bf390c046ea306da1017383a2b114e0c90c2bf7969d9b6f7e42"),
}
for item in data["files"]:
    suffix = next((s for s in rank3 if item["path"].endswith(s)), None)
    if suffix is None:
        continue
    lines, expected_hash = rank3[suffix]
    raw = (root / item["path"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == expected_hash == item["sha256"], (item["path"], digest)
    assert len(raw.decode("utf-8-sig").splitlines()) == lines == item["total_lines"]
    item["reviewed_ranges"] = [{
        "start": 1,
        "end": lines,
        "notes": "2026-11-09 contiguous full-source read via read_source.py; SHA256 verified; source-only, no math/SDK/runtime/performance proof.",
        "report_section": "../problems/10_TreeLineMinDistance.md#i",
    }]
    item["status"] = "complete"
    del rank3[suffix]
assert not rank3, f"Missing rank3 ledger entries: {sorted(rank3)}"
data["review_date"] = "2026-11-09"
if all(item["status"] == "complete" for item in data["files"]):
    data["status"] = "first_pass_source_review_complete"
target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Recorded four rank3 source files; other submissions and topic ledgers preserved.")
