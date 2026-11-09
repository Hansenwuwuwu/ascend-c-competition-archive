"""Verify archive identity, range bounds, gaps and overview. No code execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def merge(ranges, length):
    result = []
    for start, end in sorted(ranges):
        assert 1 <= start <= end <= length, (start, end, length)
        if result and start <= result[-1][1] + 1:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def main():
    index = json.loads((ROOT / "source_index.json").read_text(encoding="utf-8"))
    lookup = {x["path"]: x for x in index}
    records = {}
    rows = []
    review_dates = []
    for ledger in sorted((ROOT / "deep_audit/coverage").glob("*.json")):
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        review_dates.append(payload.get("review_date", payload.get("date", "2026-11-08")))
        count = total = complete = 0
        for item in payload["files"]:
            path = item["path"]
            assert path not in records, ("Duplicate ledger path", path)
            original = lookup[path]
            assert original["sha256"] == item["sha256"]
            raw = (ROOT / path).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == item["sha256"], path
            assert len(raw.decode("utf-8-sig").splitlines()) == item["total_lines"] == original["lines"], path
            intervals = merge([(r["start"], r["end"]) for r in item["reviewed_ranges"]], item["total_lines"])
            n = sum(b - a + 1 for a, b in intervals)
            missing, cursor = [], 1
            for a, b in intervals:
                if cursor < a:
                    missing.append([cursor, a - 1])
                cursor = b + 1
            if cursor <= item["total_lines"]:
                missing.append([cursor, item["total_lines"]])
            records[path] = {"sha256": item["sha256"], "reviewed_ranges": intervals,
                             "unreviewed_ranges": missing, "reviewed_lines": n,
                             "total_lines": item["total_lines"]}
            count += n
            total += item["total_lines"]
            complete += not missing
        rows.append({"operator": payload["operator"], "complete_files": complete, "files": len(payload["files"]),
                     "reviewed_lines": count, "total_lines": total, "ledger": ledger.name})
    assert set(records) == set(lookup), sorted(set(lookup) - set(records))
    submissions = {}
    for original in index:
        key = (original["operator"], original["rank"])
        submissions.setdefault(key, []).append(not records[original["path"]]["unreviewed_ranges"])
    out = {"status": "in_progress", "date": max(review_dates), "note": "Counts include explicit hash-identical text reuse; full text read does not imply all proof obligations discharged.",
           "total_files": len(records), "complete_files": sum(not r["unreviewed_ranges"] for r in records.values()),
           "total_submissions": len(submissions), "complete_submissions": sum(all(v) for v in submissions.values()),
           "reviewed_lines": sum(r["reviewed_lines"] for r in records.values()),
           "total_lines": sum(r["total_lines"] for r in records.values()), "operators": rows, "files": records}
    if out["complete_files"] == out["total_files"]:
        out["status"] = "first_pass_source_review_complete"
    (ROOT / "deep_audit/COVERAGE_SUMMARY.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['operator']}: {row['reviewed_lines']}/{row['total_lines']} lines, {row['complete_files']}/{row['files']} files")
    print("ARCHIVE IDENTITY AND RANGE CHECKS PASS", out["reviewed_lines"], "/", out["total_lines"])
    print("COMPLETE FILES", out["complete_files"], "/", out["total_files"], "SUBMISSIONS", out["complete_submissions"], "/", out["total_submissions"])


if __name__ == "__main__":
    main()
