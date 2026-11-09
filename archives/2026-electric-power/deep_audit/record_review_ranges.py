"""Materialize explicit reading ranges, not keyword hits or inferred source coverage.

Only the named first-pass reports and the current reader's continuous reads are
inputs to the declarations below. This does not execute any archived code.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "det": (4, "04_det", {1: [(1, 1687)], 2: [(1, 3548)], 3: [(1, 724)]}),
    "lstsq": (6, "06_lstsq", {1: [(2678, 2762)], 2: [(1, 959)], 3: [(3091, 3355)]}),
    "solve": (9, "09_solve", {1: [(1, 210), (2191, 2344)], 2: [(1, 1874)], 3: [(1, 240), (3771, 3988)]}),
    "FireCandidateDecodeFusion": (7, "07_FireCandidateDecodeFusion", {1: [(1, 230), (8530, 8774)], 2: [(1, 227)]}),
    "TreeLineMinDistance": (10, "10_TreeLineMinDistance", {}),
}


def main():
    index = json.loads((ROOT / "source_index.json").read_text(encoding="utf-8"))
    build_hashes = {x["sha256"] for x in index if x["operator"] == "SplineConv" and x["rank"] == 1 and x["path"].endswith("CMakeLists.txt")}
    for op, (number, report, kernel_ranges) in SPECS.items():
        files = []
        for item in index:
            if item["operator"] != op:
                continue
            path, rank, length = item["path"], item["rank"], item["lines"]
            digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            assert digest == item["sha256"], ("Archived source changed", path)
            ranges = []
            notes = "连续全文审读，语义行段见对应题目报告；此记录不是正确性/性能证明。"
            if path.endswith("CMakeLists.txt"):
                assert digest in build_hashes
                ranges = [(1, length)]
                notes = "与已全文审读构建文件SHA256一致，复用语义审读，不作为独立算法证据。"
            elif op == "TreeLineMinDistance":
                if rank == 1:
                    ranges = [(1, length)]
                    if path.endswith(("tree_edges.h", "tree_pairs.h", "tree_tiny.h")):
                        ref = ROOT / "extracted/TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_vector.h"
                        assert (ROOT / path).read_text(encoding="utf-8-sig").splitlines()[:217] == ref.read_text(encoding="utf-8-sig").splitlines()[:217]
                        notes = "1–217与已全文实读tree_vector.h逐行一致，复用公共头；218至末尾连续实读。"
            elif op == "FireCandidateDecodeFusion":
                if rank == 3 or "/op_host/" in path or path.endswith(("_tiling.h", "tiling_key_fire_candidate_decode_fusion.h")):
                    ranges = [(1, length)]
                elif path.endswith("op_kernel/fire_candidate_decode_fusion.cpp"):
                    ranges = kernel_ranges.get(rank, [])
            elif "/op_host/" in path or path.endswith("_tiling.h") or "/tiling_key_" in path or path.endswith("lstsq_options.h"):
                ranges = [(1, length)]
            elif path.endswith(f"op_kernel/linalg_{op}.cpp"):
                ranges = kernel_ranges[rank]
            elif op == "det" and rank == 3 and path.endswith(("det_small.h", "det_coop64.h", "det_coop64_direct_diag.h", "det_cube_wide.h", "det_cube_static.h", "det_cube_lu.h")):
                ranges = [(1, length)]
            entries = [{"start": a, "end": b, "notes": notes, "report_section": f"../problems/{report}.md"} for a, b in ranges]
            complete = bool(ranges) and sum(b - a + 1 for a, b in ranges) == length
            files.append({"path": path, "sha256": digest, "total_lines": length,
                          "status": "complete" if complete else "partial" if ranges else "not_started",
                          "reviewed_ranges": entries})
        payload = {"schema_version": 1, "problem": number, "operator": op,
                   "status": "first_pass_text_complete_proof_open" if op == "det" else "in_progress", "review_date": "2026-11-08",
                   "scope": "First-pass continuous semantic reading only; no archived code executed. Hash-identical text may reuse reading.",
                   "files": files}
        target = ROOT / f"deep_audit/coverage/{report}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(op, sum(r["end"] - r["start"] + 1 for f in files for r in f["reviewed_ranges"]), "/", sum(f["total_lines"] for f in files))


if __name__ == "__main__":
    main()
