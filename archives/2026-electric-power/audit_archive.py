"""Validate and index downloaded sources without building or executing them."""
import hashlib
import json
import re
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
SOURCE_SUFFIXES = {".cpp", ".cc", ".c", ".h", ".hpp", ".cuh", ".py", ".cmake", ".sh", ".md", ".txt", ".json", ".yaml", ".yml"}
PATTERNS = {
    "compile_time_specialization": r"if\s+constexpr|template\s*<|TILING_KEY_IS|ASCENDC_TPL_",
    "dma_and_padding": r"DataCopyPad|DataCopyExtParams|DataCopyParams|Nd2Nz|ND2NZ",
    "vector_layout": r"Transpose|Gather|Brcb|CreateVecIndex|ArithProgression|Scatter|Interleave|DeInterleave",
    "pipeline_and_sync": r"TQue|DBuff|BUFFER_NUM|double.?buffer|SetFlag|WaitFlag|CrossCore|SyncAll|UnitFlag",
    "cube": r"Mmad\(|Matmul<|matmul::|LoadData|Fixpipe|SetOrgShape|SetSingleShape",
    "reduction": r"WholeReduce|BlockReduce|RepeatReduce|ReduceSum|ReduceMax|ReduceMin",
    "arithmetic": r"Reciprocal|Div\(|Exp\(|Ln\(|Rsqrt|Sqrt\(|Axpy|Fma|Muls\(|Adds\(",
    "cache_and_registers": r"CacheMode|CACHE_MODE|__aicore__|__simt__|__launch_bounds__|__VEC_SCOPE__|MicroAPI|SetAtomic",
}


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    inventory = []
    for bundle in manifest:
        if bundle["status"] != "downloaded":
            raise RuntimeError(f"Missing archive: {bundle['name']}")
        archive = ROOT / bundle["local_path"]
        if hashlib.sha256(archive.read_bytes()).hexdigest() != bundle["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {archive}")
        destination = ROOT / "extracted" / bundle["operator"] / f"rank{bundle['decision_rank']}_submission_{bundle['submission_id']}"
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise RuntimeError(f"Bad ZIP CRC: {archive}")
            if sum(info.file_size for info in z.infolist()) > 100 * 1024 * 1024:
                raise RuntimeError(f"Oversize extraction: {archive}")
            for info in z.infolist():
                name = PurePosixPath(info.filename.replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
                    raise RuntimeError(f"Unsafe ZIP path: {info.filename}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise RuntimeError(f"ZIP symbolic link: {info.filename}")
                target = destination.joinpath(*name.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = z.read(info)
                if target.exists() and target.read_bytes() != payload:
                    raise RuntimeError(f"Refusing to overwrite modified extracted source: {target}")
                target.write_bytes(payload)
                record = {"operator": bundle["operator"], "rank": bundle["decision_rank"],
                          "submission_id": bundle["submission_id"], "path": target.relative_to(ROOT).as_posix(),
                          "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                if target.suffix.lower() in SOURCE_SUFFIXES or target.name.lower().startswith(("license", "cmakelists")):
                    text = payload.decode("utf-8-sig", errors="replace")
                    lines = text.splitlines()
                    record["lines"] = len(lines)
                    record["evidence"] = {category: [{"line": n, "text": line.strip()[:260]}
                        for n, line in enumerate(lines, 1) if re.search(pattern, line, re.I)]
                        for category, pattern in PATTERNS.items()}
                inventory.append(record)
        bundle["zip_crc_verified"] = True
        bundle["extracted_path"] = destination.relative_to(ROOT).as_posix()
        bundle["file_count"] = sum(i["submission_id"] == bundle["submission_id"] for i in inventory)
        print(f"{bundle['operator']} rank{bundle['decision_rank']}: {bundle['file_count']} files", flush=True)
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "source_index.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TOTAL {len(manifest)} bundles; {len(inventory)} files; {sum(i.get('lines', 0) for i in inventory)} text lines")
    for item in inventory:
        if "/op_kernel/" in item["path"] and item["path"].endswith((".cpp", ".h", ".hpp")):
            print(f"KERNEL {item['lines']:5d} {item['path']}")


if __name__ == "__main__":
    main()
