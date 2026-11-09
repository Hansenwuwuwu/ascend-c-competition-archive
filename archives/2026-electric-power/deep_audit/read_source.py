"""Display immutable archived UTF-8 source with stable line numbers and hash."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("start", type=int, nargs="?", default=1)
    parser.add_argument("end", type=int, nargs="?")
    args = parser.parse_args()
    path = ROOT / args.path
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    records = json.loads((ROOT / "source_index.json").read_text(encoding="utf-8"))
    expected = next(item for item in records if item["path"] == args.path.replace("\\", "/"))
    assert digest == expected["sha256"], f"Source changed: {path}"
    lines = source.decode("utf-8-sig").splitlines()
    end = args.end or len(lines)
    assert 1 <= args.start <= end <= len(lines)
    print(f"FILE {args.path}\nSHA256 {digest}\nTOTAL_LINES {len(lines)}; DISPLAY {args.start}-{end}")
    for n in range(args.start, end + 1):
        print(f"{n}: {lines[n-1]}")


if __name__ == "__main__":
    main()
