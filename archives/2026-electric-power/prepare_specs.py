"""Extract user-supplied PDF statements; preserve PDF page boundaries and hashes."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "problems"


def main():
    target = ROOT / "deep_audit/specs"
    target.mkdir(parents=True, exist_ok=True)
    converter = shutil.which("pdftotext")
    inspector = shutil.which("pdfinfo")
    assert converter and inspector
    manifest = []
    for number in range(1, 11):
        source = next(SOURCE.glob(f"{number:02d}_*.pdf"))
        result = subprocess.run([converter, "-layout", "-enc", "UTF-8", str(source), "-"], capture_output=True, check=True)
        pages = result.stdout.decode("utf-8").split("\f")
        if pages and not pages[-1].strip():
            pages.pop()
        body = "\n".join(f"\n===== PDF PAGE {index} =====\n{page}" for index, page in enumerate(pages, 1))
        output = target / f"{number:02d}.txt"
        output.write_text(body, encoding="utf-8")
        info = subprocess.run([inspector, str(source)], capture_output=True, check=True).stdout.decode("utf-8", errors="replace")
        (target / f"{number:02d}_pdfinfo.txt").write_text(info, encoding="utf-8")
        record = {"problem": number, "pdf": source.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "text": output.relative_to(ROOT).as_posix(), "pages": len(pages), "text_characters": len(body)}
        manifest.append(record)
        print(f"PROBLEM {number}: {len(pages)} pages, {len(body)} characters")
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
