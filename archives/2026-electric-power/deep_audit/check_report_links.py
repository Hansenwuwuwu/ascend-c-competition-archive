"""Read-only local Markdown link check. Does not execute archived programs."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent
count = 0
broken = []
for document in root.rglob("*.md"):
    for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
        if re.match(r"[a-zA-Z]+://|#", target):
            continue
        count += 1
        destination = re.sub(r":\d+$", "", target.split("#", 1)[0]).strip("<>")
        if not (document.parent / destination).exists():
            broken.append((str(document.relative_to(root)), target))
print("LOCAL MARKDOWN LINKS", count, "BROKEN", broken)
assert not broken
