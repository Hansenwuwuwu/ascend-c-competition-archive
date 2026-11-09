"""Download public announcement attachments; never execute downloaded content."""
import hashlib
import http.cookiejar
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://globalcomputing.feishu.cn/wiki/OpFlwqe4Ji7jLvkRWRDcIrXRn3b"
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def download(item):
    query = urllib.parse.urlencode({"mount_point": "docx_file", "mount_node_token": item["block_id"]})
    url = f"https://globalcomputing.feishu.cn/space/api/box/stream/download/all/{item['token']}/?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": SOURCE_URL})
    with OPENER.open(request, timeout=40) as response:
        data = response.read()
        content_type = response.headers.get("Content-Type", "")
    if not data.startswith(b"PK"):
        raise RuntimeError(f"Not a ZIP: {content_type}; first bytes={data[:200]!r}")
    if len(data) != item["size"]:
        raise RuntimeError(f"Size mismatch: expected {item['size']}, got {len(data)}")
    submission = re.search(r"submission_(\d+)", item["name"]).group(1)
    rank_text = next(s for s in reversed(item["context"]) if s.startswith("代码附件"))
    rank = int(re.search(r"决策顺序(\d+)", rank_text).group(1))
    target = ROOT / "raw" / item["operator"] / f"rank{rank}_submission_{submission}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != data:
        raise RuntimeError(f"Refusing to overwrite different existing archive: {target}")
    target.write_bytes(data)
    return dict(item, decision_rank=rank, submission_id=submission, source_page=SOURCE_URL,
                download_url=url, local_path=target.relative_to(ROOT).as_posix(),
                actual_size=len(data), sha256=hashlib.sha256(data).hexdigest(), status="downloaded")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    with OPENER.open(urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=40) as response:
        response.read()
    items = json.loads((ROOT / "source/attachments.json").read_text(encoding="utf-8"))
    results = []
    for item in items[:args.limit]:
        try:
            result = download(item)
            print(f"OK {result['operator']} rank{result['decision_rank']}: {result['actual_size']} bytes", flush=True)
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                error = RuntimeError(f"HTTP {error.code}: {error.read(500)!r}")
            result = dict(item, status="failed", error=str(error))
            print(f"FAILED {item['name']}: {error}", flush=True)
        results.append(result)
        (ROOT / "manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
