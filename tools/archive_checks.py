"""Static publication checks. Never import/build/run archived submissions."""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archives/2026-electric-power'
SECRET_PATTERNS = {
    'github-token': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
    'private-key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'judge-session': re.compile(r'cannjudge_auth\s*=\s*[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'),
    'literal-password': re.compile(r'(?i)\b(?:password|passwd)\s*[:=]\s*[\"\x27][A-Za-z0-9!@#$%^&*_-]{10,}[\"\x27]'),
}


def check_hashes(base, records):
    base = base.resolve()
    errors = []
    for item in records:
        relative = item['path']
        target = (base / relative).resolve()
        if not target.is_relative_to(base):
            errors.append(f'path escapes archive: {relative}')
        elif not target.is_file():
            errors.append(f'missing file: {relative}')
        elif hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
            errors.append(f'hash mismatch: {relative}')
    return errors


def check_markdown_links(root):
    errors = []
    for document in root.rglob('*.md'):
        # Original third-party/source snapshots are byte-preserved, not edited documentation.
        if any(part in {'.git', 'extracted', 'source', '__pycache__'} for part in document.relative_to(root).parts):
            continue
        for raw in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8-sig')):
            target = raw.strip().strip('<>')
            if re.match(r'^[A-Za-z]:[/\\]', target):
                errors.append(f'{document.relative_to(root)}: nonportable drive link')
                continue
            if re.match(r'^(?:https?://|mailto:|#)', target):
                continue
            filename = unquote(target.split('#', 1)[0])
            if re.search(r':\d+$', filename):
                errors.append(f'{document.relative_to(root)}: use #L line anchor')
            elif not (document.parent / filename).is_file() and not (document.parent / filename).is_dir():
                errors.append(f'{document.relative_to(root)}: missing target {filename}')
    return errors


def secret_kinds(text):
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]


def main():
    errors = []
    index = json.loads((ARCHIVE / 'source_index.json').read_text(encoding='utf-8'))
    bundles = json.loads((ARCHIVE / 'manifest.json').read_text(encoding='utf-8'))
    statements = json.loads((ARCHIVE / 'deep_audit/specs/manifest.json').read_text(encoding='utf-8'))
    errors += check_hashes(ARCHIVE, index)
    errors += check_hashes(ARCHIVE, [{'path': x['local_path'], 'sha256': x['sha256']} for x in bundles])
    errors += check_hashes(ARCHIVE, [{'path': x['pdf'], 'sha256': x['sha256']} for x in statements])
    if (len(index), len(bundles), len(statements)) != (271, 27, 10):
        errors.append('snapshot inventory must contain 271 sources, 27 ZIPs, 10 PDFs')
    if any(Path(x['pdf']).is_absolute() or re.match(r'^[A-Za-z]:', x['pdf']) for x in statements):
        errors.append('PDF manifest must use archive-relative paths')
    for bundle in bundles:
        with zipfile.ZipFile(ARCHIVE / bundle['local_path']) as zipped:
            if zipped.testzip() is not None:
                errors.append(f'ZIP CRC failed: {bundle["local_path"]}')
    summary = json.loads((ARCHIVE / 'deep_audit/COVERAGE_SUMMARY.json').read_text(encoding='utf-8'))
    if (summary['complete_files'], summary['complete_submissions'], summary['reviewed_lines']) != (271, 27, 139460):
        errors.append('source review coverage is incomplete')
    errors += check_markdown_links(ROOT)
    text_suffixes = {'.py', '.md', '.txt', '.json', '.yml', '.yaml', '.html', '.js', '.cpp', '.h', '.hpp', '.cmake'}
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.relative_to(ROOT).parts or path.suffix.lower() not in text_suffixes:
            continue
        for kind in secret_kinds(path.read_text(encoding='utf-8-sig', errors='replace')):
            errors.append(f'{path.relative_to(ROOT)}: potential {kind} (value withheld)')
    if errors:
        print('\n'.join(errors))
        return 1
    print('PUBLIC ARCHIVE CHECKS PASS: 271 source hashes, 27 ZIP hashes/CRC, 10 PDF hashes; coverage 139460/139460; portable links; credential-pattern scan.')
    print('Static/source checks only: no third-party kernel executed; pattern scanning is not a guarantee of absence of secrets.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
