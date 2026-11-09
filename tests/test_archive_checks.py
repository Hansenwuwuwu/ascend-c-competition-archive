import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from tools import archive_checks
except ImportError:
    archive_checks = None


class PublicArchiveChecksTests(unittest.TestCase):
    def require_checker(self):
        self.assertIsNotNone(archive_checks, 'public archive checker is not implemented')

    def test_hashes_accept_original_bytes_and_reject_modified_bytes(self):
        self.require_checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'original.cpp').write_bytes(b'original\r\n')
            records = [{'path': 'original.cpp', 'sha256': hashlib.sha256(b'original\r\n').hexdigest()}]
            self.assertEqual(archive_checks.check_hashes(root, records), [])
            (root / 'original.cpp').write_bytes(b'modified\n')
            self.assertEqual(len(archive_checks.check_hashes(root, records)), 1)

    def test_manifest_cannot_escape_archive_root(self):
        self.require_checker()
        with tempfile.TemporaryDirectory() as directory:
            errors = archive_checks.check_hashes(Path(directory), [{'path': '../outside', 'sha256': '0' * 64}])
            self.assertTrue(errors)

    def test_relative_markdown_code_anchor_is_portable(self):
        self.require_checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sample.cpp').write_text('int x;', encoding='utf-8')
            (root / 'README.md').write_text('[source](sample.cpp#L1)\n', encoding='utf-8')
            self.assertEqual(archive_checks.check_markdown_links(root), [])

    def test_missing_file_and_windows_drive_link_are_rejected(self):
        self.require_checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('[missing](none.md)\n[local](D:/private/source.cpp:1)', encoding='utf-8')
            self.assertEqual(len(archive_checks.check_markdown_links(root)), 2)

    def test_scanner_detects_credentials_without_returning_values(self):
        self.require_checker()
        credential = 'gh' + 'p_' + 'A' * 36
        matches = archive_checks.secret_kinds(credential)
        self.assertIn('github-token', matches)
        self.assertNotIn(credential, str(matches))
        cookie = 'cannjudge_' + 'auth=' + 'a' * 30 + '.' + 'b' * 30
        self.assertIn('judge-session', archive_checks.secret_kinds(cookie))

    def test_public_attachment_identifiers_are_not_login_secrets(self):
        self.require_checker()
        self.assertEqual(archive_checks.secret_kinds('{"token":"publicAttachmentIdentifier","document_token":"publicDocumentId"}'), [])

    def test_frozen_snapshot_summary_records_completed_source_review(self):
        summary_path = Path(__file__).resolve().parents[1] / 'archives/2026-electric-power/deep_audit/COVERAGE_SUMMARY.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'first_pass_source_review_complete')
        self.assertEqual(summary['date'], '2026-11-09')

    def test_regenerated_pdf_manifest_uses_archive_relative_paths(self):
        script = Path(__file__).resolve().parents[1] / 'archives/2026-electric-power/prepare_specs.py'
        spec = importlib.util.spec_from_file_location('statement_preparation', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'problems'
            source.mkdir()
            for number in range(1, 11):
                (source / f'{number:02d}_example.pdf').write_bytes(b'%PDF fixture')

            def external_converter(command, **kwargs):
                return SimpleNamespace(stdout=b'one page\f' if command[0] == 'pdftotext' else b'Pages: 1\n')

            with patch.object(module, 'ROOT', root), patch.object(module, 'SOURCE', source), patch.object(module.shutil, 'which', side_effect=lambda name: name), patch.object(module.subprocess, 'run', side_effect=external_converter):
                module.main()
            records = json.loads((root / 'deep_audit/specs/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual([x['pdf'] for x in records], [f'problems/{n:02d}_example.pdf' for n in range(1, 11)])
            self.assertTrue(all(x['sha256'] == hashlib.sha256(b'%PDF fixture').hexdigest() for x in records))


if __name__ == '__main__':
    unittest.main()
