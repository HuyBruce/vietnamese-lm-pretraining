"""Kiểm tra trên dữ liệu giả trong TemporaryDirectory; không đọc corpus thật."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq
import prepare_news_inventory as inventory


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="news-inventory-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "DataRaw"
        self.category = self.raw / "Báo Tin Tức" / "Thời sự" / "Chính trị"
        self.category.mkdir(parents=True)
        self.output = self.root / "DataProcessed"
        for name in ("inventory", "manifests", "reports"):
            (self.output / name).mkdir(parents=True)
        self.original = "\ufeffTiêu đề\r\nCâu thứ nhất.\r\n\r\nCâu thứ hai.\r\n".encode("utf-8")
        (self.category / "article-123.txt").write_bytes(self.original)

    def open_db(self):
        db = inventory.database(self.output, self.raw)
        self.addCleanup(db.close)
        inventory.scan(db, self.raw, inventory.source_roots(self.raw))
        return db

    def export(self, db):
        inventory.export_parquet(db, self.raw, self.output, 1, 1024 * 1024, 1024 * 1024)

    def test_preserve_content_metadata_exclusions_and_resume(self):
        (self.category / "_crawl_errors.txt").write_text("ERROR", encoding="utf-8")
        errors = self.raw / "Báo Tin Tức" / "_crawl_errors"
        errors.mkdir()
        (errors / "01.txt").write_text("ERROR", encoding="utf-8")
        db = self.open_db()
        self.export(db)
        shards = list((self.output / "inventory").glob("*.parquet"))
        self.assertEqual(len(shards), 1)
        row = pq.read_table(shards[0]).to_pylist()[0]
        self.assertEqual(row["text"].encode("utf-8"), self.original)
        self.assertEqual(row["content_sha256"], hashlib.sha256(self.original).hexdigest())
        self.assertEqual(row["source"], "baotintuc")
        self.assertEqual(row["category"], "Thời sự")
        self.assertEqual(row["subcategory"], "Chính trị")
        self.assertIsNone(row["url"])
        self.assertIsNone(row["published_at"])
        before = inventory.file_sha(shards[0])
        self.export(db)
        self.assertEqual(inventory.file_sha(shards[0]), before)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM files").fetchone()[0], 1)
        self.assertEqual((self.category / "article-123.txt").read_bytes(), self.original)

    def test_crash_after_shard_before_manifest_commit_recovers(self):
        db = self.open_db()
        with patch.object(inventory, "file_sha", side_effect=RuntimeError("simulated interruption")):
            with self.assertRaises(RuntimeError):
                self.export(db)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM shards").fetchone()[0], 0)
        self.assertEqual(db.execute("SELECT state FROM files").fetchone()[0], "pending")
        self.export(db)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM shards").fetchone()[0], 1)
        self.assertEqual(len(list((self.output / "inventory").glob("*.parquet"))), 1)

    def test_invalid_utf8_and_changed_file_are_reported(self):
        bad = self.category / "bad-456.txt"
        bad.write_bytes(b"\xff\xfe\xff")
        db = self.open_db()
        (self.category / "article-123.txt").write_text("changed", encoding="utf-8")
        self.export(db)
        self.assertEqual(inventory.write_report(db, self.raw, self.output), 2)
        report = json.loads((self.output / "reports" / "inventory_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["counts"]["error"], 2)

    def test_cleanup_moves_only_crawlers_and_is_recoverable(self):
        worker = self.raw / "baotintuc_workers"
        worker.mkdir()
        (worker / "01.py").write_text("# worker", encoding="utf-8")
        script = self.raw / "baotintuc_to_txt.py"
        script.write_text("# crawler", encoding="utf-8")
        inventory.remove_crawlers(self.raw)
        self.assertFalse(worker.exists())
        self.assertFalse(script.exists())
        self.assertEqual((self.category / "article-123.txt").read_bytes(), self.original)
        manifest = next((self.root / ".crawler_trash").rglob("restore_manifest.json"))
        pairs = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(Path(item["backup"]).exists() for item in pairs))

    def test_corrupted_shard_blocks_resume(self):
        db = self.open_db()
        self.export(db)
        shard = next((self.output / "inventory").glob("*.parquet"))
        shard.write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError, "Shard bị thiếu/thay đổi"):
            self.export(db)


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
