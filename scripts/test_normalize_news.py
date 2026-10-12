"""Kiểm thử dữ liệu giả và checkpoint bước 2; không chạy trên corpus thật."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq
import normalize_news as n


class TextTests(unittest.TestCase):
    def test_preserve_names_numbers_punctuation_and_last_sentence(self):
        text = "Hà Nội ngày 09/09/2026\nÔng Nguyễn Văn A có 1.234,56 đồng, tăng 5%.\nĐây là câu cuối bài."
        self.assertEqual(n.normalize_text(text)[0], text)

    def test_nfc_controls_spaces_and_no_blank_lines(self):
        text = "\ufeff" + unicodedata.normalize("NFD", "Việt Nam") + "\r\n  Có\t  dấu. \u200b\r\n\r\nDòng\x00cuối."
        normalized, audit = n.normalize_text(text)
        self.assertEqual(normalized, "Việt Nam\nCó dấu.\nDòng cuối.")
        self.assertTrue(audit["changed"])
        self.assertEqual(n.normalize_text(normalized)[0], normalized)

    def test_html_and_removed_content_are_audited(self):
        text = '<h1>Tiêu đề</h1><p>Nội <b>dung</b> &amp; số 10.</p><figure><img src="a"><figcaption>Ảnh này</figcaption></figure><div class="author">Nguyễn A</div><p>Câu cuối.</p>'
        result, audit = n.normalize_text(text)
        self.assertEqual(result, "Tiêu đề\nNội dung & số 10.\nCâu cuối.")
        removed = [event["text"] for event in audit["removed"]]
        self.assertIn("Ảnh này", removed)
        self.assertIn("Nguyễn A", removed)

    def test_plain_names_are_not_removed(self):
        text = "Tiêu đề\nNội dung.\nNguyễn Văn A"
        self.assertEqual(n.normalize_text(text)[0], text)

    def test_explicit_byline_and_caption(self):
        text = "Tiêu đề\nNội dung.\nẢnh: Nguyễn B\nCâu kết còn nguyên.\nTác giả: Nguyễn Văn A"
        result, audit = n.normalize_text(text)
        self.assertEqual(result, "Tiêu đề\nNội dung.\nCâu kết còn nguyên.")
        self.assertEqual(len(audit["removed"]), 2)

    def test_narrative_not_removed_for_containing_keywords(self):
        text = "Quảng cáo\nHọ chia sẻ bài viết về ảnh và tác giả.\nTác giả: đó là cách anh tự giới thiệu trong cuộc gặp."
        self.assertEqual(n.normalize_text(text)[0], text)

    def test_unknown_tags_math_and_emoji_preserved(self):
        text = "Tiêu đề\n2 < 3 và 5 > 4, <Tên> được giữ. 👩‍💻\n<p>Đoạn HTML.</p>"
        result, _ = n.normalize_text(text)
        self.assertIn("2 < 3 và 5 > 4, <Tên> được giữ. 👩‍💻", result)
        self.assertTrue(result.endswith("Đoạn HTML."))

    def test_unclosed_discard_does_not_swallow_ending(self):
        result, audit = n.normalize_text("Tiêu đề\n<figure>Câu cuối chưa chắc là chú thích.")
        self.assertIn("Câu cuối chưa chắc là chú thích.", result)
        self.assertIn("unclosed_html_preserved", audit["flags"])

    def test_entities_no_semicolon_not_corrupted(self):
        result, _ = n.normalize_text("Tiêu đề\nA &notable; &nbsp; B &#x200b; &#0; C")
        self.assertIn("&notable;", result)
        self.assertNotIn("\x00", result)
        self.assertNotIn("\u200b", result)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="normalize-news-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "normalized"
        for name in ("inventory", "manifests"):
            (self.root / name).mkdir()
        self.rows = []
        for index, text in enumerate(["Tiêu đề\r\nCâu cuối.\r\nTác giả: Nguyễn A", ""]):
            self.rows.append(dict(id=str(index), text=text, source="baotintuc",
                                 source_path=f"bao/muc/{index}.txt", published_at=None,
                                 content_sha256=hashlib.sha256(text.encode()).hexdigest()))
        self.specs = []
        for index, row in enumerate(self.rows):
            path = self.root / "inventory" / f"part-{index:06d}.parquet"
            pq.write_table(pa.Table.from_pylist([row]), path)
            self.specs.append(dict(name=path.name, rows=1, sha256=n.file_sha(path)))
        db = sqlite3.connect(self.root / "manifests" / "inventory.sqlite")
        db.executescript('''CREATE TABLE meta(key TEXT,value TEXT);
            INSERT INTO meta VALUES ('scan_complete','1');
            CREATE TABLE files(state TEXT); INSERT INTO files VALUES ('done'),('done');
            CREATE TABLE shards(name TEXT,rows INTEGER,sha256 TEXT);''')
        db.executemany("INSERT INTO shards VALUES (?,?,?)", [(s["name"],s["rows"],s["sha256"]) for s in self.specs])
        db.commit()
        db.close()

    def test_limited_run_then_resume_metadata_empty_and_audit(self):
        n.run(self.root, self.output, workers=1, max_shards=1)
        first = self.output / self.specs[0]["name"]
        original_hash = n.file_sha(first)
        n.run(self.root, self.output, workers=1)
        self.assertEqual(n.file_sha(first), original_hash)
        row = pq.read_table(first).to_pylist()[0]
        self.assertEqual(row["text"], "Tiêu đề\nCâu cuối.")
        self.assertEqual(row["content_sha256"], self.rows[0]["content_sha256"])
        self.assertEqual(row["id"], "0")
        empty = pq.read_table(self.output / self.specs[1]["name"]).to_pylist()[0]
        self.assertEqual(empty["text"], "")
        self.assertIn("empty_after_normalization", empty["normalization_flags"])
        report = json.loads((self.output / "reports" / "normalization_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["complete"])
        self.assertEqual(report["stats"]["rows"], 2)
        for spec in self.specs:
            self.assertEqual(n.file_sha(self.root / "inventory" / spec["name"]), spec["sha256"])

    def test_interrupted_commit_reprocesses_without_duplicates(self):
        with patch.object(n, "commit_result", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                n.run(self.root, self.output, workers=1)
        n.run(self.root, self.output, workers=1)
        self.assertEqual(len(list(self.output.glob("*.parquet"))), 2)
        report = json.loads((self.output / "reports" / "normalization_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["stats"]["rows"], 2)

    def test_hash_mismatch_stops(self):
        (self.root / "inventory" / self.specs[0]["name"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "Input shard thay đổi"):
            n.run(self.root, self.output, workers=1)

    def test_parallel_spawn(self):
        n.run(self.root, self.output, workers=2)
        self.assertEqual(len(list(self.output.glob("*.parquet"))), 2)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main()
