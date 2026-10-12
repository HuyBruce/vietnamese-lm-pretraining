"""Bước 2: chuẩn hóa bảo thủ, giữ nguyên input; checkpoint theo shard.

python -B scripts/normalize_news.py
python -B scripts/normalize_news.py --workers 1 --max-shards 2
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
import hashlib
import html
from html.parser import HTMLParser
from html.entities import html5
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import unicodedata

sys.dont_write_bytecode = True
from prepare_news_inventory import atomic_json, file_sha, replace_retry, run_lock, utc_now

PROJECT = Path(__file__).resolve().parents[1]
VERSION = "news-normalize-v1"
ENTITY = re.compile(r"&(?:#[0-9]+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);")
TAGS = {"html", "body", "head", "title", "meta", "link", "div", "p", "br", "hr",
        "span", "a", "b", "i", "u", "s", "strong", "em", "small", "sup", "sub",
        "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote",
        "table", "tbody", "thead", "tr", "td", "th", "section", "article", "main",
        "script", "style", "noscript", "iframe", "figure", "figcaption", "img",
        "picture", "video", "audio", "source", "button", "nav", "footer"}
BLOCK = {"div", "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
         "ul", "ol", "li", "blockquote", "table", "tr", "section", "article", "main"}
DROP = {"head", "script", "style", "noscript", "iframe", "figure", "figcaption",
        "picture", "video", "audio", "nav", "footer"}
VOID = {"br", "hr", "img", "source", "meta", "link"}
HTML_HINT = re.compile(r"<!--|</?(?:" + "|".join(sorted(TAGS)) + r")(?:\s|/?>)", re.I)
NOISE_CLASSES = {"author", "article-author", "detail-author", "detail__author",
                 "figcaption", "image-caption", "photo-caption", "photocms_caption",
                 "advertisement", "social-share", "share-buttons"}

# Chỉ toàn dòng giao diện cố định. Không xóa những câu nhắc đến các từ này.
UI_LINES = {
    "trang chủ", "quay lại trang chủ", "chia sẻ", "chia sẻ bài viết", "sao chép liên kết",
    "sao chép đường dẫn", "gửi bình luận", "đăng nhập để bình luận", "xem thêm bình luận",
    "đọc thêm", "xem thêm", "tin liên quan", "bài viết liên quan", "quảng cáo",
    "advertisement", "click để xem thêm", "nhấn để xem thêm",
}
CAPTION = re.compile(r"^(?:ảnh|photo|chú thích ảnh|nguồn ảnh|video|clip|đồ họa)\s*[:：]\s*\S", re.I)
BYLINE = re.compile(r"^(?:tác giả|nguồn|người viết|biên tập|thực hiện)\s*[:：]\s*(.+)$", re.I)
SHARE = re.compile(r"^(?:chia sẻ (?:bài viết|bài báo)(?: này)? (?:lên|qua|trên)|"
                   r"theo dõi (?:báo|chúng tôi).{0,80}(?:trên|tại))\s+"
                   r"(?:facebook|zalo|google news|twitter|youtube)[.!]?$", re.I)
MENU_PART = re.compile(r"\s*[|»›]\s*")
SPACES = re.compile(r"[^\S\n]+")


def decode_entity(match):
    entity = match.group()
    # html.unescape có thể giải mã tiền tố '&not' trong '&notable;' sai ý nghĩa.
    if entity.startswith("&#") or entity[1:] in html5:
        return html.unescape(entity)
    return entity


class FragmentParser(HTMLParser):
    """Chỉ hiểu thẻ HTML đã biết; giữ <tên riêng> và dấu so sánh nguyên bản."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts = []
        self.stack = []
        self.removed = []
        self.discard = []
        self.reason = None
        self.tags = 0

    def emit(self, value):
        (self.discard if self.stack else self.parts).append(value)

    def handle_starttag(self, tag, attrs):
        if self.stack:
            if tag not in VOID:
                self.stack.append(tag)
            return
        if tag not in TAGS:
            self.parts.append(self.get_starttag_text())
            return
        self.tags += 1
        attributes = dict(attrs)
        marked = bool(set((attributes.get("class") or "").lower().split()) & NOISE_CLASSES)
        if tag in DROP or marked:
            if tag not in VOID:
                self.stack = [tag]
                self.reason = "html_" + ("marked_noise" if marked else tag)
            return
        if tag in BLOCK:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if self.stack:
            if tag in self.stack:
                index = len(self.stack) - 1 - self.stack[::-1].index(tag)
                del self.stack[index:]
                if not self.stack:
                    self.finish_discard()
                    self.parts.append("\n")
            return
        if tag not in TAGS:
            self.parts.append(f"</{tag}>")
        elif tag in BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_data(self, data):
        self.emit(data)

    def handle_entityref(self, name):
        self.emit("&" + name + ";")

    def handle_charref(self, name):
        self.emit("&#" + name + ";")

    def handle_comment(self, data):
        self.removed.append({"reason": "html_comment", "text": data})

    def finish_discard(self):
        if self.discard:
            self.removed.append({"reason": self.reason, "text": "".join(self.discard)})
        self.discard = []
        self.reason = None


def noise_reason(line, index, total):
    # Giữ tiêu đề kể cả trùng một nhãn giao diện.
    if index > 0 and line.casefold() in UI_LINES:
        return "ui_line"
    if SHARE.fullmatch(line):
        return "share_invitation"
    parts = MENU_PART.split(line)
    if len(parts) >= 3 and parts[0].casefold() == "trang chủ" and all(len(p) <= 45 for p in parts):
        return "breadcrumb_menu"
    if index > 0 and CAPTION.match(line) and len(line) <= 300:
        return "explicit_caption"
    match = BYLINE.match(line)
    # Byline ở đầu/cuối, ngắn, có nhãn và không giống một câu kể.
    if match and (index <= 2 or index >= total - 3):
        value = match.group(1)
        if len(value) <= 140 and len(value.split()) <= 18 and not re.search(r"[.!?;]", value):
            return "explicit_byline"
    return None


def normalize_text(text):
    """Trả về (text, audit). Không sửa nội dung bằng mô hình, không loại bài."""
    if not isinstance(text, str):
        raise ValueError("Trường text không phải chuỗi UTF-8")
    counts = Counter()
    removed, flags = [], []
    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        counts["unicode_nfc"] += 1
    cleaned = []
    nfc = nfc.replace("\r\n", "\n").replace("\r", "\n")
    for char in nfc:
        category = unicodedata.category(char)
        if char in "\n\t":
            cleaned.append(char)
        elif char in "\u2028\u2029":
            cleaned.append("\n")
            counts["line_separator"] += 1
        elif category == "Cc":
            cleaned.append(" ")
            counts["control_characters"] += 1
        elif category == "Cf" and char not in "\u200c\u200d":
            counts["invisible_characters"] += 1
        else:
            cleaned.append(char)
    value = "".join(cleaned)
    # Giữ ZWJ/ZWNJ: có thể thuộc emoji hoặc ngôn ngữ được trích dẫn.
    value, decoded_count = ENTITY.subn(decode_entity, value)
    if decoded_count:
        counts["html_entities"] = decoded_count
    if HTML_HINT.search(value):
        parser = FragmentParser()
        parser.feed(value)
        parser.close()
        if parser.stack:
            # HTML dở dang: phục hồi phần chưa đóng để tránh nuốt câu cuối.
            parser.parts.extend(parser.discard)
            flags.append("unclosed_html_preserved")
        value = "".join(parser.parts)
        removed.extend(parser.removed)
        counts["html_tags"] = parser.tags
    # Entity có thể vừa tạo ra ký tự điều khiển; làm sạch lần nữa, không decode đệ quy.
    value = "".join(" " if unicodedata.category(c) == "Cc" and c not in "\n\t"
                    else "" if unicodedata.category(c) == "Cf" and c not in "\u200c\u200d"
                    else c for c in value)
    value = unicodedata.normalize("NFC", value)
    lines = [SPACES.sub(" ", line).strip() for line in value.splitlines()]
    lines = [line for line in lines if line]
    kept = []
    for index, line in enumerate(lines):
        reason = noise_reason(line, index, len(lines))
        if reason:
            removed.append({"reason": reason, "text": line})
        else:
            kept.append(line)
    result = "\n".join(kept)
    if result != value:
        counts["layout_or_removed_lines"] += 1
    for event in removed:
        counts[event["reason"]] += 1
    if "\ufffd" in result:
        flags.append("replacement_character_present")
    if not result:
        flags.append("empty_after_normalization")
    if text and len(result) < len(text) * 0.5:
        flags.append("large_text_reduction_review")
    return result, {"changed": result != text, "counts": dict(counts),
                    "removed": removed, "flags": flags}


def load_snapshot(input_root):
    manifest = input_root / "manifests" / "inventory.sqlite"
    if not manifest.is_file():
        raise RuntimeError(f"Chưa có manifest bước 1: {manifest}")
    # Kết nối read-only, đóng ngay sau khi lấy danh sách shard đã commit.
    with closing(sqlite3.connect(manifest.as_uri() + "?mode=ro", uri=True, timeout=10)) as db:
        with db:
            meta = dict(db.execute("SELECT key,value FROM meta"))
            if meta.get("scan_complete") != "1":
                raise RuntimeError("Bước 1 chưa kiểm kê xong. Hãy chạy sau khi bước 1 hoàn tất.")
            states = dict(db.execute("SELECT state,COUNT(*) FROM files GROUP BY state"))
            if states.get("pending", 0):
                raise RuntimeError("Bước 1 còn file pending. Hãy đợi bước 1 hoàn tất.")
            shards = [dict(name=n, rows=r, sha256=h)
                      for n, r, h in db.execute("SELECT name,rows,sha256 FROM shards ORDER BY name")]
    if states.get("error", 0):
        print(f"Lưu ý: {states['error']:,} file lỗi ở bước 1 không có trong Parquet.", flush=True)
    for s in shards:
        if not re.fullmatch(r"part-\d+\.parquet", s["name"]):
            raise RuntimeError("Tên shard không hợp lệ trong manifest")
    if sum(s["rows"] for s in shards) != states.get("done", 0):
        raise RuntimeError("Tổng bản ghi/shard trong manifest bước 1 không khớp")
    return shards


def init_checkpoint(output, input_root, shards):
    db = sqlite3.connect(output / "manifests" / "normalize.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS completed(name TEXT PRIMARY KEY,output_sha TEXT,
      audit_sha TEXT,stats TEXT);''')
    config = {"input": str(input_root), "version": VERSION,
              "code_sha": file_sha(Path(__file__)),
              "snapshot_sha": hashlib.sha256(json.dumps(shards, sort_keys=True).encode()).hexdigest()}
    expected = json.dumps(config, sort_keys=True)
    old = db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
    if old and old[0] != expected:
        db.close()
        raise RuntimeError("Input/quy tắc khác checkpoint cũ. Chọn --output mới để không trộn phiên bản.")
    with db:
        db.execute("INSERT OR IGNORE INTO meta VALUES ('config',?)", (expected,))
    return db


def normalize_shard(input_root, output, spec, batch_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    name = spec["name"]
    source = input_root / "inventory" / name
    if file_sha(source) != spec["sha256"]:
        raise RuntimeError(f"Input shard thay đổi: {name}")
    destination = output / name
    audit_path = output / "audit" / (name + ".jsonl")
    temp, audit_temp = destination.with_suffix(".parquet.tmp"), audit_path.with_suffix(".jsonl.tmp")
    stats = Counter(rows=0, changed=0, removed_items=0, empty=0, input_chars=0, output_chars=0)
    rule_totals, flag_totals = Counter(), Counter()
    try:
        with pq.ParquetFile(source) as parquet:
            base_schema = parquet.schema_arrow
            extras = pa.schema([
                ("normalized_sha256", pa.string()), ("normalized_byte_count", pa.int64()),
                ("normalized_char_count", pa.int64()), ("normalized_line_count", pa.int64()),
                ("normalization_changed", pa.bool_()), ("normalization_flags", pa.list_(pa.string())),
                ("normalization_version", pa.string())])
            if not {"id", "text", "content_sha256"}.issubset(set(base_schema.names)):
                raise RuntimeError(f"Thiếu cột bắt buộc trong {name}")
            if set(extras.names) & set(base_schema.names):
                raise RuntimeError("Input đã có cột normalized; không chuẩn hóa chồng lên output cũ")
            schema = pa.schema(list(base_schema) + list(extras))
            with pq.ParquetWriter(temp, schema, compression="zstd") as writer, audit_temp.open(
                    "w", encoding="utf-8", newline="\n") as audit_file:
                for batch in parquet.iter_batches(batch_size=batch_rows, use_threads=False):
                    rows = batch.to_pylist()
                    for row in rows:
                        original = row["text"]
                        if not isinstance(original, str):
                            raise ValueError(f"text không phải chuỗi: {row['id']}")
                        text, audit = normalize_text(original)
                        encoded = text.encode("utf-8")
                        digest = hashlib.sha256(encoded).hexdigest()
                        row.update(text=text, normalized_sha256=digest,
                                   normalized_byte_count=len(encoded), normalized_char_count=len(text),
                                   normalized_line_count=len(text.splitlines()),
                                   normalization_changed=audit["changed"],
                                   normalization_flags=audit["flags"], normalization_version=VERSION)
                        stats.update(rows=1, changed=int(audit["changed"]),
                                     removed_items=len(audit["removed"]), empty=int(not text),
                                     input_chars=len(original), output_chars=len(text))
                        rule_totals.update(audit["counts"])
                        flag_totals.update(audit["flags"])
                        if audit["changed"] or audit["flags"]:
                            audit_file.write(json.dumps(dict(id=row["id"],
                                source_path=row.get("source_path"),
                                original_sha256=row["content_sha256"], normalized_sha256=digest,
                                **audit), ensure_ascii=False) + "\n")
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                audit_file.flush()
                os.fsync(audit_file.fileno())
        if stats["rows"] != spec["rows"]:
            raise RuntimeError(f"Số bài trong {name} không khớp manifest")
        with temp.open("r+b") as handle:
            os.fsync(handle.fileno())
        replace_retry(temp, destination)
        replace_retry(audit_temp, audit_path)
        return dict(name=name, output_sha=file_sha(destination), audit_sha=file_sha(audit_path),
                    stats=dict(stats), rules=dict(rule_totals), flags=dict(flag_totals))
    finally:
        for path in (temp, audit_temp):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def completed_shards(db, output):
    done = set()
    for name, digest, audit_digest in db.execute("SELECT name,output_sha,audit_sha FROM completed"):
        for path, expected in [(output / name, digest),
                               (output / "audit" / (name + ".jsonl"), audit_digest)]:
            if not path.is_file() or file_sha(path) != expected:
                raise RuntimeError(f"Output đã hoàn tất bị thiếu/thay đổi: {path}")
        done.add(name)
    return done


def commit_result(db, result):
    with db:
        db.execute("INSERT INTO completed VALUES (?,?,?,?)", (
            result["name"], result["output_sha"], result["audit_sha"],
            json.dumps(result, ensure_ascii=False)))


def report(db, output, total):
    stats, rules, flags = Counter(), Counter(), Counter()
    shard_list = []
    for name, digest, data in db.execute("SELECT name,output_sha,stats FROM completed ORDER BY name"):
        record = json.loads(data)
        stats.update(record["stats"])
        rules.update(record["rules"])
        flags.update(record["flags"])
        shard_list.append(dict(name=name, sha256=digest, rows=record["stats"]["rows"]))
    atomic_json(output / "manifests" / "shards.json", shard_list)
    atomic_json(output / "reports" / "normalization_report.json", dict(
        version=VERSION, updated_at=utc_now(), completed_shards=len(shard_list),
        total_shards=total, complete=len(shard_list) == total,
        stats=dict(stats), rule_counts=dict(rules), flags=dict(flags)))


def worker_init():
    import signal
    import pyarrow as pa
    # Parent xử lý Ctrl+C; worker hoàn tất tối đa shard hiện tại rồi được đóng.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pa.set_cpu_count(1)


def run(input_root, output, workers=2, batch_rows=128, max_shards=0):
    shards = load_snapshot(input_root)
    if (output.exists() and not (output / "manifests" / "normalize.sqlite").exists()
            and any(output.glob("*.parquet"))):
        raise RuntimeError("Đầu ra có Parquet không thuộc checkpoint bước 2. Hãy chọn --output khác.")
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        for name in ("manifests", "audit", "reports"):
            (output / name).mkdir(exist_ok=True)
        with closing(init_checkpoint(output, input_root, shards)) as db:
            done = completed_shards(db, output)
            pending = [spec for spec in shards if spec["name"] not in done]
            if max_shards:
                pending = pending[:max_shards]
            print(f"Khôi phục {len(done):,}/{len(shards):,} shard; lượt này xử lý {len(pending):,} shard.", flush=True)
            finished = len(done)

            def accept(result):
                nonlocal finished
                commit_result(db, result)
                finished += 1
                print(f"[{finished:,}/{len(shards):,}] {result['name']} | "
                      f"bài {result['stats']['rows']:,} | đổi {result['stats']['changed']:,} | "
                      f"bỏ {result['stats']['removed_items']:,} mục thừa | checkpoint đã lưu", flush=True)

            try:
                if workers == 1:
                    for spec in pending:
                        accept(normalize_shard(input_root, output, spec, batch_rows))
                else:
                    # Chỉ tối đa workers shard đang chờ/chạy, không nạp toàn corpus vào RAM.
                    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
                        iterator = iter(pending)
                        active = {}
                        for _ in range(workers):
                            spec = next(iterator, None)
                            if spec:
                                active[pool.submit(normalize_shard, input_root, output, spec, batch_rows)] = spec
                        last_update = time.monotonic()
                        while active:
                            ready, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                            for future in ready:
                                active.pop(future)
                                accept(future.result())
                                spec = next(iterator, None)
                                if spec:
                                    active[pool.submit(normalize_shard, input_root, output, spec, batch_rows)] = spec
                            if time.monotonic() - last_update >= 15:
                                print("Đang xử lý: " + ", ".join(s["name"] for s in active.values()), flush=True)
                                last_update = time.monotonic()
            finally:
                report(db, output, len(shards))
    print(f"Kết quả: {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "DataProcessed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-rows", type=int, default=128)
    parser.add_argument("--max-shards", type=int, default=0, help="0 = hết; số dương để thử vài shard")
    args = parser.parse_args(argv)
    input_root = args.input.resolve()
    output = (args.output or input_root / "normalized").resolve()
    protected = [input_root / name for name in ("inventory", "manifests", "reports")]
    if (output == input_root or input_root.is_relative_to(output)
            or any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected)):
        raise RuntimeError("Đầu ra không được trùng hoặc chứa dữ liệu/manifest bước 1.")
    if not 1 <= args.workers <= 16 or args.batch_rows <= 0 or args.max_shards < 0:
        raise RuntimeError("workers: 1–16; batch-rows > 0; max-shards >= 0")
    run(input_root, output, args.workers, args.batch_rows, args.max_shards)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng. Chạy lại cùng lệnh để tiếp tục shard chưa commit.", flush=True)
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
