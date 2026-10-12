"""Bước 1: kiểm kê TXT báo chí, giữ nguyên nội dung, xuất Parquet có metadata.

python -B scripts/prepare_news_inventory.py --preview
python -B scripts/prepare_news_inventory.py --remove-crawlers

Chỉ --remove-crawlers mới dọn crawler: chuyển code, worker và cache crawler
sang .crawler_trash để có thể phục hồi. Không dừng tiến trình đang chạy.
Các shard hoàn tất được checkpoint bằng SQLite; Ctrl+C rồi chạy lại cùng lệnh.
Đây là snapshot: muốn nhập dữ liệu mới sau khi hoàn thành, dùng --output khác.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.dont_write_bytecode = True

PROJECT = Path(__file__).resolve().parents[1]
SOURCES = {
    "bao24h": ["Báo 24H"],
    "baomoi": ["Báo Mới"],
    "baotintuc": ["Báo Tin Tức"],
    "baovanhoa": ["Báo Văn Hóa"],
    "chinhphu": ["Báo Chính Phủ"],
    "dantri": ["Báo Dân Trí"],
    "thanhnien": ["Báo Thanh Niên"],
    "tuoitre": ["Báo Tuổi Trẻ"],
    "vietnamnet": ["Báo VietnamNet"],
    "vnexpress": ["Báo vnexpress", "vnexpress"],
    "vov": ["Báo VOV"],
    "vtcnews": ["Báo VTC News"],
}
SKIP_DIRS = {"logs", "status", "checkpoints", "link", "links", "cache"}
VERSION = "news-inventory-v1"


def say(message):
    print(message, flush=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    replace_retry(temp, path)


def replace_retry(temp, path):
    for attempt in range(9):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(min(0.1 * 2**attempt, 1))


def unsafe_link(path):
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def check_crawlers_stopped(raw):
    import psutil

    active = []
    prefixes = tuple(SOURCES)
    for process in psutil.process_iter(["pid", "name", "cmdline"], ad_value=None):
        if "python" not in (process.info["name"] or "").lower():
            continue
        cmd = process.info["cmdline"]
        if cmd is None:
            raise RuntimeError(f"Không đọc được lệnh Python PID {process.pid}; hãy kiểm tra trước khi chạy.")
        for arg in cmd:
            if not arg.lower().endswith(".py"):
                continue
            name = Path(arg).name.lower()
            worker = any(f"{prefix}_workers" in arg.lower() for prefix in prefixes)
            script = any(name == f"{prefix}_{suffix}.py" for prefix in prefixes
                         for suffix in ("to_txt", "parallel"))
            if script or worker:
                active.append(f"PID {process.pid}: {arg}")
    if active:
        raise RuntimeError("Hãy dừng crawler trước để snapshot ổn định:\n" + "\n".join(active))


def crawler_targets(raw):
    targets = []
    for prefix in SOURCES:
        for suffix in ("to_txt.py", "parallel.py", "workers"):
            path = raw / f"{prefix}_{suffix}"
            if path.exists():
                targets.append(path)
    cache = raw / "__pycache__"
    if cache.is_dir() and not unsafe_link(cache):
        targets.extend(p for p in cache.iterdir() if p.is_file()
                       and any(p.name.startswith(f"{prefix}_") for prefix in SOURCES)
                       and p.suffix == ".pyc")
    for path in targets:
        if unsafe_link(path) or not path.resolve().is_relative_to(raw):
            raise RuntimeError(f"Từ chối dọn đường dẫn liên kết/ngoài DataRaw: {path}")
    return targets


def remove_crawlers(raw):
    targets = crawler_targets(raw)
    if not targets:
        say("Không còn code/worker crawler cần dọn.")
        return
    # Chuyển trong cùng ổ đĩa; dữ liệu bài báo không nằm trong targets.
    trash = raw.parent / ".crawler_trash" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    trash.mkdir(parents=True, exist_ok=False)
    for path in targets:
        # Không đi theo junction/symlink trong cây worker khi di chuyển.
        if path.is_dir():
            for directory, dirs, files in os.walk(path, followlinks=False):
                if any(unsafe_link(Path(directory) / n) for n in dirs + files):
                    raise RuntimeError(f"Có liên kết trong {path}; dừng dọn để kiểm tra.")
    mapping = [{"original": str(p), "backup": str(trash / p.relative_to(raw))}
               for p in targets]
    atomic_json(trash / "restore_manifest.json", mapping)
    for item in mapping:
        source, dest = Path(item["original"]), Path(item["backup"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        say(f"Đã dọn: {source.relative_to(raw)}")
    say(f"Bản phục hồi crawler: {trash}")


def source_roots(raw):
    aliases = {alias.casefold(): source for source, names in SOURCES.items() for alias in names}
    result = []
    unknown = []
    for path in sorted(raw.iterdir()):
        if not path.is_dir() or unsafe_link(path):
            continue
        if path.name.startswith(("_", ".")) or path.name.endswith("_workers"):
            continue
        if path.name.casefold() in SKIP_DIRS:
            continue
        source = aliases.get(path.name.casefold())
        if source:
            result.append((source, path))
        else:
            unknown.append(path.name)
    if unknown:
        raise RuntimeError("Có thư mục dữ liệu chưa được ánh xạ nguồn. Bổ sung SOURCES trong script: "
                           + ", ".join(unknown))
    if not result:
        raise RuntimeError(f"Không tìm thấy thư mục báo trong {raw}")
    return result


def article_files(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(("_", "."))
                         and d.casefold() not in SKIP_DIRS and not d.endswith("_workers")
                         and not unsafe_link(Path(directory) / d))
        for filename in sorted(files):
            path = Path(directory) / filename
            # Bài báo phải có ít nhất một thư mục chuyên mục, tên không phải log nội bộ.
            if (filename.lower().endswith(".txt") and not filename.startswith(("_", "."))
                    and Path(directory) != root and not unsafe_link(path)):
                yield path


@contextmanager
def run_lock(output):
    handle = (output / "inventory.lock").open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError("Một tiến trình đang xử lý cùng thư mục đầu ra.") from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def database(output, raw):
    path = output / "manifests" / "inventory.sqlite"
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript('''
      CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS files (
        path TEXT PRIMARY KEY, id TEXT UNIQUE, source TEXT, category TEXT,
        subcategory TEXT, size INTEGER, mtime_ns INTEGER,
        state TEXT DEFAULT 'pending', shard TEXT, sha256 TEXT,
        chars INTEGER, lines INTEGER, error TEXT
      );
      CREATE INDEX IF NOT EXISTS files_state_path ON files(state,path);
      CREATE TABLE IF NOT EXISTS shards (name TEXT PRIMARY KEY, rows INTEGER, sha256 TEXT);
    ''')
    meta = dict(db.execute("SELECT key,value FROM meta"))
    if meta and (meta.get("raw_root") != str(raw) or meta.get("version") != VERSION):
        raise RuntimeError("Manifest thuộc nguồn/phiên bản khác. Hãy chọn --output mới.")
    if not meta:
        with db:
            db.executemany("INSERT INTO meta VALUES (?,?)", [
                ("raw_root", str(raw)), ("version", VERSION), ("created_at", utc_now()),
                ("scan_complete", "0")])
    return db


def scan(db, raw, roots):
    if db.execute("SELECT value FROM meta WHERE key='scan_complete'").fetchone()[0] == "1":
        say("Khôi phục danh sách file từ manifest snapshot.")
        return
    count = 0
    for source, root in roots:
        say(f"Kiểm kê {root.name}...")
        for path in article_files(root):
            stat = path.stat()
            relative = path.relative_to(raw).as_posix()
            parts = path.relative_to(root).parts
            doc_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()
            db.execute('''INSERT OR REPLACE INTO files
                (path,id,source,category,subcategory,size,mtime_ns) VALUES (?,?,?,?,?,?,?)''',
                (relative, doc_id, source, parts[0], "/".join(parts[1:-1]) or None,
                 stat.st_size, stat.st_mtime_ns))
            count += 1
            if count % 5000 == 0:
                db.commit()
                say(f"Đã kiểm kê {count:,} file TXT...")
    db.execute("UPDATE meta SET value='1' WHERE key='scan_complete'")
    db.commit()
    say(f"Danh sách snapshot: {count:,} file.")


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_parquet(db, raw, output, shard_rows, shard_bytes, max_file_bytes):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("id", pa.string()), ("source", pa.string()), ("category", pa.string()),
        ("subcategory", pa.string()), ("source_path", pa.string()),
        ("url", pa.string()), ("published_at", pa.string()),
        ("text", pa.string()), ("content_sha256", pa.string()),
        ("byte_count", pa.int64()), ("char_count", pa.int64()), ("line_count", pa.int64()),
    ])
    # Trước khi resume phải xác minh shard đã cam kết còn nguyên.
    for name, _, digest in db.execute("SELECT name,rows,sha256 FROM shards ORDER BY name"):
        path = output / "inventory" / name
        if not path.is_file() or file_sha(path) != digest:
            raise RuntimeError(f"Shard bị thiếu/thay đổi: {path}. Phục hồi hoặc dùng --output mới.")

    total = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    completed = db.execute("SELECT COUNT(*) FROM files WHERE state!='pending'").fetchone()[0]
    shard_index = db.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
    while True:
        candidates = db.execute('''SELECT path,id,source,category,subcategory,size,mtime_ns
            FROM files WHERE state='pending' ORDER BY path LIMIT ?''', (shard_rows,)).fetchall()
        if not candidates:
            break
        rows, updates = [], []
        accumulated = 0
        for relative, doc_id, source, category, subcategory, size, mtime in candidates:
            if rows and accumulated + size > shard_bytes:
                break
            path = raw / relative
            try:
                before = path.stat()
                if before.st_size != size or before.st_mtime_ns != mtime:
                    raise ValueError("File thay đổi sau khi lập snapshot")
                if size > max_file_bytes:
                    raise ValueError(f"File vượt giới hạn bộ nhớ {max_file_bytes} byte; cần kiểm tra")
                data = path.read_bytes()
                after = path.stat()
                if after.st_size != size or after.st_mtime_ns != mtime or len(data) != size:
                    raise ValueError("File bị thay đổi trong khi đọc")
                # Strict UTF-8: không âm thầm thay ký tự lỗi. Giữ BOM/newline/nội dung nguyên gốc.
                text = data.decode("utf-8")
                digest = hashlib.sha256(data).hexdigest()
                lines = len(text.splitlines())
                rows.append(dict(id=doc_id, source=source, category=category,
                                 subcategory=subcategory, source_path=relative,
                                 url=None, published_at=None, text=text,
                                 content_sha256=digest, byte_count=size,
                                 char_count=len(text), line_count=lines))
                updates.append(("done", digest, len(text), lines, None, relative))
                accumulated += size
            except (OSError, UnicodeError, ValueError) as exc:
                updates.append(("error", None, None, None, str(exc), relative))
        name = None
        if rows:
            name = f"part-{shard_index:06d}.parquet"
            dest = output / "inventory" / name
            temp = dest.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), temp, compression="zstd")
            with temp.open("r+b") as handle:
                os.fsync(handle.fileno())
            replace_retry(temp, dest)
            digest = file_sha(dest)
        # File shard hoàn tất trước khi commit manifest. Khi bị ngắt ở giữa,
        # chạy lại sẽ ghi đè cùng shard chưa commit, không thêm bản ghi trùng.
        with db:
            if rows:
                db.execute("INSERT INTO shards VALUES (?,?,?)", (name, len(rows), digest))
            for state, sha, chars, lines, error, relative in updates:
                db.execute('''UPDATE files SET state=?,sha256=?,chars=?,lines=?,error=?,shard=?
                           WHERE path=?''',
                           (state, sha, chars, lines, error, name if state == "done" else None, relative))
        completed += len(updates)
        if rows:
            shard_index += 1
        say(f"Nhập {completed:,}/{total:,} file | {shard_index:,} shard | checkpoint đã lưu")


def write_report(db, raw, output):
    counts = dict(db.execute("SELECT state,COUNT(*) FROM files GROUP BY state"))
    totals = db.execute('''SELECT COALESCE(SUM(size),0), COALESCE(SUM(chars),0),
                          COALESCE(SUM(lines),0) FROM files WHERE state='done' ''').fetchone()
    groups = []
    for source, category, subcat, count, size, errors in db.execute('''
        SELECT source,category,subcategory,COUNT(*),SUM(size),
        SUM(CASE WHEN state='error' THEN 1 ELSE 0 END)
        FROM files GROUP BY source,category,subcategory ORDER BY source,category,subcategory'''):
        groups.append(dict(source=source, category=category, subcategory=subcat,
                           files=count, bytes=size, errors=errors))
    report = dict(version=VERSION, created_at=utc_now(), raw_root=str(raw),
                  output=str(output), counts=counts, exported_bytes=totals[0],
                  exported_characters=totals[1], exported_lines=totals[2],
                  sources=len({g["source"] for g in groups}), groups=groups,
                  notes=["Một bản ghi là một bài TXT; text giữ nguyên, chưa lọc hoặc loại trùng.",
                         "url và published_at là null vì không có metadata đáng tin cậy trong TXT.",
                         "id dựa vào đường dẫn tương đối; content_sha256 dựa vào byte gốc.",
                         "File lỗi nằm trong manifest, không được âm thầm bỏ qua.",
                         "Số dòng không được coi là số câu đã xác minh; chưa tính token."])
    atomic_json(output / "reports" / "inventory_report.json", report)
    shard_list = [dict(name=n, rows=r, sha256=h)
                  for n, r, h in db.execute("SELECT name,rows,sha256 FROM shards ORDER BY name")]
    atomic_json(output / "manifests" / "shards.json", shard_list)
    error_path = output / "reports" / "read_errors.jsonl"
    with error_path.open("w", encoding="utf-8") as handle:
        for path, error in db.execute("SELECT path,error FROM files WHERE state='error' ORDER BY path"):
            handle.write(json.dumps(dict(source_path=path, error=error), ensure_ascii=False) + "\n")
    say(f"Kết quả: {counts.get('done', 0):,} bài đã nhập; {counts.get('error', 0):,} lỗi.")
    say(f"Báo cáo: {output / 'reports' / 'inventory_report.json'}")
    return counts.get("error", 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT / "DataRaw")
    parser.add_argument("--output", type=Path, default=PROJECT / "DataProcessed")
    parser.add_argument("--preview", action="store_true", help="Chỉ hiện mục tiêu, không ghi/xóa dữ liệu")
    parser.add_argument("--remove-crawlers", action="store_true", help="Chuyển code/worker crawler sang .crawler_trash")
    parser.add_argument("--retry-errors", action="store_true", help="Đọc lại file lỗi; cần file khớp snapshot ban đầu")
    parser.add_argument("--shard-rows", type=int, default=2000)
    parser.add_argument("--shard-mb", type=int, default=32)
    parser.add_argument("--max-file-mb", type=int, default=32)
    args = parser.parse_args(argv)
    raw, output = args.raw.resolve(), args.output.resolve()
    if not raw.is_dir():
        raise RuntimeError(f"Thư mục đầu vào không tồn tại: {raw}")
    if raw == output or output.is_relative_to(raw) or raw.is_relative_to(output):
        raise RuntimeError("Đầu ra và DataRaw phải là hai cây thư mục tách biệt.")
    if min(args.shard_rows, args.shard_mb, args.max_file_mb) <= 0:
        raise RuntimeError("Kích thước shard và file phải lớn hơn 0.")
    roots = source_roots(raw)
    if args.preview:
        say("Nguồn dữ liệu sẽ nhập:")
        for source, path in roots:
            say(f"  {source}: {path}")
        say("Code/worker sẽ dọn nếu dùng --remove-crawlers:")
        for path in crawler_targets(raw):
            say(f"  {path}")
        say(f"Đầu ra: {output}")
        return 0
    try:
        import pyarrow  # noqa: F401
        import psutil  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Cài thư viện: python -m pip install -r scripts/requirements.txt") from exc
    check_crawlers_stopped(raw)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        for name in ("inventory", "manifests", "reports"):
            (output / name).mkdir(exist_ok=True)
        db = database(output, raw)
        try:
            if args.remove_crawlers:
                remove_crawlers(raw)
            scan(db, raw, roots)
            if args.retry_errors:
                with db:
                    db.execute("UPDATE files SET state='pending',error=NULL WHERE state='error'")
            export_parquet(db, raw, output, args.shard_rows, args.shard_mb * 1024**2,
                           args.max_file_mb * 1024**2)
            errors = write_report(db, raw, output)
        finally:
            db.close()
    return 2 if errors else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        say("\nĐã dừng. Chạy lại cùng lệnh để tiếp tục từ shard hoàn tất gần nhất.")
        raise SystemExit(130)
    except (RuntimeError, OSError, sqlite3.Error) as exc:
        say(f"Lỗi: {exc}")
        raise SystemExit(1)
