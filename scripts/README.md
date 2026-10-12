# Bước 1: kiểm kê dữ liệu báo thô

Sau khi bước 1 hoàn tất, xem [hướng dẫn bước 2](NORMALIZE.md) để chạy
`normalize_news.py`, chuẩn hóa văn bản có checkpoint và audit.

Chạy từ thư mục project chứa `DataRaw`:

```powershell
python -m pip install -r scripts\requirements.txt
python -B scripts\prepare_news_inventory.py --preview
python -B scripts\prepare_news_inventory.py --remove-crawlers
```

`--preview` chỉ xem các nguồn và mục tiêu sẽ dọn. Lệnh cuối dọn code/worker
crawler rồi nhập dữ liệu. Script từ chối chạy khi phát hiện crawler Python còn
hoạt động. Hãy tự dừng crawler trước; script không kết thúc process.

Nếu chỉ muốn nhập dữ liệu, không dọn crawler:

```powershell
python -B scripts\prepare_news_inventory.py
```

Code crawler, toàn bộ thư mục worker (bao gồm checkpoint/status/log của crawler)
và bytecode tương ứng được chuyển vào `.crawler_trash/<thời-gian>/`.
`restore_manifest.json` lưu đường dẫn gốc và đường dẫn phục hồi từng mục.
Đây là bản phục hồi trên ổ đĩa, không phải Windows Recycle Bin; vẫn chiếm dung lượng.
Các bài TXT và log nằm bên trong thư mục báo được giữ nguyên.

## Đầu ra

```text
DataProcessed/
  inventory/part-000000.parquet
  manifests/inventory.sqlite
  manifests/shards.json
  reports/inventory_report.json
  reports/read_errors.jsonl
```

Mỗi record là một file bài báo, gồm `id`, `source`, `category`, `subcategory`,
`source_path`, `url`, `published_at`, `text`, `content_sha256`, `byte_count`,
`char_count`, `line_count`. ID ổn định theo đường dẫn tương đối trong DataRaw;
hash nội dung tính trên byte gốc. Không lọc/loại trùng, không xóa dòng cuối, không
sửa dấu/khoảng trắng/newline. File rỗng vẫn được kiểm kê và nhập.

URL và ngày đăng để null vì TXT không có metadata đáng tin cậy. Không suy đoán
URL từ tên file hoặc lấy ngày tạo file làm ngày đăng. Số dòng chưa phải số câu
đã xác minh; chưa tính token khi chưa chọn tokenizer.

Script chỉ nhận thư mục báo được khai báo trong `SOURCES`, các file `.txt` bên
trong ít nhất một thư mục chuyên mục. Bỏ các cây `_...`, `....`, worker, cache,
logs/status/checkpoints/link; không đi theo symlink/junction. Nếu có thư mục nguồn
không nhận diện, script dừng và yêu cầu bổ sung `SOURCES` thay vì bỏ qua âm thầm.

Mỗi shard tối đa mặc định 2.000 bài hoặc khoảng 32 MiB văn bản đầu vào. Đây không
phải giới hạn cứng RAM vì chuỗi Unicode và Arrow có overhead. File trên 32 MiB,
file không đọc được UTF-8 và file thay đổi sau khi lập snapshot được ghi lỗi.
Xem `read_errors.jsonl` và trạng thái `error` trong SQLite; exit code 2 báo có lỗi.

## Dừng và chạy tiếp

Nhấn Ctrl+C, sau đó chạy lại cùng lệnh. Script giữ các shard đã commit, xác minh
hash của chúng và tiếp tục phần còn thiếu. Shard chưa commit được tạo lại;
TXT nguồn không bị sửa. Bước kiểm kê danh sách có thể phải quét lại nếu dừng
trước khi lập xong danh sách. Đóng đột ngột ứng dụng vẫn có thể để lại `.tmp`;
file tạm không được dùng làm dữ liệu hoàn tất.

Manifest đại diện cho một snapshot. Giữ DataRaw ổn định trong suốt lần xử lý.
Sau khi snapshot hoàn thành, chạy lại không tự nhập thêm file mới. Để tạo snapshot
mới từ corpus đã thay đổi, dùng thư mục đầu ra mới:

```powershell
python -B scripts\prepare_news_inventory.py --output DataProcessed_v2
```

Đọc lại lỗi I/O tạm thời trên file vẫn khớp snapshot:

```powershell
python -B scripts\prepare_news_inventory.py --retry-errors
```

Thông số tùy chọn: `--raw`, `--output`, `--shard-rows`, `--shard-mb`,
`--max-file-mb`. Không dùng chung output cho nhiều snapshot. Một khóa OS ngăn hai
tiến trình cùng ghi một output; file `inventory.lock` còn lại sau khi thoát là
bình thường, khóa thực tế tự nhả theo process.

## Kiểm thử dữ liệu giả

```powershell
python -B scripts\test_prepare_news_inventory.py
```

Các test dùng thư mục tạm tự dọn, bao gồm giữ nguyên text, resume không trùng,
ngắt giữa shard và commit, báo lỗi UTF-8/file thay đổi, hash shard và phục hồi crawler.
