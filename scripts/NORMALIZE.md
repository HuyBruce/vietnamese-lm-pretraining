# Bước 2: chuẩn hóa văn bản báo chí

Để tiếp tục đến dữ liệu token sẵn sàng pre-train, xem [PRETRAIN.md](PRETRAIN.md).

Chạy sau khi bước 1 đã hoàn tất. Không cần thư viện mới ngoài
`scripts/requirements.txt` của bước 1.

Từ thư mục project:

```powershell
python -B scripts\normalize_news.py
```

Mặc định đọc `DataProcessed/manifests/inventory.sqlite` ở chế độ read-only và
các shard đã commit trong `DataProcessed/inventory`. Script từ chối chạy nếu
bước 1 còn pending. File lỗi của bước 1 không có Parquet sẽ được thông báo.
Không mở/sửa các TXT trong DataRaw.

## Kiểm tra một lượt nhỏ trước

```powershell
python -B scripts\normalize_news.py --workers 1 --max-shards 2
```

Xem báo cáo và audit, rồi chạy toàn bộ với lệnh mặc định. Hai shard đã hoàn tất
sẽ được bỏ qua. `--max-shards` là số shard mới xử lý trong lượt chạy, không phải
giới hạn toàn bộ dataset.

Máy đang chạy công việc khác hoặc ít RAM có thể dùng `--workers 1`.
Mặc định 2 process, mỗi process xử lý 1 shard và đọc từng batch 128 bài.
Giảm thêm bộ nhớ bằng `--batch-rows 32`. Không nạp 4 triệu bài vào RAM cùng lúc.

## Kết quả

```text
DataProcessed/normalized/
    part-000000.parquet
    part-000001.parquet
    audit/part-000000.parquet.jsonl
    manifests/normalize.sqlite
    manifests/shards.json
    reports/normalization_report.json
```

Đầu ra giữ nguyên số bài, ID, nguồn, đường dẫn và mọi metadata cũ.
`text` là văn bản sau chuẩn hóa. Các cột `content_sha256`, `byte_count`,
`char_count`, `line_count` từ bước 1 vẫn mô tả dữ liệu **gốc**.
Cột mới mô tả kết quả:

- `normalized_sha256`: SHA-256 của UTF-8 text mới.
- `normalized_byte_count`, `normalized_char_count`, `normalized_line_count`.
- `normalization_changed`, `normalization_flags`, `normalization_version`.

Ngay cả bản ghi rỗng cũng được giữ và đánh dấu để bước lọc chất lượng xử lý sau.
Chưa loại trùng, chưa chia train/test, chưa tokenize hoặc sửa ngữ pháp.

## Quy tắc chuẩn hóa

- Unicode NFC, xuống dòng LF; xóa BOM, soft hyphen, ký tự điều khiển và ký tự
  định dạng vô hình không cần thiết. ZWJ/ZWNJ được giữ vì có thể tạo emoji hoặc
  thuộc ngôn ngữ được trích dẫn.
- Gom khoảng trắng trong dòng, bỏ dòng trắng, bỏ khoảng trắng đầu/cuối dòng.
  Không tự tách/gộp câu. Newline cuối văn bản được loại bỏ nên nhiều bài có thể
  được đánh dấu `changed` chỉ vì thay đổi định dạng này.
- Giải mã HTML entity có dấu chấm phẩy, bỏ các thẻ HTML thông dụng.
  Giữ tên thẻ không nhận diện như `<Tên>` và dấu so sánh trong văn bản.
- Bỏ nội dung HTML có dấu hiệu cấu trúc rõ ràng: `script`, `style`, `figure`,
  `figcaption`, media, `nav`, `footer` hoặc class đánh dấu tác giả/chú thích/quảng cáo.
- Dòng giao diện phải khớp toàn dòng với danh sách cố định, hoặc mẫu chia sẻ/menu
  chặt chẽ. Dòng đầu khớp nhãn ngắn như “Quảng cáo” được giữ vì có thể là tiêu đề.
- Chú thích cần nhãn rõ ràng như `Ảnh:` và độ dài giới hạn. Byline cần nhãn như
  `Tác giả:`/`Nguồn:`, ngắn, nằm ở đầu/cuối và không giống câu kể có dấu kết câu.
- Không xóa tên người đứng riêng, không cắt dòng cuối mặc định, không xóa dòng chỉ
  vì có từ “ảnh”, “tác giả”, “quảng cáo”. Các mẫu chưa chắc chắn được giữ lại.

Đây là các quy tắc có chủ đích bảo thủ; không đảm bảo nhận diện mọi dạng HTML
hoặc tác giả của tất cả báo. Audit cho phép kiểm tra những gì đã bị loại.
Không dùng LLM, không bỏ dấu/stopword, không sửa số liệu hay đổi chữ hoa.

## Audit và chạy tiếp

Audit JSONL ghi ID, đường dẫn gốc, hash trước/sau, loại thay đổi, những phần bị
loại và flags. Muốn đối chiếu nguyên văn, tra ID trong shard bước 1 cùng tên.
Những thay đổi ký tự/định dạng được thống kê; nội dung bị xóa có chủ đích được
ghi trong `removed`. Flags như `large_text_reduction_review` hoặc
`unclosed_html_preserved` cần được xem lại trước bước 3.

Mỗi shard và audit được ghi atomic rồi mới commit checkpoint. Nếu ngắt giữa
file và commit, chạy lại sẽ tạo lại đúng shard đó. Các shard đã commit được
kiểm tra hash trước khi bỏ qua; không thêm bản ghi trùng vào kết quả.

Nhấn Ctrl+C và đợi script thoát (có thể đợi worker đang xử lý shard hiện tại),
sau đó chạy lại cùng lệnh. Không xóa checkpoint. File khóa còn trên đĩa sau khi
thoát không có nghĩa là process vẫn đang giữ khóa.

Nếu chỉnh code/quy tắc hoặc thay snapshot bước 1, chọn output mới:

```powershell
python -B scripts\normalize_news.py --output DataProcessed\normalized_v2
```

Checkpoint ghi hash code và snapshot để tránh vô tình trộn kết quả của các bộ
quy tắc khác nhau. Khi dùng dataset cho bước 3, kiểm tra `complete: true` trong
`reports/normalization_report.json`, dùng danh sách `manifests/shards.json`,
không đọc `.tmp` hay các shard chưa commit.

## Kiểm thử

```powershell
python -B scripts\test_normalize_news.py
```

Test dùng dữ liệu giả trong thư mục tạm tự dọn, bao gồm giữ nguyên câu cuối,
NFC, HTML, số/tên riêng/emoji, audit, bài rỗng, hai worker, resume và hash input.
