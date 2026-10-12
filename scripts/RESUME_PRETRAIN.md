# Tiến trình chuẩn bị pretrain

## Lượt chạy tiếp ngày 01/10/2026

Đã khởi chạy lại `finish` từ checkpoint cũ. Chưa coi dữ liệu sẵn sàng cho
đến khi manifest binary có `complete: true` và các kiểm tra cuối đạt.

- Giữ nguyên các script pipeline bị khóa hash, tokenizer 32K và context 4096.
- Đã kiểm tra checksum toàn bộ 128 Parquet sạch: không có sai lệch.
- Bộ test toàn dự án: 30/30 đạt.
- Công cụ kiểm tra bổ sung: `D:\AgentContentVietnamese\scripts\verify_pretrain_ready.py`.
- Snapshot lượt chạy tiếp: `D:\AgentContentVietnamese\DataProcessed\pretrain_ready\reports\resume_state.json` (có đường dẫn log, không phải trạng thái live).
- Sau khi pipeline hoàn tất, chạy công cụ kiểm tra trên để tạo `D:\AgentContentVietnamese\DataProcessed\pretrain_ready\reports\independent_verification.json`.
- Ghi chú tạm dừng bên dưới là lịch sử ngày 29/09, không phải trạng thái live.

## Lịch sử tạm dừng ngày 29/09/2026

Đã dừng theo yêu cầu người dùng.

- Output: DataProcessed/pretrain_ready
- Hoàn tất: lọc, exact/near dedup, benchmark decontamination, split, Parquet sạch, tokenizer 32K.
- Train: 2.957.076 bài; validation: 14.872 bài.
- Binary đã commit và xác minh checksum: 2/128 shard.
- Còn lại: tokenize, xuất binary và kiểm tra cuối. Chưa sẵn sàng pre-train.
- Trạng thái chi tiết: DataProcessed/pretrain_ready/reports/paused_state.json.

Chạy tiếp từ thư mục dự án:

```powershell
python -B -X utf8 "D:\AgentContentVietnamese\scripts\prepare_pretrain.py" --stage finish
```

Không sửa các script pipeline trước khi chạy tiếp vì checkpoint khóa hash code. Các file .tmp chưa commit sẽ được tạo lại; không cần xóa checkpoint.
