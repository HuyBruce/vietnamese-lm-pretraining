# Chuẩn bị dữ liệu pre-train từ đầu

Đầu vào là snapshot đã hoàn tất trong `DataProcessed/normalized`. Đầu ra mặc
định là `DataProcessed/pretrain_ready`. Context length 4096, seed 20260929,
validation xấp xỉ 0,5% số nhóm sau loại trùng. Không sửa DataRaw hoặc các shard
inventory/normalized.

## Chạy và tiếp tục

```powershell
python -m pip install -r scripts/requirements-pretrain.txt
python -B scripts/prepare_pretrain.py --workers 2
python -B scripts/pretrain_status.py
```

Tài nguyên Language ID và benchmark cần có trong `assets/` trước khi chạy.
Lần chạy hiện tại dùng fastText `lid.176.bin`, VMLU bản công khai từ
`ura-hcmut/vmlu_vi` tại revision `c13b946c3b8a1c41368346bcbd9a811ccebac8cc`,
và XQuAD tiếng Việt của Google DeepMind. `assets/benchmarks.json` lưu URL,
SHA-256 và phạm vi kiểm tra. Tải lại bằng `pretrain_assets.py --output
DataProcessed/pretrain_ready <URL JSON/JSONL/ZIP> ...`. Benchmark chỉ dùng
để phát hiện trùng, không được nhập vào tập train hoặc mẫu học tokenizer.

Nguồn tham khảo: [fastText Language ID](https://fasttext.cc/docs/en/language-identification.html),
[VMLU chính thức](https://github.com/ZaloAI-Jaist/VMLU),
[bản VMLU công khai được dùng](https://huggingface.co/datasets/ura-hcmut/vmlu_vi/tree/c13b946c3b8a1c41368346bcbd9a811ccebac8cc),
[XQuAD](https://github.com/google-deepmind/xquad),
[SentencePiece normalization](https://github.com/google/sentencepiece/blob/master/doc/normalization.md).
Bản VMLU được dùng là bản mirror công khai đã cố định revision; không mặc
định coi nó là toàn bộ bộ VMLU mới nhất trên website yêu cầu đăng nhập.

Có thể chạy riêng `--stage filter|exact|sketch|near|finish`. Chạy thử bằng
`--stage filter --max-shards 2`; đây chỉ là giới hạn số shard mới trong lượt
chạy. Chạy lại cùng lệnh để tiếp tục checkpoint đã commit. Không sửa bộ quy
tắc hoặc benchmark giữa một lần chạy; chọn output mới nếu thay đổi.

## Các bước xử lý

1. Xác minh SHA-256 input. Khôi phục chú thích và dòng nguồn có ý nghĩa từ
   inventory khi audit cho biết normalizer cũ đã xóa chúng. Các trường
   `normalized_*` vẫn mô tả kết quả bước cũ; `clean_*` mô tả text đầu ra mới.
2. Language ID tiếng Việt. Nếu toàn bài có score thấp, kiểm tra phần mở đầu
   để giữ các lịch thi đấu tiếng Việt chứa nhiều tên đội nước ngoài.
3. Lọc chất lượng: ít nhất 100 ký tự và 20 whitespace-token; alphabetic
   ratio >= 0,4; special ratio <= 0,3; URL ratio <= 0,3; duplicate-line ratio
   <= 0,6; unique-token ratio >= 0,1. Tiếng Việt cần score >= 0,5, hoặc phần
   đầu >= 0,9. Ngưỡng chọn sau xem mẫu 2.160 bài trong `assets/profile.json`.
   Bài lỗi mã hóa hoặc giảm nội dung lớn chưa giải thích được được tách riêng.
4. Che email, điện thoại di động Việt Nam, CMND/CCCD/số tài khoản có nhãn
   bằng `<EMAIL>`, `<PHONE>`, `<ID>`. Đây là regex cơ bản, không cam kết nhận
   diện mọi PII hoặc mọi kiểu số điện thoại. Metadata truy vết không đưa vào
   token train; bản raw và audit vẫn có thể chứa dữ liệu ban đầu.
5. Exact dedup bằng SHA-256 của NFC + lowercase + collapse whitespace;
   văn bản train vẫn giữ chữ hoa và dấu. Giữ bản chất lượng cao hơn, dài hơn,
   rồi chọn theo ID để có kết quả ổn định.
6. MinHash 64 phép hoán vị, 16 band x 4; dùng 5-gram từ phân tách theo
   ký tự chữ/số Unicode. Xác minh Jaccard >= 0,8 trên tập hash 64-bit trước
   khi gộp nhóm. Không cắt bớt bucket lớn. LSH là phương pháp xác suất,
   vẫn có thể bỏ sót cặp gần trùng không được truy xuất.
7. Kiểm tra benchmark bằng exact document/line hoặc 13-gram từ liên tiếp,
   bỏ dấu câu và chuẩn hóa chữ thường chỉ trong biểu diễn so khớp. Nếu một
   thành viên trùng benchmark, loại cả nhóm exact/near của nó. Đây là kiểm
   tra trùng từ vựng đối với các bộ đã khai báo, không phải bảo đảm mọi
   benchmark hay mọi cách diễn đạt lại đều được loại.
8. Chia tập sau loại trùng; mỗi nhóm chỉ giữ một đại diện. Shuffle bằng
   hash có seed và xuất Parquet theo split và 64 bucket.
9. Train SentencePiece BPE 32K/48K/64K trên mẫu lấy riêng từ TRAIN. So sánh
   trên mẫu TRAIN khác, không dùng validation để học tokenizer. Chọn vocab
   nhỏ nhất có tokens/character trong 3% kết quả tốt nhất; lưu các tỷ lệ
   và phân mảnh tên riêng, số, tiếng Việt và tiếng Anh.
10. Tokenize text, thêm EOS từng bài, nối stream, lưu uint16 (hoặc uint32
    nếu cần). Tokenizer identity + byte fallback giữ nguyên dấu và newline.
    Chỉ pad đuôi mỗi shard; các target padding mang giá trị -100.

Các tỷ lệ chất lượng mô tả text trước che PII; `clean_char_count`,
`clean_byte_count`, `clean_line_count`, `clean_sha256` mô tả text sau che PII.
Một dòng đầu vào được dùng như một paragraph cho các thống kê; đây không
phải phục dựng ranh giới paragraph gốc trước crawler.

## Đầu ra và điều kiện sẵn sàng

- `clean/manifest.json`: danh sách Parquet chứa text và metadata, split nằm
  trong đường dẫn partition `split=train|validation`.
- `rejected/`: ID, đường dẫn và lý do loại do ngôn ngữ/chất lượng.
- `decisions.parquet`: quyết định exact/near/benchmark, nhóm và ID đại diện.
- `tokenizer/tokenizer.model`: tokenizer được chọn.
- `tokenizer/tokenizer.vocab.jsonl`: danh sách token có escape newline để đọc
  và phân tích. File `.vocab` do SentencePiece tạo có thể chứa token xuống dòng,
  vì vậy không dùng nó như TSV; luôn nạp tokenizer bằng `.model`.
- `reports/tokenizer_comparison.json`: kết quả 32K/48K/64K.
- `binary/*.bin`: token stream little-endian; không có header trong file.
- `binary/*.documents.parquet`: ID và khoảng `[start,end)` của từng bài,
  gồm EOS tại `end-1`.
- `binary/manifest.json`: dtype, checksum, context, EOS/PAD và số sequence.
- `reports/verification.json`: kết quả kiểm tra dữ liệu đã xuất.

Chỉ sử dụng cho training khi `binary/manifest.json` có **`complete: true`**.
Giữ manifest cùng tokenizer và các file binary. Các file `.tmp` hoặc shard
không nằm trong manifest không phải dữ liệu đã hoàn tất.

## DataLoader

```python
import sys
sys.path.insert(0, "scripts")
import torch
from torch.utils.data import DataLoader
from pretrain_dataset import CausalTokenDataset

dataset = CausalTokenDataset("DataProcessed/pretrain_ready", split="train")
loader = DataLoader(dataset, batch_size=4, shuffle=True,
                    generator=torch.Generator().manual_seed(20260929),
                    num_workers=0, pin_memory=True)
for batch in loader:
    x = batch["input_ids"].to("cuda", non_blocking=True)
    targets = batch["targets"].to("cuda", non_blocking=True)
    logits = model(input_ids=x).logits
    loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=-100)
    # backward/optimizer thuộc training loop của bạn.
```

`targets` đã dịch một token; không dịch lần nữa và không truyền chúng vào
API tự dịch `labels` của model. Attention là causal thông thường và có thể
đi qua ranh giới bài. Dùng offsets để xây document-aware attention nếu
framework hỗ trợ. Khi training phân tán, dùng DistributedSampler.

```powershell
python -B scripts/test_pretrain.py
```

## Kiểm tra độc lập sau khi chạy tiếp

Không sửa các script đã khóa hash trong checkpoint chỉ để chạy kiểm tra.
Script riêng dưới đây đối chiếu checksum của corpus sạch, tokenizer, binary
và offsets; decode mẫu ở đầu/giữa/cuối các shard để so với text gốc sạch;
kiểm tra DataLoader trên cả train và validation, bao gồm shift và padding.

```powershell
python -B -X utf8 "D:\AgentContentVietnamese\scripts\test_verify_pretrain_ready.py"
python -B -X utf8 "D:\AgentContentVietnamese\scripts\verify_pretrain_ready.py"
```

Chế độ mặc định yêu cầu manifest có `complete: true` và verification của
pipeline đã đạt. Báo cáo được lưu tại
`D:\AgentContentVietnamese\DataProcessed\pretrain_ready\reports\independent_verification.json`.
Kiểm tra decode và DataLoader là lấy mẫu, không phải decode toàn bộ corpus.

Có thể thêm `--committed-only` khi pipeline còn chạy. Chế độ này chỉ kiểm tra
shard đã commit, không chạy DataLoader và không xác nhận dataset sẵn sàng;
báo cáo riêng là `reports/committed_audit.json`. Không dùng báo cáo này thay
cho `binary/manifest.json` và `reports/verification.json`.
