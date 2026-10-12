# Pretrain model tiếng Việt 110M trên GPU 8 GB

## Cấu hình đã triển khai

- Config: `D:\AgentContentVietnamese\configs\pretrain_110m.json`.
- Dense decoder-only, 12 layers, hidden 768, 12 MHA heads, head dim 64,
  SwiGLU intermediate 2048, pre-RMSNorm, RoPE theta 10000, không bias/dropout.
- Vocabulary 32000; output head dùng trực tiếp trọng số embedding.
- **109.529.856 tham số**; context 4096, giữ nguyên tokenizer/dataset đã hoàn tất.
- BF16 autocast; parameters, gradients và AdamW moments FP32.
- Micro-batch 1, accumulation 32: 131072 token positions/update.
- AdamW LR 3e-4 -> 3e-5, warmup 300 updates, cosine; betas (0.9, 0.95),
  epsilon 1e-8, weight decay 0.1 cho ma trận, không decay RMSNorm, clip norm 1.
- Một epoch: 461136 sequences, 14411 updates (update cuối không đủ batch).
- Block checkpointing, chunked/recomputed loss head 256 positions để giảm VRAM.
- SDPA chỉ cho phép Flash hoặc memory-efficient CUDA kernel, không âm thầm
  fallback sang math attention tốn bộ nhớ. Windows/PyTorch hiện tại dùng
  memory-efficient backend; không cần cài thêm Flash Attention.

Đây là base model thiên về báo chí, không phải chatbot đã instruction-tune.
Không thay tokenizer/context giữa run. `targets` trong DataLoader đã shift:
training tính cross-entropy trực tiếp, mask -100, chia gradient theo tổng số
valid targets của update, kể cả update cuối.

## Môi trường

Đã dùng PyTorch 2.7.1+cu118 có CUDA/BF16 và các dependency dữ liệu hiện có
trong `D:\AgentContentVietnamese\scripts\requirements-pretrain.txt`.
Không tự nâng cấp thư viện giữa run: checkpoint khóa config, dataset/tokenizer,
training source hashes, Python/PyTorch/CUDA version và loại device.
Chỉ thay session limits/workers hoặc đường dẫn tới cùng dữ liệu khi resume.

## Kiểm thử và benchmark trước khi train

```powershell
python -B -X utf8 "D:\AgentContentVietnamese\scripts\test_pretrain_training.py" -v
python -B -X utf8 -m unittest discover -s "D:\AgentContentVietnamese\scripts" -p "test_*.py"
python -B -X utf8 "D:\AgentContentVietnamese\scripts\benchmark_pretrain.py" --steps 3 --warmup-steps 1
```

Benchmark dùng cấu hình/model thật, forward + backward + AdamW, không tạo
checkpoint hay bắt đầu run chính. Báo cáo mặc định:
`D:\AgentContentVietnamese\runs\benchmark_110m.json`.
Chạy nhiều steps hơn để đo 30–60 phút ổn định nhiệt; benchmark ngắn chưa đủ
suy ra thời gian hoàn thành. Tốc độ không bao gồm validation/checkpoint.
Có thể benchmark `--workers 2` riêng; mặc định 0 phù hợp Windows.

Kết quả smoke benchmark trên máy hiện tại (1 warmup + 2 measured updates):
109529856 tham số, khoảng **10679 valid tokens/s**, 12,20–12,34 giây/update,
peak PyTorch allocated **2,13 GiB**, reserved **2,52 GiB**. Đây không phải toàn
bộ VRAM của hệ điều hành/driver; số hiển thị trong Task Manager/nvidia-smi
có thể lớn hơn. Cần benchmark dài trước khi dùng tốc độ này để dự báo thời gian.

Đã smoke-test cả lưu checkpoint full-size rồi khởi chạy lại process trên GPU:
resume từ step 1/cursor 32/LR kế tiếp 2e-6, train tiếp thành step 2/cursor 64,
262144 valid tokens, lưu thành công. Run thử nằm riêng tại
`D:\AgentContentVietnamese\runs\smoke_110m`; run chính mặc định chưa được tạo.
Checkpoint và logs của run thử được giữ để bạn kiểm tra, không trộn với run chính.
Đã thử thêm STOP trong accumulation trên GPU: kết thúc đúng step 3/cursor 96,
393216 valid tokens rồi lưu/thoát, không chạy update kế tiếp. Ctrl+C handler
được kiểm thử trong unit test; không mô phỏng mất điện vật lý.

## Bắt đầu và tiếp tục: cùng một lệnh

```powershell
python -B -X utf8 "D:\AgentContentVietnamese\scripts\train_pretrain.py"
```

Run mặc định: `D:\AgentContentVietnamese\runs\pretrain_110m`.
Nếu chưa có run, tạo model mới và checkpoint step 0. Nếu có checkpoint,
tự kiểm tra và resume bản hợp lệ mới nhất; không reset LR/warmup/epoch.
Không chạy hai trainer cùng run: có OS-level lock, tự giải phóng khi process chết.

Chạy một phiên tối đa khoảng 4 giờ rồi tự lưu/dừng:

```powershell
python -B -X utf8 "D:\AgentContentVietnamese\scripts\train_pretrain.py" --session-minutes 240
```

Hoặc chạy thêm 10 optimizer updates rồi lưu/dừng:

```powershell
python -B -X utf8 "D:\AgentContentVietnamese\scripts\train_pretrain.py" --session-steps 10
```

Giới hạn là **cho phiên hiện tại**, không thay tổng training budget/scheduler.
Dừng ở cuối optimizer update; validation đang chạy cũng có thể làm vượt thời
lượng phiên. Muốn dừng ngay sau batch validation hiện tại, dùng Ctrl+C/STOP.

Muốn một run mới hoàn toàn, chọn thư mục mới, không xóa/ghi đè run cũ:

```powershell
python -B -X utf8 "D:\AgentContentVietnamese\scripts\train_pretrain.py" --run-dir "D:\AgentContentVietnamese\runs\pretrain_110m_v2"
```

## Dừng chủ động

Nhấn **Ctrl+C một lần**, giữ terminal mở và đợi:

1. `Stop requested...`.
2. Hoàn thành optimizer update đang chạy (tối đa các micro-batch còn lại).
3. `Checkpoint saved: ...` và event `session_end` với `safely_saved: true`.

Sau đó mới tắt máy. Ctrl+C tiếp tục chỉ nhắc chờ, không kill trainer.
Cũng có thể yêu cầu dừng từ một terminal khác:

```powershell
New-Item -ItemType File -Force "D:\AgentContentVietnamese\runs\pretrain_110m\STOP"
```

Trainer xóa file STOP sau khi lưu thành công. Nếu STOP còn tồn tại sau sự cố,
lần chạy tiếp theo sẽ lưu/dừng an toàn; xóa STOP nếu muốn chạy tiếp.

Đóng terminal, Task Manager kill, mất điện hoặc Windows restart không đảm bảo
chạy được handler. Khi đó resume checkpoint đã commit gần nhất, phần sau nó
phải train lại. Không cam kết cứu được update đang chạy khi mất điện.

## Checkpoint/resume và an toàn dữ liệu

Thư mục: `D:\AgentContentVietnamese\runs\pretrain_110m\checkpoints`.

- Lưu mỗi **30 phút hoặc 250 optimizer steps**, điều kiện nào đến trước,
  ở ranh giới optimizer update. Lưu thêm khi dừng và kết thúc epoch.
- Giữ 3 bản gần nhất + bản best full-validation + bản final (nếu khác nhau).
- Có model, AdamW, scheduler, epoch, global step, token count, cursor,
  best loss, RNG Python/NumPy/PyTorch CPU/CUDA và contract của run.
- Shuffle tái tạo bằng seed + epoch. Cursor do trainer tăng sau update;
  DataLoader prefetch không đẩy cursor này lên trước dữ liệu đã train.
- Lưu file tạm, flush/fsync, SHA-256, rename rồi commit metadata. `latest.json`
  chỉ là gợi ý; resume duyệt commit records theo step, không tin file tạm.
- Nếu bản mới nhất hỏng: cảnh báo và thử bản trước. Không có bản hợp lệ:
  báo lỗi, không âm thầm khởi tạo lại. Config/data/code/runtime khác: từ chối.
- `weights_only=True` khi đọc; không dùng pickle tùy ý từ checkpoint bên ngoài.
- Kiểm tra dung lượng trước khi lưu, giữ dự phòng 2 GiB ngoài ước lượng bản
  mới. Chỉ dọn bản cũ sau khi bản mới đã commit. Checkpoint đầy đủ khoảng
  1,3 GB; nên dành 10–15 GB trống. Best/final có thể làm số bản vượt 3.
- Nếu lỗi/OOM giữa update: không ghi checkpoint của trạng thái dở. Khởi chạy
  lại dùng bản đã commit trước đó. Nếu hết đĩa lúc lưu, giải phóng dung lượng
  không liên quan đến checkpoint rồi resume; phần chưa lưu có thể phải chạy lại.
- Rename/fsync giảm rủi ro nhưng không thay thế backup hoặc UPS; không cam kết
  chống mọi lỗi phần cứng/filesystem. Checkpoint nằm cùng ổ không phải backup.

Khi resume, tokenizer được hash lại, manifest được hash và binary được kiểm tra
kích thước. Không hash lại toàn bộ binary mỗi phiên; nếu nghi dữ liệu bị thay đổi,
chạy lại `D:\AgentContentVietnamese\scripts\verify_pretrain_ready.py` trước.
Không tự chỉnh manifest để vượt kiểm tra.

GPU kernels có thể không deterministic từng bit. Test CPU xác minh đường đi
resume khớp liên tục, không cam kết bitwise-identical giữa GPU/môi trường khác.
Log có event session_start để nhận biết rollback khi dùng checkpoint cũ:
những dòng metrics sau checkpoint cũ có thể bị lặp step và không đại diện tiến
độ đã commit.

## Validation và vận hành

- Console hiển thị mỗi 10 updates, update đầu phiên và cuối epoch: loss của
  update đó, valid tokens, LR, gradient norm, token/s, peak allocated VRAM,
  phần trăm tiến độ, thời gian phiên và ETA **train-only** (không gồm validation/
  checkpoint, dùng thời gian update trung bình của phiên, không phải cam kết).
- JSONL và log văn bản ghi **mọi optimizer update**, độc lập chu kỳ console.
- Quick validation: 192 sequences cố định (~0,79M token positions), mỗi 250 steps.
- Full validation: mỗi 2000 steps và cuối epoch. Best chỉ so với full-validation,
  không so lẫn loss của quick subset và toàn bộ validation.
- Nếu dừng đúng lúc hết epoch trước full-validation, lần sau chạy nốt đánh giá
  cuối và lưu final; không train lại epoch đã hoàn tất.
- Một epoch hoàn tất thì chạy lại lệnh không tự bắt đầu epoch mới. Muốn đổi
  training budget cần kế hoạch/run mới rõ ràng, không sửa config để resume.
- Chạy pilot 20–50M token trước khi cam kết dài ngày; giữ nguyên config/code
  để có thể tiếp tục pilot thành run chính. Chưa có kiểm chứng chất lượng model
  chỉ từ unit tests hoặc benchmark tốc độ.

Các module mới:
`D:\AgentContentVietnamese\scripts\pretrain_model.py`,
`D:\AgentContentVietnamese\scripts\pretrain_training.py`,
`D:\AgentContentVietnamese\scripts\pretrain_checkpoint.py`,
`D:\AgentContentVietnamese\scripts\train_pretrain.py`.
Pipeline dữ liệu đã khóa hash không bị sửa.

## Log và lịch sử phục vụ báo cáo

Không cần thêm thư viện/dịch vụ. Lệnh train/resume giữ nguyên. Trong thư mục
`D:\AgentContentVietnamese\runs\pretrain_110m`:

- `D:\AgentContentVietnamese\runs\pretrain_110m\train.log`: văn bản dễ đọc,
  thời gian UTC, loss/LR/tốc độ làm tròn để xem; append qua các phiên.
- `D:\AgentContentVietnamese\runs\pretrain_110m\metrics.jsonl`: nguồn lịch sử
  append-only, số liệu đầy đủ không làm tròn, fsync mỗi sự kiện. Ghi session ID,
  cấu hình batch, thiết bị, source hashes thực chạy, resume/checkpoint/warning,
  train/validation và kết thúc/lỗi (kèm traceback). Lỗi khởi tạo trước khi vào
  logger, output thư viện ngoài và thông báo tức thời từ signal handler không
  phải toàn bộ đều được thu vào file này. STOP được ghi sau khi lưu an toàn.
- `D:\AgentContentVietnamese\runs\pretrain_110m\metrics.csv`: UTF-8 BOM cho
  Excel, có `session_id`, `history_line`, `event`, `status` và toàn bộ trường metrics.
- `D:\AgentContentVietnamese\runs\pretrain_110m\summary.json`: tổng kết máy đọc
  được, contract cấu hình/dữ liệu/runtime, các phiên chạy và checkpoint hints.
- `D:\AgentContentVietnamese\runs\pretrain_110m\report.md`: báo cáo bảng dễ đọc.

CSV/summary/report được tạo lại khi phiên thoát bình thường hoặc gặp exception
có thể xử lý. Khi kill/mất điện, chúng có thể cũ; JSONL vẫn giữ các dòng đã ghi.
Log không thay thế checkpoint hay backup. Báo cáo không nạp weights, không hash
lại checkpoint; thông tin checkpoint trong báo cáo là metadata, file cũ có thể
đã bị retention xóa. Nếu hết đĩa, việc ghi log/báo cáo có thể thất bại; lỗi train
gốc không bị lỗi xuất báo cáo che khuất.

Xuất lại bất kỳ lúc nào, kể cả run dùng log cũ (nên làm khi trainer đã dừng):

```powershell
python -B -X utf8 "D:\AgentContentVietnamese\scripts\pretrain_report.py" --run-dir "D:\AgentContentVietnamese\runs\pretrain_110m"
```

Có thể thêm `--output-dir "D:\AgentContentVietnamese\runs\reports\pretrain_110m"`
để xuất ra nơi riêng. Không sửa JSONL hay checkpoint. Dòng JSONL bị cắt do kill
được giữ trong nguồn và cảnh báo trong báo cáo; trường log cũ không có là `null`/
`N/A`, không đoán lại. Log cũ thưa không thể khôi phục loss của các step chưa ghi.

### Cách đọc số liệu

- `committed`: step nằm trong tiến độ đã có checkpoint/resume hoặc kết thúc an
  toàn; **không** chứng minh một validation observation đã được lưu vào best
  checkpoint. Best checkpoint phải xem metadata `best.json`.
- `rolled_back`: metrics của nhánh bị bỏ khi resume checkpoint có step thấp
  hơn; vẫn giữ để truy vết nhưng không đưa vào thống kê tiến độ/loss.
- `uncommitted`: metrics chưa được sự kiện lưu/resume/kết thúc an toàn xác nhận.
- Loss trung bình cân theo valid tokens và **chỉ các updates thực sự được ghi**;
  với log cũ thưa, không gọi đó là loss trung bình toàn bộ quá trình train.
- Validation quick/full tách riêng; perplexity = exp(loss), overflow ghi `null`.
- Thời gian train đã commit không gồm validation/checkpoint IO. Tổng thời gian
  phiên đã ghi gồm overhead và cả phần train lại; phiên kill cứng có thể thiếu
  thời gian kết thúc. Không cộng `tokens_seen` giữa các phiên.
- Peak allocated VRAM là của PyTorch, không phải toàn bộ VRAM hệ thống.

### Tương thích bản log cũ

`D:\AgentContentVietnamese\scripts\pretrain_logging_upgrade.json` khóa chính xác
cặp SHA-256 trước/sau của hai file thay đổi cho logging. Chỉ cặp được kiểm tra
mới được phép resume contract cũ; model/training primitives/dataset, config và
runtime vẫn phải khớp. Contract gốc trong `run.json` và checkpoint được giữ;
source hashes thực chạy ghi trong từng `session_start`. Thay đổi mã nguồn khác
không tự được chấp nhận. Không chỉnh hash để bỏ qua lỗi không tương thích.

Kiểm thử riêng logging/report và toàn bộ training:

```powershell
python -B -X utf8 -m unittest discover -s "D:\AgentContentVietnamese\scripts" -p "test_pretrain_logging.py" -v
python -B -X utf8 "D:\AgentContentVietnamese\scripts\test_pretrain_training.py" -v
```
