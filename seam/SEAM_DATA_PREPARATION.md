# Chuẩn bị dữ liệu SEAM trong dự án này

Tài liệu này mô tả **code hiện có**, đặc biệt là lệnh `Run-SEAMFull.ps1` chạy main-result sweep trên Modal. Nó không khẳng định bộ refusal công khai giống chính xác bộ refusal được mô tả trong paper.

## 1. Nguồn dữ liệu và phép chia của main run

[`scripts/modal_seam_full.py`](scripts/modal_seam_full.py) tải JSON `beavertails_with_refusals_train.json` từ kho `domenicrosati/representation-noising` (biến `PUBLIC_DATA_URL`). Mỗi bản ghi dùng các trường `prompt`, `response`, `refusal`, `is_safe` và `category`. Pipeline:

1. Giữ bản ghi có `is_safe is False` và có giá trị không rỗng ở `prompt`, `response`, `refusal`.
2. Xáo danh sách bằng `random.Random(seed).shuffle(records)`; lệnh `Run-SEAMFull.ps1` mặc định `seed=42`.
3. Yêu cầu ít nhất 5.000 bản ghi hợp lệ. Lấy **4.000 bản ghi đầu** làm `defense.json`, **1.000 bản ghi tiếp theo** làm `attack.json`.
4. Ghi SHA-256 của **tệp JSON tải về nguyên gốc** vào `manifest.json` để theo dõi nguồn.

| Tập | Nguồn | Số bản ghi | Vai trò |
|---|---|---:|---|
| Defense harmful–refusal | 4.000 dòng đầu sau lọc và xáo | 4.000 cặp | Train SEAM |
| Attack | 1.000 dòng tiếp theo | 1.000 cặp | Fine-tuning attack khi eval; không thuộc defense train |
| Benign | `yahma/alpaca-cleaned`, split `train` | 4.000 dòng | Giữ năng lực trên tác vụ lành tính khi train SEAM |
| Harmfulness test | `PKU-Alignment/BeaverTails`, split `30k_test` | Mặc định tối đa 100 prompt sau lọc | Eval, không thuộc defense train |

Hai tập defense và attack **không trùng dòng do cách cắt danh sách**, nhưng code không có bước khử trùng `prompt` hay kiểm tra trùng ngữ nghĩa. `manifest.json` ghi số lượng từng tập và SHA-256 của tệp nguồn tải về (không phải hash riêng của `defense.json`/`attack.json`).

## 2. Ba nhánh dữ liệu mà SEAM train

[`src/train.py`](src/train.py) đọc `defense.json`, đặt `defense_size=4000`, và tạo các nhánh qua [`src/data_processing/_datasets.py`](src/data_processing/_datasets.py):

| Nhánh | Chuỗi đưa vào tokenizer | Nguồn |
|---|---|---|
| Harmful | `Question: {prompt}\nAnswer:\n{response}` | `prompt` + `response` của cùng dòng BeaverTails |
| Safety / harmless | Giá trị **nguyên văn** của trường `refusal` | Cùng dòng BeaverTails với harmful |
| Benign | `Question: {instruction}\n{input}\nAnswer: {output}` | Alpaca |

Ở nhánh benign, code xáo `yahma/alpaca-cleaned` bằng `benign_seed=seed`, sau đó lấy 4.000 dòng đầu. `src/train.py` đặt `tokenizer.pad_token = tokenizer.eos_token`. `SeamTrainer` ghép ba dataloader bằng `zip` và đặt `shuffle=False`; thứ tự harmful và refusal vẫn tương ứng theo chỉ số dòng. Với batch 8 và 500 step của main run, code đi qua 4.000 cặp defense một lượt nếu run kết thúc bình thường.

Lưu ý: `data/create_refusal_gpt.py` là **script tùy chọn để tạo dữ liệu**, không được gọi trong `Run-SEAMFull.ps1`. Script đó thử `gpt-4o-mini` trước và có thể chuyển sang `gpt-4o`; không nên suy ra refusal của main run đều do GPT-4o tạo.

## 3. Giới hạn token và nhãn

`CONTEXT_LENGTH = 256` trong [`src/data_processing/_datasets.py`](src/data_processing/_datasets.py). Cả harmful, refusal và benign đều dùng `truncation=True`, `padding="max_length"`, `max_length=256`. **256 là giới hạn cho cả chuỗi prompt + answer**, không phải 256 token cho mỗi phần. Chuỗi dài hơn bị cắt; chuỗi ngắn hơn được padding.

Code tạo `labels` từ `input_ids` và che token padding bằng `-100` khi đưa vào trainer. Đây **không phải response-only labeling** cho benign. Trong [`src/core/trainer.py`](src/core/trainer.py), safety/harmful còn dùng mask dựa trên vị trí token khác nhau giữa hai chuỗi ghép cặp. Vì vậy độ dài/truncation của `response` và `refusal` có thể ảnh hưởng trực tiếp đến loss.

## 4. Dữ liệu eval sau khi train

Main run dùng `attack.json` làm đầu vào cho [`src/eval.py`](src/eval.py). Khi `attack=true`, eval fine-tune trên tối đa 1.000 mẫu harmful ở tập này; khi `attack=false`, không có bước fine-tune attack. Harmfulness được đo trên BeaverTails `30k_test` (mặc định `test_size=100`), còn utility được đo trên các benchmark được cấu hình trong [`scripts/modal_seam_full.py`](scripts/modal_seam_full.py). Không nhầm 1.000 dòng attack với 4.000 dòng defense.

## 5. File trên Modal và cách kiểm tra

Với `seed=42` và run ID `<RUN_ID>`:

| Modal volume | Đường dẫn tương đối | Nội dung |
|---|---|---|
| `seam-full-data` | `seam-full-seed42-<RUN_ID>/defense.json` | 4.000 cặp train |
| `seam-full-data` | `seam-full-seed42-<RUN_ID>/attack.json` | 1.000 cặp attack |
| `seam-full-results` | `seam-full-seed42-<RUN_ID>/manifest.json` | Seed, nguồn, SHA-256, số mẫu, cấu hình run |
| `seam-full-results` | `seam-full-seed42-<RUN_ID>/status.json` | Trạng thái từng stage |

Ví dụ xem manifest mà không cần tải toàn bộ dữ liệu:

```powershell
$runId = 'THAY_BANG_RUN_ID'
python -m modal volume get seam-full-results "seam-full-seed42-$runId/manifest.json" .\seam-manifest.json
Get-Content .\seam-manifest.json -Raw
```

Nếu muốn tái lập run đã có, cần giữ **cùng nguồn JSON và SHA-256, seed, 4.000/1.000 split, thứ tự dữ liệu, formatter và giới hạn 256 token**. Refusal công khai của RepNoise chưa được xác minh là bộ GPT-4o trong paper; vì thế kết quả từ pipeline này là một biến thể theo code/nguồn công khai, không tự động là tái lập chính xác dữ liệu paper.
