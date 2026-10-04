# SEAM: thử nghiệm 20 bước trên 2 H100

Trạng thái: **hoàn tất**. Đây là phép chạy thử ngắn, không phải tái lập main result (500 bước) hoặc đánh giá độ an toàn của paper.

## Setting và artifact

| Mục | Giá trị |
| --- | --- |
| Model | `meta-llama/Llama-2-7b-chat-hf` |
| GPU Modal | **2 × H100** (`H100:2`): `cuda:0` cho model, `cuda:1` cho hai bộ đệm gradient |
| Defense / alignment | 4.000 cặp harmful response / refusal từ BeaverTails-with-refusals, seed 42 |
| Benign | 4.000 mẫu `yahma/alpaca-cleaned` được shuffle với seed 42 |
| Optimizer | AdamW, learning rate `2e-5`, cosine scheduler, warmup ratio `0.1`, BF16 |
| SEAM | `alpha=1`, `beta=0.01`, `epsilon=0.001` |
| Batch / độ dài | 8, gradient accumulation 1, **20 optimizer steps** (160 ví dụ qua mỗi tập) |
| Thời gian train | 78,25 giây, không tính khởi tạo image/model/dataset |
| Modal training | https://modal.com/apps/nuocb0739/main/ap-yunUdAFt3ov0ojeYON8yOV |
| Modal đánh giá loss | https://modal.com/apps/nuocb0739/main/ap-RmrtK1RihBedpkGcY0BT5c |
| Checkpoint | Volume `seam-main-results`, path `seam-20steps-2h100-seed42/checkpoint-20` |
| Log local | `results/seam-20steps-loss.jsonl` và `results/seam-20steps-heldout.json` |

Tập 1.000 attack record được tách khỏi 4.000 defense record; 64 record đầu của tập attack dùng để đo loss ngoài tập train. Tương tự, 64 benign record được lấy sau 4.000 record dùng để train. Cùng một tập 64 mẫu và cách tính được áp dụng cho base và checkpoint.

## Loss trong quá trình train

Giá trị dưới đây là trung bình 5 bước đầu và 5 bước cuối, ghi *trước* mỗi optimizer update. `L_ul` là negative log của loss nhiều tầng theo mã SEAM, nên không thể diễn giải như cross-entropy thông thường. `L_up` là loss trên refusal; `L_sd` dùng gradient cosine và xấp xỉ gradient bằng perturbation. `objective proxy = L_ul + alpha L_up + beta cosine` chỉ là chỉ số để theo dõi, không phải toàn bộ loss mà Trainer báo cáo.

| Chỉ số | Bước 1–5 | Bước 16–20 | Nhận xét |
| --- | ---: | ---: | --- |
| `L_ul` | -2,1206 | -2,3131 | Giảm (đúng hướng của unlearning objective) |
| `L_up` (refusal) | 0,3111 | 0,3574 | Tăng, chưa giữ tốt alignment loss |
| Harmful SFT CE | 2,1379 | 2,5926 | Tăng (khó học harmful responses hơn) |
| Benign SFT CE | 1,5218 | 1,8053 | Tăng (dấu hiệu suy giảm utility trên batch train) |
| Cosine gradient harmful–benign | 0,4121 | 0,4051 | Vẫn dương; chưa tạo được gradient conflict rõ rệt |
| Objective proxy | -1,8053 | -1,9517 | Giảm, nhưng không chứng minh SEAM đã thành công |

Trainer báo cáo `train_loss = -1,8888`, là trung bình `L_ul + alpha L_up` của 20 bước; nó **không** bao gồm thành phần `beta L_sd` được đưa vào gradient update riêng. Các giá trị loss đều hữu hạn, không thấy NaN/Inf trong 20 bản ghi.

## Loss ngoài tập train: base so với checkpoint bước 20

CE masked cho harmful/refusal theo cùng mask khác biệt token mà code train sử dụng; CE chuẩn cho benign. Mỗi số là trung bình 8 batch × 8 mẫu (64 mẫu/tập). Thấp hơn chỉ có nghĩa mô hình dự đoán văn bản tương ứng dễ hơn; đây **không** phải harmfulness score (HS) hay zero-shot accuracy (ZS).

| Tập | Base | SEAM-20 | Thay đổi | Diễn giải |
| --- | ---: | ---: | ---: | --- |
| Harmful response | 2,1729 | 2,5615 | +0,3886 (+17,9%) | Đúng hướng nếu muốn mô hình khó khớp harmful response |
| Refusal response | 0,2762 | 0,3513 | +0,0751 (+27,2%) | Xấu hơn: loss refusal tăng |
| Benign Alpaca | 1,5176 | 1,7236 | +0,2061 (+13,6%) | Xấu hơn: loss benign tăng |

Cả 8/8 batch đều tăng loss ở cả ba tập. Kết luận ở mức phép thử: pipeline SEAM 20 bước chạy ổn trên 2 H100 và có dấu hiệu unlearning harmful responses, nhưng **đồng thời làm tăng loss refusal và benign**. Chưa thể nói mô hình an toàn hơn, giữ utility, hay tái lập main result. Cần đánh giá HS/ZS và khả năng chống harmful fine-tuning sau khi train đủ 500 bước; 20 bước dùng warmup 2 bước nên cũng không tương đương 20 bước đầu của lịch 500 bước (warmup 50 bước).

Lệnh chạy lại:

```powershell
$env:PYTHONIOENCODING='utf-8'
modal run --detach scripts/modal_seam_main.py::main --stage train
modal run --detach scripts/modal_seam_main.py::main --stage eval-loss
```
