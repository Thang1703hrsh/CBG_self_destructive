# Kiểm tra quá trình train SEAM — 2026-10-03

Đã tìm được lỗi cụ thể trong code train hiện tại: mask của logits lệch với mask
của token đích, làm mất gradient trên các token cần học. Có thêm rủi ro số học
ở perturbation BF16. Chưa đủ bằng chứng để quy toàn bộ sự suy biến của checkpoint
500 cho một nguyên nhân duy nhất.

## Bằng chứng từ checkpoint và run đã chạy

- Eval run: `defense-eval-core-20261002T122941Z-93d6228a`, hoàn tất 6/6 case.
- HF model: `ducthang1703/seam-llama2-7b-seed42-checkpoint500`.
- Revision: `cceaabce27283438526db1396132a17b564ce44a`.
- `trainer_state.json` của chính revision này: global step 500, epoch 1.0,
  batch 8, 500 bản ghi loss. Lịch learning rate bắt đầu 4e-7, kết thúc 0.
- Metadata tải từ cache HF trong profile eval; SHA-256 của các file được đối chiếu
  với manifest của run trước khi dùng. Không tải trọng số hoặc chạy GPU mới.
- SHA-256 `defense.json` local trùng nguồn đã dùng để loại overlap khi eval:
  `a277fe5dab5c38d3e1a6614b7d5184e73e2ce6d28cb380b4c2cf9df221473fd3`.
  Điều này xác nhận file eval, chưa chứng minh file này đúng từng byte với input
  của lượt train gốc, vì chưa có manifest train gốc.
- Kiểm tra ghép cặp local: cả 4.000 refusal đều bắt đầu bằng đúng câu hỏi của
  dòng tương ứng; cả 4.000 dòng unsafe có category active, không có refusal rỗng.
  Không thấy lỗi ghép nhầm harmful/refusal trong file local này.
- Clean: 677/798 response (84.84%) chứa `Хронологија` lặp lại. Clean không chạy
  optimizer attack. Không có prompt/classifier truncation ở clean.

## Lỗi đã tái hiện: mask loss sai một vị trí

Vị trí: `src/core/trainer.py`, phương thức `masked_token_ce_loss`.

Code dùng `mask[..., :-1]` để nhân logits, nhưng dùng `mask[..., 1:]` để chọn
labels. Logit ở vị trí t dự đoán token t+1. Khi mask[t]=0 và mask[t+1]=1,
logit bị nhân 0 dù target vẫn được tính loss. Cross-entropy của target này thành
giá trị hằng, gradient về model bằng 0.

Kiểm tra CPU chạy chính method lấy từ AST của file hiện tại (không chép lại công
thức để thay thế method): mask `[0,0,1,1]` cho gradient tại logit dự đoán token
được chọn đầu tiên bằng 0. Khi chỉ mask labels, gradient norm bằng 0.49046.

Tokenize 4.000 cặp defense với tokenizer của checkpoint, đúng formatter và độ dài
256 của code train:

| Kiểm tra | Kết quả |
| --- | ---: |
| Cặp có target refusal mất gradient do mask lệch | 4.000/4.000 |
| Cặp mất gradient ở target refusal được chọn đầu tiên | 4.000/4.000 |
| Target refusal hợp lệ có logit dự đoán bị nhân 0 | 6.456/395.765 |
| Cặp không có target refusal được chọn | 0 |

Mẫu đầu tiên: hai nhánh cùng có `Answer:`, harmful tiếp theo là newline, refusal
tiếp theo là token `▁I`. Mask trước `▁I` bằng 0: việc học bắt đầu trả lời bằng
`I cannot…` bị mất gradient tại token này. Đây là lỗi thật, nhưng số target mất
gradient không có nghĩa toàn bộ refusal loss không hoạt động.

Hướng sửa: giữ nguyên logits; sau causal shift, đặt labels không được chọn thành
`-100`; bảo toàn labels padding `-100`; kiểm tra batch có ít nhất một target hợp
lệ. Cần kiểm tra cả unlearning, refusal và harmful-gradient vì cả ba gọi method
này. Việc sửa làm thay đổi objective so với code hiện tại và cần train checkpoint
mới để đánh giá hiệu quả.

## Rủi ro số học: perturbation trực tiếp trên BF16

`src/train.py` nạp model bằng BF16; `temporary_parameter_update` cộng perturbation
vào `param.data` trực tiếp. Với epsilon 0.001 và direction được chuẩn hóa trên
nhiều tham số, từng cập nhật có thể nhỏ hơn độ phân giải BF16 và bị làm tròn mất.

Kiểm tra minh họa bằng chính context manager trên linear layer CPU 1.048.576
tham số, cùng epsilon 0.001:

| Kiểu trọng số | Số phần tử thay đổi khi perturb |
| --- | ---: |
| BF16 | 16.386/1.048.576 (1.56%) |
| FP32 | 1.048.576/1.048.576 (100%) |

Đây là phép kiểm tra minh họa, **không phải số đo gradient của Llama 7B**. Cần ghi
fraction tham số thực sự thay đổi, norm perturbation thực tế và norm của gradient
self-destruct trong lượt train thật. Chưa kết luận self-destruct gradient của run
500 bằng 0. Thay dtype/epsilon cần kiểm tra lại phương pháp và bộ nhớ trước khi
chạy lớn.

## Các điểm phụ

- Mask toàn 0 làm method loss hiện tại trả NaN. Code không dừng rõ ràng khi
  loss/gradient nonfinite, và bộ lọc log NaN của Trainer có thể che NaN trong log.
  Dữ liệu local được kiểm tra không có hàng refusal thiếu target; chưa thấy bằng
  chứng NaN đã xảy ra trong run 500.
- Collator trong `src/train.py` sửa `item['input_ids']` tại chỗ khi tạo labels.
  Đã tái hiện mutation. Do input batch được stack trước mutation, điều này không
  chứng minh input forward của run bị đổi thành -100; nên clone labels để bỏ tác
  dụng phụ.
- Trung bình loss log 5 bước đầu là -1.80014, 5 bước cuối -4.84050. Loss log là
  unlearning + alpha*refusal, không gồm phần gradient self-destruct. Loss âm hoặc
  giảm không chứng minh train thành công hay thất bại. Thiếu component logs để
  xác định loss refusal/benign và gradient conflict trong chính run 500.
- Artifact cũ `results/seam-20steps-heldout.json` cho thấy refusal loss tăng
  từ 0.27623 lên 0.35134 và benign loss từ 1.51758 lên 1.72363 sau 20 bước.
  Đây là tín hiệu cần theo dõi utility từ sớm, không phải log của checkpoint-500:
  run 20 bước dùng lịch warmup khác và helper refusal cũng dùng mask lệch cũ.
  Không dùng nó để chứng minh diễn biến của chính run 500 hoặc tính đúng của loss.
- Code còn những thiếu sót này từ bản git HEAD được so với working tree; chúng
  không chỉ xuất hiện do các sửa đổi launcher eval. Tuy nhiên checkpoint không
  lưu hash code train nên chưa xác minh trọn vẹn phiên bản code lịch sử của run.

## Artifact và giới hạn

Kết quả máy đọc: `tmp/audit-seam-core/training-audit.json`.
Script CPU: `tmp/audit-seam-core/audit_training.py`.
Checkpoint metadata: `tmp/audit-seam-core/checkpoint-metadata/`.

Chạy lại kiểm tra sau khi đã có metadata local:

```powershell
python -X utf8 tmp/audit-seam-core/audit_training.py
```

Chưa sửa code train và chưa submit lượt GPU mới trong lần review này. Đọc thêm
volume train `seam-full-results` ở profile `nuocb0739` bị automatic approval review
chặn do workspace khác chưa được xác nhận quyền truy cập. Để kết luận lịch sử
run cần được phép đọc manifest và `loss_components.jsonl` của lượt train gốc.
