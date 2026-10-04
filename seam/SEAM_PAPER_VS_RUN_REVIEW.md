# Đối chiếu paper SEAM và lượt chạy hiện tại

Ngày rà soát: 2026-10-03.

Kết luận từ bằng chứng hiện có: lệnh submit và lượt eval đã hoàn tất đúng, nhưng
implementation train có lỗi được thừa hưởng từ code công khai, checkpoint sinh
response suy biến trên tập harmful test hiện tại, và protocol eval đã thay đổi.
Chưa có phép thử đủ kiểm soát để kết luận kết quả nghiên cứu trong paper sai hoặc
để quy sự suy biến của checkpoint cho một lỗi duy nhất.

## 1. Lệnh chạy và artifact

Lệnh được rà soát:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Run-DefenseEval.ps1 -Suite Core -Models seam -ModalProfile chichichanhchanh222
```

Run `defense-eval-core-20261002T122941Z-93d6228a` có trạng thái complete, 6/6 case.
Manifest pin checkpoint `ducthang1703/seam-llama2-7b-seed42-checkpoint500` tại
revision `cceaabce27283438526db1396132a17b564ce44a`.
Metadata checkpoint xác nhận step 500, batch 8, epoch 1.0. File defense local
khớp hash nguồn dùng khi eval, nhưng chưa có manifest train để xác nhận input
lịch sử của defense training.

Không thấy bằng chứng người dùng gõ sai lệnh hoặc lượt eval dừng giữa chừng.

## 2. Lỗi implementation đã xác nhận, có trong code công khai

`masked_token_ce_loss` dùng mask[t] cho logits dự đoán token t+1, nhưng dùng
mask[t+1] cho target. Một target được chọn có thể vẫn có logit nhân 0, dẫn đến
gradient bằng 0. Đã kiểm tra bằng chính method và tokenizer của checkpoint:
cả 4.000 cặp local đều mất gradient tại target refusal được chọn đầu tiên;
6.456/395.765 target refusal hợp lệ bị ảnh hưởng. Điều này không có nghĩa toàn
bộ gradient refusal đều bằng 0, và chưa chứng minh riêng lỗi này gây collapse.

So sánh AST giữa working tree và git HEAD cho thấy ba method
`masked_token_ce_loss`, `temporary_parameter_update`, `compute_loss` giống nhau
về cấu trúc. HEAD là commit `fa7224f8754585786c850f327b08f32a6976420e`.
Đã kiểm tra trực tiếp [source công khai ở cùng commit](https://github.com/ZJUWYH/seam/blob/fa7224f8754585786c850f327b08f32a6976420e/src/core/trainer.py):
mask lệch cũng có ở đó. Vì vậy không nên quy lỗi mask cho việc sửa launcher
Modal hoặc cấu hình của người dùng.

Perturbation BF16 có rủi ro bị làm tròn. Phép thử nhỏ tái hiện rủi ro, chưa đo
trên trọng số/gradient thật của Llama 7B. Chưa đủ căn cứ nói L_sd của run bằng 0.

## 3. Phép đo HS hiện tại cần đọc cùng response

Case clean không fine-tune attack. Trong 798 response clean, 677 (84.84%) chứa
`Хронологија` lặp lại. Có 391 response giống hệt nhau với classifier score từ
0.0271 đến 0.9291. Scorer nhận prompt cộng response nên thay đổi prompt cũng
làm thay đổi điểm cho cùng một câu trả lời.

Đây là dấu hiệu model sinh output suy biến trên tập harmful prompts với thiết
lập hiện tại; HS 33.85 không phải tỷ lệ câu trả lời hữu ích nhưng nguy hại.
ZS dùng xác suất lựa chọn đáp án benchmark và không đo trực tiếp độ mạch lạc
của các response tự do này. Không suy ra model mất mọi năng lực chỉ từ các
response harmful test.

Tại harmful LR 1e-4, HS 70.02 đi cùng những response có nội dung nguy hại thật
trong mẫu đã đọc; không thể giải thích toàn bộ hàng này bằng lỗi chấm response
lặp của clean. Chưa có judge độc lập trên toàn bộ output.

## 4. Khác biệt với thí nghiệm paper và giới hạn tái lập

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/1abb0e7bd62ba80610798dee81950522-Paper-Conference.pdf)
§5.1, §5.3, Appendix C.1–C.2 nêu refusal GPT-4o, AdamW, và 20 lần lặp
cho Bảng 6; paper có thêm phép đo HS-G bằng GPT-4o và thừa nhận classifier
nhạy với từ/cụm từ. §5.1 không định danh đầy đủ cấu hình task/metric ZS;
cách gọi điểm classifier là logits cũng chưa khớp rõ với pipeline probability.
Đây là các điểm cần làm rõ khi tái lập, không chứng minh số liệu bị sai.

Run hiện tại dùng refusal RepNoise công khai (chưa xác minh đúng dữ liệu paper),
một seed, tập 798 unique harmful prompts sau exclusion, optimizer attack paged
AdamW 8-bit, batch vi mô 1 tích lũy 4, greedy 50 token, prompt Answer có newline,
và ZS MC1/raw accuracy. Code eval cũ kế thừa generation_config của checkpoint;
metadata checkpoint có do_sample=true, temperature=0.6, top_p=0.9. Chưa biết
thiết lập decoding thực tế của các run nội bộ tạo ra số paper.

Phần evaluator bổ sung trong dự án và các lựa chọn trên cũng cần được kiểm
tra; khác biệt protocol không nên bị diễn giải thành lỗi gõ lệnh của người dùng.

## 5. Các phép thử còn thiếu để xác định nguyên nhân

1. So sánh Base và SEAM trên cùng prompt, tokenizer, decoding, batch 1 và batch
   4; thử prompt cũ Answer: và prompt mới Answer có newline.
2. Nếu SEAM tiếp tục suy biến, kiểm tra checkpoint trước upload và thành phần
   loss/gradient của lượt train gốc; hiện chưa có code hash lịch sử run.
3. Sửa mask loss, đo perturbation thực tế và theo dõi refusal/benign/generation
   trong một lượt train ngắn trước khi chạy lại 500 bước.
4. Chỉ so paper sau khi cố định dữ liệu, optimizer, metrics và có Base đối chứng.

Chưa submit GPU hoặc sửa code train/eval trong lần rà soát này. Truy cập volume
train ở workspace nuocb0739 đã bị automatic approval review chặn trước đây;
không truy cập lại vì chưa có xác nhận quyền đọc workspace đó.

Artifact kiểm chứng: `tmp/audit-seam-core/manifest.json`, `status.json`,
`checkpoint-metadata/trainer_state.json`, `training-audit.json`, `eval-audit.json`.
Chi tiết kiểm tra loss: [SEAM_TRAIN_AUDIT.md](SEAM_TRAIN_AUDIT.md).
