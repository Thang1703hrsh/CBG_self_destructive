# Chạy SEAM sau khi sửa code

## Những gì đã sửa

- Protocol eval `defense-checkpoint-eval-v2`: generation dùng `Question: {p}\nAnswer:`;
  attack vẫn dùng `Question: {p}\nAnswer:\n{response}`. Manifest lưu hai template.
- Mask loss chỉ chọn labels sau causal shift; logits không bị nhân mask. Token trả
  lời đầu tiên có gradient, padding `-100` được giữ, token ID 0 không bị bỏ nhầm.
- Collator clone labels và dùng attention mask; không sửa dataset tại chỗ, không
  bỏ nhầm EOS thật khi EOS cũng là pad token.
- Train mặc định FP32, không BF16/FP16 autocast hoặc TF32; AdamW giữ moments FP32
  trên CPU. Bộ đệm gradient và bản sao trọng số lúc perturb cũng lưu trên CPU.
- Ghi `perturbations[].changed_fraction`, `actual_norm` và
  `weighted_sd_gradient_norm` vào `loss_components.jsonl`; dừng khi loss/gradient
  không hữu hạn, norm bằng 0 hoặc perturbation bị làm tròn hoàn toàn.
- Gradient accumulation chỉ được chia một lần. Manifest train lưu hash code và
  dữ liệu. Checkpoint và kết quả mới có run ID riêng.

Không sửa được checkpoint đã train bằng cách sửa file Python. Checkpoint hiện tại
trong `config/defense_eval.yaml` vẫn là bản cũ; các sửa train chỉ áp dụng cho run mới.

## 1. Eval lại clean của checkpoint cũ trước

Tại `C:\Users\ducth\seam`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Run-DefenseEval.ps1 -Suite Clean -Models seam -ModalProfile chichichanhchanh222
```

Một case, một L40S, không attack fine-tuning. Ghi lại run ID mới và tải kết quả:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Get-DefenseEval.ps1 -RunId RUN_ID_EVAL_MOI -ModalProfile chichichanhchanh222 -Download
```

Đọc `comparison.md`, `summary.json`, và `cases/seam/clean/responses.jsonl` trong thư
mục kết quả tải về. Kiểm tra HS, MMLU riêng, tỷ lệ lặp/rác, và câu từ chối.
Nếu chỉ cần đọc trạng thái, bỏ `-Download`. Script cho phép thư mục đích rỗng và
dùng lại kết quả đã tải đủ. Nếu lần tải trước bị gián đoạn hoặc cần cập nhật,
thêm `-Download -Force`; tùy chọn này cập nhật các file kết quả local trùng tên.
Run cũ vẫn tồn tại để đối chiếu. Không dùng `-Resume` của run v1 vì code/protocol đã đổi.
MMLU ở run SEAM cũ là 32,99%; 46,13% là ZS trung bình bốn task.

Muốn có Base đối chứng cùng protocol, có thể chạy riêng Clean với `-Models base,seam`.
Checkpoint hồi phục generation sau đổi prompt chưa chứng minh utility hoặc robustness hồi phục.

## 2. Train thử 2 bước với code mới

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Run-SEAMFull.ps1 -TrainOnly -TrainSteps 2 -ModalProfile chichichanhchanh222
```

Chế độ train mới yêu cầu **một H100 và 320 GiB RAM CPU**. AdamW và gradient offload
tốn thời gian truyền dữ liệu; có thể chậm hơn đáng kể so với BF16 trước đây.
Đây là kiểm tra vận hành trên model 7B; kiểm tra local chỉ dùng model nhỏ trên CPU.
Không mặc định train FP32 7B trên L40S 48 GB bằng đường train này.

Launcher in tên thư mục `seam-full-seed42-...` và đường dẫn status. Đọc status:

```powershell
python -X utf8 -m modal volume get seam-full-results /TEN_THU_MUC_RUN/status.json ./seam-train-status.json --force --profile chichichanhchanh222 --env main
Get-Content ./seam-train-status.json
```

Thay `TEN_THU_MUC_RUN` bằng thư mục mà launcher vừa in. Status phải là `complete`.
Tải log để kiểm tra perturbation và self-destruct gradient:

```powershell
python -X utf8 -m modal volume get seam-full-results /TEN_THU_MUC_RUN/defense/loss_components.jsonl ./seam-loss-components.jsonl --force --profile chichichanhchanh222 --env main
```

Hai perturbation phải có norm thực tế khác 0 và hữu hạn; đọc cùng cosine và norm
gradient, không suy ra bẫy hoạt động chỉ từ một con số loss. Checkpoint 2 bước là
diagnostic, không dùng làm kết quả SEAM 500 bước.

## 3. Khi run thử hoàn tất, train 500 bước từ Base

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Run-SEAMFull.ps1 -TrainOnly -TrainSteps 500 -EvalConfigPath config/defense_eval_seam_fixed.yaml -ModalProfile chichichanhchanh222
```

Chạy từ Llama-2-7B-Chat gốc. Run này tự tải dữ liệu public, lưu defense data,
checkpoint mới và manifest vào các volume cùng profile. Launcher tự tạo
`config/defense_eval_seam_fixed.yaml` trỏ đúng checkpoint mới và defense file tương ứng.
File cấu hình được tạo lúc submit; nó chưa chứng minh train đã hoàn tất.
Chờ status `complete` và checkpoint-500 tồn tại trước khi eval.
Không cần upload checkpoint sang Hugging Face: evaluator đọc trực tiếp Modal volume.

## 4. Eval checkpoint mới trên L40S

Nên xem Clean trước:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Run-DefenseEval.ps1 -ConfigPath config/defense_eval_seam_fixed.yaml -Suite Clean -Models seam -ModalProfile chichichanhchanh222
```

Sau khi kiểm tra output và utility, chạy Core:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Run-DefenseEval.ps1 -ConfigPath config/defense_eval_seam_fixed.yaml -Suite Core -Models seam -ModalProfile chichichanhchanh222
```

Core gồm clean, harmful LR 2e-5/5e-5/1e-4, benign LR 2e-5 và mixed10 LR 5e-5.
Sáu case chạy tuần tự, mỗi case một L40S. Attack fine-tuning thuộc các case này;
defense training đã thực hiện riêng ở bước 3. Lưu run ID eval mới để lấy kết quả.

## Giới hạn

Đã kiểm tra loss boundary, collator, perturbation/restore, AdamW so với PyTorch,
prompt và vòng SeamTrainer với model Llama nhỏ trên CPU. Chưa submit GPU hoặc
xác nhận bộ nhớ/tốc độ và kết quả của model 7B sau sửa. Không bảo đảm các sửa này
sẽ tạo ra đúng số của paper: refusal public, protocol attack và một seed vẫn có
khác biệt với thí nghiệm paper. Trước khi mở rộng, cần đọc Clean và kiểm tra log train.
