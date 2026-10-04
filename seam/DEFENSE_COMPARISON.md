# So sánh SEAM, SDD và CTRAP trên Modal

Hai launcher mới chạy **một seed** theo giao thức đánh giá chung của
`Run-SEAMFull.ps1`. Chúng **không** tái tạo trực tiếp bảng chính của paper SDD
hoặc CTRAP, vì ba paper dùng attack, tập đánh giá và thước đo khác nhau.
Đã kiểm tra smoke test 2 step cho SDD và CTRAP trên Modal; chưa chạy full end-to-end.

| Method | Defense training | Code thực thi | GPU đồng thời mỗi run |
| --- | --- | --- | --- |
| SEAM | 4k cặp RepNoise, 500 step, batch 8 | `src.train` | 2 H100 khi defense; 1 H100 khi eval |
| SDD | 8k cặp BeaverTails + Alpaca (mặc định) hoặc Alpaca/LIMA, 500 step, batch toàn cục 24, LR 5e-7 | SFT tái dựng từ paper (`SDD/train` không có trong repo) | 1 H100 |
| CTRAP | 5k refusal + 5k harmful + 5k UltraChat, 20 epoch, LR 5e-4, alpha/lambda 0.1, LoRA rank 32 | `CTRAPTrainer` gốc qua adapter dữ liệu | 1 H100 |

Với cùng seed, cả ba sử dụng tập attack 1k cặp ở chỉ số 4000:5000 sau khi
lọc và xáo dữ liệu RepNoise; SDD loại prompt trùng với tập attack. CTRAP dùng
thêm 1k cặp ở chỉ số 5000:6000 cho defense, nên lượng dữ liệu defense **không
bằng nhau**. Cả hai launcher mới đánh giá base/defense trước attack và sau
250 step full fine-tune ở 5 LR: `2e-5`, `5e-5`, `8e-5`, `1e-4`, `2e-4`; cùng
`src.eval`, cùng 100 BeaverTails test prompt và 4 utility task mặc định. Đây
là phép so sánh chung, không phải metric GPT-4 hay “harmfulness rate” của SDD
và không phải HS(IO)/HS(O) trong paper CTRAP.
Các sweep này và launcher SEAM đều bật gradient checkpointing ở bước attack
để giảm bộ nhớ, không đổi số step/LR của giao thức.
SDD cũng loại prompt trùng với toàn bộ `BeaverTails:30k_test` để tránh rò rỉ
giữa defense và tập đánh giá.

## Chuẩn bị

1. Cài và đăng nhập Modal: `modal setup`.
2. Tạo Modal secret `huggingface-secret` chứa `HF_TOKEN` của tài khoản có quyền
   với `meta-llama/Llama-2-7b-chat-hf`. Chỉ chế độ SDD `-AnswerPool AlpacaLima`
   mới cần tài khoản đã chấp nhận điều kiện truy cập `GAIR/lima`.
3. Đảm bảo tài khoản Modal đủ quota và kiểm tra giá H100 trước khi chạy. CTRAP
   20 epoch và 12 lượt đánh giá/attack cho mỗi run có thể rất tốn thời gian/chi phí.

## Chạy ngầm

Từ `C:\Users\ducth\seam`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDPreflight.ps1 -Seed 42
powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDFull.ps1 -PreparedRunId RUN_ID_FROM_PREFLIGHT -ValidateOnly
powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPFull.ps1 -DryRun

powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDFull.ps1 -Seed 42 -PreparedRunId RUN_ID_FROM_PREFLIGHT
powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPFull.ps1 -Seed 42
```

Để giới hạn CTRAP ở đúng 500 optimizer step như SEAM và SDD, dùng biến thể:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPFull.ps1 -Seed 42 -MaxSteps 500
```

Mặc định không truyền `-MaxSteps` vẫn là 20 epoch CTRAP. Với 5.000 mẫu safe
và batch 5, 500 step tương đương khoảng 0,5 epoch safe-data; đây chỉ là
so sánh **cùng số bước cập nhật**, không phải cùng số mẫu hay chi phí tính toán.
`manifest.json` và `defense/train_metrics.json` ghi lại giới hạn step thực tế.

### Biến thể CTRAP tiết kiệm tài nguyên trên L40S

Hai launcher sau đều mặc định 500 optimizer step, dùng worker 2 CPU, 16 GiB RAM
và 1 L40S; một coordinator CPU nhỏ vẫn hoạt động trong lúc chờ worker. Chúng
không thay đổi launcher CTRAP/H100 ở trên:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPL40STrain.ps1 -Seed 42
powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPL40SFull.ps1 -Seed 42
```

Lệnh thứ nhất chỉ train CTRAP và xuất `defense/model` cùng `defense/adapter`;
**không có metric đánh giá**. Lệnh thứ hai train rồi chạy 12 lượt eval (base và
CTRAP trước/sau 5 learning rate attack). Attack của lệnh thứ hai là **LoRA**
rank 32/alpha 4 trên q/k/v, không phải full-model attack của SEAM/SDD và
launcher CTRAP/H100. Vì vậy **không được gộp số liệu vào bảng so sánh hiện tại**
như thể cùng protocol. Cả hai launcher nạp model trực tiếp lên GPU để giảm RAM
CPU; cấu hình L40S/16 GiB chưa được smoke test từ đầu đến cuối trên Modal.

### Chỉ train 500 step rồi tự đẩy lên Hugging Face

Hai launcher sau **không chạy attack/eval**. Chúng yêu cầu tên repo model HF
riêng và tạo repo **private**, xác minh quyền ghi bằng một file nhỏ trước khi
thuê GPU. Repo đã chứa checkpoint sẽ bị từ chối để tránh ghi đè. Modal secret
`huggingface-secret` phải có `HF_TOKEN` với quyền ghi lên namespace được chọn.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDTrainPushHF.ps1 `
  -Seed 42 -AnswerPool AlpacaOnly -PreparedRunId RUN_ID_FROM_PREFLIGHT `
  -HfRepo ducthang1703/sdd-llama2-7b-seed42-step500

powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPTrainPushHF.ps1 `
  -Seed 42 -HfRepo ducthang1703/ctrap-llama2-7b-seed42-step500
```

SDD vẫn cần preflight dữ liệu hoàn tất và train full-model trên 1 H100. CTRAP
train LoRA trên 1 L40S, worker 2 CPU/16 GiB RAM. Cả hai đều chỉ upload nếu
`global_step=500`; full model nằm ở gốc repo HF, adapter CTRAP ở `adapter/`,
và `manifest.json` cùng `train_metrics.json` đi kèm. Nếu upload thất bại, model
vẫn nằm trên Modal Volume trong `defense-compare-results`. Có thể retry upload
**không train lại** bằng chính script cũ:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Run-SDDTrainPushHF.ps1 `
  -Seed 42 -PreparedRunId RUN_ID_FROM_PREFLIGHT `
  -HfRepo ducthang1703/sdd-llama2-7b-seed42-step500 -UploadOnly

powershell -ExecutionPolicy Bypass -File .\scripts\Run-CTRAPTrainPushHF.ps1 `
  -Seed 42 -RunId CTRAP_RUN_ID `
  -HfRepo ducthang1703/ctrap-llama2-7b-seed42-step500 -UploadOnly
```

SDD mặc định `-AnswerPool AlpacaOnly` để chạy được khi chưa có quyền LIMA.
Đây là biến thể SDD đầy đủ 8k cặp/500 step nhưng **khác nguồn đáp án của paper**;
`manifest.json` ghi rõ chế độ này. Khi đã có quyền LIMA, dùng
`-AnswerPool AlpacaLima` để dùng cả Alpaca và LIMA. Cả hai chế độ lấy prompt
từ BeaverTails `330k_train`, loại trùng với attack và `30k_test`.
Preflight SDD chạy ngầm trên CPU, chuẩn bị đủ 8k cặp và lưu `run-id` được in ra.
Kiểm tra `status.json` trước khi nộp full-run:

```powershell
modal volume get defense-compare-results sdd-seed42-RUN_ID_FROM_PREFLIGHT/status.json -
```

Chỉ khi `state` là `complete`, chạy `Run-SDDFull.ps1` với **chính run-id đó**,
cùng seed và answer pool. Launcher kiểm tra preflight trước khi submit; runner
kiểm tra lại manifest, SHA-256 và số lượng 8k cặp trước khi thuê H100. Dữ liệu
preflight được dùng lại, không dựng lần thứ hai. Nếu preflight bị dừng hoặc lỗi,
tạo run-id mới bằng cách chạy lại `Run-SDDPreflight.ps1`.

Sau khi lệnh nộp job thành công, có thể tắt máy; job chạy trên Modal. Chạy
hai lệnh đồng thời có thể dùng **2 H100 tổng cộng** (mỗi run một GPU). Script
ghi `run ID` riêng, không ghi đè run cũ. Theo dõi bằng `modal app list` và
`modal app logs <app-id>`; tải về bằng:

```powershell
modal volume get defense-compare-results sdd-seed42-<run-id> .\results\sdd
modal volume get defense-compare-results ctrap-seed42-<run-id> .\results\ctrap
```

Tải thêm run SEAM cùng seed rồi xuất bảng đối chiếu:

```powershell
modal volume get seam-full-results seam-full-seed42-<run-id> .\results\seam
powershell -ExecutionPolicy Bypass -File .\scripts\Compare-DefenseResults.ps1 `
  -SeamRunDir .\results\seam -SddRunDir .\results\sdd `
  -CtrapRunDir .\results\ctrap -OutputPath .\results\comparison.md
```

Script đối chiếu sẽ dừng nếu seed, SHA-256 nguồn RepNoise hoặc cỡ tập test
khác nhau. Hãy chạy lại SEAM bằng launcher hiện tại để attack cũng bật gradient
checkpointing như SDD/CTRAP.

Mỗi run có `manifest.json`, `status.json`, `defense/train_metrics.json`,
`metrics/*.json`, `summary.json` và checkpoint full model. Khi thiếu dữ liệu,
quyền truy cập LIMA (nếu chọn AlpacaLima)/Llama-2 hoặc không đủ 6k cặp RepNoise hợp lệ, script dừng
và ghi lỗi trước/sát giai đoạn liên quan; không thay dữ liệu một cách âm thầm.

## Giới hạn quan trọng

- SDD: repo tác giả hiện **không có `train/`** dù README nói tới LLaMA-Factory.
  Script dựng lại objective SFT từ paper, dùng chat template và optimizer 8-bit
  để phù hợp 1 H100; không thể khẳng định tái lập đúng code chưa được công bố.
- SDD: `GAIR/lima` là dataset có điều kiện truy cập; chế độ `AlpacaLima` sẽ
  dừng ở bước CPU nếu tài khoản chưa có quyền. `AlpacaOnly` chạy không cần
  LIMA nhưng không phải cùng answer pool với paper. Builder dùng BeaverTails
  `330k_train` vì `30k_train` chỉ cho 3.264 prompt hợp lệ sau lọc, không đủ 8k.
- CTRAP: dùng `CTRAPTrainer` gốc nhưng thay phần nạp dữ liệu hard-code và dùng
  Llama-2-7b-**chat** để chung backbone với SEAM; paper/launcher CTRAP gốc có
  thể dùng bản base. Tập refusal RepNoise công khai không chứng minh do GPT-4o
  sinh ra. Adapter giữ công thức perturbation/gradient của code CTRAP.
- CTRAP: adapter rút ngắn prompt khi cần để giữ ít nhất 32 token cho answer
  trong ngữ cảnh 256 token. Smoke test ghi nhận 1.296/5.000 prompt UltraChat
  bị rút ngắn; đây là khác biệt tiền xử lý cần báo cáo khi so sánh. Full-run
  giữ tối đa hai checkpoint huấn luyện gần nhất, mỗi 1.000 step ở chế độ 20 epoch;
  với `-MaxSteps 500`, checkpoint được lưu tại step 500.
- Tập attack/eval chung không tạo ra kết quả trực tiếp so được với số công bố
  trong từng paper. Chỉ đối chiếu các run **cùng seed và cùng protocol**.
- Chưa kiểm chứng toàn bộ 20 epoch/500 step và evaluation sweep end-to-end.
  Hãy xem lại quota/chi phí trước khi nộp cấu hình full.
