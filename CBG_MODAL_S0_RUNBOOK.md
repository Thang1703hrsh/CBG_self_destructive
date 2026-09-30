# Chạy setting đầu tiên S0 trên Modal

Launcher: `scripts/modal_cbg_s0.py`. Đây là `cbg_version: spec`, không phải V5.
Mỗi lệnh chỉ chạy một giai đoạn; không tự khởi chạy toàn bộ sweep hoặc 7B.
Bản mã trước đã chạy thành công bước `verify` trên CPU Modal (139 tests, không
skip). Sau mỗi thay đổi mã, kể cả thay đổi chia dữ liệu hiện tại, phải chạy lại
`verify` với `run-name` mới. Probe FP32 full-parameter 3B thực tế đã vượt 79.18
GiB trên H100 trước optimizer step; pilot training chưa được xác nhận bằng lượt
chạy thành công.

## 1. Cài CLI và kiểm thử CPU trước

Trong PowerShell, ở thư mục repository:

```powershell
Set-Location C:\Users\ducth\OneDrive\PC\CBG
python -m pip install -U modal PyYAML
python -m modal setup
modal run scripts/modal_cbg_s0.py --stage verify --run-name s0-first
```

`verify` dựng image theo `requirements.txt`, chạy toàn bộ tests trên CPU và lưu
`s0-first/verification.json` cùng `.log` vào Volume `cbg-spec-s0-output`.
Không cấp GPU cho bước này. Image build/CPU vẫn có thể phát sinh phí Modal.
Nếu dependency pins trên Modal không vượt qua tests, dừng và sửa môi trường;
không bỏ qua verification để chạy GPU.

Nếu gặp `verification failed or skipped required integration checks`, đó là thông
báo tổng kết; nguyên nhân nằm trong traceback của test ở `verification.log`.
Launcher hiện in 100 dòng cuối khi thất bại. Để tải log trong PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
modal volume get cbg-spec-s0-output s0-first/verification.log verification-modal.log
Get-Content -Encoding UTF8 verification-modal.log -Tail 100
```

Lỗi T1 `vmap: inplace arithmetic` với Transformers 4.49 đã được sửa bằng causal
mask 4D dựng ngoài `vmap` trong test; không thay đổi thuật toán CBG hay nới tolerance.

Trước các bước tải model, tạo Modal Secret tên **huggingface-secret**, chứa
`HF_TOKEN`, bằng dashboard Modal. Với Llama, tài khoản Hugging Face phải có quyền
truy cập model. Không ghi token vào YAML hoặc commit vào repo.

## 2. Chọn pilot và điền các quyết định

```powershell
Copy-Item config/cbg_scenarios.yaml config/cbg_s0_modal.yaml
Copy-Item config/cbg_scenario_choices.example.yaml pilot_choices.yaml
```

Trong `config/cbg_s0_modal.yaml`:

- `models.pilot`: có thể dùng `meta-llama/Llama-3.2-1B-Instruct` cho lần đầu kiểm
  tra bộ nhớ; template scenario mặc định là Llama-3.2-3B. Đổi model thì dùng
  `run-name` mới. S0 chấp nhận pilot ≤3B-class.
- `data.refusal_model`: model/revision thực sự đã sinh refusal của dữ liệu.
- `pilot.max_fd_relative_error`, `pilot.max_alignment_growth`,
  `pilot.max_benign_kl`: các ngưỡng chấp nhận, phải quyết định trước khi xem kết quả.
- Giữ `pilot.max_steps: 100`, m=8, n=32, k=4, FP32 cho phép kiểm chứng chính.
- Sau khi đo radius, đặt `search.relative_radius` thành radius đã chọn.

Trong `pilot_choices.yaml`, đặt đủ `alpha`, `beta`, `xi`, `kappa_mu`,
`relative_radius`. Có thể tham khảo các giá trị **pilot chưa hiệu chỉnh** đã có
trong `config/cbg_spec_pilot.yaml`: 1.0, 0.01, 0.001, 0.05, 0.001 tương ứng.
Chúng không phải hyperparameter tối ưu hoặc kết quả của validation search.
`relative_radius` ban đầu là tạm thời; bước radius sẽ quét cả sáu giá trị.
Giữ alpha/beta/xi/kappa không đổi giữa đo radius và train pilot.

Không cần điền search grid, judge và external baselines để làm S0.1–S0.4.

## 3. Đưa dữ liệu lên Volume

Launcher có thể tải trực tiếp ba nguồn công khai đã ghim revision vào Modal Volume,
kiểm tra schema/SHA-256 và lưu `cbg_s0_data_receipt_v2.json`. Không cần tải qua máy:

```powershell
modal run scripts/modal_cbg_s0.py --stage data
modal volume ls cbg-spec-data
```

Stage `data` dùng CPU, không cấp GPU. Nếu receipt hợp lệ đã tồn tại, lệnh trả về
`status: cached` và không tải lại. Nó ghép các harmful response trong BeaverTails
`330k_train` với refusal đã sinh sẵn trong RepNoise bằng prompt chuẩn hóa; benign
dùng `yahma/alpaca-cleaned`, evaluation dùng đúng tệp BeaverTails `30k_test`.
Tất cả nguồn đều được ghim revision. Đặt `data.refusal_model` đúng với provenance
RepNoise đã ghi trong template S0.

Sau khi hoàn tất, Volume phải có `beavertails_cbg_train.json`,
`alpaca_train.json`, `beavertails_test.json` và `cbg_s0_data_receipt_v2.json`.

Nếu muốn dùng dữ liệu riêng thay vì các nguồn mặc định, upload thủ công:

```powershell
modal volume create cbg-spec-data
modal volume put cbg-spec-data data/beavertails_cbg_train.json /beavertails_cbg_train.json
modal volume put cbg-spec-data data/alpaca_train.json /alpaca_train.json
modal volume put cbg-spec-data data/beavertails_test.json /beavertails_test.json
```

Bỏ qua lệnh `create` nếu Volume đã tồn tại. Các tệp phải là JSON arrays đúng
schema trong `CBG_SCENARIO_RUNBOOK.md`. Ba pool harmful phải tách prompt hoàn toàn
sau khi loại prompt thuộc evaluation. Mỗi pool được dùng tối đa hai response khác
nhau cho một prompt (`max_rows_per_prompt: 2`) để đạt đúng 4K defense, 1K
validation và 10K test mà không gây rò rỉ prompt giữa các pool.
Alpaca cần ≥4K output hợp lệ; BeaverTails test phải là toàn bộ split.
Không tự tải một bản dataset không rõ provenance để thay thế dữ liệu của bạn.

Launcher ánh xạ ba tên tệp cố định trên Volume sang `/vol/data/...` và tạo
manifest mới trên Linux. Không upload manifest Windows có đường dẫn `C:\...`.

```powershell
modal run scripts/modal_cbg_s0.py --stage prepare --run-name s0-first --protocol config/cbg_s0_modal.yaml --choices pilot_choices.yaml
```

Bước này dùng CPU: kiểm tra dữ liệu và choices, tạo split có hash, tải đúng
revision của model, tạo probe bank. Cache ở `cbg-spec-hf`. Không cấp GPU.

## 4. Probe bộ nhớ, sau đó đo radius

```powershell
$env:CBG_MODAL_GPU = "H100"
modal run scripts/modal_cbg_s0.py --stage probe --run-name s0-first --protocol config/cbg_s0_modal.yaml --choices pilot_choices.yaml
modal run --detach scripts/modal_cbg_s0.py --stage radius --run-name s0-first --protocol config/cbg_s0_modal.yaml --choices pilot_choices.yaml
```

`probe` chạy đúng **một optimizer step**, ghi log và checkpoint vào `probe/`.
`radius` là S0.2: so FD với exact HVP, giữ cùng samples/signs. Exact HVP có
thể tốn bộ nhớ hơn probe. Lượt chạy 3B FP32 full-parameter trên H100 đã dùng
78.66/79.18 GiB rồi OOM khi cần cấp thêm 768 MiB ở forward đầu tiên. Dùng H200
cho run này; nếu H200 không khả dụng, tạo run mới với `reference_device: cpu`
để đổi bộ nhớ GPU lấy thời gian CPU. Không bật
BF16 hoặc giảm m/n âm thầm vì sẽ thay setting cần kiểm chứng. Multi-GPU sharding
chưa được triển khai; đừng dùng `H100:2` như cách cộng VRAM cho một model.

Modal có thể nâng yêu cầu H100 sang H200; loại GPU thực nhận được ghi trong
`hardware-<stage>.json`. Nếu cần so sánh chi phí trên đúng phần cứng, kiểm tra
metadata từng run. `--detach` cho phép job tiếp tục khi máy local ngắt kết nối.

```powershell
modal volume get cbg-spec-s0-output s0-first/radius.json output/modal-s0/radius.json
```

Chọn r có relative error dưới ngưỡng đã đặt và có ít nhất một radius liền kề
cũng đạt ngưỡng. Cập nhật **cả** `pilot_choices.yaml: relative_radius` và
`config/cbg_s0_modal.yaml: search.relative_radius`. Đừng đổi các ngưỡng acceptance
sau khi thấy kết quả. Nếu không có vùng đạt yêu cầu, chưa chạy train 100 bước.

## 5. Pilot 100 bước và kiểm tra gate

```powershell
modal run --detach scripts/modal_cbg_s0.py --stage train --run-name s0-first --protocol config/cbg_s0_modal.yaml --choices pilot_choices.yaml
modal run scripts/modal_cbg_s0.py --stage gate --run-name s0-first --protocol config/cbg_s0_modal.yaml --choices pilot_choices.yaml
```

`train` khởi tạo lại từ model gốc, không tiếp tục từ optimizer step của probe.
Trước cấp GPU có CPU preflight kiểm tra verification/data/structure và radius.
Lưu checkpoint mỗi 25 bước, giữ tối đa hai checkpoint, model cuối ở `pilot/final`.
Log, weights, config, hardware, radius, gate đều ở Volume output. Để resume:

```powershell
modal run --detach scripts/modal_cbg_s0.py --stage train --run-name s0-first --protocol config/cbg_s0_modal.yaml --choices pilot_choices.yaml --resume checkpoint-50
```

Chỉ dùng tên checkpoint thực sự đã lưu. Chạy lại từ đầu thì dùng run-name mới.
Volume được commit khi hàm kết thúc hoặc có exception thông thường; không đảm
bảo lưu được checkpoint mới nhất nếu container bị kill đột ngột trong lúc ghi.

```powershell
modal volume ls cbg-spec-s0-output s0-first
modal volume get cbg-spec-s0-output s0-first output/modal-s0
```

S0 gate phải `passed` trước khi materialize S1. Không dùng một run có NaN/OOM,
geometry không giảm, KL vượt ngưỡng hoặc chỉ có probe một bước làm bằng chứng
đã hoàn thành pilot. Chạy S0.5 và validation search theo scenario runbook sau đó.

Tài liệu Modal: [run/detach](https://modal.com/docs/cli/latest/run),
[GPU](https://modal.com/docs/guide/gpu), [Volumes](https://modal.com/docs/cli/latest/volume),
[Secrets](https://modal.com/docs/guide/secrets).
