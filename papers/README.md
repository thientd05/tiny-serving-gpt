# Thư viện paper tham khảo

30 paper, tải ngày 2026-10-10. Mục đích: nguồn tra cứu khi cài đặt và khi viết nhận xét —
mọi tuyên bố về thuật toán trong báo cáo phải chỉ được về một paper ở đây.

Quy ước tên file: `<tên-gọn>-<mã-arxiv>.pdf`. File PDF **không commit** (xem `.gitignore`);
mất thì tải lại theo mã arXiv trong tên file.

---

## 00-background — nền tảng, đọc để hiểu bài toán

| File | Vì sao cần |
|---|---|
| `attention-is-all-you-need-1706.03762` | Kiến trúc gốc. Cần để biết chính xác KV cache là cái gì và vì sao nó phình theo độ dài. |
| `gqa-grouped-query-attention-2305.13245` | Model của ta có 32 head attention nhưng chỉ 8 bộ K/V — chính là kỹ thuật trong paper này. Nó đã tiết kiệm KV cache 4 lần **ngay trong model**, nên là mốc nền mà mọi thuật toán của ta cộng thêm lên. |
| `llama3-herd-of-models-2407.21783` | Báo cáo chính thức của model ta dùng. Tra tham số, cách huấn luyện, cách xử lý context dài. |
| `efficiently-scaling-transformer-inference-2211.05102` | **Paper kinh điển quan trọng nhất cho dự án này.** Nó phân tích pha sinh token bị nghẽn ở băng thông bộ nhớ chứ không phải ở sức tính — đây là gốc rễ giải thích vì sao GPU 1650 yếu vẫn có thể cải thiện nhiều bằng tối ưu bộ nhớ. |

## 01-kv-memory — quản lý và tái dùng bộ nhớ KV (trọng tâm số 1)

| File | Vì sao cần |
|---|---|
| `vllm-pagedattention-2309.06180` | Thuật toán cốt lõi: cấp bộ nhớ KV theo từng khối nhỏ như phân trang của hệ điều hành, thay vì cấp sẵn một vùng liền theo độ dài tối đa. Quyết định số người online cùng lúc trên 4GB VRAM. *(Copy từ dự án `cuda-llm`.)* |
| `vattention-virtual-memory-kv-2405.04437` | Làm được việc của PagedAttention nhưng dùng bộ nhớ ảo của CUDA để giữ vùng KV liền mạch, nhờ đó **không phải sửa kernel attention**. Đối chứng trực tiếp với PagedAttention và là ứng viên tốt nhất cho phần tự viết CUDA. |
| `sglang-radixattention-2312.07104` | Tái dùng phần ngữ cảnh trùng nhau giữa các request bằng một cây tiền tố. Đúng kịch bản web chat: chung system prompt, và lượt chat sau chứa trọn lượt trước. |
| `prompt-cache-modular-reuse-2311.04934` | Tái dùng cả những đoạn trùng **không nằm ở đầu** prompt. Mạnh hơn prefix cache nhưng phải xem kỹ chỗ nó đánh đổi độ chính xác — có thể không còn loss-less. |

## 02-batching-scheduling — nhận và xếp lịch request

| File | Vì sao cần |
|---|---|
| `orca-iteration-level-scheduling-osdi22` | Paper kinh điển về continuous batching: xếp lịch lại sau **mỗi bước sinh token** nên request mới vào được ngay, request xong thì rời ngay. (Hội nghị OSDI'22, không có trên arXiv.) |
| `sarathi-chunked-prefill-2308.16369` | Cắt prompt dài thành từng khúc để xen kẽ với việc sinh token của người khác, tránh tình trạng một prompt dài làm cả hệ thống đứng vài giây. |
| `sarathi-serve-throughput-latency-2403.02310` | Bản hoàn chỉnh của ý trên, kèm cách định nghĩa và đo các chỉ số độ trễ. **Dùng paper này làm chuẩn cho bộ chỉ số đo của dự án.** |
| `fastserve-preemptive-scheduling-2305.05920` | Cho phép tạm dừng request đang chạy để nhường chỗ việc ngắn, tránh người hỏi một câu ngắn phải chờ sau người sinh 2000 token. |
| `vtc-fair-scheduling-2401.00588` | Chia GPU công bằng giữa nhiều người dùng. Cần khi mô phỏng tình huống thật: một người spam request không được làm chết trải nghiệm của người khác. |

## 03-kernels — tối ưu mức kernel (trọng tâm Giai đoạn 2)

| File | Vì sao cần |
|---|---|
| `flashattention-2205.14135` | Bản gốc, và là bản **còn hỗ trợ GPU Turing (sm75) như máy này**. Gộp các bước attention để không ghi bảng điểm trung gian ra bộ nhớ chậm. |
| `flashattention2-2307.08691` | Bản nhanh hơn nhưng **yêu cầu GPU đời Ampere trở lên** → gần chắc không chạy được ở đây. Đọc để biết nó tăng tốc bằng cách nào, rồi xét xem phần nào bắt chước được trên sm75. Đây là lý do chính đáng để tự viết kernel. |
| `flashdecoding-plusplus-2311.01282` | Tối ưu riêng cho **pha sinh token** (chỉ một token mới, context rất dài) — đúng phần chiếm hầu hết thời gian của một server chat. |
| `hydragen-shared-prefix-attention-2402.05099` | Khi nhiều người chung một prefix, gộp phép tính trên đoạn đó cho cả lô thành phép nhân ma trận lớn. Biến việc nghẽn băng thông thành việc dùng sức tính — rất đáng thử trên GPU không có Tensor Core. |

## 04-speculative — sinh nhiều token mỗi lượt, vẫn giữ nguyên output

Cơ chế kiểm tra lại của nhóm này đảm bảo kết quả **giống y hệt** bản thường, nên vẫn là loss-less.

| File | Vì sao cần |
|---|---|
| `speculative-decoding-leviathan-2211.17192` | Bản gốc: dùng một model nháp đoán trước nhiều token, model chính kiểm một lượt. Cần đọc phần chứng minh "không đổi output". |
| `speculative-sampling-deepmind-2302.01318` | Cùng ý tưởng, trình bày rõ hơn phần toán của bước kiểm. |
| `medusa-multi-head-decoding-2401.10774` | Không cần model nháp riêng, gắn thêm vài đầu dự đoán vào chính model — hợp hơn với ta vì VRAM không đủ chỗ cho model nháp thứ hai. |
| `eagle-2401.15077` | Bản đoán chính xác hơn Medusa. Đọc để chọn, nhưng cần cân nhắc chi phí VRAM phát sinh. |

## 05-offload-multiturn — tận dụng RAM CPU, phục vụ chat nhiều lượt

| File | Vì sao cần |
|---|---|
| `flexgen-single-gpu-throughput-2303.06865` | Paper kinh điển về chạy model lớn trên **một GPU yếu** bằng cách đẩy dữ liệu xuống RAM/đĩa và lên lịch truyền thật khéo. Hoàn cảnh gần ta nhất. |
| `cachedattention-multiturn-2403.19708` | Trong web chat, người dùng đọc câu trả lời mất vài chục giây, KV cache của họ nằm chiếm VRAM vô ích. Paper này đẩy nó xuống RAM rồi nạp lại khi họ gõ tiếp. Rất đúng bài toán của ta. |
| `mooncake-kvcache-centric-2407.00079` | Hệ thống thật coi KV cache là trung tâm, chia tầng bộ nhớ. Ta không có nhiều máy, nhưng cách chia tầng VRAM → RAM thì áp dụng được. |

## 06-lossy-not-used — nhóm lossy, tải về để lập luận vì sao LOẠI

Dự án **không dùng** nhóm này (model 1B bị hỏng chứ không chỉ kém đi). Nhưng loại thì phải
loại có căn cứ: đọc để chỉ ra chính xác chúng đánh đổi cái gì, và mức model nào thì chịu được.

| File | Nó làm gì |
|---|---|
| `h2o-heavy-hitter-oracle-2306.14048` | Giữ lại các token "quan trọng", bỏ phần còn lại. |
| `streamingllm-attention-sinks-2309.17453` | Chỉ giữ vài token đầu cộng một cửa sổ gần nhất. |
| `snapkv-2404.14469` | Tỉa KV cache ngay sau khi đọc prompt. |
| `kivi-2bit-kv-quantization-2402.02750` | Nén KV cache xuống 2 bit. |

## 07-disaggregation — đọc để phân tích, không triển khai

Cần từ 2 GPU trở lên, ta chỉ có 1. Đọc để biết **vì sao** người ta tách và lấy ý tưởng
xếp lịch áp dụng trong phạm vi một GPU.

| File | Nó làm gì |
|---|---|
| `distserve-2401.09670` | Tách việc đọc prompt và việc sinh token ra 2 nhóm máy khác nhau vì chúng có tính chất đối lập. |
| `splitwise-2311.18677` | Cùng hướng, phân tích sâu phần điện năng và loại GPU phù hợp cho từng pha. |
