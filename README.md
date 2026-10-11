# tiny-serving-gpt

Một hệ thống serving LLM nhỏ, phát triển **dần theo từng giai đoạn** trên đúng một GPU yếu
(GTX 1650, 4 GB, Turing sm75) với một model nhỏ (Llama 3.2 1B Instruct).

Mục tiêu không phải là chạy cho xong một model, mà là dựng một hệ thống **hành xử như server
serving thật**: request đến bất kỳ lúc nào, prompt và câu trả lời dài ngắn khác nhau, nhiều
người dùng cùng lúc. Trọng tâm nghiên cứu là các thuật toán tối ưu KV cache thuộc nhóm
**loss-less** — không được đổi kết quả model sinh ra, chỉ được đổi tốc độ và bộ nhớ.

## Cách dự án này được tổ chức

Mỗi giai đoạn thêm **đúng một cơ chế**, và cơ chế đó phải sửa một hạn chế **đo được** của
giai đoạn ngay trước. Mỗi giai đoạn chia hai phần:

1. **Build hệ thống** — thêm cơ chế, kiểm nó chạy đúng như đặc tả.
2. **Đo tìm nút thắt** — đo hiệu năng để chỉ ra điểm nghẽn còn lại. Điểm nghẽn đó chính là
   lý do tồn tại của giai đoạn sau.

Không giai đoạn nào được tuyên bố "nhanh hơn" mà không có số đo kèm điều kiện đo.

| Giai đoạn | Cơ chế thêm vào |
|---|---|
| **G0** | base thuần Python, xếp lịch theo **lô cố định** (static batching) |
| G1 | cùng cơ chế, viết bằng C++ / graph tĩnh |
| G2 | continuous batching (xếp lịch lại sau mỗi token) |
| G3 | cấp bộ nhớ KV theo khối — PagedAttention ⟷ vAttention |
| G4 | kernel attention hợp nhất có sẵn trong thư viện |
| G5 | xếp lịch prompt dài — cắt thành khúc ⟷ tạm dừng request |
| G6 | prefix cache (cây tiền tố) |
| G7 | gộp phép tính trên prefix dùng chung |
| G8 | đẩy KV của phiên đang nghỉ xuống RAM CPU |
| G9 | speculative decoding |
| G10 | lớp chọn cơ chế theo hoàn cảnh |
| G11 | tự viết kernel CUDA, ở những chỗ **đã đo được** là thư viện chưa đủ tốt trên sm75 |

## Bố trí mã nguồn

```
common/     nạp model, bộ chỉ số, phát lại workload, đọc trạng thái máy, cấu hình
serving/    hệ thống serving — MỘT codebase duy nhất, phát triển dần qua các giai đoạn
stages/     mỗi giai đoạn một thư mục: cấu hình bật/tắt cơ chế + ghi chú riêng của nó
tests/      kiểm chức năng (chạy không cần GPU, dùng model giả)
```

## Chạy

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# chạy một kịch bản tải qua hệ thống
.venv/bin/python -m serving --config stages/G0-base-python/g0.json \
    --workload alpaca --n 200 --pattern poisson --rate 2.0

# kiểm chức năng (không cần GPU)
.venv/bin/python -m pytest tests/ -q

# lớp HTTP để thử tay (không dùng khi đo — xem serving/server.py)
.venv/bin/python -m serving --serve --port 8000
curl -s localhost:8000/health
```

Model và dataset không nằm trong repo. `models/` và `data/external/` là symlink sang nơi đã
có sẵn trên máy; `requirements.txt` ghi rõ phiên bản đã **đo được** là chạy trên sm75.

## Giai đoạn hiện tại: G0

Hệ thống G0 làm đúng một việc, lặp mãi: GPU rỗi thì lấy tối đa `B` request đang chờ ở đầu
hàng đợi (**không** đợi cho đủ `B`), đệm trái cho bằng prompt dài nhất, gọi `generate()` một
lần cho cả lô, chờ **cả lô** xong, rồi quay lại đầu.

Đây là baseline "static batching" mà các paper continuous batching (Orca, vLLM) lấy làm mốc
so sánh. Hai hạn chế nằm ngay trong cơ chế, và chúng là phần việc của G2/G3:

- trong lúc lô đang chạy, người đến sau không vào được, và chỗ của người đã sinh xong không
  được dùng lại;
- prompt ngắn bị đệm cho bằng prompt dài nhất, tốn cả bộ nhớ lẫn phép tính cho token đệm.

G0 cũng là nơi **sinh ra** mốc tham chiếu token-exact (bản `B=1`) mà mọi giai đoạn sau phải
đối chiếu để chứng minh mình không làm đổi kết quả.
