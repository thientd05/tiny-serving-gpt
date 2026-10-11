# G0 — base: serving thuần Python, chạy từng lô cố định

## Cơ chế

```
GPU rỗi  →  lấy tối đa B request đang chờ ở đầu hàng đợi (KHÔNG đợi cho đủ B)
         →  đệm trái cho bằng prompt dài nhất trong lô
         →  gọi generate() một lần cho cả lô
         →  chờ CẢ LÔ sinh xong
         →  quay lại đầu
```

Attention **eager**, KV cache cấp liền mạch (`DynamicCache`), greedy, dtype **bfloat16**.

## Vì sao base là "lô cố định" chứ không phải "một request một lúc"

`generate()` nhận cả lô sẵn, nên một base chạy một request một lúc là base **yếu giả tạo**:
mọi giai đoạn sau sẽ thắng chỉ nhờ biết dùng batch, che mất phần cải thiện thật của thuật
toán. Base phải là bản tốt nhất làm được trong khuôn khổ lô cố định.

## Hai núm điều chỉnh

| Núm | Mặc định | Ý nghĩa |
|---|---|---|
| `batch_size` (`B`) | xem `g0.json` | số request tối đa trong một lô |
| `batch_timeout_s` | `0.0` | đợi thêm bao lâu để gom lô to hơn. `0` = không đợi, đúng đặc tả G0 |

## Sinh đúng `output_len` token, bỏ qua EOS

Mỗi mẫu workload đã ghi sẵn `output_len`, và hệ thống sinh **đúng** số đó
(`min_new_tokens == max_new_tokens`, không dừng theo EOS).

Lý do: nếu để EOS dừng sớm, hai lần chạy khác `B` sẽ dừng ở chỗ khác nhau, và "lệch token"
lẫn luôn vào "dừng khác chỗ" — không còn tách được cái nào. Bỏ qua EOS làm mỗi kịch bản tải
phát lại **y hệt** qua mọi giai đoạn G0→G11, đúng mục đích mà `output_len` được ghi sẵn.

## Hạn chế còn lại (phần việc của giai đoạn sau)

| Hạn chế | Giai đoạn sửa |
|---|---|
| Lô đóng kín: người đến sau phải đợi hết lô; chỗ của người xong sớm không được dùng lại | G2 |
| Prompt ngắn bị đệm cho bằng prompt dài nhất → tốn bộ nhớ và phép tính cho token đệm | G2, G3 |
| KV cache cấp liền mạch theo độ dài lớn nhất | G3 |
| Attention eager giữ cả bảng điểm trong bộ nhớ | G4 |
| Một prompt dài vào là cả hệ thống đứng | G5 |

## Vai trò trong cả lộ trình

G0 là nơi **sinh ra** mốc tham chiếu token-exact (bản `B=1`), không phải nơi đối chiếu với
mốc nào — chưa có mốc nào trước nó. Mọi giai đoạn sau phải khớp từng token với mốc này để
được gọi là loss-less.

Số đo và kết luận của G0 nằm ở báo cáo riêng, không nằm trong file này.
