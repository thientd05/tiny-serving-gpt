"""Model giả để kiểm chính sách xếp lịch mà không cần GPU.

Vì sao cần: các phép kiểm "lấy tối đa B", "lô đóng kín", "đúng thứ tự FIFO",
"sinh đúng output_len" là chuyện của vòng lặp xếp lịch, không phải chuyện của
model. Tách ra thì test chạy trong vài giây, không tranh GPU với phép đo, và
khi test đỏ thì biết chắc lỗi nằm ở engine.

Model giả sinh token theo một hàm thuần: token thứ k của một request = hàm băm
của (token cuối của prompt, k). Nhờ đó kiểm được cả tính tái lập và việc đệm
trái không làm lẫn ngữ cảnh giữa các request trong cùng lô.
"""
from __future__ import annotations

import torch


class _GenCfgHolder:
    eos_token_id = None
    do_sample = False


class FakeModel:
    """Chỉ cài đúng phần `generate()` mà engine dùng."""

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id
        self.generation_config = _GenCfgHolder()
        self.calls: list[dict] = []     # ghi lại từng lần gọi, để test kiểm

    @staticmethod
    def _tok(seed: int, k: int) -> int:
        return 1000 + (seed * 7919 + k * 104729) % 50000

    def generate(self, input_ids, attention_mask=None, generation_config=None,
                 stopping_criteria=None, **kw):
        n_new = generation_config.max_new_tokens
        B, L = input_ids.shape
        self.calls.append({"batch": B, "padded_len": L, "n_new": n_new,
                           "ids": input_ids.clone(),
                           "mask": None if attention_mask is None
                                   else attention_mask.clone()})

        # Lấy token cuối ở ĐÚNG cột cuối cùng — chính là điều mà một model
        # đệm-trái thật dựa vào. Cố ý KHÔNG tự đoán kiểu đệm: nếu engine đệm
        # phải thì cột cuối là token đệm, mọi request trong lô cùng seed, và
        # test đệm trái sẽ đỏ. Model giả mà "thông minh" hơn engine thì phép
        # kiểm mất khả năng thất bại (RISKS.md mục C).
        last_real = input_ids[:, -1]

        new = torch.empty((B, n_new), dtype=input_ids.dtype)
        for i in range(B):
            seed = int(last_real[i])
            for k in range(n_new):
                new[i, k] = self._tok(seed, k)

        if stopping_criteria:
            for _ in range(n_new):
                for c in stopping_criteria:
                    c(input_ids, None)
        return torch.cat([input_ids, new.to(input_ids.device)], dim=1)


class FakeTok:
    pad_token_id = 0
    eos_token_id = 128001
