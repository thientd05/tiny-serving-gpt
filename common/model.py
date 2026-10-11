"""Nạp model và ép về chế độ mà G0 yêu cầu.

Ba thứ phải ép tường minh, vì mặc định của thư viện không khớp yêu cầu G0:

1. **Greedy.** `generation_config.json` của checkpoint bật lấy mẫu
   (`do_sample: true, temperature: 0.6, top_p: 0.9`). Không ghi đè thì kết quả
   không tái lập được, và cả dự án mất mốc tham chiếu.
2. **Attention eager.** transformers v5 mặc định dùng SDPA. G0 phải eager —
   kernel hợp nhất là việc của G4, và phần cải thiện của nó phải đo được riêng.
3. **dtype.** Chốt ở Phần 0 (xem .claude/env_report.md). Giữ nguyên qua G0→G11;
   đổi giữa đường là mọi so sánh mất giá trị.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from common.config import ServingConfig

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}


def greedy_config(output_len: int, ignore_eos: bool, pad_token_id: int,
                  eos_token_id: Any = None) -> GenerationConfig:
    """Sinh **đúng** output_len token, tham lam, không lấy mẫu.

    Vì sao `min_new_tokens == max_new_tokens` và bỏ qua EOS: mỗi mẫu workload đã
    ghi sẵn `output_len`. Nếu để EOS dừng sớm thì hai lần chạy khác `B` dừng ở
    chỗ khác nhau, và "lệch token" lẫn luôn vào "dừng khác chỗ" — không còn đo
    được cái nào. Bỏ qua EOS làm một kịch bản tải phát lại y hệt qua mọi bậc.
    """
    return GenerationConfig(
        do_sample=False, num_beams=1,
        temperature=None, top_p=None, top_k=None,            # tránh cảnh báo v5
        max_new_tokens=output_len, min_new_tokens=output_len,
        eos_token_id=None if ignore_eos else eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=True,
    )


def load_tokenizer(cfg: ServingConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    tok.padding_side = "left"          # G0: đệm trái
    if tok.pad_token_id is None:
        # Llama 3.2 không có pad token. Giá trị không ảnh hưởng kết quả vì
        # attention mask che hết, nhưng phải có một id hợp lệ để dựng tensor.
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: ServingConfig):
    if cfg.dtype not in DTYPES:
        raise ValueError(f"dtype phải thuộc {sorted(DTYPES)}, nhận {cfg.dtype!r}")
    tok = load_tokenizer(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=DTYPES[cfg.dtype],            # v5: `dtype`, không phải `torch_dtype`
        attn_implementation=cfg.attn_impl,
    )
    model = model.to(cfg.device).eval()
    model.generation_config.do_sample = False   # chặn mặc định của checkpoint

    got = model.config._attn_implementation
    if got != cfg.attn_impl:
        raise RuntimeError(
            f"xin attention {cfg.attn_impl!r} nhưng thư viện dùng {got!r}")

    torch.manual_seed(cfg.seed)
    return model, tok


def model_facts(model) -> dict:
    """Thông số kiến trúc dùng để tính ngân sách KV cache."""
    c = model.config
    kv_heads = getattr(c, "num_key_value_heads", c.num_attention_heads)
    head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    bytes_per = next(model.parameters()).element_size()
    return {
        "layers": c.num_hidden_layers,
        "q_heads": c.num_attention_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": bytes_per,
        # 2 vì phải giữ cả K và V
        "kv_bytes_per_token": 2 * kv_heads * head_dim * c.num_hidden_layers * bytes_per,
        "max_position_embeddings": c.max_position_embeddings,
        "vocab_size": c.vocab_size,
    }
