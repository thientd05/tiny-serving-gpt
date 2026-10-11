"""Cấu hình hệ thống serving.

Một dataclass duy nhất giữ mọi núm điều chỉnh. Mỗi lần đo lưu lại đúng cấu hình
đã chạy ra file, để sau này biết con số đo được sinh ra trong điều kiện nào.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ServingConfig:
    # --- model
    model_path: str = str(ROOT / "models" / "Llama-3.2-1B-Instruct")
    dtype: str = "float16"          # chốt ở Phần 0; xem .claude/env_report.md
    attn_impl: str = "eager"        # G0 bắt buộc eager
    device: str = "cuda"

    # --- xếp lịch (núm điều chỉnh của G0)
    batch_size: int = 8             # B: số request tối đa trong một lô
    batch_timeout_s: float = 0.0    # 0 = không đợi, lấy luôn cái gì đang chờ

    # --- giới hạn
    max_context: int = 4096         # trần độ dài prompt + output nhận vào

    # --- sinh token
    ignore_eos: bool = True         # sinh đúng output_len, xem README G0
    seed: int = 0

    # --- ghi chép
    run_id: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ServingConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"cấu hình có khóa lạ: {sorted(unknown)}")
        return cls(**raw)

    def replace(self, **kw) -> "ServingConfig":
        d = self.to_dict()
        d.update(kw)
        return ServingConfig(**d)
