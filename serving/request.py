"""Một request trong hệ thống."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    prompt_ids: list[int]
    output_len: int          # số token phải sinh, đã ghi sẵn trong workload
    arrival_t: float         # giây, theo đồng hồ thật (time.perf_counter)

    # điền dần trong lúc chạy
    schedule_t: float = math.nan
    first_token_t: float = math.nan
    last_token_t: float = math.nan
    batch_id: int = -1
    batch_size: int = 0
    pad_tokens: int = 0
    output_ids: list[int] = field(default_factory=list)
    inter_token_s: list[float] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + self.output_len
