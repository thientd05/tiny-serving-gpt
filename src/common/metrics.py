"""Bộ chỉ số.

Nguyên tắc (CLAUDE.md mục 9): báo cáo **phân phối**, không bao giờ chỉ trung
bình. Với serving, một prompt dài làm cả lô đứng gần như không nhích trung bình
nhưng phá p99 — mà đó đúng là thứ người dùng cảm nhận.

Ba mức ghi chép:
  RequestRecord — một request
  BatchRecord   — một lô
  RunRecord     — một lần chạy (cấu hình + điều kiện máy)
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ------------------------------------------------------------------ phân phối
def percentile(xs: list[float], q: float) -> float:
    """Phân vị theo nội suy tuyến tính. q tính theo phần trăm (50, 95, 99)."""
    if not xs:
        return math.nan
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summary(xs: list[float]) -> dict:
    """p50/p95/p99 + min/max + n. Trung bình có mặt nhưng không bao giờ đứng một mình."""
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "min": min(xs),
        "max": max(xs),
        "mean": sum(xs) / len(xs),
    }


# ------------------------------------------------------------------ bản ghi
@dataclass
class RequestRecord:
    rid: int
    arrival_t: float                  # lúc request vào hàng đợi
    prompt_len: int
    output_len_req: int

    schedule_t: float = math.nan      # lúc được xếp vào một lô
    first_token_t: float = math.nan   # lúc có token đầu tiên
    last_token_t: float = math.nan
    output_len_actual: int = 0

    batch_id: int = -1
    batch_size: int = 0
    pad_tokens: int = 0               # số token đệm request này phải gánh

    # thời gian giữa hai token liên tiếp, giây. Lấy từ CUDA event của lô.
    inter_token_s: list[float] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)

    @property
    def queue_wait_s(self) -> float:
        return self.schedule_t - self.arrival_t

    @property
    def ttft_s(self) -> float:
        """Thời gian tới token đầu, tính từ lúc request đến — gồm cả chờ hàng đợi."""
        return self.first_token_t - self.arrival_t

    @property
    def e2e_s(self) -> float:
        return self.last_token_t - self.arrival_t

    @property
    def norm_latency_s(self) -> float:
        """Độ trễ chuẩn hóa theo số token sinh ra (chỉ số của Sarathi-Serve)."""
        return self.e2e_s / self.output_len_actual if self.output_len_actual else math.nan

    def to_dict(self, with_ids: bool = False) -> dict:
        d = asdict(self)
        if not with_ids:
            d.pop("output_ids")
        d.update(queue_wait_s=self.queue_wait_s, ttft_s=self.ttft_s,
                 e2e_s=self.e2e_s, norm_latency_s=self.norm_latency_s)
        return d


@dataclass
class BatchRecord:
    batch_id: int
    size: int
    rids: list[int]
    max_prompt_len: int
    total_prompt_tokens: int          # tổng token prompt thật
    pad_tokens_total: int             # tổng token đệm — phép tính bỏ đi
    decode_steps: int
    t_start: float
    t_end: float
    prefill_ms: float = math.nan      # đo bằng CUDA event
    decode_ms: float = math.nan
    step_ms: list[float] = field(default_factory=list)  # từng bước, CUDA event
    wall_ms: float = math.nan         # đồng hồ thật, để so với tổng CUDA event
    peak_vram_mib: float = math.nan
    # Cỡ bảng điểm attention ở bước prefill. Với attention eager, bảng này
    # tốn B×head×L×L và softmax còn tạo thêm một bản fp32 — ở G0 chính nó đặt
    # trần bộ nhớ, chứ không phải ngân sách KV cache. Ghi lại để quy trách
    # nhiệm đúng chỗ khi OOM.
    attn_score_mib: float = math.nan
    kv_mib: float = math.nan
    oom: bool = False

    @property
    def pad_waste_frac(self) -> float:
        tot = self.total_prompt_tokens + self.pad_tokens_total
        return self.pad_tokens_total / tot if tot else 0.0

    @property
    def cpu_overhead_ms(self) -> float:
        """Đồng hồ thật trừ thời gian kernel chạy trên GPU.

        Đây là con số Phần 2 của G0 cần: phần thời gian KHÔNG phải việc của GPU,
        tức phần mà G1 (viết lại bằng C++) hứa cắt.
        """
        gpu = (0.0 if math.isnan(self.prefill_ms) else self.prefill_ms) + \
              (0.0 if math.isnan(self.decode_ms) else self.decode_ms)
        return self.wall_ms - gpu

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(pad_waste_frac=self.pad_waste_frac,
                 cpu_overhead_ms=self.cpu_overhead_ms)
        return d


@dataclass
class RunRecord:
    """Điều kiện của một lần chạy. Không khóa được xung GPU nên xung + nhiệt độ
    phải đi kèm mọi kết quả (RISKS.md mục A)."""
    run_id: str
    config: dict
    workload: str
    workload_sha256: str
    timeline_sha256: str = ""
    git_commit: str = ""
    torch_version: str = ""
    transformers_version: str = ""
    t_start: float = math.nan
    t_end: float = math.nan
    gpu_samples: list[dict] = field(default_factory=list)  # xung, nhiệt, VRAM
    host_ram: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ ghi file
def git_commit(root: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "<không có commit>"
    except Exception:  # noqa: BLE001
        return "<không lấy được>"


class RunWriter:
    """Ghi ra results/<stage>/<run_id>.{requests,batches}.jsonl + .run.json

    Ghi từng dòng ngay khi có, không giữ trong bộ nhớ — mất tiến trình giữa đường
    thì phần đã đo vẫn còn.
    """

    def __init__(self, out_dir: str | Path, run_id: str, with_ids: bool = False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.with_ids = with_ids
        self._req = (self.dir / f"{run_id}.requests.jsonl").open("w")
        self._bat = (self.dir / f"{run_id}.batches.jsonl").open("w")

    def request(self, r: RequestRecord) -> None:
        self._req.write(json.dumps(r.to_dict(self.with_ids)) + "\n")
        self._req.flush()

    def batch(self, b: BatchRecord) -> None:
        self._bat.write(json.dumps(b.to_dict()) + "\n")
        self._bat.flush()

    def finish(self, run: RunRecord) -> Path:
        p = self.dir / f"{self.run_id}.run.json"
        p.write_text(json.dumps(run.to_dict(), indent=2) + "\n")
        self._req.close()
        self._bat.close()
        return p
