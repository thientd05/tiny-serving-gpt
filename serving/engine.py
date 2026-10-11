"""G0 — serving thuần Python, chạy từng lô cố định (static batching).

Cơ chế, đúng một đoạn:

    GPU rỗi  →  lấy tối đa B request đang chờ ở đầu hàng đợi (KHÔNG đợi cho đủ B)
             →  đệm trái cho bằng prompt dài nhất trong lô
             →  gọi generate() một lần cho cả lô
             →  chờ CẢ LÔ sinh xong
             →  quay lại đầu

Đây là baseline "static batching" mà Orca (OSDI'22) và vLLM lấy làm mốc so sánh.

Hai hạn chế nằm ngay trong cơ chế, và đó là phần việc của bậc sau:
  - trong lúc lô đang chạy, người đến sau không vào được, và chỗ của người đã
    sinh xong không được dùng lại  → G2 (continuous batching) chữa
  - prompt ngắn bị đệm cho bằng prompt dài nhất → tốn cả bộ nhớ KV lẫn phép tính
    cho token rác; engine ghi lại tỉ lệ đó (`pad_waste_frac`) để G2/G3 có số
    trước/sau

Vì sao không chạy một request một lúc: `generate()` nhận cả lô sẵn, nên "một
request một lúc" là base yếu giả tạo — mọi bậc sau sẽ thắng chỉ nhờ biết dùng
batch, che mất phần cải thiện thật của thuật toán.
"""
from __future__ import annotations

import math
import time
from collections import deque

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from common.config import ServingConfig
from common.metrics import BatchRecord, RequestRecord
from common.model import greedy_config
from serving.request import Request


def attn_score_bytes(batch: int, seq_len: int, n_heads: int) -> int:
    """Bộ nhớ bảng điểm attention ở bước prefill, với attention eager.

    transformers tạo bảng điểm ở dtype của model rồi gọi softmax với
    `dtype=torch.float32`, tức thêm một bản fp32 nữa. Nên đỉnh là
    B × head × L × L × (2 + 4) byte.

    Con số này tăng theo BÌNH PHƯƠNG độ dài prompt, nên nó — không phải ngân
    sách KV cache — mới là thứ đặt trần cho G0. Việc thay bằng kernel không
    vật chất hóa bảng điểm là của G4.
    """
    return batch * n_heads * seq_len * seq_len * 6


def kv_bytes(batch: int, seq_len: int, kv_heads: int, head_dim: int,
             layers: int, dtype_bytes: int = 2) -> int:
    return batch * seq_len * 2 * kv_heads * head_dim * layers * dtype_bytes


class _StepTimer(StoppingCriteria):
    """Móc đo: transformers gọi stopping criteria đúng một lần mỗi bước sinh token.

    Mỗi lần gọi thì ghi một CUDA event. Hết lô mới sync một lần rồi đọc toàn bộ
    event — nên phép đo gần như không thêm chi phí vào thứ đang đo.

    Vì sao không đọc time.perf_counter() mỗi bước: CUDA chạy bất đồng bộ nên con
    số đó vô nghĩa; mà gọi synchronize() mỗi bước thì chính phép đo làm chậm
    cái đang đo.
    """

    def __init__(self, n_steps: int, cuda: bool = True):
        self.cuda = cuda
        self.events = ([torch.cuda.Event(enable_timing=True)
                        for _ in range(n_steps + 1)] if cuda else [])
        self.i = 0

    def start(self) -> None:
        if self.cuda:
            self.events[0].record()
        self.i = 1

    def __call__(self, input_ids, scores, **kw):  # noqa: ANN001
        if self.cuda and self.i < len(self.events):
            self.events[self.i].record()
        self.i += 1
        return torch.zeros(input_ids.shape[0], dtype=torch.bool,
                           device=input_ids.device)

    def step_ms(self) -> list[float]:
        """Thời gian từng bước, ms. Bước 0 gồm cả prefill.

        Lưu ý khi đọc số: mốc này là mốc **trên stream**, nên khoảng GPU ngồi
        chờ CPU cũng tính vào. Muốn tách GPU bận khỏi GPU chờ thì phải dùng
        profiler — xem .claude/scripts/measure_cpu_overhead.py.
        """
        if not self.cuda:
            return []
        torch.cuda.synchronize()
        n = min(self.i, len(self.events))
        return [self.events[k].elapsed_time(self.events[k + 1])
                for k in range(n - 1)]


class Engine:
    """Hệ thống serving G0. Một hàng đợi FIFO, một vòng lặp."""

    def __init__(self, cfg: ServingConfig, model, tok):
        self.cfg = cfg
        self.model = model
        self.tok = tok
        self.pad_id = tok.pad_token_id
        self.cuda = str(cfg.device).startswith("cuda") and torch.cuda.is_available()
        self.queue: deque[Request] = deque()
        self.next_batch_id = 0
        self.batches: list[BatchRecord] = []
        # Chỗ để harness kiểm đúng cắm thêm hook vào generate() (ví dụ
        # logits_processor để ghi khoảng cách hai ứng viên đầu bảng). Vòng đo
        # hiệu năng để rỗng — không hook nào được nằm trên đường đo.
        self.extra_generate_kwargs: dict = {}

    # ------------------------------------------------------------- hàng đợi
    def submit(self, req: Request) -> None:
        if req.total_len > self.cfg.max_context:
            raise ValueError(
                f"request {req.rid}: prompt {req.prompt_len} + output "
                f"{req.output_len} = {req.total_len} vượt max_context "
                f"{self.cfg.max_context}")
        self.queue.append(req)

    def _take_batch(self) -> list[Request]:
        """Lấy tối đa B request đang chờ. KHÔNG đợi cho đủ B."""
        n = min(self.cfg.batch_size, len(self.queue))
        return [self.queue.popleft() for _ in range(n)]

    # ------------------------------------------------------------- chạy lô
    def run_batch(self, reqs: list[Request]) -> BatchRecord:
        """Chạy hết một lô. Trả về bản ghi của lô."""
        bid = self.next_batch_id
        self.next_batch_id += 1

        max_prompt = max(r.prompt_len for r in reqs)
        # Cả lô chạy cùng số bước: generate() không thả ai ra giữa đường. Người
        # sinh xong sớm vẫn giữ chỗ tới khi người dài nhất xong — đúng hạn chế
        # mà G2 chữa.
        steps = max(r.output_len for r in reqs)

        ids = torch.full((len(reqs), max_prompt), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(reqs), max_prompt), dtype=torch.long)
        for i, r in enumerate(reqs):
            ids[i, max_prompt - r.prompt_len:] = torch.tensor(r.prompt_ids)
            mask[i, max_prompt - r.prompt_len:] = 1
            r.pad_tokens = max_prompt - r.prompt_len
            r.batch_id = bid
            r.batch_size = len(reqs)
        ids = ids.to(self.cfg.device)
        mask = mask.to(self.cfg.device)

        cc = getattr(self.model, "config", None)
        n_heads = getattr(cc, "num_attention_heads", 0) or 0
        score_mib = attn_score_bytes(len(reqs), max_prompt, n_heads) / 2**20 \
            if n_heads else math.nan
        kv_mib = (kv_bytes(len(reqs), max_prompt + steps,
                           getattr(cc, "num_key_value_heads", 0) or 0,
                           getattr(cc, "head_dim", 0) or 0,
                           getattr(cc, "num_hidden_layers", 0) or 0) / 2**20
                  if n_heads else math.nan)

        gc = greedy_config(steps, self.cfg.ignore_eos, self.pad_id,
                           getattr(self.model.generation_config, "eos_token_id", None))
        timer = _StepTimer(steps, cuda=self.cuda)

        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        timer.start()
        try:
            with torch.no_grad():
                out = self.model.generate(
                    ids, attention_mask=mask, generation_config=gc,
                    stopping_criteria=StoppingCriteriaList([timer]),
                    **self.extra_generate_kwargs)
        except torch.OutOfMemoryError as exc:
            if self.cuda:
                torch.cuda.empty_cache()
            raise torch.OutOfMemoryError(
                f"lô {bid} hết VRAM: B={len(reqs)}, prompt dài nhất "
                f"{max_prompt}, sinh {steps} token.\n"
                f"  bảng điểm attention (eager, gồm bản fp32 của softmax): "
                f"{score_mib:.0f} MiB\n"
                f"  KV cache cho cả lô:                                   "
                f"{kv_mib:.0f} MiB\n"
                f"  → bảng điểm lớn gấp {score_mib/max(kv_mib,1e-9):.1f} lần KV. "
                f"Đây là trần của attention eager, không phải trần ngân sách KV; "
                f"việc sửa thuộc G4.\n"
                f"  nguyên văn: {exc}") from exc
        if self.cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        step_ms = timer.step_ms()
        gen = out[:, max_prompt:].tolist()

        # Chia thời gian về từng request. TTFT lấy mốc đồng hồ thật ở đầu lô
        # cộng độ lệch từ CUDA event — vì chỉ CUDA event mới biết GPU xong lúc nào.
        for i, r in enumerate(reqs):
            r.output_ids = gen[i][:r.output_len]
            r.inter_token_s = [m / 1000.0 for m in step_ms[1:r.output_len]]
            if step_ms:
                r.first_token_t = t0 + step_ms[0] / 1000.0
                r.last_token_t = t0 + sum(step_ms[:r.output_len]) / 1000.0
            else:  # không đo được event thì rơi về đồng hồ thật
                r.first_token_t = r.last_token_t = t1

        rec = BatchRecord(
            batch_id=bid, size=len(reqs), rids=[r.rid for r in reqs],
            max_prompt_len=max_prompt,
            total_prompt_tokens=sum(r.prompt_len for r in reqs),
            pad_tokens_total=sum(r.pad_tokens for r in reqs),
            decode_steps=steps, t_start=t0, t_end=t1,
            prefill_ms=step_ms[0] if step_ms else math.nan,
            decode_ms=sum(step_ms[1:]) if len(step_ms) > 1 else 0.0,
            step_ms=step_ms, wall_ms=(t1 - t0) * 1000.0,
            peak_vram_mib=(torch.cuda.max_memory_allocated() / 2**20
                           if self.cuda else math.nan),
            attn_score_mib=score_mib, kv_mib=kv_mib,
        )
        self.batches.append(rec)
        return rec

    # ------------------------------------------------------------ vòng lặp
    def run(self, requests: list[Request], t_origin: float | None = None,
            on_batch=None) -> list[RequestRecord]:
        """Chạy một kịch bản tải. `requests` phải sắp theo arrival_t tăng dần.

        arrival_t tính theo giây từ lúc bắt đầu. Vòng lặp **ngủ thật** tới thời
        điểm đến kế tiếp, nên thời gian chờ hàng đợi đo được là thời gian chờ thật.
        """
        pending = deque(requests)
        done: list[RequestRecord] = []
        t0 = t_origin if t_origin is not None else time.perf_counter()

        def admit(now: float) -> None:
            while pending and t0 + pending[0].arrival_t <= now:
                r = pending.popleft()
                r.arrival_t = t0 + r.arrival_t      # đổi sang mốc tuyệt đối
                self.submit(r)

        while pending or self.queue:
            now = time.perf_counter()
            admit(now)

            if not self.queue:
                # GPU rỗi mà chưa ai đến: ngủ tới thời điểm đến kế tiếp
                time.sleep(max(0.0, t0 + pending[0].arrival_t - now))
                continue

            if self.cfg.batch_timeout_s > 0 and len(self.queue) < self.cfg.batch_size:
                # Núm điều chỉnh: đợi thêm một chút để gom lô to hơn. Mặc định
                # tắt (= 0), vì spec G0 là "không đợi cho đủ B".
                deadline = now + self.cfg.batch_timeout_s
                while (len(self.queue) < self.cfg.batch_size and pending
                       and time.perf_counter() < deadline):
                    time.sleep(0.001)
                    admit(time.perf_counter())

            batch = self._take_batch()
            for r in batch:
                r.schedule_t = time.perf_counter()
            rec = self.run_batch(batch)
            if on_batch:
                on_batch(rec)
            for r in batch:
                done.append(_to_record(r))
        return done


def _to_record(r: Request) -> RequestRecord:
    rec = RequestRecord(rid=r.rid, arrival_t=r.arrival_t, prompt_len=r.prompt_len,
                        output_len_req=r.output_len)
    rec.schedule_t = r.schedule_t
    rec.first_token_t = r.first_token_t
    rec.last_token_t = r.last_token_t
    rec.output_len_actual = len(r.output_ids)
    rec.batch_id = r.batch_id
    rec.batch_size = r.batch_size
    rec.pad_tokens = r.pad_tokens
    rec.inter_token_s = r.inter_token_s
    rec.output_ids = r.output_ids
    return rec
