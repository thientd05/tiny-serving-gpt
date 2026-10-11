"""CLI chạy hệ thống serving trên một kịch bản tải.

Ví dụ:
  .venv/bin/python -m serving --config stages/G0-base-python/g0.json \
      --workload alpaca --n 200 --pattern poisson --rate 2.0
  .venv/bin/python -m serving --serve --port 8000        # lớp HTTP demo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import ServingConfig                        # noqa: E402
from common.gpuinfo import GpuMonitor, cuda_vram, host_ram     # noqa: E402
from common.metrics import RunRecord, RunWriter, git_commit, summary  # noqa: E402
from common.model import load_model                            # noqa: E402
from common.workload import Replay, build_replay, load_workload  # noqa: E402
from serving.engine import Engine                              # noqa: E402
from serving.request import Request                            # noqa: E402


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="serving", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=ROOT / "stages" / "G0-base-python" / "g0.json")
    p.add_argument("--workload", default="alpaca")
    p.add_argument("--n", type=int, default=200, help="số request")
    p.add_argument("--pattern", default="poisson",
                   choices=["poisson", "burst", "all_at_once"])
    p.add_argument("--rate", type=float, default=2.0, help="request/giây")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None, help="ghi đè B")
    p.add_argument("--replay", type=Path, default=None,
                   help="phát lại đúng một kịch bản đã lưu")
    p.add_argument("--save-replay", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "g0")
    p.add_argument("--run-id", default="")
    p.add_argument("--save-tokens", action="store_true",
                   help="ghi cả token sinh ra (cần cho mốc tham chiếu)")
    p.add_argument("--serve", action="store_true", help="chạy lớp HTTP demo")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main() -> None:
    a = build_args()
    cfg = ServingConfig.load(a.config) if a.config.exists() else ServingConfig()
    if a.batch_size:
        cfg = cfg.replace(batch_size=a.batch_size)

    model, tok = load_model(cfg)

    if a.serve:
        from serving.server import serve
        serve(Engine(cfg, model, tok), port=a.port)
        return

    wl = load_workload(a.workload)
    if a.replay:
        rep = Replay.load(a.replay)
        if rep.workload_sha256 != wl.sha256:
            raise SystemExit(f"{a.replay} dựng từ bản workload khác — hash lệch")
    else:
        rep = build_replay(wl, a.n, a.pattern, a.rate, a.seed,
                           max_prompt_len=cfg.max_context - 1024)
    if a.save_replay:
        print(f"kịch bản lưu ở {a.save_replay} (hash {rep.save(a.save_replay)[:12]})")

    reqs = [Request(rid=i, prompt_ids=wl.samples[idx]["prompt"],
                    output_len=wl.samples[idx]["output_len"], arrival_t=at)
            for i, (idx, at) in enumerate(zip(rep.indices, rep.arrivals))]
    reqs = [r for r in reqs if r.total_len <= cfg.max_context]

    run_id = a.run_id or f"{a.workload}_{a.pattern}_r{a.rate}_B{cfg.batch_size}_s{a.seed}"
    w = RunWriter(a.out, run_id, with_ids=a.save_tokens)
    print(f"[{run_id}] {len(reqs)} request, B={cfg.batch_size}, "
          f"timeout={cfg.batch_timeout_s}s, dtype={cfg.dtype}, attn={cfg.attn_impl}")

    t0 = time.time()
    with GpuMonitor(period_s=2.0) as mon:
        done = engine_run(cfg, model, tok, reqs, w)
    t1 = time.time()

    run = RunRecord(
        run_id=run_id, config=cfg.to_dict(), workload=a.workload,
        workload_sha256=wl.sha256, timeline_sha256=rep.spec_sha256(),
        git_commit=git_commit(ROOT), t_start=t0, t_end=t1,
        gpu_samples=mon.samples, host_ram=host_ram(),
        extra={"gpu_drift": mon.drift_report(), "vram": cuda_vram(),
               "n_requests": len(done)})
    import torch, transformers
    run.torch_version, run.transformers_version = torch.__version__, transformers.__version__
    path = w.finish(run)

    report(done, t1 - t0, mon.drift_report())
    print(f"\nGhi ra: {path.parent}/{run_id}.*")


def engine_run(cfg, model, tok, reqs, writer):
    eng = Engine(cfg, model, tok)
    done = eng.run(reqs, on_batch=writer.batch)
    for r in done:
        writer.request(r)
    return done


def report(done, wall_s: float, drift: dict) -> None:
    ttft = [r.ttft_s for r in done]
    tbt = [x for r in done for x in r.inter_token_s]
    qw = [r.queue_wait_s for r in done]
    out_tok = sum(r.output_len_actual for r in done)

    def line(name: str, s: dict, unit: str = "ms", scale: float = 1000) -> None:
        if not s.get("n"):
            print(f"  {name:28s} (không có số)")
            return
        print(f"  {name:28s} p50 {s['p50']*scale:8.1f}  p95 {s['p95']*scale:8.1f}  "
              f"p99 {s['p99']*scale:8.1f}  max {s['max']*scale:9.1f}  [{unit}]")

    print("\n--- phân phối độ trễ (KHÔNG báo cáo trung bình một mình) ---")
    line("thời gian tới token đầu", summary(ttft))
    line("thời gian chờ hàng đợi", summary(qw))
    line("thời gian giữa hai token", summary(tbt))
    print("\n--- thông lượng ---")
    print(f"  {out_tok} token ra / {wall_s:.1f} s = {out_tok/wall_s:.1f} token/s")
    print(f"  {len(done)} request / {wall_s:.1f} s = {len(done)/wall_s:.2f} request/s")
    print(f"\n--- điều kiện máy ---")
    print(f"  xung {drift.get('clock_min_mhz')}–{drift.get('clock_max_mhz')} MHz, "
          f"tụt {drift.get('clock_drop_pct', 0):.1f}% "
          f"→ {'ĐẠT' if drift.get('ok') else 'KHÔNG ĐẠT, kết quả không đáng so'}, "
          f"nhiệt tối đa {drift.get('temp_max_c')}°C")


if __name__ == "__main__":
    main()
