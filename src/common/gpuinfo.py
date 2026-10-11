"""Đọc trạng thái máy để đi kèm mọi kết quả đo.

Vì sao cần: không khóa được xung GPU trên máy này (RISKS.md mục A — thiếu quyền
root). Chạy sweep nhiều giờ thì GPU nóng lên và tụt tần số, lần chạy sau chậm
hơn **vì nhiệt, không vì thuật toán**. Cách chống duy nhất còn lại là ghi xung +
nhiệt độ vào từng kết quả rồi kiểm lại khi so sánh.
"""
from __future__ import annotations

import subprocess
import threading
import time

_QUERY = ("clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,"
          "memory.used,memory.total")


def gpu_sample() -> dict:
    """Một lần đọc nvidia-smi. Trả dict rỗng nếu không đọc được."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return {}
        vals = [v.strip() for v in r.stdout.strip().split(",")]
        keys = ["clock_sm_mhz", "clock_mem_mhz", "temp_c", "power_w",
                "util_pct", "vram_used_mib", "vram_total_mib"]

        def num(x: str) -> float | None:
            try:
                return float(x)
            except ValueError:
                return None

        d = {k: num(v) for k, v in zip(keys, vals)}
        d["t"] = time.time()
        return d
    except Exception:  # noqa: BLE001
        return {}


def host_ram() -> dict:
    """RAM host theo MiB. Máy này RAM rất chật nên phải đo lại trước mỗi lần chạy."""
    try:
        info = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0])  # kB
        return {
            "total_mib": info["MemTotal"] / 1024,
            "available_mib": info["MemAvailable"] / 1024,
            "free_mib": info["MemFree"] / 1024,
            "swap_total_mib": info.get("SwapTotal", 0) / 1024,
            "swap_free_mib": info.get("SwapFree", 0) / 1024,
        }
    except Exception:  # noqa: BLE001
        return {}


def cuda_vram() -> dict:
    """VRAM theo con số của chính CUDA (khác nvidia-smi: không gồm tiến trình khác)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {
            "free_mib": free / 2**20,
            "total_mib": total / 2**20,
            "allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
    except Exception:  # noqa: BLE001
        return {}


class GpuMonitor:
    """Lấy mẫu xung/nhiệt độ trong lúc chạy, trên một luồng riêng.

    Dùng: with GpuMonitor() as m: ...  rồi đọc m.samples và m.drift_report()
    """

    def __init__(self, period_s: float = 2.0):
        self.period_s = period_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = gpu_sample()
            if s:
                self.samples.append(s)
            self._stop.wait(self.period_s)

    def __enter__(self) -> "GpuMonitor":
        s = gpu_sample()
        if s:
            self.samples.append(s)
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=self.period_s + 1)
        s = gpu_sample()
        if s:
            self.samples.append(s)

    def drift_report(self, tol_pct: float = 5.0, min_util_pct: float = 50.0) -> dict:
        """Xung có tụt quá tol_pct trong lúc chạy không — nếu có thì kết quả
        của lần chạy này phải bị đánh dấu, không đem so với lần chạy khác.

        Chỉ tính trên các mẫu GPU **đang bận** (util >= min_util_pct). Lúc rỗi
        GPU tụt xung về 300 MHz để tiết kiệm điện; tính cả mẫu đó vào thì mọi
        lần chạy đều báo "tụt 80%", che mất trôi nhiệt thật.
        """
        busy = [s for s in self.samples
                if (s.get("util_pct") or 0) >= min_util_pct]
        used = busy or self.samples
        clocks = [s["clock_sm_mhz"] for s in used
                  if s.get("clock_sm_mhz") is not None]
        temps = [s["temp_c"] for s in used if s.get("temp_c") is not None]
        if not clocks:
            return {"ok": None, "note": "không đọc được xung"}
        drop = (max(clocks) - min(clocks)) / max(clocks) * 100
        return {
            "n_samples": len(self.samples), "n_busy_samples": len(busy),
            "basis": "busy" if busy else "tất cả (không bắt được mẫu nào lúc bận)",
            "clock_max_mhz": max(clocks), "clock_min_mhz": min(clocks),
            "clock_drop_pct": drop,
            "temp_start_c": temps[0] if temps else None,
            "temp_max_c": max(temps) if temps else None,
            "ok": drop <= tol_pct,
            "note": "" if drop <= tol_pct
                    else f"xung tụt {drop:.1f}% > {tol_pct}% — kết quả không đáng so",
        }
