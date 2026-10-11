"""Nạp workload và phát lại kịch bản tải.

File workload có sẵn (data/external/*.json, dùng lại từ cuda-llm) đã tokenize
bằng tokenizer Llama 3.2 và đã ghi sẵn output_len từng mẫu, nhưng **không có
thời điểm đến**. Nên thời điểm đến phải tự sinh ở đây.

Mỗi kịch bản sinh ra được lưu thành file kèm sha256, để mọi bậc G0→G11 phát lại
y hệt một dòng request. Không có điều đó thì G2 so với G0 trên hai dòng request
khác nhau, và con số so sánh vô nghĩa.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
WORKLOADS = ROOT / "data" / "workloads"

PATTERNS = ("poisson", "burst", "all_at_once")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True,
                                   separators=(",", ":")).encode())


# ------------------------------------------------------------------- nạp file
@dataclass
class Workload:
    name: str
    meta: dict
    samples: list[dict]          # mỗi mẫu: prompt (list[int]), prompt_len, output_len
    sha256: str

    def __len__(self) -> int:
        return len(self.samples)


def load_workload(name: str, path: str | Path | None = None) -> Workload:
    p = Path(path) if path else EXTERNAL / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"không thấy workload: {p}")
    raw = json.loads(p.read_text())
    if "samples" not in raw:
        raise ValueError(f"{p} không có khóa 'samples'")
    for i, s in enumerate(raw["samples"][:1]):
        missing = {"prompt", "prompt_len", "output_len"} - set(s)
        if missing:
            raise ValueError(f"{p} mẫu {i} thiếu khóa {sorted(missing)}")
    return Workload(name=name, meta=raw.get("meta", {}),
                    samples=raw["samples"], sha256=sha256_file(p))


# -------------------------------------------------------------- thời điểm đến
def build_arrivals(n: int, pattern: str, rate_qps: float, seed: int,
                   burst_size: int = 8, burst_every_s: float = 10.0) -> list[float]:
    """Sinh n thời điểm đến, tính theo giây từ lúc bắt đầu, đã sắp tăng dần.

    poisson     — khoảng cách giữa hai request theo phân phối mũ (tải ngẫu nhiên
                  đúng kiểu nhiều người dùng độc lập)
    burst       — nền poisson, cộng thêm từng đợt burst_size request dồn vào cùng
                  một lúc, cứ burst_every_s giây một đợt
    all_at_once — mọi request đến ở t=0: trường hợp bão hòa, dùng để quét B và
                  đo thông lượng trần
    """
    if pattern not in PATTERNS:
        raise ValueError(f"pattern phải thuộc {PATTERNS}, nhận {pattern!r}")
    if pattern == "all_at_once":
        return [0.0] * n
    rng = np.random.default_rng(seed)
    if rate_qps <= 0:
        raise ValueError("rate_qps phải > 0 với pattern poisson/burst")

    if pattern == "poisson":
        gaps = rng.exponential(1.0 / rate_qps, size=n)
        return np.cumsum(gaps).tolist()

    # burst: chia n thành phần nền và phần dồn cục
    n_burst_slots = max(1, int(n / (burst_size + rate_qps * burst_every_s)))
    n_in_bursts = min(n, n_burst_slots * burst_size)
    n_base = n - n_in_bursts
    base = np.cumsum(rng.exponential(1.0 / rate_qps, size=n_base)) if n_base else np.array([])
    horizon = float(base[-1]) if n_base else n_in_bursts / rate_qps
    bursts: list[float] = []
    k = 0
    t = burst_every_s
    while k < n_in_bursts:
        take = min(burst_size, n_in_bursts - k)
        bursts.extend([min(t, horizon)] * take)
        k += take
        t += burst_every_s
    return sorted(base.tolist() + bursts)


# --------------------------------------------------------------- kịch bản tải
@dataclass
class Replay:
    """Một kịch bản tải cụ thể: ai đến lúc nào, sinh bao nhiêu token.

    Lưu ra file được, nạp lại được — đó là cách mọi bậc dùng chung một dòng request.
    """
    workload: str
    workload_sha256: str
    pattern: str
    rate_qps: float
    seed: int
    indices: list[int]           # chỉ số mẫu trong workload gốc
    arrivals: list[float]
    note: str = ""

    def spec_sha256(self) -> str:
        return sha256_json({"workload": self.workload,
                            "workload_sha256": self.workload_sha256,
                            "pattern": self.pattern, "rate_qps": self.rate_qps,
                            "seed": self.seed, "indices": self.indices,
                            "arrivals": [round(a, 9) for a in self.arrivals]})

    def __len__(self) -> int:
        return len(self.indices)

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["spec_sha256"] = self.spec_sha256()
        p.write_text(json.dumps(d, indent=1) + "\n")
        return d["spec_sha256"]

    @classmethod
    def load(cls, path: str | Path) -> "Replay":
        d = json.loads(Path(path).read_text())
        want = d.pop("spec_sha256", None)
        r = cls(**d)
        if want and r.spec_sha256() != want:
            raise ValueError(f"{path}: hash không khớp — file đã bị sửa")
        return r


def build_replay(wl: Workload, n: int, pattern: str, rate_qps: float, seed: int,
                 max_prompt_len: int | None = None, note: str = "") -> Replay:
    """Chọn n mẫu (không lặp lại mẫu nào) và gán thời điểm đến.

    Không lặp cùng một prompt trong một workload: lặp prompt làm prefix cache ăn
    gian so với tải thật (RISKS.md mục C).
    """
    rng = np.random.default_rng(seed)
    pool = range(len(wl))
    if max_prompt_len is not None:
        pool = [i for i in pool if wl.samples[i]["prompt_len"] <= max_prompt_len]
    if n > len(pool):
        raise ValueError(f"xin {n} mẫu nhưng chỉ có {len(pool)} mẫu hợp lệ")
    idx = rng.choice(np.asarray(pool), size=n, replace=False)
    idx.sort()
    return Replay(workload=wl.name, workload_sha256=wl.sha256, pattern=pattern,
                  rate_qps=rate_qps, seed=seed, indices=idx.tolist(),
                  arrivals=build_arrivals(n, pattern, rate_qps, seed), note=note)


# ------------------------------------------------- bộ kiểm đúng (W6) cố định
def build_w6(per_source: int = 150, max_output_len: int = 128, seed: int = 20261011,
             sources: tuple[str, ...] = ("alpaca", "sharegpt"),
             max_prompt_len: int | None = None, name: str = "w6_correctness",
             out: str | Path | None = None) -> dict:
    """Bộ kiểm đúng: nhỏ, cố định, phủ cả prompt ngắn/trung/dài.

    Phân tầng theo prompt_len (chia ba theo phân vị) để không bị toàn prompt ngắn
    — vì lỗi quản lý bộ nhớ KV lộ ra ở prompt dài rõ hơn.
    """
    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    for name in sources:
        wl = load_workload(name)
        ok = [i for i, s in enumerate(wl.samples)
              if s["output_len"] <= max_output_len
              and (max_prompt_len is None or s["prompt_len"] <= max_prompt_len)]
        lens = np.array([wl.samples[i]["prompt_len"] for i in ok])
        edges = np.percentile(lens, [33.3, 66.7])
        tiers = [
            [i for i, L in zip(ok, lens) if L <= edges[0]],
            [i for i, L in zip(ok, lens) if edges[0] < L <= edges[1]],
            [i for i, L in zip(ok, lens) if L > edges[1]],
        ]
        base, extra = divmod(per_source, 3)
        for t, tier in enumerate(tiers):
            k = base + (1 if t < extra else 0)
            if k > len(tier):
                raise ValueError(f"{name}: tầng {t} chỉ có {len(tier)} mẫu, xin {k}")
            for i in rng.choice(np.asarray(tier), size=k, replace=False):
                s = wl.samples[int(i)]
                picked.append({"source": name, "source_index": int(i),
                               "prompt": s["prompt"], "prompt_len": s["prompt_len"],
                               "output_len": s["output_len"]})
    for rid, s in enumerate(picked):
        s["rid"] = rid

    doc = {
        "meta": {
            "name": name,
            "max_prompt_len": max_prompt_len,
            "purpose": "kiểm đúng: so khớp từng token với mốc G0 B=1",
            "seed": seed, "per_source": per_source,
            "max_output_len": max_output_len, "sources": list(sources),
            "n": len(picked),
            "prompt_len": {
                "min": min(s["prompt_len"] for s in picked),
                "p50": float(np.percentile([s["prompt_len"] for s in picked], 50)),
                "max": max(s["prompt_len"] for s in picked),
            },
            "total_output_tokens": sum(s["output_len"] for s in picked),
        },
        "samples": picked,
    }
    doc["meta"]["content_sha256"] = sha256_json(doc["samples"])
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1) + "\n")
    return doc
