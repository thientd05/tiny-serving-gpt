"""Phần 1 của G0 — kiểm hệ thống chạy ĐÚNG NHƯ YÊU CẦU.

Đây không phải test hiệu năng. Mỗi test ứng với một dòng trong spec G0
(.claude/PLAN-G0.md, mục "Test — chỉ kiểm hệ thống chạy đúng yêu cầu").

Mỗi test được thiết kế để CÓ THỂ thất bại: không test nào chỉ kiểm "chạy không
crash".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from g0_base_python.config import ServingConfig          # noqa: E402
from g0_base_python.engine import Engine                # noqa: E402
from g0_base_python.request import Request              # noqa: E402
from g0_base_python.tests.fake_model import FakeModel, FakeTok  # noqa: E402


def mk_engine(**kw) -> Engine:
    cfg = ServingConfig(device="cpu", **kw)
    return Engine(cfg, FakeModel(), FakeTok())


def mk_reqs(spec: list[tuple[int, int, float]]) -> list[Request]:
    """spec = [(prompt_len, output_len, arrival_t), ...]"""
    return [Request(rid=i, prompt_ids=list(range(500, 500 + pl)),
                    output_len=ol, arrival_t=at)
            for i, (pl, ol, at) in enumerate(spec)]


# ------------------------------------------------------------- thứ tự FIFO
def test_fifo_order():
    eng = mk_engine(batch_size=1)
    reqs = mk_reqs([(10, 2, 0.0), (10, 2, 0.01), (10, 2, 0.02)])
    done = eng.run(reqs)
    assert [r.rid for r in done] == [0, 1, 2]
    assert [r.batch_id for r in done] == [0, 1, 2]


# -------------------------------------------- lấy tối đa B, KHÔNG đợi đủ B
def test_takes_at_most_B():
    eng = mk_engine(batch_size=4)
    reqs = mk_reqs([(10, 2, 0.0)] * 10)
    eng.run(reqs)
    assert [c["batch"] for c in eng.model.calls] == [4, 4, 2]


def test_does_not_wait_to_fill_B():
    """Hàng đợi có 3 cái mà B=8 thì chạy ngay với 3, không đợi thêm."""
    eng = mk_engine(batch_size=8)
    eng.run(mk_reqs([(10, 2, 0.0)] * 3))
    assert [c["batch"] for c in eng.model.calls] == [3]


# ---------------------------------------------------------- lô đóng kín
def test_batch_is_closed_while_running():
    """Request đến TRONG LÚC lô đang chạy không được vào lô đó.

    Đây đúng là hạn chế G2 chữa, nên nó phải là hành vi KIỂM ĐƯỢC của G0, không
    phải chuyện ngầm hiểu.
    """
    eng = mk_engine(batch_size=8)
    # một lô chậm (output_len lớn) khởi động ở t=0; người thứ hai đến ở t=0.05
    reqs = mk_reqs([(10, 200, 0.0), (10, 2, 0.05)])
    done = eng.run(reqs)
    assert len(eng.model.calls) == 2, "phải là HAI lô, không gộp làm một"
    assert eng.model.calls[0]["batch"] == 1
    assert eng.model.calls[1]["batch"] == 1
    r1 = next(r for r in done if r.rid == 1)
    assert r1.batch_id == 1
    assert r1.queue_wait_s > 0, "người đến sau phải chờ hết lô trước"


# ------------------------------------------------------- đúng số token
def test_exact_output_len_per_request():
    """Mỗi request ra ĐÚNG output_len token, dù cả lô chạy theo người dài nhất."""
    eng = mk_engine(batch_size=4)
    reqs = mk_reqs([(10, 3, 0.0), (10, 7, 0.0), (10, 5, 0.0)])
    done = eng.run(reqs)
    got = {r.rid: r.output_len_actual for r in done}
    assert got == {0: 3, 1: 7, 2: 5}
    # cả lô vẫn chạy 7 bước — chỗ của người xong sớm KHÔNG được dùng lại
    assert eng.model.calls[0]["n_new"] == 7


# -------------------------------------------------------- đệm trái đúng
def test_left_padding_does_not_change_output():
    """Prompt ngắn nằm cùng lô với prompt dài phải ra CÙNG kết quả như chạy riêng.

    Nếu đệm trái tính sai (vị trí RoPE lệch, mask sai) thì test này đỏ.
    """
    alone = mk_engine(batch_size=1)
    d1 = alone.run(mk_reqs([(8, 4, 0.0)]))
    together = mk_engine(batch_size=2)
    d2 = together.run(mk_reqs([(8, 4, 0.0), (40, 4, 0.0)]))
    short_alone = d1[0].output_ids
    short_batched = next(r for r in d2 if r.rid == 0).output_ids
    assert short_batched == short_alone
    assert together.model.calls[0]["padded_len"] == 40


def test_padding_is_on_the_left():
    """Phép kiểm cấu trúc: token đệm phải nằm ở ĐẦU hàng, không phải cuối.

    Cần cả test này bên cạnh test trên, vì một model giả "thông minh" có thể
    tự đoán được kiểu đệm và làm test kia đi qua dù engine đệm sai.
    """
    eng = mk_engine(batch_size=2)
    eng.run(mk_reqs([(8, 2, 0.0), (30, 2, 0.0)]))
    call = eng.model.calls[0]
    mask = call["mask"]
    short_row = mask[0].tolist()          # request prompt 8 trong lô rộng 30
    assert short_row[:22] == [0] * 22, "token đệm không nằm bên trái"
    assert short_row[22:] == [1] * 8
    assert call["ids"][0, -1].item() == 507, "cột cuối phải là token thật cuối"


def test_pad_tokens_recorded():
    eng = mk_engine(batch_size=2)
    done = eng.run(mk_reqs([(10, 2, 0.0), (30, 2, 0.0)]))
    pads = {r.rid: r.pad_tokens for r in done}
    assert pads == {0: 20, 1: 0}
    b = eng.batches[0]
    assert b.pad_tokens_total == 20
    assert b.pad_waste_frac == pytest.approx(20 / 60)


# -------------------------------------------------------------- tái lập
def test_reproducible():
    a = mk_engine(batch_size=4).run(mk_reqs([(10, 5, 0.0)] * 6))
    b = mk_engine(batch_size=4).run(mk_reqs([(10, 5, 0.0)] * 6))
    assert [r.output_ids for r in a] == [r.output_ids for r in b]


# ------------------------------------------------------------ max_context
def test_rejects_too_long():
    eng = mk_engine(batch_size=4, max_context=64)
    with pytest.raises(ValueError, match="vượt max_context"):
        eng.submit(mk_reqs([(60, 10, 0.0)])[0])


# ------------------------------------------------------ chỉ số khớp thực tế
def test_metrics_consistent():
    eng = mk_engine(batch_size=4)
    done = eng.run(mk_reqs([(10, 3, 0.0), (20, 6, 0.0)]))
    for r in done:
        assert r.output_len_actual == len(r.output_ids) == r.output_len_req
        assert r.schedule_t >= r.arrival_t
        assert r.last_token_t >= r.first_token_t
    b = eng.batches[0]
    assert b.total_prompt_tokens == 30
    assert b.size == 2 and sorted(b.rids) == [0, 1]
    assert b.decode_steps == 6


# ----------------------------------------- timeout gom lô (núm điều chỉnh)
def test_batch_timeout_collects_more():
    """Bật timeout thì lô gom được thêm người đến muộn một chút."""
    off = mk_engine(batch_size=8, batch_timeout_s=0.0)
    off.run(mk_reqs([(10, 2, 0.0), (10, 2, 0.03)]))
    on = mk_engine(batch_size=8, batch_timeout_s=0.2)
    on.run(mk_reqs([(10, 2, 0.0), (10, 2, 0.03)]))
    assert [c["batch"] for c in off.model.calls] == [1, 1]
    assert [c["batch"] for c in on.model.calls] == [2], \
        "timeout bật mà vẫn chia hai lô → núm không có tác dụng"


# ---------------------------------------- thời gian chờ hàng đợi là thật
def test_queue_wait_is_real_time():
    eng = mk_engine(batch_size=1)
    t = time.perf_counter()
    done = eng.run(mk_reqs([(10, 2, 0.0), (10, 2, 0.15)]))
    elapsed = time.perf_counter() - t
    assert elapsed >= 0.15, "engine phải ngủ thật tới thời điểm đến kế tiếp"
    assert next(r for r in done if r.rid == 1).queue_wait_s < 0.05
