"""Lớp HTTP mỏng — CHỈ để demo, KHÔNG dùng khi đo.

Vì sao không đo qua HTTP: lớp này cộng thêm vài ms vào đúng con số "thời gian
tới token đầu" mà cả dự án đang cần đo chính xác, và trên máy chỉ còn ~2 GiB RAM
thì thêm một web server là thêm một biến số không kiểm soát được. Harness đo gọi
thẳng vào Engine.

Dùng stdlib `http.server`, không cần fastapi/uvicorn — một phụ thuộc ít đi là
một thứ ít phải kiểm tương thích sm75 hơn.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serving.engine import Engine
from serving.request import Request


def serve(engine: Engine, port: int = 8000) -> None:
    lock = threading.Lock()
    counter = {"rid": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # bớt ồn
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "stage": "G0",
                                 "batch_size": engine.cfg.batch_size,
                                 "dtype": engine.cfg.dtype,
                                 "attn": engine.cfg.attn_impl,
                                 "queued": len(engine.queue),
                                 "batches_done": engine.next_batch_id})
            else:
                self._json(404, {"error": "chỉ có /health và POST /generate"})

        def do_POST(self):
            if self.path != "/generate":
                return self._json(404, {"error": "chỉ có POST /generate"})
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                return self._json(400, {"error": f"JSON sai: {e}"})

            ids = req.get("prompt_ids")
            if ids is None and "prompt" in req:
                ids = engine.tok(req["prompt"], add_special_tokens=True)["input_ids"]
            if not ids:
                return self._json(400, {"error": "cần 'prompt' hoặc 'prompt_ids'"})
            out_len = int(req.get("output_len", 32))

            with lock:       # G0 chạy một lô một lúc; HTTP chỉ xếp hàng vào
                counter["rid"] += 1
                r = Request(rid=counter["rid"], prompt_ids=list(ids),
                            output_len=out_len, arrival_t=0.0)
                t0 = time.perf_counter()
                try:
                    engine.submit(r)
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                batch = engine._take_batch()
                for q in batch:
                    q.schedule_t = time.perf_counter()
                rec = engine.run_batch(batch)

            return self._json(200, {
                "rid": r.rid,
                "output_ids": r.output_ids,
                "text": engine.tok.decode(r.output_ids, skip_special_tokens=True),
                "batch_id": rec.batch_id, "batch_size": rec.size,
                "latency_s": time.perf_counter() - t0,
            })

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"G0 đang nghe ở http://127.0.0.1:{port}  "
          f"(GET /health, POST /generate)")
    print("LƯU Ý: lớp HTTP này chỉ để demo, số đo trong báo cáo không đi qua đây.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
