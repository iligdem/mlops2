import http.client
import json
import os
import statistics
import threading
import time
import uuid

TOK = os.environ["TOK"]
IMG = open("/tmp/dog.jpg", "rb").read()
DEADLINE = time.time() + 20
LOCK = threading.Lock()
STAT = {"ok": 0, "fail": 0, "codes": {}, "vis": [], "ttft": []}


def vision():
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"d.jpg\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + IMG + f"\r\n--{boundary}--\r\n".encode()
    conn = http.client.HTTPConnection("localhost", 8090, timeout=60)
    started = time.perf_counter()
    conn.request(
        "POST",
        "/vision",
        body,
        {"Authorization": "Bearer " + TOK, "Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    resp = conn.getresponse()
    resp.read()
    return resp.status, (time.perf_counter() - started) * 1000


def chat():
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": "Name a color."}],
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": True,
        }
    )
    conn = http.client.HTTPConnection("localhost", 8090, timeout=120)
    started = time.perf_counter()
    ttft = None
    conn.request(
        "POST",
        "/chat",
        body,
        {"Authorization": "Bearer " + TOK, "Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    while True:
        line = resp.readline()
        if not line:
            break
        text = line.decode("utf-8", "replace").strip()
        if text == "data: [DONE]":
            break
        if text.startswith("data:") and ttft is None:
            ttft = (time.perf_counter() - started) * 1000
    return resp.status, ttft if ttft is not None else -1


def worker():
    while time.time() < DEADLINE:
        for fn in (vision, chat):
            if time.time() >= DEADLINE:
                break
            try:
                status, value = fn()
            except Exception as exc:
                status, value = type(exc).__name__, -1
            with LOCK:
                STAT["codes"][str(status)] = STAT["codes"].get(str(status), 0) + 1
                if status == 200:
                    STAT["ok"] += 1
                    (STAT["vis"] if fn is vision else STAT["ttft"]).append(value)
                else:
                    STAT["fail"] += 1


def p95(values):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[int(len(ordered) * 0.95) - 1], 1)


threads = [threading.Thread(target=worker) for _ in range(20)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

total = STAT["ok"] + STAT["fail"]
print("всего запросов:", total, "| успешно:", STAT["ok"], "| доля успеха:", round(STAT["ok"] / max(total, 1), 4))
print("коды ответов:", STAT["codes"])
print("5xx:", sum(count for code, count in STAT["codes"].items() if code.startswith("5")))
print(
    "vision: n=%d p50=%s p95=%s max=%s ms"
    % (
        len(STAT["vis"]),
        round(statistics.median(STAT["vis"]), 1) if STAT["vis"] else None,
        p95(STAT["vis"]),
        round(max(STAT["vis"]), 1) if STAT["vis"] else None,
    )
)
print(
    "chat TTFT: n=%d p50=%s p95=%s max=%s ms"
    % (
        len(STAT["ttft"]),
        round(statistics.median(STAT["ttft"]), 1) if STAT["ttft"] else None,
        p95(STAT["ttft"]),
        round(max(STAT["ttft"]), 1) if STAT["ttft"] else None,
    )
)
