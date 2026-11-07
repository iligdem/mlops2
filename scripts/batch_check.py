from __future__ import annotations

import http.client
import re
import subprocess
import sys


def metric(name: str) -> int:
    conn = http.client.HTTPConnection("localhost", 8090, timeout=30)
    conn.request("GET", "/metrics/triton")
    resp = conn.getresponse()
    text = resp.read().decode("utf-8", "replace")
    match = re.search(rf'{name}\{{model="convnext_tiny"[^}}]*\}}\s+([0-9.]+)', text)
    if not match:
        raise SystemExit(f"метрика {name} не найдена")
    return int(float(match.group(1)))


count_before = metric("nv_inference_count")
exec_before = metric("nv_inference_exec_count")

subprocess.run([sys.executable, "scripts/loadtest.py"], check=True)

count_after = metric("nv_inference_count")
exec_after = metric("nv_inference_exec_count")

delta_count = count_after - count_before
delta_exec = exec_after - exec_before
print()
print(f"инференсов: +{delta_count}, запусков модели: +{delta_exec}")
if delta_exec:
    ratio = delta_count / delta_exec
    print(f"соотношение: {ratio:.2f} картинок на запуск (порог 1.5) -> {'ПРОХОДИТ' if ratio >= 1.5 else 'НЕ ПРОХОДИТ'}")
