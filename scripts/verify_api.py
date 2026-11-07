from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import os
import re
import time
import uuid

HOST = os.environ.get("GW_HOST", "localhost")
PORT = int(os.environ.get("GW_PORT", "8090"))
SECRET = os.environ["JWT_SECRET"]
ASSETS = os.environ.get("ASSETS", "/tmp/assets")

PASS = []
FAIL = []


def check(name: str, got, expected) -> None:
    ok = got == expected
    (PASS if ok else FAIL).append((name, got, expected))
    print(f"{'OK  ' if ok else 'FAIL'} {name}: получено {got}, ожидалось {expected}")


def b64(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def make_token(payload: dict, secret: str) -> str:
    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = b64(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = head + b"." + body
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return (signing_input + b"." + b64(sig)).decode()


now = int(time.time())
claims = {"sub": "student", "role": "student", "iat": now, "exp": now + 86400}
VALID = make_token(claims, SECRET)
WRONG_SECRET = make_token(claims, "definitely-not-our-secret")
NONE_ALG = (
    b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    + b"."
    + b64(json.dumps({"sub": "student", "role": "student"}).encode())
    + b"."
).decode()
GARBAGE = "abc.def.ghi"


def get(path: str, token: str | None = None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    headers = {} if token is None else {"Authorization": "Bearer " + token}
    conn.request("GET", path, None, headers)
    resp = conn.getresponse()
    body = resp.read()
    return resp.status, body


def post_json(path: str, payload, token: str | None = None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=180)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    conn.request("POST", path, json.dumps(payload), headers)
    resp = conn.getresponse()
    return resp.status, resp.read()


def post_file(path: str, filename: str, data: bytes, token: str | None = None, field: str = "file"):
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=180)
    headers = {"Content-Type": "multipart/form-data; boundary=" + boundary}
    if token:
        headers["Authorization"] = "Bearer " + token
    conn.request("POST", path, body, headers)
    resp = conn.getresponse()
    return resp.status, resp.read()


def asset(name: str) -> bytes:
    with open(os.path.join(ASSETS, name), "rb") as fh:
        return fh.read()


print("=== КРИТЕРИЙ 2: авторизация ===")
check("/info без токена", get("/info")[0], 401)
check("/info мусорный токен", get("/info", GARBAGE)[0], 401)
check("/info чужой секрет", get("/info", WRONG_SECRET)[0], 401)
check("/info alg=none", get("/info", NONE_ALG)[0], 401)
check("/info валидный", get("/info", VALID)[0], 200)
check("/vision без токена", post_file("/vision", "text.txt", asset("text.txt"))[0], 401)
check("/chat без токена", post_json("/chat", {"messages": [{"role": "user", "content": "hi"}]})[0], 401)
check("/vision валидный", post_file("/vision", "rgb.png", asset("rgb.png"), VALID)[0], 200)

print("\n=== КРИТЕРИЙ 8: устойчивость входов ===")
check("PNG с альфой", post_file("/vision", "a.png", asset("rgba.png"), VALID)[0], 200)
check("grayscale", post_file("/vision", "g.png", asset("gray.png"), VALID)[0], 200)
check("CMYK", post_file("/vision", "c.tiff", asset("cmyk.tiff"), VALID)[0], 200)
check("palette PNG", post_file("/vision", "p.png", asset("palette.png"), VALID)[0], 200)
check("8000x8000", post_file("/vision", "h.jpg", asset("huge.jpg"), VALID)[0], 200)
check("1x1", post_file("/vision", "t.png", asset("tiny.png"), VALID)[0], 200)
check("пустой файл", post_file("/vision", "e.bin", asset("empty.bin"), VALID)[0], 400)
check("текст вместо картинки", post_file("/vision", "t.txt", asset("text.txt"), VALID)[0], 400)
check("случайные байты", post_file("/vision", "r.bin", asset("random.bin"), VALID)[0], 400)
check("нет поля file", post_file("/vision", "x.png", asset("rgb.png"), VALID, field="upload")[0], 422)
check("битый multipart", post_json("/vision", {"nope": 1}, VALID)[0], 422)

print("\n=== КРИТЕРИЙ 9: метрики vLLM ===")
status, body = get("/metrics/vllm")
text = body.decode("utf-8", "replace")
check("/metrics/vllm статус", status, 200)
print("    request_success_total:", "есть" if "vllm:request_success_total" in text else "НЕТ")
print("    kv_cache_usage:", "есть" if ("vllm:kv_cache_usage_perc" in text or "vllm:gpu_cache_usage_perc" in text) else "НЕТ")
m = re.search(r'vllm:request_success_total\{[^}]*model_name="([^"]+)"', text)
print("    model_name в метрике:", m.group(1) if m else "не найден")

print("\n=== КРИТЕРИЙ 9: точный инкремент nv_inference_count ===")


def triton_metric(name: str) -> int:
    _, raw = get("/metrics/triton")
    txt = raw.decode("utf-8", "replace")
    match = re.search(rf'{name}\{{model="convnext_tiny"[^}}]*\}}\s+([0-9.]+)', txt)
    return int(float(match.group(1))) if match else -1


before = triton_metric("nv_inference_count")
before_exec = triton_metric("nv_inference_exec_count")
sent = 5
for _ in range(sent):
    post_file("/vision", "rgb.png", asset("rgb.png"), VALID)
after = triton_metric("nv_inference_count")
after_exec = triton_metric("nv_inference_exec_count")
check("прирост nv_inference_count", after - before, sent)
print(f"    exec_count: {before_exec} -> {after_exec} (запусков модели: {after_exec - before_exec})")

print("\n=== ИТОГО ===")
print(f"пройдено: {len(PASS)}, провалено: {len(FAIL)}")
for name, got, expected in FAIL:
    print(f"  FAIL {name}: got={got} expected={expected}")
