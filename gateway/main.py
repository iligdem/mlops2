"""Real-time inference gateway.

Единая точка входа: JWT-авторизация, /info, /vision (Triton + препроцессинг
convnext_tiny), /chat (SSE-прокси в vLLM), health-ручки бэкендов и проброс их
Prometheus-метрик. Контракт — API_CONTRACT.md.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import jwt
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator

JWT_SECRET = os.environ.get("JWT_SECRET", "")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET не задан")

CV_MODEL = os.environ.get("CV_MODEL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen2.5-1.5B-Instruct")
STUDENT_NAME = os.environ.get("STUDENT_NAME", "")
STUDENT_EMAIL = os.environ.get("STUDENT_EMAIL", "")
GPU_NAME = os.environ.get("GPU_NAME", "unknown")
VERSION = os.environ.get("GATEWAY_VERSION", "1.0")

TRITON_URL = os.environ.get("TRITON_URL", "http://triton:8000").rstrip("/")
TRITON_METRICS_URL = os.environ.get("TRITON_METRICS_URL", "http://triton:8002/metrics")
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8000").rstrip("/")
VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://vllm:8000/metrics")
NODE_METRICS_URL = os.environ.get("NODE_METRICS_URL", "http://node-exporter:9100/metrics")

TRITON_TIMEOUT_S = float(os.environ.get("TRITON_TIMEOUT_S", "10"))
VLLM_TIMEOUT_S = float(os.environ.get("VLLM_TIMEOUT_S", "120"))
PREPROCESS_WORKERS = int(os.environ.get("PREPROCESS_WORKERS", "4"))

INPUT_NAME = "INPUT__0"
OUTPUT_NAME = "OUTPUT__0"

# Параметры препроцессинга convnext_tiny — взяты из
# ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms():
#   Resize(236, BILINEAR) -> CenterCrop(224) -> ToTensor -> Normalize(mean, std)
RESIZE_SIZE = 236
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

Image.MAX_IMAGE_PIXELS = None

_LABELS = [
    line.strip()
    for line in (Path(__file__).parent / "imagenet_classes.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

_pool = ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS, thread_name_prefix="prep")


async def _warmup() -> None:
    """Первый запрос к vLLM стоит десятки секунд — прогреваем оба бэкенда заранее."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    for _ in range(60):
        try:
            async with _client(VLLM_TIMEOUT_S) as client:
                r = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
                if r.status_code == 200:
                    break
        except httpx.HTTPError:
            pass
        await asyncio.sleep(5)

    try:
        dummy = np.zeros((1, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        await _triton_infer(dummy)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_warmup())
    yield
    task.cancel()
    await _triton_client.aclose()


app = FastAPI(title="Real-time Inference Gateway", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


def _authorize(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="empty token")
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid token") from None


def _client(timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=timeout_s, write=timeout_s, pool=5.0),
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
    )


_triton_client = _client(TRITON_TIMEOUT_S)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/triton/health")
async def triton_health() -> JSONResponse:
    try:
        r = await _triton_client.get(f"{TRITON_URL}/v2/health/ready")
        ready = r.status_code == 200
    except httpx.HTTPError:
        ready = False
    if not ready:
        return JSONResponse(status_code=503, content={"backend": "triton", "ready": False, "model": CV_MODEL})
    return JSONResponse(content={"backend": "triton", "ready": True, "model": CV_MODEL})


@app.get("/vllm/health")
async def vllm_health() -> JSONResponse:
    async with _client(10) as client:
        try:
            r = await client.get(f"{VLLM_URL}/v1/models")
            if r.status_code != 200:
                raise ValueError("not ready")
            models = [m.get("id") for m in r.json().get("data", [])]
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
            return JSONResponse(status_code=503, content={"backend": "vllm", "ready": False, "models": []})
    return JSONResponse(content={"backend": "vllm", "ready": True, "models": models})


async def _proxy_metrics(url: str) -> PlainTextResponse:
    async with _client(10) as client:
        try:
            r = await client.get(url)
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="backend metrics unavailable") from None
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="backend metrics unavailable")
    return PlainTextResponse(r.text, status_code=200)


@app.get("/metrics/triton")
async def metrics_triton() -> PlainTextResponse:
    return await _proxy_metrics(TRITON_METRICS_URL)


@app.get("/metrics/vllm")
async def metrics_vllm() -> PlainTextResponse:
    return await _proxy_metrics(VLLM_METRICS_URL)


@app.get("/metrics/node")
async def metrics_node() -> PlainTextResponse:
    return await _proxy_metrics(NODE_METRICS_URL)


@app.get("/info")
def info(_: dict = Depends(_authorize)) -> dict:
    return {
        "student": {"name": STUDENT_NAME, "email": STUDENT_EMAIL},
        "models": {"vision": CV_MODEL, "llm": LLM_MODEL},
        "gpu": GPU_NAME,
        "version": VERSION,
    }


def _preprocess(raw: bytes) -> np.ndarray:
    """Байты картинки → батч [1,3,224,224] FP32, ровно как weights.transforms()."""
    with Image.open(io.BytesIO(raw)) as src:
        src.load()
        img = src.convert("RGB")

    w, h = img.size
    if w <= h:
        new_w, new_h = RESIZE_SIZE, int(RESIZE_SIZE * h / w)
    else:
        new_w, new_h = int(RESIZE_SIZE * w / h), RESIZE_SIZE
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = int(round((new_w - CROP_SIZE) / 2.0))
    top = int(round((new_h - CROP_SIZE) / 2.0))
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN.transpose(1, 2, 0)) / STD.transpose(1, 2, 0)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)


async def _triton_infer(batch: np.ndarray) -> np.ndarray:
    body = {
        "inputs": [
            {
                "name": INPUT_NAME,
                "shape": list(batch.shape),
                "datatype": "FP32",
                "parameters": {"binary_data_size": int(batch.nbytes)},
            }
        ],
        "outputs": [{"name": OUTPUT_NAME, "parameters": {"binary_data": True}}],
    }
    head = json.dumps(body, separators=(",", ":")).encode("utf-8")
    r = await _triton_client.post(
        f"{TRITON_URL}/v2/models/{CV_MODEL}/infer",
        content=head + batch.tobytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(len(head)),
        },
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="triton inference failed")

    header_len = int(r.headers.get("Inference-Header-Content-Length", "0") or 0)
    payload = r.content
    if header_len:
        meta = json.loads(payload[:header_len])
        raw = payload[header_len:]
    else:
        meta = r.json()
        raw = b""

    out = meta["outputs"][0]
    shape = tuple(out["shape"])
    size = out.get("parameters", {}).get("binary_data_size")
    if size:
        logits = np.frombuffer(raw[: int(size)], dtype=np.float32).reshape(shape)
    else:
        logits = np.asarray(out["data"], dtype=np.float32).reshape(shape)
    return logits


def _softmax_top5(logits: np.ndarray) -> list[dict[str, Any]]:
    vec = logits.reshape(-1).astype(np.float64)
    vec = vec - vec.max()
    exp = np.exp(vec)
    probs = exp / exp.sum()
    idx = np.argsort(-probs)[:5]
    return [
        {"class_id": int(i), "label": _LABELS[int(i)], "score": float(probs[int(i)])}
        for i in idx
    ]


@app.post("/vision")
async def vision(file: UploadFile = File(...), _: dict = Depends(_authorize)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    loop = asyncio.get_running_loop()
    try:
        batch = await loop.run_in_executor(_pool, _preprocess, raw)
    except Exception:
        raise HTTPException(status_code=400, detail="not an image") from None

    started = time.perf_counter()
    logits = await _triton_infer(batch)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

    return {"model": CV_MODEL, "predictions": _softmax_top5(logits), "latency_ms": latency_ms}


async def _stream_vllm(client: httpx.AsyncClient, response: httpx.Response) -> AsyncIterator[bytes]:
    done = False
    try:
        async for line in response.aiter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                yield b"data: [DONE]\n\n"
                break
            if data:
                yield f"data: {data}\n\n".encode("utf-8")
    except httpx.HTTPError:
        yield b'data: {"error": {"message": "upstream stream interrupted"}}\n\n'
        return
    finally:
        await response.aclose()
        await client.aclose()

    if not done:
        yield b"data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: Request, _: dict = Depends(_authorize)):
    raw = await request.body()
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None

    if not isinstance(payload, dict) or not payload.get("messages"):
        raise HTTPException(status_code=400, detail="messages is required")

    stream = bool(payload.get("stream", True))
    payload["model"] = LLM_MODEL

    client = _client(VLLM_TIMEOUT_S)
    try:
        req = client.build_request("POST", f"{VLLM_URL}/v1/chat/completions", json=payload)
        response = await client.send(req, stream=True)
    except httpx.HTTPError:
        await client.aclose()
        raise HTTPException(status_code=502, detail="vLLM unavailable") from None

    if response.status_code >= 500:
        detail = (await response.aread())[:500].decode("utf-8", "replace")
        await response.aclose()
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"vLLM error: {detail}")

    if response.status_code >= 400:
        detail = (await response.aread())[:500].decode("utf-8", "replace")
        await response.aclose()
        await client.aclose()
        raise HTTPException(status_code=400, detail=f"vLLM rejected request: {detail}")

    if not stream:
        try:
            body = await response.aread()
        finally:
            await response.aclose()
            await client.aclose()
        return JSONResponse(content=json.loads(body))

    return StreamingResponse(
        _stream_vllm(client, response),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
