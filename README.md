# Real-time NLP/CV инференс — ДЗ

Шлюз к Triton (CV-модель варианта) и vLLM (Qwen2.5-1.5B-Instruct): JWT-авторизация,
`/vision`, `/chat` со стримингом, health-ручки бэкендов и проброс их метрик.

Постановка и критерии — в [ЗАДАНИЕ.md](ЗАДАНИЕ.md), контракт ручек — в [API_CONTRACT.md](API_CONTRACT.md).

## Требования

- NVIDIA GPU + драйвер + `nvidia-container-toolkit`
- Docker и Docker Compose v2
- Python 3.10–3.13 на хосте (для экспорта ONNX; под 3.14 torch не ставится)

## Запуск

```bash
cp .env.example .env
# заполнить: JWT_SECRET (openssl rand -hex 32), STUDENT_NAME, STUDENT_EMAIL, CV_MODEL

make variant   # проверить, что CV_MODEL совпадает с вариантом по email
make init      # экспорт весов → model_repository/<CV_MODEL>/1/model.onnx
make up        # поднять стек
make ps        # дождаться healthy у gateway, triton, vllm
make token     # Bearer-токен для грейдера
make smoke     # дёрнуть ручки
```

`base_url` — адрес gateway, доступный грейдеру, порт берётся из `GATEWAY_PORT`.

## Чего нет в архиве

| Отсутствует | Как получить |
|---|---|
| `model_repository/<CV_MODEL>/1/model.onnx` | `make init` |
| веса `Qwen/Qwen2.5-1.5B-Instruct` | скачиваются vLLM при первом старте |
| `.env` | создаётся из `.env.example` |

## Состав

- `gateway/` — FastAPI-шлюз (ручки, JWT, препроцессинг, SSE)
- `model_repository/<CV_MODEL>/config.pbtxt` — конфиг модели для Triton
- `docker-compose.yml` — Triton, vLLM, gateway, Prometheus, node-exporter
- `scripts/` — `variant.py`, `export_model.py`, `make_token.py`, `smoke.sh`
- `prometheus/prometheus.yml`
