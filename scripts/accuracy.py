from __future__ import annotations

import http.client
import json
import os
import uuid

ROOT = os.environ.get("IMAGENETTE", "/tmp/imagenette2-160/val")
PER_CLASS = int(os.environ.get("PER_CLASS", "5"))
HOST, PORT = "localhost", 8090

CLASSES = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}

with open("gateway/imagenet_classes.txt", encoding="utf-8") as fh:
    LABELS = [line.strip() for line in fh if line.strip()]

bad = [(idx, name, LABELS[idx]) for idx, name in CLASSES.values() if LABELS[idx] != name]
if bad:
    raise SystemExit(f"несовпадение разметки: {bad}")
print("соответствие wnid → индекс ImageNet проверено по imagenet_classes.txt")

token = os.environ["TOK"]


def predict(path: str):
    with open(path, "rb") as fh:
        data = fh.read()
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=120)
    conn.request(
        "POST",
        "/vision",
        body,
        {"Authorization": "Bearer " + token, "Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    resp = conn.getresponse()
    payload = resp.read()
    if resp.status != 200:
        return None
    return json.loads(payload)["predictions"]


ours = {}
top1 = top5 = total = 0
per_class_ok = {}

for wnid, (idx, name) in CLASSES.items():
    folder = os.path.join(ROOT, wnid)
    files = sorted(os.listdir(folder))[:PER_CLASS]
    ok5 = 0
    for filename in files:
        path = os.path.join(folder, filename)
        preds = predict(path)
        total += 1
        if preds is None:
            print(f"  ОШИБКА {filename}")
            continue
        ids = [p["class_id"] for p in preds]
        ours[path] = preds
        hit5 = idx in ids
        hit1 = ids[0] == idx
        top5 += int(hit5)
        top1 += int(hit1)
        ok5 += int(hit5)
        print(f"  {filename[:28]:28} ожидали {idx:4} {name:16} -> got {ids[0]:4} {preds[0]['label'][:20]:20} score={preds[0]['score']:.4f} top5={'Y' if hit5 else 'N'}")
    per_class_ok[name] = f"{ok5}/{len(files)}"

print()
print(f"top-1 accuracy: {top1}/{total} = {top1 / total:.3f}")
print(f"top-5 accuracy: {top5}/{total} = {top5 / total:.3f}")
print("по классам (top-5):", per_class_ok)

with open("/tmp/ours.json", "w", encoding="utf-8") as fh:
    json.dump(ours, fh)
print("результаты сохранены в /tmp/ours.json")
