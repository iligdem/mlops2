from __future__ import annotations

import json

import torch
import torchvision
from PIL import Image

with open("/tmp/ours.json", encoding="utf-8") as fh:
    ours = json.load(fh)

weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = torchvision.models.convnext_tiny(weights=weights).eval()
transform = weights.transforms()

worst = 0.0
mismatch = 0
checked = 0

for path, preds in sorted(ours.items()):
    with Image.open(path) as src:
        src.load()
        img = src.convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(transform(img).unsqueeze(0)), 1)[0]
    ref_idx = int(probs.argmax())
    ref_score = float(probs[ref_idx])

    got_idx = preds[0]["class_id"]
    got_score = preds[0]["score"]
    diff = abs(ref_score - got_score)
    worst = max(worst, diff)
    checked += 1
    if ref_idx != got_idx:
        mismatch += 1
        print(f"РАСХОЖДЕНИЕ КЛАССА {path.split('/')[-1][:28]}: gateway {got_idx} vs эталон {ref_idx}")

print(f"сверено картинок: {checked}")
print(f"расхождений по top-1 классу: {mismatch}")
print(f"максимальное |Δscore|: {worst:.4f} (допуск грейдера ~0.02)")
print("ИТОГ:", "СОВПАДАЕТ" if mismatch == 0 and worst < 0.02 else "ЕСТЬ ПРОБЛЕМЫ")
