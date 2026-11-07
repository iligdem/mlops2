from __future__ import annotations

import os
import random

from PIL import Image

OUT = os.environ.get("ASSETS", "/assets")
os.makedirs(OUT, exist_ok=True)
rnd = random.Random(0)


def noise(mode: str, size: tuple[int, int]) -> Image.Image:
    payload = bytes(rnd.randrange(256) for _ in range(size[0] * size[1] * len(mode)))
    return Image.frombytes(mode, size, payload)


noise("RGB", (300, 200)).save(f"{OUT}/rgb.png")
noise("RGBA", (300, 200)).save(f"{OUT}/rgba.png")
noise("L", (300, 200)).save(f"{OUT}/gray.png")
noise("CMYK", (300, 200)).save(f"{OUT}/cmyk.tiff")
noise("P", (300, 200)).save(f"{OUT}/palette.png")
Image.new("RGB", (8000, 8000), (120, 80, 40)).save(f"{OUT}/huge.jpg", quality=60)
Image.new("RGB", (1, 1), (10, 200, 30)).save(f"{OUT}/tiny.png")
open(f"{OUT}/empty.bin", "wb").close()
with open(f"{OUT}/text.txt", "w", encoding="utf-8") as fh:
    fh.write("this is definitely not an image")
with open(f"{OUT}/random.bin", "wb") as fh:
    fh.write(bytes(rnd.randrange(256) for _ in range(5000)))

print("assets:", sorted(os.listdir(OUT)))
