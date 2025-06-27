from pix2text import Pix2Text
from PIL import Image
import json
import os

pix2text = Pix2Text(det_model='mfd', ocr_model='mfr')

with open("results_step1.json", "r", encoding="utf-8") as f:
    blocks = json.load(f)

for block in blocks:
    if block["type"] == "math":
        try:
            img = Image.open(block["image"])
            result = pix2text.recognize(img)
            block["text"] = result[0]["text"] if isinstance(result, list) else str(result)
        except Exception as e:
            block["text"] = f"[수식 인식 오류: {e}]"

with open("results_final.json", "w", encoding="utf-8") as f:
    json.dump(blocks, f, ensure_ascii=False, indent=2)