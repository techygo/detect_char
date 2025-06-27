import os
import cv2
import numpy as np
from PIL import Image
from craft_text_detector import Craft
import easyocr
import json

craft = Craft(output_dir=None, crop_type="box", cuda=False)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

os.makedirs("cropped_blocks", exist_ok=True)

def is_math_like(text):
    return any(sym in text for sym in ['=', '+', '-', '\\', '^', '_', '∫', '∑'])

image_path = "math_eq.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape

craft_result = craft.detect_text(image)
boxes = craft_result["boxes"]

result_data = []

for i, box in enumerate(boxes):
    x_min = max(0, int(np.min(box[:,0])))
    y_min = max(0, int(np.min(box[:,1])))
    x_max = min(width, int(np.max(box[:,0])))
    y_max = min(height, int(np.max(box[:,1])))

    crop_img = image_rgb[y_min:y_max, x_min:x_max]
    pil_crop = Image.fromarray(crop_img)
    crop_path = f"cropped_blocks/block_{i:03d}.png"
    pil_crop.save(crop_path)

    temp_text = " ".join(reader.readtext(crop_img, detail=0, paragraph=True))
    block_type = "math" if is_math_like(temp_text) else "text"

    result_data.append({
        "index": i,
        "type": block_type,
        "text": temp_text,
        "image": crop_path
    })

with open("results_step1.json", "w", encoding="utf-8") as f:
    json.dump(result_data, f, ensure_ascii=False, indent=2)

craft.unload_craftnet_model()
craft.unload_refinenet_model()