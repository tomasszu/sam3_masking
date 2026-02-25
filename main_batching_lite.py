from transformers import Sam3Processor, Sam3Model

import torch

from PIL import Image
from sam3.visualization_utils import plot_results
import matplotlib.pyplot as plt

import torch.nn.functional as F

def merge_results(results):
    scores = []
    boxes = []
    masks = []

    # making sure that we dont pass along empty batches
    for r in results:
        if r["masks"].numel() == 0:
            continue  # skip empty detections

        scores.append(r["scores"])
        boxes.append(r["boxes"])
        masks.append(r["masks"])

    # if no batch returns a mask
    if len(masks) == 0:
        return {"scores": torch.empty(0),
                "boxes": torch.empty(0),
                "masks": torch.empty(0)}

    return {
        "scores": torch.cat(scores, dim=0),
        "boxes": torch.cat(boxes, dim=0),
        "masks": torch.cat(masks, dim=0),
    }



device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

#image

image = Image.open("/home/toms.zinars/tomass/construction_photos/stock_photos/2.jpg")
# ja CUDA out of memory:

image = image.resize((2024,1520))
width, height = image.size

images = []

text_prompts = ["concrete", "ceiling", "pipe or metal", "car or person"]

inputs = processor(images=[image] * len(text_prompts), text=text_prompts, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# print("_______________outputs______________________")
# print(outputs)

# Post-process results for both images
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.45,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)
# print("_______________results______________________")
# print(results)

for i in range(len(results)):
    print(f"Image {i}: {len(results[i]['masks'])} objects found")


merged_results = merge_results(results)
plot_results(image, merged_results)
plt.waitforbuttonpress()
#plt.savefig()
