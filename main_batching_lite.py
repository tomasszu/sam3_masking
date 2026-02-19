from transformers import Sam3Processor, Sam3Model

import torch

from PIL import Image
from sam3.visualization_utils import plot_results
import matplotlib.pyplot as plt

def merge_results(results):
    merged = {
        "scores": torch.cat([r["scores"] for r in results], dim=0),
        "boxes":  torch.cat([r["boxes"] for r in results], dim=0),
        "masks":  torch.cat([r["masks"] for r in results], dim=0),
    }
    return merged



device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

#image

image = Image.open("/home/toms.zinars/tomass/construction_photos/rosbag_fisheye/2.png")
# ja CUDA out of memory:
image = image.resize((2024,1520))
width, height = image.size

images = []

text_prompts = ["cobblestone", "building facade"]

for i in range(text_prompts.__len__()):
    images.append(image)

inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results for both images
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.45,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)

for i in range(len(results)):
    print(f"Image {i}: {len(results[i]['masks'])} objects found")


merged_results = merge_results(results)
plot_results(image, merged_results)
plt.waitforbuttonpress()
#plt.savefig()
