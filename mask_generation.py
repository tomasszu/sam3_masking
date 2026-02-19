from transformers import pipeline
from PIL import Image
import requests
import numpy as np
import matplotlib
import torch
from torchvision.transforms.functional import to_pil_image

def overlay_masks(image, masks):
    # masks = list of boolean tensors, vini jakonverte uz float/int mask
    image = image.convert("RGBA")
    
    n_masks = len(masks)
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):

        mask.squeeze()

        # --- Convert bool mask -> uint8 in [0,255]
        if mask.dtype == torch.bool:
            mask = mask.to(torch.uint8) * 255

        mask = to_pil_image(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

generator = pipeline("mask-generation", model="facebook/sam3", device=0)
image_url = "/home/toms.zinars/tomass/construction_photos/5.jpg"
outputs = generator(image_url, points_per_batch=64)

print(f"Output length: " + str(len(outputs["masks"])))  # Number of masks generated
print(f"Outputs: \n" + str(outputs))

image = Image.open("/home/toms.zinars/tomass/construction_photos/5.jpg")
masked_img = overlay_masks(image,outputs["masks"])

masked_img.show()


