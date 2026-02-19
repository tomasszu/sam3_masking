import torch
import numpy as np
import matplotlib.cm as cm
from transformers import Sam3Processor, Sam3Model
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F

def extract_raw_logit_map(semantic_seg, image):
    # semantic_seg: [B,1,Hs,Ws]
    logits = semantic_seg[0, 0]   # [Hs, Ws]
    
    # Upsample logits to image size
    H, W = image.size[1], image.size[0]
    logits_up = torch.nn.functional.interpolate(
        logits.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False
    ).squeeze()    # [H,W]
    
    return logits_up.cpu().numpy()

import numpy as np
from PIL import Image
import matplotlib.cm as cm

def overlay_heatmap(image, heat):
    """heat: numpy array [H,W] with arbitrary float values"""
    image = ImageOps.grayscale(image)
    image = image.convert("RGBA")
    
    # Normalize heat → [0,1]
    h_min, h_max = heat.min(), heat.max()
    if h_max == h_min:
        heat_norm = np.zeros_like(heat)
    else:
        heat_norm = (heat - h_min) / (h_max - h_min)
    
    # Convert to color map (RGBA, 0–255)
    cmap = cm.get_cmap("jet")
    colors = (cmap(heat_norm) * 255).astype(np.uint8)  # [H,W,4]

    heatmap_img = Image.fromarray(colors, mode="RGBA")
    
    # Blend 50%
    return Image.blend(image, heatmap_img, alpha=0.5)

##### Caur Transformers package ####

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available - {torch.cuda.is_available()}")

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Load an image
image = Image.open("/home/toms.zinars/tomass/construction_photos/2.jpg")
text_prompt = "column"
# inference_state = processor.set_image(image)

inputs = processor(images=image, text=text_prompt, return_tensors="pt").to("cuda")

with torch.no_grad():

    outputs = model(**inputs)

# Instance segmentation masks

instance_masks = torch.sigmoid(outputs.pred_masks)  # [batch, num_queries, H, W]

# Semantic segmentation (single channel)

semantic_seg = outputs.semantic_seg  # [batch, 1, H, W]

print(f"Instance masks: {instance_masks.shape}")

print(f"Semantic segmentation: {semantic_seg.shape}")

logits = extract_raw_logit_map(outputs.semantic_seg, image)
overlay = overlay_heatmap(image, logits)

overlay.show()


