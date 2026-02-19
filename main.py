import os

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

import torch

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# Load the model
model = build_sam3_image_model()

#image

image = Image.open("/home/toms.zinars/tomass/construction_photos/2.jpg")
width, height = image.size
processor = Sam3Processor(model, confidence_threshold=0.5)
inference_state = processor.set_image(image)

#Text

processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="column")

img0 = image.copy()
plot_results(img0, inference_state)

plt.waitforbuttonpress()


