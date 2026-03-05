# This code will call the sam3 batch inference class on a bunch of images from a consecutive photo track

from PIL import Image
from sam3.visualization_utils import plot_results
import matplotlib.pyplot as plt

import os
import shutil
import time

from sam3_batch_inference_class import Sam3BatchInference

text_prompts = ["concrete", "painted wall", "windows/frames", "scaffolding", "people", "sand and gravel", "stairs", "tarps/plastic sheeting", "construction equipment"]

sam3 = Sam3BatchInference(text_prompts)

folder_path = r"/home/toms.zinars/tomass/data/rosbags/rosbag2_2025_09_09-19_49_04_converted_bag"

# Get only files, sorted alphabetically by name
files = sorted(
    (entry for entry in os.scandir(folder_path) if entry.is_file()),
    key=lambda e: e.name
)

for i, entry in enumerate(files):
    if i % 15 == 0:

        with Image.open(entry.path) as image:

            s = time.time()
            results, im_resized = sam3.predict(image)
            print("Done in ", (time.time() - s) * 1e3, "ms")

        plot_results(im_resized, results)

        plt.savefig(f"track1_results/masked/{entry.name}")
        plt.close("all")
        
        shutil.copyfile(entry.path, f"track1_results/orig/{entry.name}")

