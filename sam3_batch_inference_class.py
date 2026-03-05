from transformers import Sam3Processor, Sam3Model

import torch

class Sam3BatchInference:
    def __init__(self, prompts):
        self.text_prompts = prompts

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        
    def merge_results(self, results):
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
    
    def predict(self, image):
        image = image.resize((2024,1520))

        with torch.no_grad():

            inputs = self.processor(images=[image] * len(self.text_prompts), text=self.text_prompts, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.45,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )

            for i in range(len(results)):
                print(f"mask {i}: {len(results[i]['masks'])} objects found")

            merged_results = self.merge_results(results)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return merged_results, image