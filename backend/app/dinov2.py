import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class DinoV2:
    def __init__(self, model_path="models/dino"):
        print("-Loading DINO Processor-")
        self.processor = AutoImageProcessor.from_pretrained(
            model_path,
            use_fast=True
        )

        print("-Loading DINO Model-")
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda"
        )

        self.model.eval()

        print("-DINO Loaded-")

    def embed_image(self, image: Image.Image) -> np.ndarray:

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # token 0 is the CLS token, DINOv2's global descriptor for the whole
        # image. The patch tokens after it are per-region and not useful here.
        last_hidden = outputs.last_hidden_state  
        cls_embedding = last_hidden[:, 0, :]

        vec = cls_embedding.squeeze().float().cpu().numpy()

        vec = vec / np.linalg.norm(vec)  # unit length to match the IP index

        return vec
    
if __name__ == "__main__":
    img = Image.open("testing/test_images/Aconitum_heterophyllum_test.jpg").convert("RGB")
    dino = DinoV2()
    emb = dino.embed_image(img)
    print("Embedding shape:", emb.shape)
    print("L2 norm:", np.linalg.norm(emb))