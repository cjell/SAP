import io
import requests
import numpy as np
from typing import Optional, Dict, Any, List
from PIL import Image

from .retrieval import Retriever
from .rrf import fuse_results_rrf


class Router:
    def __init__(self):
        self.llava_url = "https://cjell-NepalRag.hf.space/llava"
        self.dino_url = "https://cjell-NepalRag.hf.space/dino"
        self.embed_url = "https://cjell-NepalRag.hf.space/embed"

        self.retriever = Retriever(router=self, base_dir="vector_stores")
        
    def run_llava(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("image.jpg", buf, "image/jpeg")}
        r = requests.post(self.llava_url, files=files)
        r.raise_for_status()
        return r.json()["caption"]



    def run_dino(self, img: Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("image.jpg", buf, "image/jpeg")}
        r = requests.post(self.dino_url, files=files)
        r.raise_for_status()
        return np.array(r.json()["embedding"])



    def run_text_embed(self, text: str):
        r = requests.post(self.embed_url, json={"text": text})
        r.raise_for_status()
        return np.array(r.json()["embedding"])



    def handle_query(
        self,
        text: Optional[str],
        image: Optional[Image.Image],
        top_k: int = 5
    ) -> Dict[str, Any]:

        has_text = bool(text and text.strip())
        has_image = image is not None

        caption = None
        identified_plant = None
        fused_ranked: List[Dict[str, Any]] = []


        if has_image:
            mode = "image+text" if has_text else "image"

            caption = self.run_llava(image)

            image_vec = self.run_dino(image)
            caption_vec = self.run_text_embed(caption)

            image_results = self.retriever.search_image(image_vec, top_k)
            caption_results = self.retriever.search_caption(caption_vec, top_k)

            fused_for_id = fuse_results_rrf(
                {"image": image_results, "caption": caption_results},
                k_rrf=60
            )

            chosen_id = None
            chosen_name = None

            for item in fused_for_id:
                pid = item.get("plant_id")
                pname = item.get("plant_name")
                if pid or pname:
                    chosen_id = pid
                    chosen_name = pname
                    break

            if chosen_id:
                matches = [
                    dict(meta)
                    for meta in self.retriever.text_metadata
                    if meta.get("plant_id") == chosen_id
                ]

                if matches:
                    identified_plant = matches[0]
                    fused_ranked = matches

        else:
            mode = "text"

            if has_text:
                text_vec = self.run_text_embed(text)
                text_results = self.retriever.search_text(text_vec, top_k)

                fused_ranked = fuse_results_rrf(
                    {"text": text_results},
                    k_rrf=60
                )
            else:
                fused_ranked = []

        return {
            "mode": mode,
            "generated_caption": caption,
            "fused_ranked": fused_ranked,
            "identified_plant": identified_plant,
        }
