from typing import Optional, Dict, Any, List
from PIL import Image

from .llava_next import LLaVANextCaptioner
from .text_embedder import TextEmbedder
from .dinov2 import DinoV2
from .retrieval import Retriever
from .rrf import fuse_results_rrf


class Router:
    def __init__(self):
        self.llava = LLaVANextCaptioner(model_path="models/llava-next")
        self.text_embedder = TextEmbedder(model_path="models/text")
        self.dino = DinoV2(model_path="models/dino")

        self.retriever = Retriever(
            text_embedder=self.text_embedder,
            dino=self.dino,
            base_dir="vector_stores"
        )

    def handle_query(
        self,
        text: Optional[str],
        image: Optional[Image.Image],
        top_k: int = 5
    ) -> Dict[str, Any]:

        has_text = bool(text and text.strip())
        has_image = image is not None
        caption: Optional[str] = None

        identified_plant: Optional[Dict[str, Any]] = None
        fused_ranked: List[Dict[str, Any]] = []


        if has_image:
            mode = "image+text" if has_text else "image"

            caption = self.llava.caption(image)

            image_results = self.retriever.search_image(image, top_k)
            caption_results = self.retriever.search_caption(caption, top_k)

            arms_for_id = {
                "image": image_results,
                "caption": caption_results,
            }
            fused_for_id = fuse_results_rrf(arms_for_id, k_rrf=60)

            chosen_plant_id = None
            chosen_plant_name = None
            for item in fused_for_id:
                pid = item.get("plant_id")
                pname = item.get("plant_name")
                if pid or pname:
                    chosen_plant_id = pid
                    chosen_plant_name = pname
                    break

            plant_metadata_items: List[Dict[str, Any]] = []
            if chosen_plant_id:
                for meta in self.retriever.text_metadata:
                    if meta.get("plant_id") == chosen_plant_id:
                        enriched = dict(meta)
                        enriched["source"] = enriched.get("source", "plant_metadata")
                        enriched["id"] = enriched.get("id", f"plant_{chosen_plant_id}")
                        enriched.setdefault("rank", 0)
                        enriched.setdefault("faiss_distance", 0.0)
                        plant_metadata_items.append(enriched)

                if plant_metadata_items:
                    identified_plant = plant_metadata_items[0]

            fused_ranked = plant_metadata_items


        else:
            mode = "text"
            if has_text:
                text_results = self.retriever.search_text(text, top_k)
                fused_ranked = fuse_results_rrf({"text": text_results}, k_rrf=60)
            else:
                fused_ranked = []


        return {
            "mode": mode,
            "generated_caption": caption,
            "fused_ranked": fused_ranked,
            "identified_plant": identified_plant,
        }
