from __future__ import annotations

from typing import Optional, Dict, Any, List

from PIL import Image

from .llava_next import LLaVANextCaptioner
from .text_embedder import TextEmbedder
from .dinov2 import DinoV2
from .retrieval import Retriever
from .rrf import fuse_results_rrf


class Router:
    def __init__(self) -> None:
        self.llava = LLaVANextCaptioner(
            model_path="backend/models/llava-next"
        )
        self.text_embedder = TextEmbedder(
            model_path="backend/models/text"
        )
        self.dino = DinoV2(
            model_path="backend/models/dino"
        )

        self.retriever = Retriever(
            text_embedder=self.text_embedder,
            dino=self.dino,
            base_dir="backend/vector_stores"
        )

    def handle_query(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:

        text = (text or "").strip()
        has_text = len(text) > 0
        has_image = image is not None

        if not has_text and not has_image:
            return {
                "mode": "empty",
                "message": "No text or image provided.",
            }

        mode: str
        caption: Optional[str] = None

        if has_image:
            mode = "image" if not has_text else "image+text"
            caption = self.llava.caption(image)
        else:
            mode = "text"

        arm_results: Dict[str, List[Dict[str, Any]]] = {
            "text": [],
            "caption": [],
            "image": [],
        }

        if mode == "text":
            arm_results["text"] = self.retriever.search_text(text, top_k=top_k)
            arm_results["caption"] = self.retriever.search_caption(text, top_k=top_k)

        elif mode == "image":
            arm_results["image"] = self.retriever.search_image(image, top_k=top_k)

            if caption:
                arm_results["caption"] = self.retriever.search_caption(caption, top_k=top_k)
                arm_results["text"] = self.retriever.search_text(caption, top_k=top_k)

        else:
            arm_results["image"] = self.retriever.search_image(image, top_k=top_k)
            if caption:
                arm_results["caption"] = self.retriever.search_caption(caption, top_k=top_k)
                arm_results["text"] = self.retriever.search_text(
                    text + " " + caption,
                    top_k=top_k
                )
            else:
                arm_results["caption"] = self.retriever.search_caption(text, top_k=top_k)
                arm_results["text"] = self.retriever.search_text(text, top_k=top_k)

        fused = fuse_results_rrf(arm_results, k_rrf=60)


        return {
            "mode": mode,
            "query_text": text if has_text else None,
            "generated_caption": caption,
            "per_arm": arm_results,
            "fused_ranked": fused,
        }
