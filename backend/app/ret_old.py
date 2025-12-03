from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json

import numpy as np
import faiss
from PIL import Image

from .text_embedder import TextEmbedder
from .dinov2 import DinoV2


class Retriever:
    def __init__(
        self,
        text_embedder: TextEmbedder,
        dino: DinoV2,
        base_dir: str = "vector_stores"
    ) -> None:

        self.text_embedder = text_embedder
        self.dino = dino

        base_path = Path(base_dir)


        text_dir = base_path / "text_faiss"
        self.text_index = faiss.read_index(str(text_dir / "index.faiss"))
        with open(text_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.text_metadata: List[Dict[str, Any]] = json.load(f)

  
        cap_dir = base_path / "caption_faiss"
        self.captions_index = faiss.read_index(str(cap_dir / "index.faiss"))
        with open(cap_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.captions_metadata: List[Dict[str, Any]] = json.load(f)

        img_dir = base_path / "image_faiss"
        self.images_index = faiss.read_index(str(img_dir / "index.faiss"))
        with open(img_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.images_metadata: List[Dict[str, Any]] = json.load(f)

    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        vec = self.text_embedder.embed(query).astype("float32").reshape(1, -1)
        distances, indices = self.text_index.search(vec, top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self.text_metadata):
                continue

            meta = dict(self.text_metadata[idx])
            meta["faiss_distance"] = float(dist)
            meta["rank"] = rank

            meta["id"] = meta.get("id", f"text_{idx}")
            meta["source"] = meta.get("source", "text")

            results.append(meta)

        return results


    def search_caption(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        vec = self.text_embedder.embed(query).astype("float32").reshape(1, -1)
        distances, indices = self.captions_index.search(vec, top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self.captions_metadata):
                continue

            meta = dict(self.captions_metadata[idx])
            meta["faiss_distance"] = float(dist)
            meta["rank"] = rank

            meta["id"] = meta.get("id", f"caption_{idx}")
            meta["source"] = meta.get("source", "caption")

            results.append(meta)

        return results

 
    def search_image(self, image: Image.Image, top_k: int = 5) -> List[Dict[str, Any]]:
        if image is None:
            return []

        img_rgb = image.convert("RGB")
        vec = self.dino.embed_image(img_rgb).astype("float32").reshape(1, -1)

        distances, indices = self.images_index.search(vec, top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self.images_metadata):
                continue    

            meta = dict(self.images_metadata[idx])
            meta["faiss_distance"] = float(dist)
            meta["rank"] = rank

            meta["id"] = meta.get("id", f"image_{idx}")
            meta["source"] = meta.get("source", "image")

            results.append(meta)

        return results
