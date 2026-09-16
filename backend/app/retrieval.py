# Handles retrieval to vector databases (/vectorstores)


from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json

import numpy as np
import faiss
from PIL import Image


class Retriever:
    """Thin wrapper over the three FAISS stores.

    Each store ships as an index.faiss plus a metadata.json written at build time,
    and row i of the index lines up with entry i of that list. Rebuilding one
    without the other silently returns the wrong plants, so they travel together.
    """

    def __init__(
        self,
        router,
        base_dir: str = "vector_stores"
    ) -> None:

        self.router = router

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

    def search_text(self, query, top_k: int = 5) -> List[Dict[str, Any]]:

        if query is None:
            return []

        # takes either raw text or an already-embedded vector, since the router
        # sometimes has one in hand and there's no point paying for the call twice
        if isinstance(query, str):
            if not query.strip():
                return []
            vec = np.array(self.router.run_text_embed(query), dtype="float32").reshape(1, -1)

        else:
            vec = np.array(query, dtype="float32").reshape(1, -1)

        distances, indices = self.text_index.search(vec, top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            # faiss pads with -1 when the store holds fewer than top_k rows
            if idx < 0 or idx >= len(self.text_metadata):
                continue

            # copy, otherwise the scores below get written back into the store
            meta = dict(self.text_metadata[idx])
            meta["faiss_distance"] = float(dist)
            meta["rank"] = rank
            meta["id"] = meta.get("id", f"text_{idx}")
            meta["source"] = meta.get("source", "text")

            results.append(meta)

        return results

    def search_caption(self, query, top_k: int = 5) -> List[Dict[str, Any]]:

        if query is None:
            return []

        if isinstance(query, str):
            if not query.strip():
                return []
            vec = np.array(self.router.run_text_embed(query), dtype="float32").reshape(1, -1)

        else:
            vec = np.array(query, dtype="float32").reshape(1, -1)

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

    # image side only ever gets a vector - DINO runs on the remote space, so the
    # router has already done the embedding by the time we get here
    def search_image(self, image_vec, top_k: int = 5) -> List[Dict[str, Any]]:

        if image_vec is None:
            return []

        vec = np.array(image_vec, dtype="float32").reshape(1, -1)

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
