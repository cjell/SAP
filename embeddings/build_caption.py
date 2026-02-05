# Builds vectorstore for image caption embeddings produced by LLaVA-Next

import sys, json, faiss
import numpy as np
from PIL import Image
import glob

sys.path.append("backend/app")
from llava_next import LLaVANextCaptioner
from text_embedder import TextEmbedder


IMAGES_DIR = "data/images/*"
OUT_INDEX = "backend/vector_stores/caption_faiss/index.faiss"
OUT_META = "backend/vector_stores/caption_faiss/metadata.json"


def clean_llava_output(text):
    if "assistant" in text:
        text = text.split("assistant", 1)[-1]
    return text.strip()


def build_caption_index():
    print("\nBuilding Captions\n")

    paths = sorted(glob.glob(IMAGES_DIR))
    if not paths:
        raise RuntimeError("No images found in data/images.")

    llava = LLaVANextCaptioner()
    embedder = TextEmbedder()

    vectors = []
    metadata = []

    for i, img_path in enumerate(paths):
        img = Image.open(img_path).convert("RGB")

        raw = llava.caption(img)
        caption = clean_llava_output(raw)

        vec = embedder.embed(caption)

        vectors.append(vec)
        metadata.append({
            "id": f"caption_{i}",
            "source": "caption",
            "image_path": img_path,
            "caption": caption
        })

        print(f"  captioned {i+1}/{len(paths)}")

    matrix = np.vstack(vectors).astype("float32")
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, OUT_INDEX)

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n-Saved caption index + metadata-\n")


if __name__ == "__main__":
    build_caption_index()
