import sys, json, faiss
import numpy as np
from PIL import Image
import glob

sys.path.append("backend/app")
from llava_next import LLaVANextCaptioner
from text_embedder import TextEmbedder


IMAGES_DIR = "data/images/*"
OUT_INDEX = "backend/vector_stores/caption_faiss/index.faiss"
OUT_META  = "backend/vector_stores/caption_faiss/metadata.json"


def clean_llava_output(text):
    if "assistant" in text:
        text = text.split("assistant", 1)[-1]
    return text.strip()


def build_caption_index():
    print("\nBuilding Caption FAISS store...\n")

    paths = sorted(glob.glob(IMAGES_DIR))
    if not paths:
        raise RuntimeError("No images found in data/images")

    llava = LLaVANextCaptioner()
    embedder = TextEmbedder()

    metadata = []
    vectors = []

    for i, img_path in enumerate(paths):
        img = Image.open(img_path).convert("RGB")

        raw = llava.caption(img)
        caption = clean_llava_output(raw)

        vec = embedder.embed(caption)

        plant_id = img_path.split("/")[-1].split("\\")[-1].split(".")[0].lower()
        plant_name = plant_id.replace("_", " ").title()

        metadata.append({
            "id": f"caption_{i}",
            "source": "caption",
            "image_path": img_path,
            "caption": caption,
            "plant_id": plant_id,
            "plant_name": plant_name
        })

        vectors.append(vec)

        print(f" captioned {i+1}/{len(paths)}")

    matrix = np.vstack(vectors).astype("float32")
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, OUT_INDEX)

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved caption index + metadata\n")


if __name__ == "__main__":
    build_caption_index()
