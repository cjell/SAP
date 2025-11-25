import sys, json, faiss
import numpy as np
import glob
from PIL import Image

sys.path.append("backend/app")
from dinov2 import DinoV2


IMAGES_DIR = "data/images/*"
OUT_INDEX = "backend/vector_stores/image_faiss/index.faiss"
OUT_META  = "backend/vector_stores/image_faiss/metadata.json"


def build_image_index():
    print("\nBuilding Image FAISS store...\n")

    paths = sorted(glob.glob(IMAGES_DIR))
    if not paths:
        raise RuntimeError("No images found in data/images")

    dino = DinoV2()

    metadata = []
    vectors = []

    for i, img_path in enumerate(paths):
        img = Image.open(img_path).convert("RGB")
        vec = dino.embed_image(img)

        plant_id = img_path.split("/")[-1].split("\\")[-1].split(".")[0].lower()
        plant_name = plant_id.replace("_", " ").title()

        metadata.append({
            "id": f"image_{i}",
            "source": "image",
            "image_path": img_path,
            "plant_id": plant_id,
            "plant_name": plant_name
        })

        vectors.append(vec)

        print(f" embedded {i+1}/{len(paths)}")

    matrix = np.vstack(vectors).astype("float32")
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, OUT_INDEX)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved image index + metadata\n")


if __name__ == "__main__":
    build_image_index()
