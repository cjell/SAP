import sys, json, faiss
import numpy as np
from PyPDF2 import PdfReader

sys.path.append("backend/app")

from text_embedder import TextEmbedder


PDF_PATH = "data/Indogangetic.pdf"
PLANT_PATH = "data/plant_data.json"

OUT_INDEX = "backend/vector_stores/text_faiss/index.faiss"
OUT_META = "backend/vector_stores/text_faiss/metadata.json"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80


def extract_pdf_chunks(path):
    reader = PdfReader(path)
    chunks, meta = [], []
    chunk_id = 0

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        text = text.replace("\n", " ").strip()
        L = len(text)
        start = 0

        while start < L:
            end = min(start + CHUNK_SIZE, L)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                meta.append({
                    "id": f"pdf_{chunk_id}",
                    "source": "pdf",
                    "page": page_idx,
                    "chunk_index": chunk_id,
                    "text": chunk
                })
                chunk_id += 1

            if end == L:
                break
            start += (CHUNK_SIZE - CHUNK_OVERLAP)

    return chunks, meta


def load_plant_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks, meta = [], []
    for plant in data:
        info = plant.get("info", "").strip()
        if not info:
            continue

        chunks.append(info)
        meta.append({
            "id": f"plant_{plant['id']}",
            "source": "plant_metadata",
            "plant_id": plant["id"],
            "name": plant.get("name", ""),
            "text": info
        })

    return chunks, meta


def build_text_index():
    print("\nBuilding Text\n")

    pdf_chunks, pdf_meta = extract_pdf_chunks(PDF_PATH)
    plant_chunks, plant_meta = load_plant_metadata(PLANT_PATH)

    texts = pdf_chunks + plant_chunks
    metadata = pdf_meta + plant_meta

    print(f"-Total chunks: {len(texts)}-")

    embedder = TextEmbedder()
    vectors = []

    for i, t in enumerate(texts):
        vectors.append(embedder.embed(t))
        if (i + 1) % 25 == 0:
            print(f"-embedded {i+1}/{len(texts)}-")

    matrix = np.vstack(vectors).astype("float32")
    dim = matrix.shape[1]

    print(f"-Matrix shape: {matrix.shape}-")

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, OUT_INDEX)

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n-Saved text index + metadata-\n")


if __name__ == "__main__":
    build_text_index()
