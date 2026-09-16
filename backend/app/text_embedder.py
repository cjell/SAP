import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_path="models/text"):

        print('-Loading SentenceTransformer-')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"-SentenceTransformer Ready on {self.device}-")

        self.model = SentenceTransformer(
            model_path,
            device=self.device
        )
    def embed(self, text: str) -> np.ndarray:

        # e5 was trained with "query:" / "passage:" prefixes and retrieval
        # noticeably degrades without them
        formatted = "query: " + text.strip()

        with torch.inference_mode():
            embedding = self.model.encode(
                formatted,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=False 
            )

        vec = embedding.cpu().numpy()

        # normalising by hand instead of letting sentence-transformers do it, so
        # the zero vector case doesn't come back as nan. FAISS uses IndexFlatIP,
        # and inner product on unit vectors is just cosine similarity.
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm


if __name__ == "__main__":
    embedder = TextEmbedder()
    vec = embedder.embed("Test embedding: Aconitum is a Himalayan medicinal plant.")
    print("Shape:", vec.shape)
    print("Norm:", np.linalg.norm(vec))