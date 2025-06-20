from sentence_transformers import SentenceTransformer, SimilarityFunction
from hydra.utils import get_object
# distiluse-base-multilingual-cased-v1


class Similarity:
    def __init__(self, cfg):
        self.sbert = SentenceTransformer(
            cfg.uncertainty.similarity_model,
            similarity_fn_name=SimilarityFunction[cfg.uncertainty.similarity_fn_name]
        )

    def __call__(self, y, y_prime):
        y_enc = self.sbert.encode(
            y, convert_to_tensor=True,
            normalize_embeddings=True
        )
        y_prime_enc = self.sbert.encode(
            y_prime, convert_to_tensor=True,
            normalize_embeddings=True
        )
        return self.sbert.similarity(y_enc, y_prime_enc).mean(dim=0)
