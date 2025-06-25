from sentence_transformers import SentenceTransformer, SimilarityFunction


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

    def encode_generated(self, generated):
        encodings = self.sbert.encode(
            generated, convert_to_tensor=True,
            normalize_embeddings=True
        )

        similarities = self.sbert.similarity(encodings, encodings)
        return similarities, {y: idx for idx, y in enumerate(generated)}
        # encoding_map = {}
        # for idx, y in enumerate(generated):
        #     encoding_map[y] = encodings[idx, :]

        # return encoding_map
