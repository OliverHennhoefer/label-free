import numpy as np
import faiss


def sireos(
    scores: np.ndarray,
    data: np.ndarray,
    quantile: float = 0.01,
) -> float:

    n_samples, n_features = np.shape(data)

    # Pairwise distance to set the parameter of the heat kernel using FAISS
    data_float32 = data.astype(np.float32)
    index = faiss.IndexFlatL2(n_features)
    index.add(data_float32)

    # Chunked computation to avoid memory issues
    chunk_size = min(5000, n_samples)
    all_distances = []
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        d_chunk, _ = index.search(data_float32[i:end_idx], n_samples)
        all_distances.append(d_chunk)

    D = np.vstack(all_distances)
    D = np.sqrt(D)
    t = np.quantile(D[np.nonzero(D)], 0.01)

    norm_scores = scores / scores.sum()

    # Computing the index
    distances, _ = index.search(data_float32, n_samples)
    distances = distances[:, 1:]
    exp_terms = np.exp(-distances / (2 * t * t))
    score = np.sum(np.mean(exp_terms, axis=1) * norm_scores.flatten())

    return score
