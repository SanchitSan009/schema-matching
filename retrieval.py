"""Bounded per-column cosine retrieval with exact-small and HNSW backends."""

import numpy as np
from embeddings import unit_vectors


def neighbors(vectors, k=15, threshold=0.9, method="auto"):
    n = len(vectors)
    if type(k) is not int or k < 1 or not -1 <= threshold <= 1 or method not in {"auto", "exact", "hnsw"}:
        raise ValueError("invalid retrieval configuration")
    if n < 2:
        return [], dict(method="none", directed_edges=0, degree_drops=0)
    vectors = np.ascontiguousarray(unit_vectors(vectors, n))
    selected = "exact" if method == "auto" and n <= 512 else "hnsw" if method == "auto" else method
    if selected == "exact" and n > 5000:
        raise ValueError("exact search is limited to 5000 columns; use hnsw")
    candidates = {}
    if selected == "hnsw":
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install faiss-cpu from schema_matching/requirements.txt for HNSW retrieval") from exc
        faiss.omp_set_num_threads(2)
        index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 100
        index.hnsw.efSearch = max(64, 4 * k)
        index.add(vectors)
    directed = 0
    for start in range(0, n, 256):
        if selected == "hnsw":
            scores, indices = index.search(vectors[start:start + 256], min(k + 1, n))
        else:
            matrix = vectors[start:start + 256] @ vectors.T
            indices = np.argsort(-matrix, axis=1, kind="stable")[:, :min(k + 1, n)]
            scores = np.take_along_axis(matrix, indices, axis=1)
        for offset, (row_ids, row_scores) in enumerate(zip(indices, scores)):
            i = start + offset
            count = 0
            for j, score in zip(row_ids, row_scores):
                j = int(j)
                if j < 0 or j == i or score < threshold or count >= k:
                    continue
                count += 1
                directed += 1
                pair = tuple(sorted((i, j)))
                candidates[pair] = max(candidates.get(pair, -1), float(np.clip(score, -1, 1)))
    # Union top-k can create hubs with unbounded incoming degree. Cap final degree.
    degrees = np.zeros(n, dtype=int)
    retained, drops = [], 0
    for (i, j), score in sorted(candidates.items(), key=lambda p: (-p[1], p[0])):
        if degrees[i] >= k or degrees[j] >= k:
            drops += 1
            continue
        degrees[i] += 1
        degrees[j] += 1
        retained.append((i, j, score))
    return sorted(retained), dict(method=selected, directed_edges=directed, degree_drops=drops,
        max_final_degree=int(degrees.max()), theoretical_pair_bound=n * k // 2,
        approximation=selected == "hnsw")
