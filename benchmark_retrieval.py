import json
import time
import numpy as np
from embeddings import unit_vectors
from retrieval import neighbors
from candidates import HERE


def main():
    n,d,k=10000,64,15
    vectors=unit_vectors(np.random.default_rng(42).normal(size=(n,d)),n)
    start=time.perf_counter()
    edges,stats=neighbors(vectors,k,-1,"hnsw")
    elapsed=time.perf_counter()-start
    # Measure final degree-capped edge recall against exact neighbors for 100 queries.
    retained={i:set() for i in range(100)}
    for i,j,_ in edges:
        if i in retained: retained[i].add(j)
        if j in retained: retained[j].add(i)
    recalls=[]
    for i in range(100):
        scores=vectors @ vectors[i]
        scores[i]=-np.inf
        truth=set(np.argsort(-scores)[:k].tolist())
        recalls.append(len(truth & retained[i])/k)
    report=dict(synthetic=True,columns=n,dimensions=d,k=k,seconds=elapsed,pairs=len(edges),
                final_edge_recall_at_k=float(np.mean(recalls)),queries_measured=100,search=stats,
                scope="Measures index build, search, and degree-capping on synthetic vectors; excludes embeddings, LLM calls, serialization and clustering. Million-column operation not verified.")
    output=HERE/"retrieval_scale_report.json"
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
