import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DOCS = json.loads((ROOT / "cases.json").read_text())
QUESTIONS = [(q, i) for i, (_, _, *qs) in enumerate(DOCS) for q in qs]
QUESTIONS += [(q, None) for q in (
    "Do you offer a student discount?",
    "Can I ship an order internationally?",
    "Do vendors provide sewing lessons?",
    "What is the loyalty program enrollment fee?",
)]
VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
INDEX = VECTORIZER.fit_transform([doc[1] for doc in DOCS])
PROJECTION = TruncatedSVD(n_components=5, random_state=42).fit(INDEX)
DENSE_INDEX = PROJECTION.transform(INDEX)
app = FastAPI(title="RAG Evaluation Dashboard")


def evaluate():
    results = {}
    for method in ("Keyword TF-IDF", "Dense LSA"):
        rows = []
        for question, expected in QUESTIONS:
            start = perf_counter()
            query = VECTORIZER.transform([question])
            scores = (cosine_similarity(query, INDEX)[0] if method == "Keyword TF-IDF"
                      else cosine_similarity(PROJECTION.transform(query), DENSE_INDEX)[0])
            order = np.argsort(-scores, kind="stable")[:3]
            ranking = [int(i) for i in order if scores[i] > 1e-9]
            elapsed = (perf_counter() - start) * 1000
            top = ranking[0] if ranking else None
            rows.append({
                "question": question, "expected": DOCS[expected][0] if expected is not None else None,
                "pass": top == expected, "hit_at_3": expected is not None and expected in ranking,
                "answer": DOCS[top][1] if top is not None else "No evidence retrieved.",
                "evidence": [{"id": DOCS[i][0], "text": DOCS[i][1], "score": round(float(scores[i]), 3)}
                             for i in ranking],
                "latency_ms": elapsed,
            })
        answerable = [row for row in rows if row["expected"]]
        unanswerable = [row for row in rows if not row["expected"]]
        results[method] = {
            "top1": sum(row["pass"] for row in answerable) / len(answerable),
            "hit3": sum(row["hit_at_3"] for row in answerable) / len(answerable),
            "abstention": sum(row["pass"] for row in unanswerable) / len(unanswerable),
            "p95_ms": round(float(np.percentile([row["latency_ms"] for row in rows], 95)), 2),
            "cases": rows,
        }
    return results


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "index.html")


@app.get("/api/evaluate")
def benchmark():
    return evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(evaluate(), indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    else:
        print(report)
