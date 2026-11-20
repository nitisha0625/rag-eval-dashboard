# RAG Evaluation Dashboard

A local benchmark for retrieval over fictional Mana Dhobi support documentation.
Compare sparse TF-IDF with dense latent semantic analysis (LSA) embeddings, then
inspect retrieved passages for individual questions. Both methods run offline.

## Run locally

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app
```

Open http://127.0.0.1:8000. The dashboard runs the benchmark on load. Select a
retriever and open a failed case to compare expected and retrieved passages.

To reproduce the report from the command line:

```bash
python app.py --output benchmark.json
```

## Evaluation

There are nine short passages, 36 answerable questions with one labeled passage
each, and four out-of-scope questions. Top-1 accuracy and Hit@3 use answerable
questions. Abstention uses out-of-scope questions. p95 latency covers retrieval
and ranking on all questions and varies by machine.

The first version returns the highest-ranked passage verbatim and abstains when
every similarity score is nonpositive. An irrelevant passage on an out-of-scope
question counts as a false answer. The demo measures retrieval and abstention.
A generative answer stage and citation-level evaluation are future work.
The small handcrafted fixture is suitable for a demo and functional regression
checks. The scores do not establish performance on larger corpora.
