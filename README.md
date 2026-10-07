# GraphQA3GPP

Code and evaluation data for the paper *GraphQA3GPP*: question answering over 3GPP specifications with a knowledge graph beside BM25 and dense retrieval. Live system: https://www.ask3gpp.online/

The pipeline combines **dense retrieval** (`e5-base-v2` embeddings), **BM25**, **graph retrieval** (Neo4j Cypher written by an LLM over a knowledge graph built from the specifications) and a **cross-encoder reranker**, and answers technical questions about 4G/5G specifications with citations to `spec_id` and clause.

Licence: MIT. The 3GPP specifications are available from the 3GPP file server and are not included. TeleQnA is distributed by its authors and is not redistributed here.

## Repository layout

```
GraphQA3GPP/
├── document_processing/    # DOCX → JSON chunking pipeline
├── kg_builder/             # JSON → Neo4j knowledge graph + embeddings
├── rag-engine/             # FastAPI retrieval and answering service
├── frontend/               # Vite + React UI
├── scripts/                # start-deps, stop, rebuild-kg
└── data/3gpp_clauseqa/     # 3GPP-ClauseQA evaluation set + pooled labels
```

- **`document_processing/`** — single-file pipeline: download → extract DOCX → chunk by clause → JSON.
- **`kg_builder/`** — loads the JSON into Neo4j (Document, Chunk, Term, Message, ServiceOperation, Parameter, Step, … nodes and their edges) and creates the vector index.
- **`rag-engine/`** — FastAPI on `:8000`; `POST /api/query` streams the stages (intent → retrieval → rerank → answer → sources) over SSE, `POST /api/cypher` is a read-only Cypher tester. Prompts are in `rag-engine/llm/prompts.py`.
- **`frontend/`** — chat UI, Cypher tester and benchmark run viewer, port `:3000`.

### Retrieval modes

- **`fixed`** — dense + LLM-generated Cypher → RRF fusion → cross-encoder rerank (the full system of the paper). BM25-only, dense-only and BM25+dense configurations are selected through the request body.
- **`react_agent`** — adaptive ReAct loop in which a planner LLM chooses a tool each iteration from `vector | cypher | expand_term | inspect_chunk | finish`.

## 3GPP-ClauseQA

[`data/3gpp_clauseqa/3gpp_clauseqa.json`](data/3gpp_clauseqa/3gpp_clauseqa.json) holds the 3,000 evaluation questions used in the paper: 1,000 factoid, 1,000 procedure and 1,000 requirement questions, each with chunk-level gold labels and a reference answer. Every record has

| field | meaning |
|---|---|
| `qid` | question id |
| `question` | question text |
| `question_class` | `factoid` / `procedure` / `requirement` |
| `question_subclass`, `question_type` | authoring template and subject (message, service operation, information element, …) |
| `gold_chunk_ids` | clause ids that answer the question, e.g. `ts_23_502_4.2.2.2.2` |
| `reference_answer` | reference answer written from the gold clauses |
| `spec_series`, `structure_tags` | specification series; `cross_specification` / `cross_section` / `multi_hop` tags |

[`data/3gpp_clauseqa/pooling/`](data/3gpp_clauseqa/pooling/) holds the extended labels from TREC-style pooling on a stratified sample of 900 questions: `sample_900.json` (the sample), `pool.json` (13,012 question–clause pairs, the top-5 of seven retrieval configurations plus the authored gold), `judgments.json` (one blind judgment per pair) and `summary.json` (per-class Recall@5 under authored and pooled labels). The script that produced them is [`data/3gpp_clauseqa/pool_relabel.py`](data/3gpp_clauseqa/pool_relabel.py).

Per-question results of every run in the paper, including TeleQnA, are browsable on the public run viewer: https://www.ask3gpp.online/run.html?run=compare_kg_bench_3000_v5 and https://www.ask3gpp.online/run.html?run=teleqna10k_27092026. The run directories themselves are not in the repository because of their size.

## External services

- **Neo4j** — Docker container, ports `7474` / `7687`.
- **Ollama** — local, configured through `OLLAMA_URL`.

Copy `.env.example` to `.env` and set `NEO4J_*`, `OLLAMA_URL` and `JSON_DIR`.

## Setup

```bash
# Node dependencies for the root and the frontend
npm run install:all

# Python virtual environment
python -m venv venv
source venv/bin/activate
pip install -r document_processing/requirements.txt fastapi uvicorn python-dotenv \
    sentence-transformers neo4j tqdm requests
```

## Running the system

```bash
npm run dev:local  # rag-engine (8000) + frontend (3000), no tunnel — the usual way to run locally
npm run dev        # the same plus the cloudflared tunnel (public demo)
npm run stop       # stop everything, including the Neo4j container
```

Both start scripts first run `scripts/start-deps.sh`, which starts the Neo4j container if port 7474 does not answer and checks that Ollama is reachable. `dev:local` needs no cloudflared configuration.

Individual services:

```bash
npm run dev:rag        # FastAPI :8000
npm run dev:frontend   # Vite :3000
npm run dev:tunnel     # cloudflared (optional, public demo only)
```

## Building the knowledge graph

```bash
npm run rebuild-kg                  # full: clean + graph + embeddings
npm run rebuild-kg:kg-only          # graph only
npm run rebuild-kg:embed-only       # embeddings only
```

Re-chunk already extracted DOCX files into JSON (no download):

```bash
python document_processing/download_and_process_3gpp.py process-local \
  --input  document_processing/data/rel18_extracted \
  --output 3GPP_JSON_DOC/processed_json_v6
```
