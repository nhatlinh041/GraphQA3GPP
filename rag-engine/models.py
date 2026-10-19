"""
Shared model loading — embedding + cross-encoder.
Loaded once at startup to avoid per-request overhead.
"""
import os

# Run the retrieval models on CPU by default (2026-07-12). e5-base-v2 (110M) and
# MiniLM-L6 (22M) are tiny and run fast on CPU, but on the GPU they compete for
# VRAM with the Ollama LLM (9.3 GB qwen3:14b) — together they overflow a 16 GB
# card and thrash. Pinning them to CPU keeps the GPU exclusively for the LLM.
# PROJECT RULE (2026-09-27): both retrieval models run on CPU for EVERY run, cloud
# generator or local. The env override is deliberately gone: .env had silently
# carried RETRIEVAL_DEVICE=cuda from the cloud-generator runs, and because the
# uvicorn reloader passes its own environment to each worker (load_dotenv never
# overrides an existing var), editing .env did not even take effect on reload.
RETRIEVAL_DEVICE = "cpu"
RERANKER_DEVICE = "cpu"

# Hard-hide the GPU from torch when running the retrieval models on CPU. Setting
# device="cpu" alone still lets torch initialise a CUDA context (~300 MB) at
# import — with CUDA_VISIBLE_DEVICES="" torch sees no GPU at all, so this process
# touches ZERO VRAM. Ollama runs as a SEPARATE process (Windows host, port 11435)
# and is unaffected — it keeps its own GPU access. MUST be set before torch is
# imported (i.e. before importing sentence_transformers below).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sentence_transformers import SentenceTransformer, CrossEncoder

# Embedding model (e5-base-v2, same as current RAG system)
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
# Cross-encoder for reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embedding_model: SentenceTransformer | None = None
_reranker_model: CrossEncoder | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=RETRIEVAL_DEVICE)
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        # Always CPU, regardless of RETRIEVAL_DEVICE (project rule since 2026-09-27):
        # every run must rerank on the same device so latency and scores stay
        # comparable across runs. .env had silently carried RETRIEVAL_DEVICE=cuda
        # from the cloud-generator runs, so a run believed to be on CPU was not.
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=RERANKER_DEVICE)
    return _reranker_model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using e5 model."""
    model = get_embedding_model()
    # e5 requires "query: " / "passage: " prefix
    return model.encode(texts, normalize_embeddings=True).tolist()
