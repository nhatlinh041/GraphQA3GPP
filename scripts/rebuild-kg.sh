#!/usr/bin/env bash
# Rebuild the Knowledge Graph — clears zombie schema metadata before building.
#
# Usage (run from the repo root):
#   bash scripts/rebuild-kg.sh                 # full: clean + KG + embeddings
#   bash scripts/rebuild-kg.sh kg-only         # clean + KG (no embeddings)
#   bash scripts/rebuild-kg.sh embed-only      # embeddings only (KG already built)
#   SKIP_RESTART=1 bash scripts/rebuild-kg.sh  # skip the docker restart (keeps zombie tokens)
#
# Cleanup pipeline (full/kg-only):
#   1. Drop ALL constraints + indexes (vector and legacy ones included)
#   2. Batched DETACH DELETE all nodes (10k/batch)
#   3. Restart the neo4j-server Docker container (clears zombie label/type tokens)
#   4. Set up the schema (10 node labels + 18 edge types, see builder.py CYPHER_CONSTRAINTS)
#   5. Load JSON → build graph
#   6. (full) Embedder + vector index
#
# Env vars (or declare them in .env):
#   JSON_DIR        — directory of processed JSON (default: ../3GPP_JSON_DOC/processed_json_v6)
#   NEO4J_URI       — (default: neo4j://localhost:7687)
#   NEO4J_USER      — (default: neo4j)
#   NEO4J_PASSWORD  — (default: password)
#   SKIP_RESTART    — set to 1 to skip the docker restart (keeps zombie metadata)
#   NEO4J_CONTAINER — container name (default: neo4j-server)

set -e
DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ok()   { echo -e "\033[32m✓\033[0m $*"; }
info() { echo -e "\033[36mℹ\033[0m $*"; }
warn() { echo -e "\033[33m⚠\033[0m $*"; }
err()  { echo -e "\033[31m✗\033[0m $*"; exit 1; }

# Preserve variables the caller passed on the command line: `. .env` under `set -a`
# OVERWRITES them, so `JSON_DIR=... bash scripts/rebuild-kg.sh full` was once silently
# swallowed by .env — the script printed "JSON dir: ...v4" and built the wrong corpus.
# The caller must win over .env.
_CALLER_JSON_DIR="${JSON_DIR:-}"
_CALLER_REEMBED_ALL="${REEMBED_ALL:-}"
_CALLER_SKIP_RESTART="${SKIP_RESTART:-}"

# Load .env if present
[ -f "$DEMO_DIR/.env" ] && { set -a; . "$DEMO_DIR/.env"; set +a; }

[ -n "$_CALLER_JSON_DIR" ]     && JSON_DIR="$_CALLER_JSON_DIR"
[ -n "$_CALLER_REEMBED_ALL" ]  && REEMBED_ALL="$_CALLER_REEMBED_ALL"
[ -n "$_CALLER_SKIP_RESTART" ] && SKIP_RESTART="$_CALLER_SKIP_RESTART"

# Default JSON dir: use ./3GPP_JSON_DOC if present (portable), else the parent dir
if [ -d "$DEMO_DIR/3GPP_JSON_DOC/processed_json_v6" ]; then
  JSON_DIR_DEFAULT="$DEMO_DIR/3GPP_JSON_DOC/processed_json_v6"
else
  JSON_DIR_DEFAULT="$DEMO_DIR/../3GPP_JSON_DOC/processed_json_v6"
fi
JSON_DIR="${JSON_DIR:-$JSON_DIR_DEFAULT}"
MODE="${1:-full}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-neo4j-server}"
SKIP_RESTART="${SKIP_RESTART:-0}"

info "Demo dir:    $DEMO_DIR"
info "JSON dir:    $JSON_DIR"
info "Mode:        $MODE"
info "Container:   $NEO4J_CONTAINER"
info "Skip restart: $SKIP_RESTART"

# Activate the venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
  VENV="$DEMO_DIR/../.venv"
  [ -d "$VENV" ] || err "venv not found at $VENV. Run: python -m venv $VENV && pip install -r requirements.txt"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
fi

# ── Helper: restart the Neo4j Docker container to clear zombie label/type tokens ──
restart_neo4j() {
  if [ "$SKIP_RESTART" = "1" ]; then
    warn "Skipping Docker restart (SKIP_RESTART=1) — zombie label/type tokens will persist."
    return
  fi

  if ! command -v docker &>/dev/null; then
    warn "Docker not available — skipping restart, zombie tokens will persist."
    return
  fi

  if ! docker ps --filter "name=^${NEO4J_CONTAINER}$" --format "{{.Names}}" 2>/dev/null | grep -q "${NEO4J_CONTAINER}"; then
    warn "Container ${NEO4J_CONTAINER} is not running — skipping restart."
    return
  fi

  info "Restarting Docker container '${NEO4J_CONTAINER}' to clear zombie tokens..."
  docker restart "$NEO4J_CONTAINER" >/dev/null

  # Wait for Neo4j HTTP to be ready (max 30s)
  info "Waiting for Neo4j HTTP..."
  for i in $(seq 1 30); do
    if curl -sf --max-time 3 http://localhost:7474 > /dev/null 2>&1; then
      ok "Neo4j ready sau ${i}s"
      # Give Bolt another 2s to settle
      sleep 2
      return
    fi
    sleep 1
  done
  err "Neo4j not ready after 30s. Check: docker logs ${NEO4J_CONTAINER}"
}

# ── Phase 1: Full clean (drop schema + delete data) ─────────────────────────
run_clean() {
  python - <<'PYEOF'
import sys, os
from pathlib import Path
demo_dir = Path(os.environ["DEMO_DIR"])
sys.path.insert(0, str(demo_dir))

from kg_builder import KGBuilder

builder = KGBuilder()
if not builder.verify_connection():
    sys.exit(1)
print("[kg] === Phase 1: Full clean ===")
builder.clean_all()
builder.close()
PYEOF
}

# ── Phase 2: Build KG (after restart, fresh schema) ─────────────────────────
run_build_kg() {
  python - <<'PYEOF'
import sys, os
from pathlib import Path
demo_dir = Path(os.environ["DEMO_DIR"])
sys.path.insert(0, str(demo_dir))

from kg_builder import KGBuilder

json_dir = Path(os.environ.get("JSON_DIR", ""))
builder = KGBuilder()
if not builder.verify_connection():
    sys.exit(1)
print("[kg] === Phase 2: Build KG ===")
builder.setup_schema()
builder.load_json_dir(json_dir)
builder.print_stats()
builder.validate()
builder.close()
PYEOF
}

# ── Phase 3: Embeddings + vector index ──────────────────────────────────────
run_embed() {
  python - <<'PYEOF'
import sys, os
from pathlib import Path
demo_dir = Path(os.environ["DEMO_DIR"])
sys.path.insert(0, str(demo_dir))

from kg_builder import Embedder

print("[kg] === Phase 3: Embeddings ===")
embedder = Embedder()
embedder.create_vector_index()
# REEMBED_ALL=1 → drop old embeddings and re-embed EVERYTHING. Required whenever the
# passage formula changes (e.g. adding section_title) so the vector space stays
# consistent; otherwise embed_all_chunks() only touches chunks whose embedding IS NULL
# (i.e. none).
if os.getenv("REEMBED_ALL", "").lower() in ("1", "true", "yes"):
    embedder.clear_embeddings()
embedder.embed_all_chunks()
embedder.close()
PYEOF
}

export DEMO_DIR="$DEMO_DIR"
export JSON_DIR="$JSON_DIR"

# ── Main pipeline ───────────────────────────────────────────────────────────
case "$MODE" in
  full)
    run_clean
    restart_neo4j
    run_build_kg
    run_embed
    ;;
  kg-only)
    run_clean
    restart_neo4j
    run_build_kg
    ;;
  embed-only)
    run_embed
    ;;
  clean-only)
    run_clean
    restart_neo4j
    ;;
  *)
    err "Unknown mode: $MODE. Use: full | kg-only | embed-only | clean-only"
    ;;
esac

ok "Rebuild complete."
