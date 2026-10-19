#!/usr/bin/env bash
# Start Neo4j + check Ollama before launching app services.
set -e

# Load demo/.env if present (script runs from demo/)
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

CONTAINER="neo4j-server"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"
# Accept LOCAL_LLM_URL (old project format) by stripping the /api/* path
if [ -n "$LOCAL_LLM_URL" ] && [ -z "$OLLAMA_URL" ]; then
  OLLAMA_URL="$(echo "$LOCAL_LLM_URL" | sed -E 's|^(https?://[^/]+).*|\1|')"
fi
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# ── helpers ──────────────────────────────────────────────────────────────────
ok()   { echo -e "\033[32m✓\033[0m $*"; }
info() { echo -e "\033[36mℹ\033[0m $*"; }
warn() { echo -e "\033[33m⚠\033[0m $*"; }
err()  { echo -e "\033[31m✗\033[0m $*"; exit 1; }

# ── docker guard ─────────────────────────────────────────────────────────────
# `sudo -n` (non-interactive) is required, not plain `sudo`: this script runs from
# `npm run dev`, which has no terminal, so a plain `sudo` blocks forever waiting for a
# password it can never be given. Observed 2026-09-01 — the log stopped at
# "Checking Neo4j..." and never continued, leaving rag-engine and the frontend unstarted
# while Neo4j sat there stopped. With -n, a missing passwordless-sudo fails immediately
# and falls through to the error below, which says what to do.
if docker info &>/dev/null; then
  DOCKER="docker"
elif sudo -n docker info &>/dev/null 2>&1; then
  DOCKER="sudo docker"
else
  err "Docker is not running (or needs a password for sudo). Start Docker first."
fi

# ── Neo4j: check if already reachable on the port ────────────────────────────
info "Checking Neo4j..."

if curl -sf --max-time 3 http://localhost:7474 > /dev/null 2>&1; then
  ok "Neo4j already reachable on port 7474 — skipping container start."
else
  # Port free — safe to start/create container
  RUNNING=$($DOCKER ps --filter "name=^${CONTAINER}$" --format "{{.Names}}" 2>/dev/null)

  if [ -n "$RUNNING" ]; then
    ok "Neo4j container already running."
  else
    EXISTS=$($DOCKER ps -a --filter "name=^${CONTAINER}$" --format "{{.Names}}" 2>/dev/null)

    if [ -n "$EXISTS" ]; then
      info "Starting existing container..."
      $DOCKER start "$CONTAINER" > /dev/null || err "Failed to start $CONTAINER. Check: docker logs $CONTAINER"
    else
      info "Creating new Neo4j container..."
      $DOCKER run -d \
        --name "$CONTAINER" \
        -p 7474:7474 -p 7687:7687 \
        -e "NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASSWORD}" \
        neo4j:2026.05.0 > /dev/null || err "Failed to create Neo4j container."
    fi

    # Wait for Neo4j HTTP to be ready (up to 30s)
    info "Waiting for Neo4j to be ready..."
    for i in $(seq 1 30); do
      if curl -sf --max-time 3 http://localhost:7474 > /dev/null 2>&1; then
        ok "Neo4j is ready."
        break
      fi
      [ "$i" -eq 30 ] && err "Neo4j did not start in 30s. Check: $DOCKER logs $CONTAINER"
      sleep 1
    done
  fi
fi

# ── Ollama ────────────────────────────────────────────────────────────────────
info "Checking Ollama at $OLLAMA_URL..."
if curl -sf --max-time 5 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
  ok "Ollama is reachable."
else
  warn "Ollama not reachable at $OLLAMA_URL — LLM calls will fail."
  warn "Start Ollama: ollama serve"
fi

ok "Dependencies ready. Starting app services..."
