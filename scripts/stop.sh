#!/usr/bin/env bash
# Stop all dev services started by `npm run dev` plus the Neo4j container.
# Idempotent: safe to run when nothing (or only some) services are up.
set -u

# ANSI colors matching existing ✓/⚠ style.
G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
ok()   { echo -e "${G}✓${N} $*"; }
warn() { echo -e "${Y}⚠${N} $*"; }
err()  { echo -e "${R}✗${N} $*"; }

# Configurable ports; defaults match package.json dev scripts.
RAG_PORT="${RAG_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-neo4j-server}"
GRACE_SECONDS="${GRACE_SECONDS:-5}"

# Resolve listening PIDs on a TCP port. Prefers lsof; falls back to ss.
pids_on_port() {
  local port="$1" pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null || true)"
  fi
  if [ -z "$pids" ] && command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnpH "sport = :$port" 2>/dev/null \
      | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
  fi
  echo "$pids"
}

# Resolve PIDs whose full argv matches an extended regex.
pids_by_pattern() {
  pgrep -f -- "$1" 2>/dev/null || true
}

# Send SIGTERM, wait up to GRACE_SECONDS, escalate to SIGKILL on survivors.
term_then_kill() {
  local label="$1"; shift
  local pids=("$@")
  [ "${#pids[@]}" -eq 0 ] && { warn "$label not running."; return 0; }

  local alive=()
  for p in "${pids[@]}"; do
    kill -0 "$p" 2>/dev/null && alive+=("$p")
  done
  [ "${#alive[@]}" -eq 0 ] && { warn "$label already exited."; return 0; }

  kill -TERM "${alive[@]}" 2>/dev/null || true

  local waited=0 limit_ticks=$(( GRACE_SECONDS * 4 ))
  while [ "$waited" -lt "$limit_ticks" ]; do
    local still=()
    for p in "${alive[@]}"; do
      kill -0 "$p" 2>/dev/null && still+=("$p")
    done
    [ "${#still[@]}" -eq 0 ] && { ok "$label stopped (PIDs: ${alive[*]})."; return 0; }
    alive=("${still[@]}")
    sleep 0.25
    waited=$(( waited + 1 ))
  done

  kill -KILL "${alive[@]}" 2>/dev/null || true
  ok "$label force-killed (PIDs: ${alive[*]})."
}

# Stop uvicorn reloader + worker for main:app (handles --reload zombie case).
stop_rag() {
  local pat='uvicorn .*main:app'
  local pids; pids="$(pids_by_pattern "$pat")"
  local port_pids; port_pids="$(pids_on_port "$RAG_PORT")"
  local all; all="$(printf '%s\n%s\n' "$pids" "$port_pids" | awk 'NF' | sort -u)"
  # shellcheck disable=SC2086
  term_then_kill "RAG (uvicorn :$RAG_PORT)" $all
}

# Stop Vite dev server by listener on FRONTEND_PORT.
stop_frontend() {
  local pids; pids="$(pids_on_port "$FRONTEND_PORT")"
  # shellcheck disable=SC2086
  term_then_kill "Frontend (Vite :$FRONTEND_PORT)" $pids
}

# Stop only our cloudflared instance, identified by config path.
stop_tunnel() {
  local pat='cloudflared.*cloudflared-demo\.yml'
  local pids; pids="$(pids_by_pattern "$pat")"
  # shellcheck disable=SC2086
  term_then_kill "Cloudflared tunnel" $pids
}

# Stop Neo4j container with sudo fallback and timeout guard.
stop_neo4j() {
  if ! command -v docker >/dev/null 2>&1; then
    warn "docker not installed; skipping Neo4j."
    return 0
  fi
  if timeout 10 docker stop "$NEO4J_CONTAINER" >/dev/null 2>&1 \
     || timeout 10 sudo docker stop "$NEO4J_CONTAINER" >/dev/null 2>&1; then
    ok "Neo4j ($NEO4J_CONTAINER) stopped."
  else
    warn "Could not stop $NEO4J_CONTAINER (may not be running)."
  fi
}

stop_rag
stop_frontend
stop_tunnel
stop_neo4j

ok "All stop steps completed."
