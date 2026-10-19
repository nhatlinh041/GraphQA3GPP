#!/usr/bin/env bash
# Copy the expert-review database (paper data, gitignored) to ~/bench_logs with a
# timestamp. Uses sqlite's online backup so a copy taken mid-write is consistent.
#   bash scripts/backup-review-db.sh            # one-off
#   crontab: 0 * * * * bash /path/to/GraphQA3GPP/scripts/backup-review-db.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="rag-engine/data/review.sqlite"
DST_DIR="${HOME}/bench_logs/review_db_backup"
mkdir -p "$DST_DIR"
[ -f "$SRC" ] || { echo "no $SRC"; exit 0; }
DST="$DST_DIR/review_$(date +%Y%m%d_%H%M%S).sqlite"
.venv/bin/python - "$SRC" "$DST" <<'PY'
import sqlite3, sys
src, dst = sqlite3.connect(sys.argv[1]), sqlite3.connect(sys.argv[2])
src.backup(dst); dst.close(); src.close()
PY
ls -t "$DST_DIR"/review_*.sqlite | tail -n +49 | xargs -r rm --   # keep the newest 48
echo "backup -> $DST"
