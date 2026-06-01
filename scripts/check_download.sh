#!/usr/bin/env bash
# Quick progress check on the in-flight TRM checkpoint download.
# Prints percent complete, bytes, and a rough ETA.

set -u
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/experiments/trm_arc_v2_public/step_723914"
EXPECTED=2467988810

if [[ ! -f "$DEST" ]]; then
  echo "no partial file at: $DEST"
  exit 1
fi

ACTUAL="$(stat -f '%z' "$DEST" 2>/dev/null || stat -c '%s' "$DEST")"
PCT=$(awk "BEGIN {printf \"%.2f\", 100 * $ACTUAL / $EXPECTED}")
GB=$(awk "BEGIN {printf \"%.3f\", $ACTUAL / 1e9}")

echo "downloaded : ${GB} GB / 2.468 GB  (${PCT}%)"
if [[ "$ACTUAL" -ge "$EXPECTED" ]]; then
  echo "status     : complete (verify sha with: bash scripts/fetch_trm_checkpoint.sh)"
else
  echo "status     : in progress"
fi
