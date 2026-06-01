#!/usr/bin/env bash
# Resilient checkpoint fetch: loop curl-with-resume until the file reaches
# the expected size. Each curl invocation gets a fresh connection, which
# sidesteps mid-stream drops (curl exit 56) that --retry doesn't catch.
#
# Idempotent: if the file is already complete + sha256-correct, exits 0.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/experiments/trm_arc_v2_public"
URL_BASE="https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v2_public"

NAME=step_723914
EXPECTED_SIZE=2467988810
EXPECTED_SHA=8d7036b97e7ea38c7dd29d01216bfcfc4e212af3024d5233fe40dd3059e8f4a9

mkdir -p "$DEST_DIR"
out="${DEST_DIR}/${NAME}"

# Pull small companion files first; one shot is fine for these.
for small in all_config.yaml losses.py trm.py; do
  small_out="${DEST_DIR}/${small}"
  if [[ ! -s "$small_out" ]]; then
    echo "[fetch] ${small}"
    curl -L --fail --silent --output "$small_out" "${URL_BASE}/${small}"
  fi
done

size_of() {
  if stat --version >/dev/null 2>&1; then stat -c '%s' "$1"
  else stat -f '%z' "$1"; fi
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

attempt=0
max_attempts=50
while :; do
  current_size=0
  if [[ -f "$out" ]]; then
    current_size="$(size_of "$out")"
  fi
  pct=$(awk "BEGIN {printf \"%.2f\", 100 * $current_size / $EXPECTED_SIZE}")
  echo "[attempt $((attempt+1))/${max_attempts}] have ${current_size}/${EXPECTED_SIZE} bytes (${pct}%)"

  if [[ "$current_size" -ge "$EXPECTED_SIZE" ]]; then
    break
  fi

  # Each invocation: one curl, one fresh connection. Generous timeouts so a
  # slow stream doesn't get killed, but if we stop receiving entirely the
  # `--speed-time` + `--speed-limit` pair will reset us so the loop can
  # reconnect.
  curl --location \
       --continue-at - \
       --output "$out" \
       --connect-timeout 30 \
       --speed-time 30 \
       --speed-limit 1024 \
       --retry 0 \
       "${URL_BASE}/${NAME}"
  rc=$?
  echo "[attempt $((attempt+1))] curl exit=${rc}"

  attempt=$((attempt+1))
  if [[ "$attempt" -ge "$max_attempts" ]]; then
    echo "ERROR: gave up after ${max_attempts} attempts (size $(size_of "$out")/$EXPECTED_SIZE)" >&2
    exit 1
  fi
  # Small backoff between attempts to avoid hammering when something is wrong.
  sleep 3
done

actual_size="$(size_of "$out")"
if [[ "$actual_size" != "$EXPECTED_SIZE" ]]; then
  echo "ERROR: size ${actual_size} != ${EXPECTED_SIZE}" >&2
  exit 1
fi

echo "[verify] sha256..."
actual_sha="$(sha256_of "$out")"
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: sha mismatch (got ${actual_sha})" >&2
  exit 1
fi
echo "OK. ${out} (${actual_size} bytes, sha matches)"
ls -lh "$DEST_DIR"
