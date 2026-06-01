#!/usr/bin/env bash
# Fetch the ARC Prize verification TRM checkpoint for ARC-AGI-2 public eval.
#
#   huggingface.co/arcprize/trm_arc_prize_verification  (MIT, 2.47 GB)
#
# Idempotent: skips files that already exist and pass the size/sha checks;
# resumes partial downloads via `curl -C -`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/experiments/trm_arc_v2_public"
BASE_URL="https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v2_public"

# (path inside arc_v2_public, expected size in bytes, sha256 if known)
# sha256 is only checked for the LFS-backed checkpoint; small text files are
# size-only since they may legitimately change if upstream re-renders YAML etc.
FILES=(
  "step_723914|2467988810|8d7036b97e7ea38c7dd29d01216bfcfc4e212af3024d5233fe40dd3059e8f4a9"
  "all_config.yaml|1008|"
  "losses.py|3988|"
  "trm.py|13214|"
)

mkdir -p "$DEST"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

size_of() {
  if stat --version >/dev/null 2>&1; then
    stat -c '%s' "$1"           # GNU
  else
    stat -f '%z' "$1"            # BSD / macOS
  fi
}

for spec in "${FILES[@]}"; do
  IFS='|' read -r name expected_size expected_sha <<< "$spec"
  out="${DEST}/${name}"

  if [[ -f "$out" ]]; then
    actual_size="$(size_of "$out")"
    if [[ "$actual_size" == "$expected_size" ]]; then
      if [[ -z "$expected_sha" ]]; then
        echo "[skip] ${name} (size ok, no sha to check)"
        continue
      fi
      echo "[check] ${name} sha256..."
      actual_sha="$(sha256_of "$out")"
      if [[ "$actual_sha" == "$expected_sha" ]]; then
        echo "[skip] ${name} (size + sha ok)"
        continue
      fi
      echo "[warn] ${name} sha mismatch (got ${actual_sha}); re-downloading"
      rm -f "$out"
    fi
  fi

  echo "[fetch] ${name} (${expected_size} bytes)"
  curl --location --fail --retry 3 --retry-delay 5 \
       --continue-at - \
       --output "$out" \
       "${BASE_URL}/${name}"

  actual_size="$(size_of "$out")"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "ERROR: ${name} downloaded size ${actual_size}, expected ${expected_size}" >&2
    exit 1
  fi
  if [[ -n "$expected_sha" ]]; then
    actual_sha="$(sha256_of "$out")"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      echo "ERROR: ${name} sha256 ${actual_sha} != expected ${expected_sha}" >&2
      exit 1
    fi
  fi
done

echo
echo "OK. Checkpoint ready at: ${DEST}"
ls -lh "${DEST}"
