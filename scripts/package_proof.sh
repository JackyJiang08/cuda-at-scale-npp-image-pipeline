#!/usr/bin/env bash
#
# Collects the run logs, the contact sheets, and a sample of the edge maps
# into results/proof_of_execution.tar.gz, which is the file to upload to the
# assignment's proof-of-execution field.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

RESULTS_DIR="results"
STAGING_DIR="$RESULTS_DIR/.staging/proof_of_execution"
ARCHIVE="$RESULTS_DIR/proof_of_execution.tar.gz"
SAMPLE_COUNT="${SAMPLE_COUNT:-24}"

if [ ! -d "$RESULTS_DIR/logs" ]; then
  echo "no results/logs directory; run ./run.sh first" >&2
  exit 1
fi

rm -rf "$RESULTS_DIR/.staging"
mkdir -p "$STAGING_DIR/logs" "$STAGING_DIR/edge_maps" "$STAGING_DIR/inputs"

cp -R "$RESULTS_DIR/logs/." "$STAGING_DIR/logs/"
for extra in contact_sheet.png pairs samples samples_fixed; do
  if [ -e "$RESULTS_DIR/$extra" ]; then
    cp -R "$RESULTS_DIR/$extra" "$STAGING_DIR/"
  fi
done

# Include a sample of the edge maps together with the matching inputs so the
# before/after relationship is visible without downloading the whole dataset.
count=0
for edge_path in $(find data/output -name '*_edges.png' | sort); do
  [ "$count" -ge "$SAMPLE_COUNT" ] && break
  stem="$(basename "$edge_path" _edges.png)"
  cp "$edge_path" "$STAGING_DIR/edge_maps/"
  for candidate in data/input/"$stem".png data/input/"$stem".jpg; do
    if [ -f "$candidate" ]; then
      cp "$candidate" "$STAGING_DIR/inputs/"
      break
    fi
  done
  count=$((count + 1))
done

cat > "$STAGING_DIR/README.txt" <<EOF
Proof of execution for the CUDA NPP batch edge-detection pipeline.

logs/            build, host tests, environment (nvidia-smi, nvcc), the full
                 run over the whole dataset, a stream-scaling sweep, and the
                 error-handling checks.
inputs/          a sample of the original images.
edge_maps/       the corresponding GPU-produced edge maps (<name>_edges.png).
pairs/           the same images rendered side by side, input on the left.
samples/         a run with --save-stages --dilate, showing the grayscale,
                 blurred, and gradient-magnitude intermediates.
samples_fixed/   the same images with a fixed threshold instead of Otsu.
contact_sheet.png  every processed image as an input/edge thumbnail pair.

Generated $(date -u '+%Y-%m-%dT%H:%M:%SZ') by scripts/package_proof.sh.
EOF

tar -czf "$ARCHIVE" -C "$RESULTS_DIR/.staging" proof_of_execution
rm -rf "$RESULTS_DIR/.staging"

echo "wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
