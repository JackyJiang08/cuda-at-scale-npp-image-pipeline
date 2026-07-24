#!/usr/bin/env bash
#
# End-to-end proof-of-execution driver.
#
# Builds the project, runs the host tests, processes the whole image set on
# the GPU under several configurations, sweeps the stream count to show how
# throughput scales, renders before/after contact sheets, and packages
# everything under results/.
#
# Usage: ./run.sh [input_dir]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

INPUT_DIR="${1:-data/input}"
OUTPUT_DIR="data/output"
RESULTS_DIR="results"
LOG_DIR="$RESULTS_DIR/logs"
BIN="bin/edge_pipeline"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$RESULTS_DIR/samples"

echo "==> Building"
make -j"$(nproc 2>/dev/null || echo 4)" 2>&1 | tee "$LOG_DIR/00_build.log"

echo "==> Host tests (no GPU required)"
make test 2>&1 | tee "$LOG_DIR/01_host_tests.log"

echo "==> Google C++ Style check"
if python3 -c "import cpplint" 2>/dev/null; then
  make lint 2>&1 | tee "$LOG_DIR/01b_lint.log"
  echo "cpplint reported no findings" | tee -a "$LOG_DIR/01b_lint.log"
else
  echo "cpplint not installed; skipping (pip install cpplint)" \
      | tee "$LOG_DIR/01b_lint.log"
fi

echo "==> Environment"
{
  echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "host: $(uname -a)"
  echo
  nvidia-smi || echo "nvidia-smi unavailable"
  echo
  "${CUDA_PATH:-/usr/local/cuda}/bin/nvcc" --version || true
} 2>&1 | tee "$LOG_DIR/02_environment.log"

echo "==> Usage message"
"$BIN" --help 2>&1 | tee "$LOG_DIR/03_help.log"

# The headline run: every image, Otsu thresholding, four concurrent streams.
echo "==> Full dataset, default settings"
"$BIN" --input "$INPUT_DIR" --output "$OUTPUT_DIR" --streams 4 --verbose \
       --log "$LOG_DIR/04_full_run.log"

# A second configuration exercising the optional stages so the intermediate
# images land in results/samples for inspection.
echo "==> Sample run with intermediate stages and dilation"
"$BIN" --input "$INPUT_DIR" --output "$RESULTS_DIR/samples" --streams 2 \
       --limit 8 --save-stages --dilate --verbose \
       --log "$LOG_DIR/05_stages_run.log"

# The 3x3 mask on its own. The 5x5 NPP Gaussian was reverse engineered from
# a real run; this pass gives the same evidence for the 3x3 path, both as
# saved stage images and as an agreement number.
echo "==> 3x3 Gaussian, both engines"
"$BIN" --input "$INPUT_DIR" --output "$RESULTS_DIR/samples_gauss3" \
       --streams 4 --gauss-size 3 --engine both --limit 8 --save-stages \
       --verbose --log "$LOG_DIR/06b_gauss3_verify.log"
grep -E "speedup|edge map match|otsu threshold" \
     "$LOG_DIR/06b_gauss3_verify.log" || true

# A fixed threshold run, to show the non-Otsu path works too.
echo "==> Fixed-threshold run"
"$BIN" --input "$INPUT_DIR" --output "$RESULTS_DIR/samples_fixed" \
       --streams 4 --limit 8 --threshold 40 \
       --log "$LOG_DIR/06_fixed_threshold.log"

# Correctness and baseline: every image is processed twice, once by the NPP
# and custom-kernel path and once by the host reference, and the two edge
# maps are compared pixel by pixel.
echo "==> GPU versus host reference, full dataset"
"$BIN" --input "$INPUT_DIR" --output "$OUTPUT_DIR" --streams 4 \
       --engine both --verbose --log "$LOG_DIR/07_gpu_vs_cpu.log"
grep -E "speedup|edge map match|otsu threshold|worst image" \
     "$LOG_DIR/07_gpu_vs_cpu.log" || true

echo "==> Stream scaling sweep"
: > "$LOG_DIR/08_stream_scaling.log"
for streams in 1 2 4 8; do
  echo "--- streams=$streams" | tee -a "$LOG_DIR/08_stream_scaling.log"
  "$BIN" --input "$INPUT_DIR" --output "$OUTPUT_DIR" --streams "$streams" \
         --log "$LOG_DIR/08_stream_scaling.log" >/dev/null
done
echo "throughput by stream count:"
grep -E "throughput|wall clock" "$LOG_DIR/08_stream_scaling.log" || true

echo "==> Error handling checks (these are expected to fail cleanly)"
{
  echo "--- missing --input"
  "$BIN" || echo "exit=$?"
  echo "--- bad stream count"
  "$BIN" --input "$INPUT_DIR" --streams 0 || echo "exit=$?"
  echo "--- nonexistent input directory"
  "$BIN" --input no/such/dir || echo "exit=$?"
} 2>&1 | tee "$LOG_DIR/09_error_handling.log"

echo "==> Contact sheets"
if python3 -c "import PIL" 2>/dev/null; then
  python3 scripts/make_contact_sheet.py \
      --input "$INPUT_DIR" --output "$OUTPUT_DIR" \
      --dest "$RESULTS_DIR" 2>&1 | tee "$LOG_DIR/10_contact_sheets.log"
else
  echo "Pillow not installed; skipping contact sheets" \
      | tee "$LOG_DIR/10_contact_sheets.log"
fi

echo "==> Inventory"
{
  echo "inputs:  $(find "$INPUT_DIR" -type f | wc -l) files"
  echo "outputs: $(find "$OUTPUT_DIR" -type f | wc -l) files"
  du -sh "$INPUT_DIR" "$OUTPUT_DIR" "$RESULTS_DIR"
} 2>&1 | tee "$LOG_DIR/11_inventory.log"

bash scripts/package_proof.sh

cat <<'EOF'

Done. To publish the evidence:

    git add -A results data/output
    git commit -m "Add proof of execution artifacts"
    git push

The archive results/proof_of_execution.tar.gz is what to upload to the
assignment's "proof of execution" field.
EOF
