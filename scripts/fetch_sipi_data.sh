#!/usr/bin/env bash
#
# Rebuilds data/input from the original USC-SIPI image database.
#
# The repository already ships the converted images, so this script is only
# needed to reproduce that conversion from the upstream source or to pull in
# additional volumes. It downloads the requested SIPI volumes and converts the
# TIFF originals to PNG, which is what the pipeline reads.
#
# Usage: ./scripts/fetch_sipi_data.sh [volume ...]
#        default volumes: misc textures
#        other options:   aerials sequences
#
# Requires curl, unzip, and Python with Pillow.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

VOLUMES=("$@")
if [ ${#VOLUMES[@]} -eq 0 ]; then
  VOLUMES=(misc textures)
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
DEST_DIR="data/input"
mkdir -p "$DEST_DIR"

for volume in "${VOLUMES[@]}"; do
  url="https://sipi.usc.edu/database/${volume}.zip"
  echo "==> Downloading $url"
  curl -fSL --retry 3 -o "$WORK_DIR/$volume.zip" "$url"
  unzip -q -o "$WORK_DIR/$volume.zip" -d "$WORK_DIR/extracted"
done

echo "==> Converting TIFF originals to PNG in $DEST_DIR"
python3 - "$WORK_DIR/extracted" "$DEST_DIR" <<'PYTHON'
import os
import sys

from PIL import Image

source_root, dest_dir = sys.argv[1], sys.argv[2]
converted = 0
for root, _, names in os.walk(source_root):
    volume = os.path.basename(root)
    for name in sorted(names):
        if not name.lower().endswith((".tiff", ".tif")):
            continue
        stem = os.path.splitext(name)[0].replace(".", "_")
        target = os.path.join(dest_dir, "{}_{}.png".format(volume, stem))
        with Image.open(os.path.join(root, name)) as image:
            # The pipeline handles 8-bit grayscale and RGB; collapse anything
            # else (palette, RGBA, 16-bit) onto one of those two.
            if image.mode not in ("L", "RGB"):
                image = image.convert("RGB")
            image.save(target, optimize=True)
        converted += 1
print("converted {} images into {}".format(converted, dest_dir))
PYTHON

echo "==> $(find "$DEST_DIR" -type f | wc -l) images now in $DEST_DIR"
