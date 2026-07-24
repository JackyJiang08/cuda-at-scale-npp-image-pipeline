#!/usr/bin/env python3
"""Render before/after evidence images from a pipeline run.

Produces two things under --dest:

  pairs/<name>_before_after.png   one input beside its edge map
  contact_sheet.png               a grid of every processed image

Only Pillow is required; the pipeline itself does not depend on Python.
"""

import argparse
import os
import sys

try:
    from PIL import Image
except ImportError:  # pragma: no cover - handled by the caller
    sys.exit("Pillow is required: pip install pillow")

EDGE_SUFFIX = "_edges.png"
THUMB_SIZE = 128
GRID_PADDING = 4


def find_pairs(input_dir, output_dir):
    """Matches each <name>_edges.png output back to the input it came from."""
    inputs = {}
    for name in os.listdir(input_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tga"):
            inputs[stem] = os.path.join(input_dir, name)

    pairs = []
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(EDGE_SUFFIX):
            continue
        stem = name[: -len(EDGE_SUFFIX)]
        if stem in inputs:
            pairs.append((stem, inputs[stem], os.path.join(output_dir, name)))
    return pairs


def write_side_by_side(stem, before_path, after_path, dest_dir):
    """Writes one input and its edge map into a single image."""
    before = Image.open(before_path).convert("L")
    after = Image.open(after_path).convert("L")
    if before.size != after.size:
        after = after.resize(before.size)

    width, height = before.size
    canvas = Image.new("L", (width * 2 + GRID_PADDING, height), color=128)
    canvas.paste(before, (0, 0))
    canvas.paste(after, (width + GRID_PADDING, 0))
    out_path = os.path.join(dest_dir, stem + "_before_after.png")
    canvas.save(out_path)
    return out_path


def write_contact_sheet(pairs, dest_path, columns=10):
    """Writes a grid of input/edge thumbnail pairs for the whole run."""
    if not pairs:
        return None
    cell_width = THUMB_SIZE * 2 + GRID_PADDING
    cell_height = THUMB_SIZE
    rows = (len(pairs) + columns - 1) // columns
    sheet = Image.new(
        "L",
        (columns * (cell_width + GRID_PADDING), rows * (cell_height + GRID_PADDING)),
        color=64,
    )
    for index, (_, before_path, after_path) in enumerate(pairs):
        before = Image.open(before_path).convert("L").resize(
            (THUMB_SIZE, THUMB_SIZE))
        after = Image.open(after_path).convert("L").resize(
            (THUMB_SIZE, THUMB_SIZE))
        column = index % columns
        row = index // columns
        x = column * (cell_width + GRID_PADDING)
        y = row * (cell_height + GRID_PADDING)
        sheet.paste(before, (x, y))
        sheet.paste(after, (x + THUMB_SIZE + GRID_PADDING, y))
    sheet.save(dest_path)
    return dest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input image directory")
    parser.add_argument("--output", required=True,
                        help="directory holding *_edges.png results")
    parser.add_argument("--dest", required=True,
                        help="where to write the contact sheets")
    parser.add_argument("--pairs", type=int, default=12,
                        help="number of full-size before/after images")
    args = parser.parse_args()

    pairs = find_pairs(args.input, args.output)
    if not pairs:
        sys.exit("no matching input/output pairs found; run the pipeline first")

    pair_dir = os.path.join(args.dest, "pairs")
    os.makedirs(pair_dir, exist_ok=True)

    # Sample evenly across the sorted list so the examples are not all from
    # one part of the dataset.
    step = max(1, len(pairs) // max(1, args.pairs))
    written = 0
    for stem, before_path, after_path in pairs[::step][: args.pairs]:
        write_side_by_side(stem, before_path, after_path, pair_dir)
        written += 1

    sheet_path = write_contact_sheet(pairs, os.path.join(args.dest,
                                                         "contact_sheet.png"))
    print("matched {} input/output pairs".format(len(pairs)))
    print("wrote {} before/after images to {}".format(written, pair_dir))
    print("wrote contact sheet to {}".format(sheet_path))


if __name__ == "__main__":
    main()
