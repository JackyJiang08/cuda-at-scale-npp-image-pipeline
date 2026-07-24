# Project Description

## What I built

A command-line tool that applies a multi-stage edge-detection pipeline to a
whole directory of images on the GPU. It ships with 103 images from the
USC-SIPI database (46.3 megapixels total, sizes from 256x256 to 1024x1024,
both grayscale and colour), and one invocation processes all of them.

Per image, the work is: upload, RGB-to-gray, Gaussian smoothing, horizontal
and vertical Sobel derivatives, gradient magnitude, a 256-bin histogram, an
Otsu threshold, binarization with an edge-pixel count, an optional dilation,
and download. Five of those stages are NPP calls, two are CUDA kernels I
wrote, and only the threshold arithmetic and file I/O run on the host.

## Thought process

I started from the requirement that the work be genuinely GPU-bound and at
scale, and worked backwards to a pipeline where every pixel touch happens on
the device. Edge detection was a good fit: it decomposes into stages that NPP
already provides, but it has two gaps NPP does not fill — combining the two
Sobel responses into a magnitude, and thresholding against a value that is not
known until runtime. Those gaps are exactly where custom kernels belong, so
the project ends up demonstrating both library use and kernel authorship
without either feeling bolted on.

Two decisions shaped the rest of it.

The first was to make the threshold data-dependent. A hard-coded threshold
would have kept everything asynchronous, but it produces poor results across a
dataset this varied — a texture image and a low-contrast aerial photo need very
different cut-offs. Computing a histogram on the device and running Otsu on
the host per image gives much better output, at the cost of one
`cudaStreamSynchronize` in the middle of every image. I kept the data-driven
version and dealt with the stall through concurrency instead, and left
`--threshold <value>` available so the difference is measurable.

The second was to keep dependencies near zero. The course template uses
FreeImage, which I found more trouble than it was worth to build. Two vendored
public-domain `stb` headers decode and encode PNG, JPEG, BMP, and TGA in about
fifteen lines of wrapper code, so the project needs nothing beyond the CUDA
toolkit. I also kept CUDA headers out of the public interface by putting all
device state behind a `GpuPipeline::Impl`, which meant the driver, the
argument parsing, and the I/O stayed ordinary C++ that compiles and runs
without a GPU.

## Issues encountered

**Concurrency with NPP.** My first sketch used the plain NPP entry points with
`nppSetStream`. That is process-global state, so it is unusable from several
host threads at once — one worker changing the stream underneath another is a
race. The fix was to switch every call to its `_Ctx` variant and give each
worker its own `NppStreamContext`, built from `nppGetStreamContext` and then
pointed at that worker's stream. That is what makes N workers safe, and it is
the single most important detail in the GPU code.

**A signature that changed between CUDA versions.**
`nppiHistogramEvenGetBufferSize_8u_C1R` returns its size through an `int*` in
CUDA 11 and a `size_t*` in CUDA 12. Since I could not be sure which toolkit the
lab would have, I checked the headers from both releases, confirmed the
difference, and selected the type with `#if NPP_VERSION_MAJOR >= 12`. I then
type-checked the whole pipeline against both header sets. That step caught
nothing else, but it turned "this should compile" into "this does compile."

**A test that was wrong, not the code.** I wrote an Otsu test asserting that a
distribution with 50,000 pixels spread across bins 0-49 and 150 pixels near bin
194 would produce a threshold between the two clusters. It failed. Working
through the arithmetic, splitting the wide low block near its middle really
does yield higher between-class variance than isolating the tiny bright tail —
Otsu was behaving correctly and my expectation was wrong. I replaced it with
two narrow modes of unequal mass, which is what a gradient-magnitude histogram
actually looks like. A good reminder that a failing test is a question, not a
verdict.

**Mixed resolutions.** The dataset mixes 256x256 and 1024x1024 images, and
reallocating pitched device memory per frame was wasteful. Buffers now only
ever grow, so allocation settles after the first few large images.

## Lessons learned

The interesting engineering was not in the kernels, which are short and
unremarkable, but in keeping the device fed. A single-stream version spends a
large share of its time in PNG decoding and in the mid-pipeline sync, with the
GPU idle. Giving each worker its own stream, buffers, and pinned staging
memory is what turns a correct program into a fast one.

I also got more value than expected from separating host logic from device
logic. Because argument parsing, Otsu, and image I/O have no CUDA in them, I
could unit-test them anywhere and had 65 assertions passing before the code
ever reached a GPU. That is a pattern worth keeping for CUDA projects
generally.

## Results

The edge maps are clean: object boundaries in the Miscellaneous volume come
out as connected contours, and the Textures volume produces the dense
high-frequency response you would expect. Per-image Otsu thresholds vary
widely across the dataset, which is the clearest evidence that the histogram
stage is doing real work — a fixed threshold visibly over- or under-detects on
the same images.

`run.sh` sweeps `--streams 1 2 4 8` over the identical dataset so the
throughput difference from overlapping streams is visible directly in the
logs, alongside per-image upload, compute, and download times measured with
CUDA events.
