# GPU Batch Edge Detection with CUDA NPP

A command-line tool that runs a multi-stage edge-detection pipeline on the GPU
across a whole directory of images. Every pixel operation happens on the
device: five stages come from the CUDA NPP library and two are custom CUDA
kernels written for this project. The host only decodes PNGs, picks a
threshold from a GPU-computed histogram, and writes results back out.

The bundled dataset is **103 images** (256x256 to 1024x1024, grayscale and
colour) from the USC-SIPI image database, so a default run processes
**46.3 megapixels** in one invocation.

- [What it does](#what-it-does)
- [Repository layout](#repository-layout)
- [Building](#building)
- [Running](#running)
- [Running on Colab](#running-on-colab)
- [Command line reference](#command-line-reference)
- [How it scales](#how-it-scales)
- [Correctness](#correctness)
- [Output](#output)
- [Results](#results)
- [Dataset](#dataset)
- [Testing](#testing)
- [Design notes](#design-notes)

## What it does

For each input image the pipeline performs:

| # | Stage | Implementation |
|---|-------|----------------|
| 1 | Upload, pinned and asynchronous | `cudaMemcpy2DAsync` |
| 2 | RGB to grayscale (colour inputs only) | `nppiRGBToGray_8u_C3C1R_Ctx` |
| 3 | Gaussian smoothing, 3x3 or 5x5 | `nppiFilterGaussBorder_8u_C1R_Ctx` |
| 4 | Horizontal Sobel derivative | `nppiFilterSobelHorizBorder_8u16s_C1R_Ctx` |
| 5 | Vertical Sobel derivative | `nppiFilterSobelVertBorder_8u16s_C1R_Ctx` |
| 6 | Gradient magnitude, `sqrt(gx²+gy²)` scaled to 8-bit | **custom kernel** `GradientMagnitudeKernel` |
| 7 | 256-bin histogram of the magnitude | `nppiHistogramEven_8u_C1R_Ctx` |
| 8 | Otsu threshold from that histogram | host, `ComputeOtsuThreshold` |
| 9 | Binarization plus an edge-pixel count | **custom kernel** `BinarizeKernel` |
| 10 | Optional 3x3 dilation | `nppiDilate3x3Border_8u_C1R_Ctx` |
| 11 | Download of the requested planes | `cudaMemcpy2DAsync` |

Stage 9 also reduces the edge-pixel count on the device (warp shuffle, then
shared memory, then one `atomicAdd` per block) so the reported edge coverage
costs no extra pass over the image.

## Repository layout

```
include/imgpipe/     public headers (CUDA-free, so the driver is plain C++)
src/
  main.cc            worker threads, logging, summary statistics
  cli.cc             argument parsing and validation
  image.cc           PNG/JPEG/BMP/TGA decode and encode, directory listing
  otsu.cc            Otsu threshold selection from a histogram
  cpu_reference.cc   the same pipeline on the host: oracle and baseline
  gpu_pipeline.cu    NPP stage sequence, stream and buffer management
  kernels.cu         the two custom CUDA kernels
tests/host_tests.cc  tests for everything that does not need a GPU
scripts/             data fetch, contact sheets, proof packaging
notebooks/           Colab notebook that builds and runs the whole thing
docs/                project description and presentation script
third_party/stb/     vendored public-domain image codecs
data/input/          103 USC-SIPI images
run.sh               end-to-end build, run, and evidence collection
```

## Building

Requires a CUDA toolkit (tested against the NPP headers shipped with CUDA
11.8 and 12.3) and a C++14 compiler.

```bash
make                       # builds bin/edge_pipeline
```

If the toolkit is not at `/usr/local/cuda`, or you want to target one
architecture instead of the default fat binary:

```bash
make CUDA_PATH=/opt/cuda GENCODE="-arch=sm_75"
```

CMake works too:

```bash
cmake -S . -B build && cmake --build build -j
```

## Running

The simplest run processes the bundled dataset with default settings:

```bash
./bin/edge_pipeline --input data/input --output data/output --verbose
```

`run.sh` does everything the assignment asks for in one go — build, host
tests, environment capture, a full-dataset run, a stage-dump run, a
fixed-threshold run, a stream-scaling sweep, error-handling checks, contact
sheets, and a packaged archive:

```bash
./run.sh
```

It leaves logs in `results/logs/`, images in `data/output/` and
`results/`, and a ready-to-upload `results/proof_of_execution.tar.gz`.

## Running on Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JackyJiang08/cuda-at-scale-npp-image-pipeline/blob/main/notebooks/run_on_colab.ipynb)

If you do not have an NVIDIA GPU to hand, `notebooks/run_on_colab.ipynb`
builds and runs the whole project on a free Colab T4 and downloads the
evidence archive at the end. Open it in Colab, set *Runtime -> Change runtime
type -> GPU*, and run the cells in order; it takes about five minutes and
needs no configuration, because the dataset travels with the repository.

## Command line reference

```
Required:
  --input <dir>          Directory of input images (png, jpg, bmp, tga).

Options:
  --output <dir>         Output directory (default: data/output).
  --engine <name>        gpu (NPP and custom kernels), cpu (host reference),
                         or both to run each image through the two and
                         report the speedup and pixel agreement
                         (default: gpu).
  --streams <n>          Concurrent CUDA streams and worker threads, 1-32
                         (default: 4).
  --gauss-size <3|5>     Gaussian smoothing mask size (default: 5).
  --sobel-scale <f>      Scale applied to gradient magnitude before clamping
                         to 8-bit (default: 0.25).
  --threshold <v|auto>   Binarization threshold 0-255, or 'auto' for
                         per-image Otsu (default: auto).
  --dilate               Thicken edges with a 3x3 dilation.
  --save-stages          Also write gray, blurred, and gradient images.
  --limit <n>            Process at most n images (default: all).
  --device <n>           CUDA device index (default: 0).
  --log <file>           Append the run log to this file as well as stdout.
  --verbose              Print one line per image.
  --help                 Show this message and exit.
```

Examples:

```bash
# Eight streams, dilated edges, intermediate stages saved
./bin/edge_pipeline --input data/input --output data/output \
                    --streams 8 --dilate --save-stages --verbose

# A fixed threshold and a lighter blur, on the first 20 images only
./bin/edge_pipeline --input data/input --threshold 40 \
                    --gauss-size 3 --limit 20 --log run.log
```

Exit status is 0 when every image succeeded, 1 on a runtime failure, and 2 on
a bad command line.

## How it scales

Work is handed out to `--streams` worker threads through a shared atomic
cursor. Each worker owns a CUDA stream, an `NppStreamContext` bound to that
stream, its own pitched device buffers, and its own pinned staging memory, so
one image's PNG decode overlaps another's kernels and a third's download.
The `_Ctx` variants of the NPP entry points are used throughout, which is what
makes concurrent NPP calls from several host threads safe.

Buffers grow to fit the largest image seen and are then reused, so a
mixed-resolution dataset stops reallocating after the first few images rather
than churning on every frame.

`run.sh` sweeps `--streams 1 2 4 8` over the same dataset and records
throughput for each. Measured on an A100-SXM4-40GB over all 103 images:

| streams | wall clock | images/s | megapixels/s | speedup |
| --- | --- | --- | --- | --- |
| 1 | 2.368 s | 43.50 | 19.54 | 1.00x |
| 2 | 1.206 s | 85.40 | 38.36 | 1.96x |
| 4 | 0.628 s | 163.97 | 73.66 | 3.77x |
| 8 | 0.433 s | 237.75 | 106.80 | **5.47x** |

Scaling stays close to linear to four workers and is still worth 45% on the
step to eight, which is the mid-pipeline histogram sync being hidden behind
other workers' kernels rather than stalling the device.

## Correctness

An edge map looks plausible whether or not it is right, so the project ships
a second implementation of the entire pipeline on the host
(`src/cpu_reference.cc`) and a mode that runs both and compares them:

```bash
./bin/edge_pipeline --input data/input --engine both --streams 4 --verbose
# or
make verify
```

Every image goes through the NPP-and-custom-kernel path and through the host
reference, the two edge maps are compared pixel by pixel, and the run ends
with a block like this:

```
=== gpu versus host reference ===
kernel time      : gpu 166.5 ms, cpu 3855.3 ms
speedup          : 23.2x on summed per-image compute time
edge map match   : 99.9998% of 46.27 megapixels identical
worst image      : textures_texmos3_s512 at 99.979%
otsu threshold   : 103 of 103 images chose the same threshold on both engines
```

The host reference is deliberately independent: it convolves directly rather
than calling anything from NPP. It is also run through the same worker pool,
so the speedup is measured against a busy multi-core host rather than a
single idle core.

Building the reference implementation is also how the project learned what
NPP actually computes, and it took two passes to get right.

The first comparison run on an A100 disagreed on the Otsu threshold for 102
of 103 images, always in the same direction. A one-sided error is systematic,
not noise. Comparing the saved stage images one at a time showed the
grayscale stage matching bit for bit and the divergence starting at the blur,
so a least-squares fit of a free 5x5 kernel to one NPP input/output pair
pinned down the mask. It was not the separable binomial kernel that seemed
obvious. That correction took threshold agreement to 82 of 103.

It was still wrong. The 3x3 verification pass added at the same time came
back agreeing on 1 of 8 images, which showed the first fix had found a better
approximation rather than the actual filter. Sweeping the standard deviation
against the saved stage images settled it: `nppiFilterGaussBorder` uses a
true sampled Gaussian, with sigma 1.0 for the 3x3 and 1.4 for the 5x5, and it
**truncates** where the obvious implementation rounds.

| 5x5 kernel | rounding | pixels identical to NPP |
| --- | --- | --- |
| binomial / 256 | round | 29.7% |
| binomial / 256 | truncate | 40.0% |
| integer approximation / 159 | truncate | 89.4% |
| **Gaussian, sigma 1.4** | **truncate** | **99.99%** |

With that in place the two engines pick the same Otsu threshold on **all 103
images** at 5x5 and all 8 at 3x3, and the edge maps agree on **99.9998%** of
46.27 megapixels, with the 3x3 pass fully bit-identical. Feeding NPP's own blurred image into the
host Sobel reproduces its gradient image bit for bit, so every stage after
the blur is exact and the residual disagreement is bounded at one grey level,
coming from fixed-point precision inside NPP that cannot be recovered from
outside the library.

One detail is worth recording because it is so easy to get backwards:
normalising the Gaussian weights up front matches NPP better than
accumulating unnormalised and dividing at the end, by exactly one image out
of 103. The two differ by a single ULP, which is invisible under rounding and
decisive under truncation.

`--engine cpu` runs the host reference on its own and is the one mode that
needs no CUDA device at all, which makes it a convenient way to check the
tool works before moving to a GPU machine.

## Output

For every input `name.png`:

- `name_edges.png` — the binary edge map (always written)
- `name_gray.png`, `name_blur.png`, `name_grad.png` — with `--save-stages`

The run log ends with a per-image table and a summary:

```
image                              size      ch  thr  edge%   up(ms)  gpu(ms)  down(ms)
misc_4_1_01                       256x256   3   23      9.50    0.052    0.184     0.061
...
=== summary ===
images processed : 103 of 103 (0 failed)
pixels processed : 46.27 megapixels
wall clock       : 0.639 s using 4 stream(s)
throughput       : 161.27 images/s, 72.44 megapixels/s
gpu time (sum)   : upload 5.8 ms, compute 171.3 ms, download 5.7 ms
gpu time (mean)  : 1.663 ms of kernel time per image
```

That block is from the committed run in [results/logs/04_full_run.log](results/logs/04_full_run.log).

## Results

Everything below is committed under [results/](results/) and was produced by
one `./run.sh` on a Colab A100-SXM4-40GB (sm_80, 108 SMs), CUDA runtime 12.8,
NPP 12.3.3.

| | |
| --- | --- |
| Images processed | 103 of 103, 0 failed |
| Data | 46.27 megapixels in one invocation |
| Wall clock, 4 streams | 0.639 s |
| Throughput, 4 streams | 161.27 images/s, 72.44 megapixels/s |
| Throughput, 8 streams | 237.75 images/s, 106.80 megapixels/s |
| Kernel time per image | 1.663 ms mean |
| Speedup over host reference | 23.2x on summed compute time |
| Edge maps identical to host | 99.9998% of 46.27 megapixels |
| Otsu threshold agreement | 103 of 103 images |
| Host tests | 108 of 108 passing |
| cpplint | no findings |

What the logs hold:

| file | what it shows |
| --- | --- |
| `00_build.log` | clean build from source, no warnings |
| `01_host_tests.log`, `01b_lint.log` | tests and the style check |
| `02_environment.log` | `nvidia-smi` and `nvcc --version` for the machine |
| `04_full_run.log` | the headline run, one line per image plus the summary |
| `05_stages_run.log` | intermediate stages and dilation |
| `06_fixed_threshold.log` | the non-Otsu path |
| `06b_gauss3_verify.log` | 3x3 mask, both engines compared |
| `07_gpu_vs_cpu.log` | full dataset, both engines compared |
| `08_stream_scaling.log` | the `--streams 1 2 4 8` sweep |
| `09_error_handling.log` | bad arguments rejected cleanly |
| `11_inventory.log` | file counts and sizes in and out |

`results/contact_sheet.png` shows every input beside its edge map, and
`results/pairs/` holds individual before/after pairs.

Regenerating `results/proof_of_execution.tar.gz`, the archive for the
assignment's upload field, is a matter of running `./run.sh`; it is not
committed because every file in it is already here loose.

## Dataset

`data/input/` holds 103 images converted to PNG from the
[USC-SIPI image database](https://sipi.usc.edu/database/), volumes *Miscellaneous*
(39 images) and *Textures* (64 images). Sizes are 256x256, 512x512, and
1024x1024; 89 are grayscale and 14 are colour, which exercises both the
one-channel and the RGB-to-gray paths.

To rebuild the directory from the upstream source, or to add the larger
*aerials* volume:

```bash
./scripts/fetch_sipi_data.sh                 # misc + textures
./scripts/fetch_sipi_data.sh aerials         # adds 38 large images
```

The pipeline reads any PNG, JPEG, BMP, or TGA directory, so pointing
`--input` at your own data works without changes.

## Testing

Argument parsing, Otsu, the host reference pipeline, and image/directory I/O
are covered by tests that need no GPU:

```bash
make test
```

The sources are checked against the Google C++ Style Guide with
[cpplint](https://github.com/cpplint/cpplint):

```bash
pip install cpplint
make lint      # reports nothing when the tree is clean
```

Two cpplint categories are suppressed and both are noted in the Makefile:
`legal/copyright`, because the licence lives in `LICENSE` rather than in a
per-file banner, and `build/include_subdir`, because `src/kernels.h` is a
private header included by name from its own directory.

## Design notes

**Why a host round trip for the threshold.** Otsu's threshold depends on the
whole histogram, so the binarization kernel cannot launch until the host has
seen it. That forces one `cudaStreamSynchronize` per image mid-pipeline. With
several streams in flight the other workers keep the device busy during that
stall, which is a large part of why the stream count matters. Passing
`--threshold <value>` removes the sync entirely and is measurably faster.

**Why a second implementation.** The GPU path chains six NPP entry points and
two hand-written kernels. Without an independent implementation there is
nothing to check the output against beyond eyeballing it, and eyeballing an
edge map does not distinguish a correct detector from one whose Sobel signs
are swapped. The host reference costs about 250 lines and turns "the images
look right" into a number.

**Why `stb` instead of FreeImage.** The only third-party dependency is two
vendored public-domain headers, so the project builds with nothing beyond the
CUDA toolkit.

**Portability.** `nppiHistogramEvenGetBufferSize_8u_C1R` returns its size
through an `int*` in CUDA 11 and a `size_t*` in CUDA 12; the code selects the
right type from `NPP_VERSION_MAJOR`. The build was checked against both
versions' headers.

## Licence

MIT — see [LICENSE](LICENSE). The vendored stb headers are public domain. The
USC-SIPI images are redistributed here for education and research; see
[data/README.md](data/README.md).
