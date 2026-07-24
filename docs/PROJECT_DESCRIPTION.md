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
FreeImage; I chose not to take on a separately built image library at all, so
that the project would compile anywhere the CUDA toolkit does. Two vendored
public-domain `stb` headers decode and encode PNG, JPEG, BMP, and TGA behind
about fifteen lines of wrapper code. I also kept CUDA headers out of the public interface by putting all
device state behind a `GpuPipeline::Impl`, which meant the driver, the
argument parsing, and the I/O stayed ordinary C++ that compiles and runs
without a GPU.

The third decision came later, after the pipeline already worked. I had no way
to tell whether it worked *correctly*. An edge map is a picture of edges; it
looks convincing whether or not the Sobel signs are swapped, whether the blur
is normalised, and whether the border handling matches what NPP does. So I
wrote the whole pipeline a second time on the host, directly rather than
through any NPP call, and added `--engine both` to run every image through
each path and compare the results pixel by pixel. It doubles as the
performance baseline, and because it runs through the same worker pool the
comparison is against a busy multi-core host rather than one idle core.

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

**The two engines did not agree, and running that down was the most
interesting work in the project.** The first comparison run on the A100 came
back with the edge maps 96.3% identical, which sounds fine, and with the two
engines agreeing on the Otsu threshold for exactly **1 of 103 images**, which
does not. Worse, the host threshold was *always* higher than the GPU's, by one
to eight bins. A one-sided error is never noise; something was systematically
different.

The saved stage images made it tractable. Comparing them one stage at a time
showed the grayscale conversion was 100% bit-identical, so BT.601 was right,
and the divergence appeared at the blur: only 41.7% of blurred pixels matched,
with the host's mean brightness half a grey level above NPP's. That half-level
offset is the fingerprint of truncation versus rounding, and the fact that the
error survived into the gradient meant the kernel itself was wrong too.

Rather than keep guessing at masks, I solved for it. Taking one NPP
input/output pair, a least-squares fit of a general 5x5 kernel over the
interior pixels recovered a mask with a centre tap near 15/159 and off-centre
taps near 12/159 and 9/159 — which is not the binomial outer product I had
assumed, but the 159-divisor Gaussian approximation from the Canny literature,
and it is not even separable. Testing the candidates directly confirmed it:

| 5x5 kernel | rounding | pixels identical to NPP | max error |
| --- | --- | --- | --- |
| binomial / 256 | round | 41.7% | 16 |
| binomial / 256 | truncate | 58.9% | 16 |
| Canny / 159 | round | 50.5% | 2 |
| Canny / 159 | truncate | 95.4% | 1 |

With the blur corrected, threshold agreement went from 1 of 103 images to 82
of 103 exactly and 102 of 103 within a single bin. Better still, feeding NPP's
own blurred image into the host Sobel and magnitude stages reproduces NPP's
gradient image *bit for bit*, which isolates the entire remaining disagreement
to one stage and bounds it at one grey level.

The one image that still disagrees, `misc_ruler_512`, turned out not to be a
defect either. Its gradient histogram has two nearly equal Otsu optima — 98.3%
and 100.0% of the peak between-class variance — so a one-level difference
anywhere upstream is enough to flip the argmax from 25 to 32. That is a
property of a photograph of a ruler, which is mostly hard black-on-white
edges, and no amount of matching NPP's arithmetic would make it stable.

Two lessons stuck. The first is that the *direction* and *distribution* of a
disagreement carry more information than its size: 96.3% agreement looked
acceptable and was hiding a real bug, while the one-sided threshold error gave
it away immediately. The second is that when a library's behaviour is
undocumented, fitting a model to its input/output pairs beats reading forum
posts about what the mask probably is.

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
logic. Because argument parsing, Otsu, the host reference pipeline, and image
I/O have no CUDA in them, I could unit-test them anywhere and had over a
hundred assertions passing before the code ever reached a GPU. That is a
pattern worth keeping for CUDA projects generally.

The broader lesson is about what counts as evidence. "The output images look
like edges" is not evidence, and neither is "it runs without an error". A
reference implementation is more work than a visual check, but it converts a
vague impression into a percentage that either holds up or does not — and when
it does not, the shape of the failure points at the cause.

## Results

The pipeline instruments itself with CUDA events, so every run reports, per
image, the upload time, the kernel time, the download time, the Otsu threshold
it selected, and the fraction of pixels classified as edge. The run then
summarizes total megapixels, wall clock, images per second, and megapixels per
second. `run.sh` sweeps `--streams 1 2 4 8` over the identical dataset, so the
throughput gain from overlapping streams is a direct measurement in the logs
rather than an assertion.

The measured numbers from the lab run are in `results/logs/`, and
`results/contact_sheet.png` shows every input beside its edge map. Two things
are worth looking at in those artifacts: whether the selected threshold varies
across the dataset, which is what tells you the histogram stage is doing real
work rather than reproducing a constant; and how the throughput line moves as
the stream count rises, which is where the concurrency design either pays off
or does not.

I want to be straightforward about the development conditions, because they
shaped the engineering. I wrote and validated this on a machine with no CUDA
device, so until the lab run there was no execution evidence at all. That
constraint is the reason for the verification strategy described above —
type-checking against two real toolkit header sets and pushing every piece of
GPU-independent logic behind a unit-testable boundary. It is a decent
substitute for running the code, but it is not the same thing, and I would not
claim a result I had not measured.
