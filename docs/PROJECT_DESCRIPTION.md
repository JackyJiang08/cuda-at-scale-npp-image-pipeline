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
does not. Worse, the host threshold was *always* higher than the GPU's, never
lower. A one-sided error is systematic; something was genuinely different.

The saved stage images made it tractable. Comparing them one stage at a time
showed the grayscale conversion was 100% bit-identical, so BT.601 was right,
and the divergence appeared at the blur: only 41.7% of blurred pixels matched,
with the host's mean brightness half a grey level above NPP's. That half-level
offset is the fingerprint of truncation versus rounding, and the fact that the
error survived into the gradient meant the kernel itself was wrong too.

Rather than keep guessing at masks, I solved for one. Taking a single NPP
input/output pair, a least-squares fit of a general 5x5 kernel over the
interior pixels recovered tap ratios that were clearly not the binomial outer
product I had assumed. Switching to the integer approximation those ratios
resembled took agreement from 41.7% of pixels to 95.4% and threshold agreement
from 1 image to 82.

**And that fix was still wrong, which is the part I am gladdest about.** In the
same change I had added a 3x3 verification pass to `run.sh`, on the principle
that the 5x5 mask had been reverse engineered and the 3x3 had only been assumed.
The next run came back with the 3x3 path agreeing on the threshold for 1 of 8
images. If I had only checked the 5x5 I would have shipped a plausible-looking
95% and called it precision loss.

The 3x3 fit pointed at ratios matching a true Gaussian rather than any integer
mask, so I swept the standard deviation against the saved stage images for both
sizes. The optimum was unambiguous, and it is simply what the function says on
the label: NPP's Gaussian is a sampled Gaussian, sigma 1.0 for the 3x3 and 1.4
for the 5x5, truncated rather than rounded.

| 5x5 kernel | rounding | pixels identical to NPP |
| --- | --- | --- |
| binomial / 256 | round | 29.7% |
| binomial / 256 | truncate | 40.0% |
| integer approximation / 159 | truncate | 89.4% |
| Gaussian, sigma 1.4 | truncate | 99.99% |

The two engines now choose the same Otsu threshold on **all 103 images** at 5x5
and all 8 at 3x3, including `misc_ruler_512`, which had been the worst case
throughout. That image is a genuinely marginal one — its gradient histogram has
two Otsu optima within 1.7% of each other, so a single grey level anywhere
upstream flips the argmax — and getting it to agree was the clearest signal
that the blur was finally right rather than merely close.

There is a last detail I would not have believed before measuring it.
Normalising the Gaussian weights up front agrees with NPP on all 103 images;
accumulating unnormalised and dividing once at the end agrees on 102. The two
differ by one ULP. Under rounding that is invisible, but the filter truncates,
so a value landing at 199.9999999 instead of 200.0 becomes a different pixel.
The same ULP is why the unit test asserting "a constant image is unchanged by
the blur" started failing: that test encoded a rounding assumption the device
does not have, so I corrected the test to assert what a truncating filter
actually guarantees — that a flat field stays flat, edges included, within one
level — which still catches the normalisation and border bugs it was there to
catch.

Three lessons stuck. The *direction* of a disagreement carries more information
than its size: 96.3% agreement looked acceptable and was hiding a real bug,
while the one-sided threshold error gave it away immediately. When a library's
behaviour is undocumented, fitting a model to its input/output pairs beats
reasoning about what the mask probably is. And a verification pass is worth
most on the path you did *not* investigate — the 3x3 check existed only because
the 5x5 had needed work, and it is the reason the first fix did not ship as the
final answer.

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
it selected, and the fraction of pixels classified as edge. Everything below
came from one `./run.sh` on a Colab A100-SXM4-40GB and is committed under
`results/`.

The headline run processes all 103 images, 46.27 megapixels, in **0.639 s** of
wall clock at four streams: 161.27 images/s, 72.44 megapixels/s, averaging
1.663 ms of kernel time per image. Nothing failed.

The stream sweep is the measurement I care most about, because it is the one
that tests the design rather than the hardware:

| streams | wall clock | images/s | speedup |
| --- | --- | --- | --- |
| 1 | 2.368 s | 43.50 | 1.00x |
| 2 | 1.206 s | 85.40 | 1.96x |
| 4 | 0.628 s | 163.97 | 3.77x |
| 8 | 0.433 s | 237.75 | 5.47x |

Scaling is near linear to four workers and still gains 45% on the step to
eight. That is the mid-pipeline histogram synchronisation being covered by
other workers' kernels instead of stalling the device, which is exactly what
the per-worker stream and buffer design was for. It also says the remaining
limit is elsewhere: at 256x256 the kernels are launch bound rather than
bandwidth bound, which is where I would go next.

Against the host reference, run through the same worker pool so the comparison
is to a busy multi-core host rather than one idle core, the GPU is **23.2x**
faster on summed per-image compute time, 166.5 ms against 3855.3 ms.

The correctness numbers matter more to me than the speed. The two engines
agree on the Otsu threshold for **103 of 103 images** at 5x5 and 8 of 8 at
3x3, and the edge maps are identical on **99.9998%** of 46.27 megapixels, with
the 3x3 comparison bit-identical outright. The worst single image,
`textures_texmos3_s512`, still agrees on 99.979% of its pixels. Given that the
two implementations share no code below the Otsu function and reach the answer
by different routes, that is a much stronger statement than the images looking
right.

One number in the logs should not be read as a benchmark: the 3x3 verification
pass reports a 0.2x "speedup", because eight small images are nowhere near
enough to amortise CUDA context and NPP initialisation, so almost all of its
GPU column is one-off startup. That pass exists to check agreement; the
full-dataset run is the timing measurement.

I want to be straightforward about the development conditions, because they
shaped the engineering. I wrote this on a machine with no CUDA device, so
every GPU number here comes from running the project on a rented A100 through
the Colab notebook in `notebooks/`, and for most of the project's life there
was no execution evidence at all. That constraint is why so much effort went
into things that can be checked without a GPU: keeping CUDA headers out of the
public interface so the driver, argument parsing, and I/O stay ordinary C++;
type-checking against two real toolkit header sets; and, in the end, writing
the whole pipeline a second time on the host.

That last decision is the one I would repeat. It was meant as a substitute for
having a GPU, and it turned into the thing that found a real bug the GPU alone
would never have surfaced — twice over, since the first fix was also wrong and
only the verification pass caught it. Going from "the edge maps look like
edges" to "the two implementations agree on 103 of 103 thresholds and
99.9998% of 46 megapixels" is the difference between believing the code works
and knowing it.
