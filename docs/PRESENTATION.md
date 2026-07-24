# Presentation script

A slide-by-slide plan for the 5-10 minute recorded demonstration, with what
to say and roughly how long to spend. Total as written is about 8 minutes,
which leaves room to slow down without running past 10.

The numbers below are the measured ones from the committed run in
`results/logs/`, on a Colab A100-SXM4-40GB. The same values appear in the
README, so the two never drift apart.

Recording setup that works: screen share with the repository open in one
window and a terminal in another, so the demo is a live run rather than a
walk through static screenshots.

---

## Slide 1 - Title and the one-sentence version (0:00-0:30)

> "GPU batch edge detection with CUDA NPP. It is a command line tool that
> runs an eleven-stage edge detection pipeline over a whole directory of
> images entirely on the GPU: five stages come from NPP, two are CUDA kernels
> I wrote, and the default dataset is 103 images from the USC-SIPI database,
> 46.3 megapixels in one invocation."

Show the repository URL on the slide.

## Slide 2 - Why this problem (0:30-1:15)

The goal was to build something genuinely GPU-bound rather than a wrapper
around one library call, and to end up with an artifact I would actually
reuse.

Two things made edge detection the right choice:

- It decomposes into stages NPP already provides, so it exercises the
  library properly rather than through a single entry point.
- It has two gaps NPP does not fill. Combining the two Sobel responses into
  a magnitude, and thresholding against a value that is not known until
  runtime. Those gaps are exactly where a custom kernel belongs, so the
  project demonstrates library use and kernel authorship without either
  feeling bolted on.

## Slide 3 - The pipeline (1:15-2:30)

Walk the table of the eleven stages, calling out which are NPP and which are
mine. Keep it moving; the point is the shape, not every entry.

Emphasise three things:

1. Every pixel touch happens on the device. The host only decodes PNGs,
   picks a threshold, and writes files.
2. `GradientMagnitudeKernel` computes `clamp(scale * sqrt(gx^2 + gy^2))` per
   pixel over the pitched `Npp16s` Sobel planes.
3. `BinarizeKernel` thresholds *and* counts edge pixels in the same pass,
   reducing warp-shuffle, then shared memory, then one `atomicAdd` per block,
   so the edge coverage statistic costs no extra pass over the image.

Show `src/kernels.cu` on screen for this slide. It is short enough to read.

## Slide 4 - How it scales (2:30-3:30)

This is the part I would want a reviewer to look at hardest.

- Work is handed out to `--streams` worker threads through a shared atomic
  cursor, so a mixed-resolution dataset stays balanced.
- Each worker owns a CUDA stream, an `NppStreamContext` bound to that
  stream, its own pitched device buffers, and its own pinned staging memory.
- The `_Ctx` variants of the NPP entry points are used throughout. This is
  not cosmetic: `nppSetStream` is process-global, so calling NPP from several
  host threads without a per-thread context is a race. This was the single
  most important thing I learned about NPP.
- Buffers grow to fit the largest image seen and are then reused, so the
  dataset stops reallocating after the first few images.

Put the stream-scaling numbers on this slide. Measured on the A100 over all
103 images:

| streams | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| images/s | 43.50 | 85.40 | 163.97 | 237.75 |
| speedup | 1.00x | 1.96x | 3.77x | 5.47x |

Near linear to four workers, still 45% on the step to eight. Say what that
means: the mid-pipeline histogram sync is being covered by other workers
rather than stalling the device.

## Slide 5 - The interesting design tension (3:30-4:30)

Otsu's threshold depends on the whole histogram, so the binarization kernel
cannot launch until the host has seen it. That forces a
`cudaStreamSynchronize` in the middle of every image.

I kept the data-dependent version anyway, because a fixed threshold produces
visibly worse results across a dataset that mixes texture close-ups with
low-contrast aerial photographs. The stall is dealt with through concurrency
instead: while one worker waits on its histogram, the others keep the device
busy. `--threshold <value>` removes the sync entirely, so the cost of the
decision is measurable rather than assumed.

This is the slide where the trade-off gets named out loud: correctness of the
output was worth a mid-pipeline sync, given that the sync could be hidden.

## Slide 6 - Live demo (4:30-6:30)

Two minutes, run for real:

```bash
./bin/edge_pipeline --help
./bin/edge_pipeline --input data/input --output data/output --streams 4 --verbose
```

Let the per-image lines scroll, then stop on the summary block and read out
the images processed, megapixels, wall clock, and throughput.

Then show one before/after pair on screen (`results/contact_sheet_*.png` or a
single input beside its `_edges.png`), so there is visual evidence and not
only a log.

If a live run is risky in the recording environment, play back the Colab
notebook output instead - but say clearly that this is a recording of the
run, not a fresh one.

## Slide 7 - Verifying it is actually right (6:30-7:30)

An edge map looks plausible even when it is wrong, so plausibility is not
evidence. The project ships a second, independent implementation of the whole
pipeline on the host, and `--engine both` runs every image through both and
compares:

```bash
./bin/edge_pipeline --input data/input --engine both --streams 4
```

Results to report:

- **99.9998%** of 46.27 megapixels identical, worst single image 99.979%
- **103 of 103** images chose the same Otsu threshold on both engines, and
  8 of 8 on the 3x3 pass, which is bit-identical outright
- **23.2x** on summed per-image compute time against the multi-threaded host
  reference, 166.5 ms against 3855.3 ms

Then tell the debugging story, because it is the strongest thing in the
project, and it has two acts.

**Act one.** The first comparison run agreed on the Otsu threshold for **1
image out of 103**, and the host value was always higher, never lower.
One-sided error means a systematic difference, not noise. The saved stage
images localised it: grayscale was 100% bit-identical, the blur was not. A
least-squares fit of a free 5x5 kernel to one NPP input/output pair showed
the mask was not the separable binomial one I had assumed. Correcting it took
agreement to 82 of 103.

**Act two, which is the better half.** That same change added a 3x3
verification pass, because the 5x5 had been reverse engineered while the 3x3
had only been assumed. The next run showed the 3x3 agreeing on 1 of 8 images.
The first fix had found a good approximation, not the truth. Sweeping sigma
against the saved stages settled it: `nppiFilterGaussBorder` is a true sampled
Gaussian, sigma 1.0 at 3x3 and 1.4 at 5x5, and it truncates rather than rounds.

Put this table on the slide; it makes the point in one look.

| 5x5 kernel | rounding | identical to NPP |
| --- | --- | --- |
| binomial / 256 | round | 29.7% |
| integer approximation / 159 | truncate | 89.4% |
| Gaussian, sigma 1.4 | truncate | 99.99% |

Land the result: the two engines now agree on the threshold for **all 103
images**, including `misc_ruler_512`, which had been the worst case throughout
and whose gradient histogram has two Otsu optima within 1.7% of each other.
NPP's own blur fed through the host Sobel reproduces its gradient bit for bit.

If there is time, the one-ULP detail is worth thirty seconds: normalising the
Gaussian weights up front agrees on 103 images, dividing at the end agrees on
102, and the difference is invisible under rounding but decisive under
truncation. It is a good illustration of how little slack a bit-exact
comparison leaves.

The line to say out loud on this slide: the 3x3 check existed only because the
5x5 had needed work, and it is the only reason the first fix did not ship as
the final answer.

## Slide 8 - What I would do next (7:30-8:00)

Say these as concrete next steps, not as a wish list:

- Replace the mid-pipeline host round trip with a device-side Otsu, so the
  whole image stays asynchronous end to end. The 256-bin reduction is small
  enough to do in one block.
- Add a non-maximum suppression stage and hysteresis to make it a true Canny
  detector, both of which are custom-kernel work.
- Batch small images into a single launch. At 256x256 the kernels are launch
  bound rather than bandwidth bound, and the stream sweep shows it.
- Profile with Nsight Systems to confirm the overlap looks the way the design
  says it does, rather than inferring it from throughput alone.

Close on the repository URL again.

---

## Checklist before recording

- [ ] `./run.sh` has completed and `results/` holds the current logs
- [ ] Numbers above still match `results/logs/` if the run was repeated
- [ ] Terminal font large enough to read at the recording resolution
- [ ] Video is between 5 and 10 minutes
- [ ] Repository URL is visible on the first and last slide
