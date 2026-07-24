# Presentation script

A slide-by-slide plan for the 5-10 minute recorded demonstration, with what
to say and roughly how long to spend. Total as written is about 8 minutes,
which leaves room to slow down without running past 10.

Numbers in `[brackets]` come from `results/logs/`; they are filled in from
the recorded run before presenting, and the same values appear in the README.

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

Put the stream-scaling numbers on this slide:
`[streams 1 / 2 / 4 / 8 throughput, from results/logs/08_stream_scaling.log]`

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

- `[edge map match %]` of `[compared megapixels]` megapixels identical
- `[n of 103]` images chose the same Otsu threshold on both engines
- `[speedup]x` on summed per-image compute time against the multi-threaded
  host reference

Be honest about the residual disagreement: NPP does not document the internal
precision of `nppiFilterGauss`, so a small number of pixels sit on the far
side of the threshold from where the host reference puts them. The
disagreement is concentrated at exactly the pixels whose gradient magnitude
lands within a step or two of the threshold, which is what you would expect
from rounding rather than from a logic error.

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
- [ ] Bracketed numbers above replaced with the measured values
- [ ] Terminal font large enough to read at the recording resolution
- [ ] Video is between 5 and 10 minutes
- [ ] Repository URL is visible on the first and last slide
