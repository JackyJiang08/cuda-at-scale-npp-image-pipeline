// Types shared by the GPU and CPU implementations of the pipeline.
//
// Kept free of CUDA headers so that host-only translation units, including
// the CPU reference engine and the tests, can include it.

#ifndef INCLUDE_IMGPIPE_PIPELINE_TYPES_H_
#define INCLUDE_IMGPIPE_PIPELINE_TYPES_H_

#include "imgpipe/image.h"

namespace imgpipe {

// Intermediate results of one pass through the pipeline. `edges` is always
// produced; the other stages are only filled when Options::save_stages is set.
struct StageImages {
  Image gray;
  Image blurred;
  Image gradient;
  Image edges;
};

// Per-image measurements. The GPU engine fills these from CUDA events; the
// CPU engine reports all of its work as `compute_ms` because it never copies
// across a bus.
struct FrameTiming {
  double upload_ms = 0.0;
  double compute_ms = 0.0;
  double download_ms = 0.0;
  int threshold = 0;
  // Share of pixels classified as edge, in [0, 1], counted by the
  // binarization step and therefore measured before any dilation.
  double edge_fraction = 0.0;
};

}  // namespace imgpipe

#endif  // INCLUDE_IMGPIPE_PIPELINE_TYPES_H_
