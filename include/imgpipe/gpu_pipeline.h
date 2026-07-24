// GPU edge-detection pipeline built on NPP plus two custom CUDA kernels.
//
// This header is deliberately free of CUDA headers so that the driver code
// stays ordinary host C++; all device state lives behind GpuPipeline::Impl.

#ifndef INCLUDE_IMGPIPE_GPU_PIPELINE_H_
#define INCLUDE_IMGPIPE_GPU_PIPELINE_H_

#include <memory>
#include <string>

#include "imgpipe/cli.h"
#include "imgpipe/image.h"
#include "imgpipe/pipeline_types.h"

namespace imgpipe {

// Selects `device` and writes a human-readable description of it to
// `*description`. Returns false and fills `*error` if the device is
// unavailable.
bool SelectDevice(int device, std::string* description, std::string* error);

// Owns one CUDA stream and one set of device buffers. Instances are not
// thread-safe; the driver creates one per worker thread so that uploads,
// kernels, and downloads from different images overlap.
class GpuPipeline {
 public:
  GpuPipeline();
  ~GpuPipeline();

  GpuPipeline(const GpuPipeline&) = delete;
  GpuPipeline& operator=(const GpuPipeline&) = delete;

  // Creates the stream and NPP context. Device buffers are allocated lazily
  // and grow to fit the largest image seen so far. Returns false and fills
  // `*error` on failure.
  bool Initialize(const Options& options, std::string* error);

  // Runs grayscale conversion, Gaussian smoothing, Sobel gradients, gradient
  // magnitude, histogram-driven thresholding, and optional dilation on
  // `input`, writing results to `*stages` and measurements to `*timing`.
  // Returns false and fills `*error` on failure.
  bool Process(const Image& input, StageImages* stages, FrameTiming* timing,
               std::string* error);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace imgpipe

#endif  // INCLUDE_IMGPIPE_GPU_PIPELINE_H_
