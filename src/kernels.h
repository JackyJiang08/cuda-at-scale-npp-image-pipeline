// Custom CUDA kernels that fill the gaps between the NPP primitives used by
// the pipeline. Declared separately from the pipeline so the device code
// lives in a single translation unit.

#ifndef IMGPIPE_SRC_KERNELS_H_
#define IMGPIPE_SRC_KERNELS_H_

#include <cuda_runtime.h>
#include <npp.h>

namespace imgpipe {

// Combines the horizontal and vertical Sobel responses into an 8-bit
// gradient magnitude image: dst = clamp(scale * sqrt(gx^2 + gy^2), 0, 255).
//
// All steps are in bytes. `gradient` must have room for `width` x `height`
// pixels. Returns the launch status.
cudaError_t LaunchGradientMagnitude(const Npp16s* gx, int gx_step,
                                    const Npp16s* gy, int gy_step,
                                    Npp8u* gradient, int gradient_step,
                                    int width, int height, float scale,
                                    cudaStream_t stream);

// Writes 255 where `gradient` exceeds `threshold` and 0 elsewhere, and adds
// the number of above-threshold pixels into `*edge_count`, which the caller
// must have zeroed. Returns the launch status.
cudaError_t LaunchBinarize(const Npp8u* gradient, int gradient_step,
                           Npp8u* binary, int binary_step, int width,
                           int height, int threshold, unsigned int* edge_count,
                           cudaStream_t stream);

}  // namespace imgpipe

#endif  // IMGPIPE_SRC_KERNELS_H_
