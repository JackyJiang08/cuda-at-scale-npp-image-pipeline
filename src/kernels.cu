#include "kernels.h"

#include <cuda_runtime.h>

namespace imgpipe {
namespace {

// One warp per row segment works well for the image sizes in the dataset;
// 32x8 keeps coalesced loads while giving enough blocks to fill the device.
constexpr int kBlockWidth = 32;
constexpr int kBlockHeight = 8;

// Returns the byte-stepped address of row `y` in a pitched device image.
template <typename T>
__device__ inline const T* ConstRow(const T* base, int step_bytes, int y) {
  return reinterpret_cast<const T*>(
      reinterpret_cast<const unsigned char*>(base) +
      static_cast<size_t>(y) * static_cast<size_t>(step_bytes));
}

template <typename T>
__device__ inline T* MutableRow(T* base, int step_bytes, int y) {
  return reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(base) +
                              static_cast<size_t>(y) *
                                  static_cast<size_t>(step_bytes));
}

__global__ void GradientMagnitudeKernel(const Npp16s* gx, int gx_step,
                                        const Npp16s* gy, int gy_step,
                                        Npp8u* gradient, int gradient_step,
                                        int width, int height, float scale) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const float dx = static_cast<float>(ConstRow(gx, gx_step, y)[x]);
  const float dy = static_cast<float>(ConstRow(gy, gy_step, y)[x]);
  const float magnitude = scale * sqrtf(dx * dx + dy * dy);
  const float clamped = fminf(fmaxf(magnitude, 0.0f), 255.0f);
  MutableRow(gradient, gradient_step, y)[x] =
      static_cast<Npp8u>(clamped + 0.5f);
}

__global__ void BinarizeKernel(const Npp8u* gradient, int gradient_step,
                               Npp8u* binary, int binary_step, int width,
                               int height, int threshold,
                               unsigned int* edge_count) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int is_edge = 0;
  if (x < width && y < height) {
    const int value = ConstRow(gradient, gradient_step, y)[x];
    is_edge = value > threshold ? 1u : 0u;
    MutableRow(binary, binary_step, y)[x] =
        static_cast<Npp8u>(is_edge ? 255 : 0);
  }

  // Reduce within the warp, then across warps through shared memory, so each
  // block contributes a single atomic to the global counter.
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    is_edge += __shfl_down_sync(0xffffffffu, is_edge, offset);
  }

  __shared__ unsigned int warp_totals[kBlockWidth * kBlockHeight / 32];
  const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  const int lane = thread_id % warpSize;
  const int warp_id = thread_id / warpSize;
  const int warp_count = (blockDim.x * blockDim.y + warpSize - 1) / warpSize;
  if (lane == 0) warp_totals[warp_id] = is_edge;
  __syncthreads();

  if (thread_id == 0) {
    unsigned int block_total = 0;
    for (int i = 0; i < warp_count; ++i) block_total += warp_totals[i];
    if (block_total > 0) atomicAdd(edge_count, block_total);
  }
}

dim3 GridFor(int width, int height) {
  return dim3((width + kBlockWidth - 1) / kBlockWidth,
              (height + kBlockHeight - 1) / kBlockHeight);
}

}  // namespace

cudaError_t LaunchGradientMagnitude(const Npp16s* gx, int gx_step,
                                    const Npp16s* gy, int gy_step,
                                    Npp8u* gradient, int gradient_step,
                                    int width, int height, float scale,
                                    cudaStream_t stream) {
  if (width <= 0 || height <= 0) return cudaErrorInvalidValue;
  const dim3 block(kBlockWidth, kBlockHeight);
  GradientMagnitudeKernel<<<GridFor(width, height), block, 0, stream>>>(
      gx, gx_step, gy, gy_step, gradient, gradient_step, width, height, scale);
  return cudaGetLastError();
}

cudaError_t LaunchBinarize(const Npp8u* gradient, int gradient_step,
                           Npp8u* binary, int binary_step, int width,
                           int height, int threshold, unsigned int* edge_count,
                           cudaStream_t stream) {
  if (width <= 0 || height <= 0) return cudaErrorInvalidValue;
  const dim3 block(kBlockWidth, kBlockHeight);
  BinarizeKernel<<<GridFor(width, height), block, 0, stream>>>(
      gradient, gradient_step, binary, binary_step, width, height, threshold,
      edge_count);
  return cudaGetLastError();
}

}  // namespace imgpipe
