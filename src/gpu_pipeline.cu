#include "imgpipe/gpu_pipeline.h"

#include <cuda_runtime.h>
#include <npp.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "imgpipe/otsu.h"
#include "kernels.h"

namespace imgpipe {
namespace {

// Number of histogram levels; NPP produces nLevels - 1 bins.
constexpr int kHistogramLevels = kHistogramBins + 1;

// Stage images that can be downloaded in one batch when --save-stages is on.
constexpr int kMaxDownloadPlanes = 4;

bool CudaFailed(cudaError_t status, const char* what, std::string* error) {
  if (status == cudaSuccess) return false;
  *error = std::string(what) + " failed: " + cudaGetErrorString(status);
  return true;
}

bool NppFailed(NppStatus status, const char* what, std::string* error) {
  // Positive NppStatus values are warnings, not failures.
  if (status >= NPP_SUCCESS) return false;
  std::ostringstream out;
  out << what << " failed: NppStatus " << static_cast<int>(status);
  *error = out.str();
  return true;
}

NppiMaskSize GaussMaskSize(int size) {
  return size == 3 ? NPP_MASK_SIZE_3_X_3 : NPP_MASK_SIZE_5_X_5;
}

// Copies a tightly packed host plane out of a pinned staging buffer.
void CopyPlaneToImage(const unsigned char* source, int width, int height,
                      Image* image) {
  image->width = width;
  image->height = height;
  image->channels = 1;
  image->pixels.assign(source, source + static_cast<size_t>(width) * height);
}

}  // namespace

bool SelectDevice(int device, std::string* description, std::string* error) {
  int device_count = 0;
  if (CudaFailed(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount",
                 error)) {
    return false;
  }
  if (device_count <= 0) {
    *error = "no CUDA-capable device was found";
    return false;
  }
  if (device < 0 || device >= device_count) {
    std::ostringstream out;
    out << "requested device " << device << " but only " << device_count
        << " device(s) are present";
    *error = out.str();
    return false;
  }
  if (CudaFailed(cudaSetDevice(device), "cudaSetDevice", error)) return false;

  cudaDeviceProp properties;
  if (CudaFailed(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties", error)) {
    return false;
  }

  int runtime_version = 0;
  int driver_version = 0;
  cudaRuntimeGetVersion(&runtime_version);
  cudaDriverGetVersion(&driver_version);

  std::ostringstream out;
  out << "device " << device << ": " << properties.name << " (sm_"
      << properties.major << properties.minor << ", "
      << properties.multiProcessorCount << " SMs, "
      << (properties.totalGlobalMem >> 20) << " MiB global memory)\n"
      << "CUDA runtime " << runtime_version / 1000 << "."
      << (runtime_version % 1000) / 10 << ", driver "
      << driver_version / 1000 << "." << (driver_version % 1000) / 10
      << ", NPP " << NPP_VERSION_MAJOR << "." << NPP_VERSION_MINOR << "."
      << NPP_VERSION_PATCH;
  *description = out.str();
  return true;
}

// Holds every CUDA resource used by one worker: a stream, an NPP context
// bound to it, pitched device buffers sized to the largest image seen so far,
// pinned staging memory, and the events used for timing.
class GpuPipeline::Impl {
 public:
  ~Impl() { ReleaseAll(); }

  bool Initialize(const Options& options, std::string* error) {
    options_ = options;
    if (CudaFailed(cudaStreamCreate(&stream_), "cudaStreamCreate", error)) {
      return false;
    }
    if (NppFailed(nppGetStreamContext(&npp_context_), "nppGetStreamContext",
                  error)) {
      return false;
    }
    npp_context_.hStream = stream_;
    if (CudaFailed(cudaStreamGetFlags(stream_, &npp_context_.nStreamFlags),
                   "cudaStreamGetFlags", error)) {
      return false;
    }
    for (cudaEvent_t& event : events_) {
      if (CudaFailed(cudaEventCreate(&event), "cudaEventCreate", error)) {
        return false;
      }
    }
    if (CudaFailed(cudaMalloc(&device_histogram_,
                              kHistogramBins * sizeof(Npp32s)),
                   "cudaMalloc(histogram)", error)) {
      return false;
    }
    if (CudaFailed(cudaMalloc(&device_edge_count_, sizeof(unsigned int)),
                   "cudaMalloc(edge counter)", error)) {
      return false;
    }
    if (CudaFailed(cudaHostAlloc(&host_histogram_,
                                 kHistogramBins * sizeof(Npp32s),
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(histogram)", error)) {
      return false;
    }
    if (CudaFailed(cudaHostAlloc(&host_edge_count_, sizeof(unsigned int),
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(edge counter)", error)) {
      return false;
    }
    return true;
  }

  bool Process(const Image& input, StageImages* stages, FrameTiming* timing,
               std::string* error) {
    const int width = input.width;
    const int height = input.height;
    if (input.IsEmpty() || (input.channels != 1 && input.channels != 3)) {
      *error = "expected a non-empty 1- or 3-channel image";
      return false;
    }
    // The current device is per-thread, and Initialize ran on the main
    // thread. Without this, a worker on a non-default --device would launch
    // against device 0. It is a no-op once the thread is already bound.
    if (CudaFailed(cudaSetDevice(options_.device), "cudaSetDevice", error)) {
      return false;
    }
    if (!EnsureCapacity(width, height, error)) return false;

    const NppiSize roi = {width, height};
    const NppiPoint origin = {0, 0};

    if (!Upload(input, error)) return false;

    if (CudaFailed(cudaEventRecord(events_[2], stream_), "event record",
                   error)) {
      return false;
    }
    if (input.channels == 3 &&
        NppFailed(nppiRGBToGray_8u_C3C1R_Ctx(device_rgb_, rgb_step_,
                                             device_gray_, gray_step_, roi,
                                             npp_context_),
                  "nppiRGBToGray_8u_C3C1R_Ctx", error)) {
      return false;
    }
    if (NppFailed(nppiFilterGaussBorder_8u_C1R_Ctx(
                      device_gray_, gray_step_, roi, origin, device_blurred_,
                      blurred_step_, roi, GaussMaskSize(options_.gauss_size),
                      NPP_BORDER_REPLICATE, npp_context_),
                  "nppiFilterGaussBorder_8u_C1R_Ctx", error)) {
      return false;
    }
    if (NppFailed(nppiFilterSobelHorizBorder_8u16s_C1R_Ctx(
                      device_blurred_, blurred_step_, roi, origin,
                      device_gradient_x_, gradient_x_step_, roi,
                      NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, npp_context_),
                  "nppiFilterSobelHorizBorder_8u16s_C1R_Ctx", error)) {
      return false;
    }
    if (NppFailed(nppiFilterSobelVertBorder_8u16s_C1R_Ctx(
                      device_blurred_, blurred_step_, roi, origin,
                      device_gradient_y_, gradient_y_step_, roi,
                      NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, npp_context_),
                  "nppiFilterSobelVertBorder_8u16s_C1R_Ctx", error)) {
      return false;
    }
    if (CudaFailed(LaunchGradientMagnitude(
                       device_gradient_x_, gradient_x_step_, device_gradient_y_,
                       gradient_y_step_, device_gradient_, gradient_step_,
                       width, height, options_.sobel_scale, stream_),
                   "gradient magnitude kernel", error)) {
      return false;
    }
    if (CudaFailed(cudaMemsetAsync(device_edge_count_, 0,
                                   sizeof(unsigned int), stream_),
                   "cudaMemsetAsync(edge counter)", error)) {
      return false;
    }

    int threshold = options_.fixed_threshold;
    if (options_.threshold_mode == ThresholdMode::kOtsu) {
      if (!EnsureHistogramScratch(roi, error)) return false;
      if (NppFailed(nppiHistogramEven_8u_C1R_Ctx(
                        device_gradient_, gradient_step_, roi,
                        device_histogram_, kHistogramLevels, 0,
                        kHistogramBins, device_histogram_scratch_,
                        npp_context_),
                    "nppiHistogramEven_8u_C1R_Ctx", error)) {
        return false;
      }
      if (CudaFailed(cudaMemcpyAsync(host_histogram_, device_histogram_,
                                     kHistogramBins * sizeof(Npp32s),
                                     cudaMemcpyDeviceToHost, stream_),
                     "cudaMemcpyAsync(histogram)", error)) {
        return false;
      }
    }
    if (CudaFailed(cudaEventRecord(events_[3], stream_), "event record",
                   error)) {
      return false;
    }

    if (options_.threshold_mode == ThresholdMode::kOtsu) {
      // The threshold is data-dependent, so the host has to see the histogram
      // before the binarization kernel can be launched.
      if (CudaFailed(cudaStreamSynchronize(stream_),
                     "cudaStreamSynchronize(histogram)", error)) {
        return false;
      }
      threshold = ComputeOtsuThreshold(host_histogram_, kHistogramBins);
    }

    if (CudaFailed(cudaEventRecord(events_[4], stream_), "event record",
                   error)) {
      return false;
    }
    if (CudaFailed(LaunchBinarize(device_gradient_, gradient_step_,
                                  device_binary_, binary_step_, width, height,
                                  threshold, device_edge_count_, stream_),
                   "binarize kernel", error)) {
      return false;
    }
    const Npp8u* edges = device_binary_;
    int edges_step = binary_step_;
    if (options_.dilate) {
      if (NppFailed(nppiDilate3x3Border_8u_C1R_Ctx(
                        device_binary_, binary_step_, roi, origin,
                        device_dilated_, dilated_step_, roi,
                        NPP_BORDER_REPLICATE, npp_context_),
                    "nppiDilate3x3Border_8u_C1R_Ctx", error)) {
        return false;
      }
      edges = device_dilated_;
      edges_step = dilated_step_;
    }
    if (CudaFailed(cudaEventRecord(events_[5], stream_), "event record",
                   error)) {
      return false;
    }

    if (!Download(edges, edges_step, width, height, stages, error)) {
      return false;
    }

    timing->threshold = threshold;
    timing->edge_fraction =
        static_cast<double>(*host_edge_count_) /
        (static_cast<double>(width) * static_cast<double>(height));
    float upload_ms = 0.0f;
    float compute_a_ms = 0.0f;
    float compute_b_ms = 0.0f;
    float download_ms = 0.0f;
    cudaEventElapsedTime(&upload_ms, events_[0], events_[1]);
    cudaEventElapsedTime(&compute_a_ms, events_[2], events_[3]);
    cudaEventElapsedTime(&compute_b_ms, events_[4], events_[5]);
    cudaEventElapsedTime(&download_ms, events_[6], events_[7]);
    timing->upload_ms = upload_ms;
    timing->compute_ms = compute_a_ms + compute_b_ms;
    timing->download_ms = download_ms;
    return true;
  }

 private:
  bool Upload(const Image& input, std::string* error) {
    const size_t row_bytes = input.RowBytes();
    std::memcpy(host_upload_, input.pixels.data(), input.SizeBytes());
    if (CudaFailed(cudaEventRecord(events_[0], stream_), "event record",
                   error)) {
      return false;
    }
    Npp8u* destination = input.channels == 3 ? device_rgb_ : device_gray_;
    const int destination_step = input.channels == 3 ? rgb_step_ : gray_step_;
    if (CudaFailed(cudaMemcpy2DAsync(destination, destination_step,
                                     host_upload_, row_bytes, row_bytes,
                                     input.height, cudaMemcpyHostToDevice,
                                     stream_),
                   "cudaMemcpy2DAsync(upload)", error)) {
      return false;
    }
    return !CudaFailed(cudaEventRecord(events_[1], stream_), "event record",
                       error);
  }

  // Queues every requested output plane into the pinned download buffer and
  // waits once, so extra stages cost bandwidth but not extra round trips.
  bool Download(const Npp8u* edges, int edges_step, int width, int height,
                StageImages* stages, std::string* error) {
    const size_t plane_bytes = static_cast<size_t>(width) * height;
    const Npp8u* sources[kMaxDownloadPlanes] = {edges, nullptr, nullptr,
                                                nullptr};
    int steps[kMaxDownloadPlanes] = {edges_step, 0, 0, 0};
    int plane_count = 1;
    if (options_.save_stages) {
      sources[plane_count] = device_gray_;
      steps[plane_count++] = gray_step_;
      sources[plane_count] = device_blurred_;
      steps[plane_count++] = blurred_step_;
      sources[plane_count] = device_gradient_;
      steps[plane_count++] = gradient_step_;
    }

    if (CudaFailed(cudaEventRecord(events_[6], stream_), "event record",
                   error)) {
      return false;
    }
    for (int plane = 0; plane < plane_count; ++plane) {
      unsigned char* target = host_download_ + plane * plane_bytes;
      if (CudaFailed(cudaMemcpy2DAsync(target, width, sources[plane],
                                       steps[plane], width, height,
                                       cudaMemcpyDeviceToHost, stream_),
                     "cudaMemcpy2DAsync(download)", error)) {
        return false;
      }
    }
    if (CudaFailed(cudaMemcpyAsync(host_edge_count_, device_edge_count_,
                                   sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync(edge counter)", error)) {
      return false;
    }
    if (CudaFailed(cudaEventRecord(events_[7], stream_), "event record",
                   error)) {
      return false;
    }
    if (CudaFailed(cudaStreamSynchronize(stream_),
                   "cudaStreamSynchronize(download)", error)) {
      return false;
    }

    CopyPlaneToImage(host_download_, width, height, &stages->edges);
    if (options_.save_stages) {
      CopyPlaneToImage(host_download_ + plane_bytes, width, height,
                       &stages->gray);
      CopyPlaneToImage(host_download_ + 2 * plane_bytes, width, height,
                       &stages->blurred);
      CopyPlaneToImage(host_download_ + 3 * plane_bytes, width, height,
                       &stages->gradient);
    }
    return true;
  }

  // Grows the device and pinned buffers so they cover `width` x `height`.
  // Capacity only ever increases, so a mixed-resolution dataset settles after
  // the first few images instead of reallocating per frame.
  bool EnsureCapacity(int width, int height, std::string* error) {
    if (width <= capacity_width_ && height <= capacity_height_) return true;
    const int new_width = std::max(width, capacity_width_);
    const int new_height = std::max(height, capacity_height_);
    ReleaseBuffers();

    device_rgb_ = nppiMalloc_8u_C3(new_width, new_height, &rgb_step_);
    device_gray_ = nppiMalloc_8u_C1(new_width, new_height, &gray_step_);
    device_blurred_ = nppiMalloc_8u_C1(new_width, new_height, &blurred_step_);
    device_gradient_ = nppiMalloc_8u_C1(new_width, new_height, &gradient_step_);
    device_binary_ = nppiMalloc_8u_C1(new_width, new_height, &binary_step_);
    device_dilated_ = nppiMalloc_8u_C1(new_width, new_height, &dilated_step_);
    device_gradient_x_ =
        nppiMalloc_16s_C1(new_width, new_height, &gradient_x_step_);
    device_gradient_y_ =
        nppiMalloc_16s_C1(new_width, new_height, &gradient_y_step_);
    if (device_rgb_ == nullptr || device_gray_ == nullptr ||
        device_blurred_ == nullptr || device_gradient_ == nullptr ||
        device_binary_ == nullptr || device_dilated_ == nullptr ||
        device_gradient_x_ == nullptr || device_gradient_y_ == nullptr) {
      *error = "out of device memory while allocating image buffers";
      return false;
    }

    const size_t plane_bytes =
        static_cast<size_t>(new_width) * static_cast<size_t>(new_height);
    if (CudaFailed(cudaHostAlloc(&host_upload_, plane_bytes * 3,
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(upload)", error)) {
      return false;
    }
    if (CudaFailed(cudaHostAlloc(&host_download_,
                                 plane_bytes * kMaxDownloadPlanes,
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(download)", error)) {
      return false;
    }

    capacity_width_ = new_width;
    capacity_height_ = new_height;
    return true;
  }

  bool EnsureHistogramScratch(NppiSize roi, std::string* error) {
    // NPP changed this out parameter from int to size_t in CUDA 12.
#if NPP_VERSION_MAJOR >= 12
    size_t required = 0;
#else
    int required = 0;
#endif
    if (NppFailed(nppiHistogramEvenGetBufferSize_8u_C1R(roi, kHistogramLevels,
                                                        &required),
                  "nppiHistogramEvenGetBufferSize_8u_C1R", error)) {
      return false;
    }
    const size_t needed = static_cast<size_t>(required);
    if (needed <= histogram_scratch_bytes_ &&
        device_histogram_scratch_ != nullptr) {
      return true;
    }
    if (device_histogram_scratch_ != nullptr) {
      cudaFree(device_histogram_scratch_);
      device_histogram_scratch_ = nullptr;
    }
    if (CudaFailed(cudaMalloc(&device_histogram_scratch_, needed),
                   "cudaMalloc(histogram scratch)", error)) {
      return false;
    }
    histogram_scratch_bytes_ = needed;
    return true;
  }

  void ReleaseBuffers() {
    for (Npp8u** buffer : {&device_rgb_, &device_gray_, &device_blurred_,
                           &device_gradient_, &device_binary_,
                           &device_dilated_}) {
      if (*buffer != nullptr) {
        nppiFree(*buffer);
        *buffer = nullptr;
      }
    }
    for (Npp16s** buffer : {&device_gradient_x_, &device_gradient_y_}) {
      if (*buffer != nullptr) {
        nppiFree(*buffer);
        *buffer = nullptr;
      }
    }
    if (host_upload_ != nullptr) {
      cudaFreeHost(host_upload_);
      host_upload_ = nullptr;
    }
    if (host_download_ != nullptr) {
      cudaFreeHost(host_download_);
      host_download_ = nullptr;
    }
  }

  void ReleaseAll() {
    ReleaseBuffers();
    if (device_histogram_ != nullptr) cudaFree(device_histogram_);
    if (device_histogram_scratch_ != nullptr) {
      cudaFree(device_histogram_scratch_);
    }
    if (device_edge_count_ != nullptr) cudaFree(device_edge_count_);
    if (host_histogram_ != nullptr) cudaFreeHost(host_histogram_);
    if (host_edge_count_ != nullptr) cudaFreeHost(host_edge_count_);
    for (cudaEvent_t& event : events_) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  Options options_;
  cudaStream_t stream_ = nullptr;
  NppStreamContext npp_context_ = {};
  cudaEvent_t events_[8] = {};

  Npp8u* device_rgb_ = nullptr;
  Npp8u* device_gray_ = nullptr;
  Npp8u* device_blurred_ = nullptr;
  Npp8u* device_gradient_ = nullptr;
  Npp8u* device_binary_ = nullptr;
  Npp8u* device_dilated_ = nullptr;
  Npp16s* device_gradient_x_ = nullptr;
  Npp16s* device_gradient_y_ = nullptr;
  int rgb_step_ = 0;
  int gray_step_ = 0;
  int blurred_step_ = 0;
  int gradient_step_ = 0;
  int binary_step_ = 0;
  int dilated_step_ = 0;
  int gradient_x_step_ = 0;
  int gradient_y_step_ = 0;

  Npp32s* device_histogram_ = nullptr;
  Npp8u* device_histogram_scratch_ = nullptr;
  size_t histogram_scratch_bytes_ = 0;
  unsigned int* device_edge_count_ = nullptr;

  unsigned char* host_upload_ = nullptr;
  unsigned char* host_download_ = nullptr;
  Npp32s* host_histogram_ = nullptr;
  unsigned int* host_edge_count_ = nullptr;

  int capacity_width_ = 0;
  int capacity_height_ = 0;
};

GpuPipeline::GpuPipeline() : impl_(new Impl) {}
GpuPipeline::~GpuPipeline() = default;

bool GpuPipeline::Initialize(const Options& options, std::string* error) {
  return impl_->Initialize(options, error);
}

bool GpuPipeline::Process(const Image& input, StageImages* stages,
                          FrameTiming* timing, std::string* error) {
  return impl_->Process(input, stages, timing, error);
}

}  // namespace imgpipe
