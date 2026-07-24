#include "imgpipe/cpu_reference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "imgpipe/otsu.h"

namespace imgpipe {
namespace {

// The masks NPP applies for nppiFilterGauss, stored unnormalised. Both are
// symmetric under reflection in either axis, so only one quadrant is kept,
// indexed as [abs(dy)][abs(dx)] with the centre tap at [0][0].
//
//     3x3, divided by 16        5x5, divided by 159
//        1  2  1                   2  4  5  4  2
//        2  4  2                   4  9 12  9  4
//        1  2  1                   5 12 15 12  5
//                                  4  9 12  9  4
//                                  2  4  5  4  2
//
// The 5x5 is *not* the binomial outer product; it is the 159-divisor
// approximation familiar from the Canny literature, and unlike the binomial
// it is not separable. This was established by least-squares fitting a 5x5
// kernel to an (input, output) pair from a real NPP run and then confirming
// the candidate directly: the binomial reproduces 41.7% of NPP's output
// pixels exactly, this mask reproduces 95.4% with a maximum error of one
// grey level. See docs/PROJECT_DESCRIPTION.md.
const int kGauss3Quadrant[2][2] = {{4, 2}, {2, 1}};
const int kGauss3Divisor = 16;
const int kGauss5Quadrant[3][3] = {{15, 12, 5}, {12, 9, 4}, {5, 4, 2}};
const int kGauss5Divisor = 159;

// Clamps `value` into [0, limit - 1], which is how NPP_BORDER_REPLICATE
// behaves at the edges of the ROI.
inline int ClampIndex(int value, int limit) {
  if (value < 0) return 0;
  if (value >= limit) return limit - 1;
  return value;
}

inline unsigned char SaturateToByte(int value) {
  return static_cast<unsigned char>(std::min(255, std::max(0, value)));
}

// Reads a single-channel pixel with replicated borders.
inline int SampleReplicated(const Image& image, int x, int y) {
  const int cx = ClampIndex(x, image.width);
  const int cy = ClampIndex(y, image.height);
  return image.pixels[static_cast<size_t>(cy) * image.width + cx];
}

// The 3x3 Sobel masks. Which of the two is called "horizontal" differs
// between conventions, but the gradient magnitude is symmetric in the two
// responses, so the choice does not affect the output.
const int kSobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
const int kSobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

}  // namespace

double AgreementStats::MatchFraction() const {
  if (pixels == 0) return 0.0;
  return static_cast<double>(matching) / static_cast<double>(pixels);
}

void ConvertRgbToGray(const Image& rgb, Image* gray) {
  gray->Allocate(rgb.width, rgb.height, 1);
  const size_t pixel_count =
      static_cast<size_t>(rgb.width) * static_cast<size_t>(rgb.height);
  for (size_t i = 0; i < pixel_count; ++i) {
    const float r = static_cast<float>(rgb.pixels[3 * i + 0]);
    const float g = static_cast<float>(rgb.pixels[3 * i + 1]);
    const float b = static_cast<float>(rgb.pixels[3 * i + 2]);
    const float luma = 0.299f * r + 0.587f * g + 0.114f * b;
    gray->pixels[i] = SaturateToByte(static_cast<int>(luma + 0.5f));
  }
}

void FilterGaussian(const Image& gray, int mask_size, Image* blurred) {
  const bool small = mask_size == 3;
  const int radius = small ? 1 : 2;
  const int divisor = small ? kGauss3Divisor : kGauss5Divisor;

  blurred->Allocate(gray.width, gray.height, 1);
  for (int y = 0; y < gray.height; ++y) {
    for (int x = 0; x < gray.width; ++x) {
      int accumulator = 0;
      for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
          const int dy = ky < 0 ? -ky : ky;
          const int dx = kx < 0 ? -kx : kx;
          const int weight =
              small ? kGauss3Quadrant[dy][dx] : kGauss5Quadrant[dy][dx];
          accumulator += weight * SampleReplicated(gray, x + kx, y + ky);
        }
      }
      // NPP truncates here rather than rounding. Matching that matters more
      // than it looks: rounding biases every blurred pixel up by half a grey
      // level on average, which survives into the gradient magnitude and
      // pulls the Otsu threshold off by several bins.
      blurred->pixels[static_cast<size_t>(y) * gray.width + x] =
          SaturateToByte(accumulator / divisor);
    }
  }
}

void FilterSobel(const Image& gray, std::vector<int>* gx,
                 std::vector<int>* gy) {
  const size_t pixel_count =
      static_cast<size_t>(gray.width) * static_cast<size_t>(gray.height);
  gx->assign(pixel_count, 0);
  gy->assign(pixel_count, 0);

  for (int y = 0; y < gray.height; ++y) {
    for (int x = 0; x < gray.width; ++x) {
      int sum_x = 0;
      int sum_y = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          const int sample = SampleReplicated(gray, x + kx, y + ky);
          const int tap = (ky + 1) * 3 + (kx + 1);
          sum_x += kSobelX[tap] * sample;
          sum_y += kSobelY[tap] * sample;
        }
      }
      const size_t index = static_cast<size_t>(y) * gray.width + x;
      (*gx)[index] = sum_x;
      (*gy)[index] = sum_y;
    }
  }
}

void GradientMagnitude(const std::vector<int>& gx, const std::vector<int>& gy,
                       int width, int height, float scale, Image* gradient) {
  gradient->Allocate(width, height, 1);
  const size_t pixel_count =
      static_cast<size_t>(width) * static_cast<size_t>(height);
  for (size_t i = 0; i < pixel_count; ++i) {
    const float dx = static_cast<float>(gx[i]);
    const float dy = static_cast<float>(gy[i]);
    // Matches GradientMagnitudeKernel, including the rounding, so that a
    // disagreement points at the filters rather than at this step.
    const float magnitude = scale * std::sqrt(dx * dx + dy * dy);
    const float clamped = std::min(std::max(magnitude, 0.0f), 255.0f);
    gradient->pixels[i] = static_cast<unsigned char>(clamped + 0.5f);
  }
}

void ComputeHistogram(const Image& gray, int* histogram) {
  for (int bin = 0; bin < kHistogramBins; ++bin) histogram[bin] = 0;
  for (unsigned char value : gray.pixels) ++histogram[value];
}

void Binarize(const Image& gradient, int threshold, Image* binary,
              size_t* edge_count) {
  binary->Allocate(gradient.width, gradient.height, 1);
  size_t count = 0;
  for (size_t i = 0; i < gradient.pixels.size(); ++i) {
    const bool is_edge = gradient.pixels[i] > threshold;
    binary->pixels[i] = is_edge ? 255 : 0;
    if (is_edge) ++count;
  }
  *edge_count = count;
}

void Dilate3x3(const Image& binary, Image* dilated) {
  dilated->Allocate(binary.width, binary.height, 1);
  for (int y = 0; y < binary.height; ++y) {
    for (int x = 0; x < binary.width; ++x) {
      int best = 0;
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          best = std::max(best, SampleReplicated(binary, x + kx, y + ky));
        }
      }
      dilated->pixels[static_cast<size_t>(y) * binary.width + x] =
          static_cast<unsigned char>(best);
    }
  }
}

AgreementStats CompareImages(const Image& left, const Image& right) {
  AgreementStats stats;
  if (left.width != right.width || left.height != right.height ||
      left.channels != right.channels || left.pixels.size() !=
                                             right.pixels.size()) {
    return stats;
  }
  double difference_sum = 0.0;
  for (size_t i = 0; i < left.pixels.size(); ++i) {
    const int difference =
        std::abs(static_cast<int>(left.pixels[i]) - right.pixels[i]);
    if (difference == 0) ++stats.matching;
    difference_sum += difference;
    stats.max_difference = std::max(stats.max_difference,
                                    static_cast<double>(difference));
  }
  stats.pixels = left.pixels.size();
  if (stats.pixels > 0) {
    stats.mean_difference = difference_sum / static_cast<double>(stats.pixels);
  }
  return stats;
}

bool RunCpuPipeline(const Options& options, const Image& input,
                    StageImages* stages, FrameTiming* timing,
                    std::string* error) {
  if (input.IsEmpty() || (input.channels != 1 && input.channels != 3)) {
    *error = "expected a non-empty 1- or 3-channel image";
    return false;
  }

  const auto start = std::chrono::steady_clock::now();

  Image gray;
  if (input.channels == 3) {
    ConvertRgbToGray(input, &gray);
  } else {
    gray = input;
  }

  Image blurred;
  FilterGaussian(gray, options.gauss_size, &blurred);

  std::vector<int> gx;
  std::vector<int> gy;
  FilterSobel(blurred, &gx, &gy);

  Image gradient;
  GradientMagnitude(gx, gy, blurred.width, blurred.height, options.sobel_scale,
                    &gradient);

  int threshold = options.fixed_threshold;
  if (options.threshold_mode == ThresholdMode::kOtsu) {
    int histogram[kHistogramBins];
    ComputeHistogram(gradient, histogram);
    threshold = ComputeOtsuThreshold(histogram, kHistogramBins);
  }

  Image binary;
  size_t edge_count = 0;
  Binarize(gradient, threshold, &binary, &edge_count);

  if (options.dilate) {
    Image dilated;
    Dilate3x3(binary, &dilated);
    binary.pixels.swap(dilated.pixels);
  }

  const auto finish = std::chrono::steady_clock::now();

  timing->upload_ms = 0.0;
  timing->download_ms = 0.0;
  timing->compute_ms =
      std::chrono::duration<double, std::milli>(finish - start).count();
  timing->threshold = threshold;
  timing->edge_fraction =
      static_cast<double>(edge_count) /
      (static_cast<double>(input.width) * static_cast<double>(input.height));

  stages->edges.width = binary.width;
  stages->edges.height = binary.height;
  stages->edges.channels = 1;
  stages->edges.pixels.swap(binary.pixels);
  if (options.save_stages) {
    stages->gray.pixels.swap(gray.pixels);
    stages->gray.width = gray.width;
    stages->gray.height = gray.height;
    stages->gray.channels = 1;
    stages->blurred.pixels.swap(blurred.pixels);
    stages->blurred.width = blurred.width;
    stages->blurred.height = blurred.height;
    stages->blurred.channels = 1;
    stages->gradient.pixels.swap(gradient.pixels);
    stages->gradient.width = gradient.width;
    stages->gradient.height = gradient.height;
    stages->gradient.channels = 1;
  }
  return true;
}

}  // namespace imgpipe
