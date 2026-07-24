// Single-threaded host implementation of the same edge-detection pipeline.
//
// This exists for two reasons. It is a correctness oracle: the GPU path
// chains six NPP entry points and two hand-written kernels, and without an
// independent implementation there is nothing to check the edge maps
// against. It is also the performance baseline the GPU numbers are quoted
// against, run through the same worker pool so that the comparison is
// against a busy multi-core host rather than a single idle core.
//
// The stages mirror the device pipeline exactly: BT.601 luma, a separable
// binomial blur with replicated borders, 3x3 Sobel derivatives, a scaled
// gradient magnitude, a 256-bin histogram, an Otsu threshold, binarization,
// and an optional 3x3 dilation.

#ifndef INCLUDE_IMGPIPE_CPU_REFERENCE_H_
#define INCLUDE_IMGPIPE_CPU_REFERENCE_H_

#include <cstddef>
#include <string>
#include <vector>

#include "imgpipe/cli.h"
#include "imgpipe/image.h"
#include "imgpipe/pipeline_types.h"

namespace imgpipe {

// Runs the whole pipeline on the host under `options`, writing results to
// `*stages` and timings to `*timing`. Only `timing->compute_ms`,
// `timing->threshold`, and `timing->edge_fraction` are meaningful; there is
// no transfer to measure. Returns false and fills `*error` on bad input.
bool RunCpuPipeline(const Options& options, const Image& input,
                    StageImages* stages, FrameTiming* timing,
                    std::string* error);

// Individual stages, exposed so the tests can check them one at a time.

// Converts an interleaved RGB image to single-channel luma using the BT.601
// weights NPP applies in nppiRGBToGray_8u_C3C1R.
void ConvertRgbToGray(const Image& rgb, Image* gray);

// Blurs a single-channel image with the separable binomial kernel NPP uses
// for nppiFilterGauss: [1 2 1]/4 for `mask_size` 3, [1 4 6 4 1]/16 for 5.
// Borders replicate the edge pixel, matching NPP_BORDER_REPLICATE.
void FilterGaussian(const Image& gray, int mask_size, Image* blurred);

// Fills `*gx` and `*gy` with the 3x3 Sobel derivatives of `gray`, one entry
// per pixel in row-major order, with replicated borders.
void FilterSobel(const Image& gray, std::vector<int>* gx, std::vector<int>* gy);

// Writes clamp(scale * sqrt(gx^2 + gy^2), 0, 255) into `*gradient`.
void GradientMagnitude(const std::vector<int>& gx, const std::vector<int>& gy,
                       int width, int height, float scale, Image* gradient);

// Accumulates a 256-bin histogram of `gray` into `histogram`, which must have
// room for kHistogramBins entries and is overwritten.
void ComputeHistogram(const Image& gray, int* histogram);

// Writes 255 where `gradient` exceeds `threshold` and 0 elsewhere, and
// returns the number of above-threshold pixels through `*edge_count`.
void Binarize(const Image& gradient, int threshold, Image* binary,
              size_t* edge_count);

// 3x3 maximum filter with replicated borders, matching nppiDilate3x3Border.
void Dilate3x3(const Image& binary, Image* dilated);

// How closely two single-channel images of the same size agree.
struct AgreementStats {
  size_t pixels = 0;
  size_t matching = 0;      // Pixels that are bit-identical.
  double max_difference = 0.0;
  double mean_difference = 0.0;

  // Share of pixels that agree exactly, in [0, 1]. Returns 0 for empty input.
  double MatchFraction() const;
};

// Compares two single-channel images. Returns an all-zero result when the
// dimensions differ.
AgreementStats CompareImages(const Image& left, const Image& right);

}  // namespace imgpipe

#endif  // INCLUDE_IMGPIPE_CPU_REFERENCE_H_
