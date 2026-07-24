// Otsu's method for choosing a binarization threshold from a histogram.

#ifndef INCLUDE_IMGPIPE_OTSU_H_
#define INCLUDE_IMGPIPE_OTSU_H_

namespace imgpipe {

// Number of bins in the histograms produced by the pipeline.
constexpr int kHistogramBins = 256;

// Returns the intensity threshold in [0, bin_count - 1] that maximizes
// between-class variance for the distribution in `histogram`, following
// Otsu (1979). Pixels with an intensity strictly greater than the returned
// value belong to the foreground class.
//
// `histogram` must point to `bin_count` non-negative counts. Returns 0 for an
// empty or degenerate histogram.
int ComputeOtsuThreshold(const int* histogram, int bin_count);

}  // namespace imgpipe

#endif  // INCLUDE_IMGPIPE_OTSU_H_
