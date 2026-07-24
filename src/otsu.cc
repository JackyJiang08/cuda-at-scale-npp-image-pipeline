#include "imgpipe/otsu.h"

namespace imgpipe {

int ComputeOtsuThreshold(const int* histogram, int bin_count) {
  if (histogram == nullptr || bin_count <= 1) return 0;

  double total = 0.0;
  double weighted_sum = 0.0;
  for (int bin = 0; bin < bin_count; ++bin) {
    const double count = static_cast<double>(histogram[bin]);
    total += count;
    weighted_sum += count * bin;
  }
  if (total <= 0.0) return 0;

  // Sweep every split point, tracking the class that falls at or below the
  // candidate threshold. Between-class variance can be evaluated in O(1) per
  // step from the running background weight and sum.
  double background_weight = 0.0;
  double background_sum = 0.0;
  double best_variance = -1.0;
  int best_threshold = 0;
  for (int bin = 0; bin < bin_count - 1; ++bin) {
    background_weight += static_cast<double>(histogram[bin]);
    if (background_weight <= 0.0) continue;
    const double foreground_weight = total - background_weight;
    if (foreground_weight <= 0.0) break;

    background_sum += static_cast<double>(histogram[bin]) * bin;
    const double background_mean = background_sum / background_weight;
    const double foreground_mean =
        (weighted_sum - background_sum) / foreground_weight;
    const double mean_gap = background_mean - foreground_mean;
    const double variance =
        background_weight * foreground_weight * mean_gap * mean_gap;
    if (variance > best_variance) {
      best_variance = variance;
      best_threshold = bin;
    }
  }
  return best_threshold;
}

}  // namespace imgpipe
