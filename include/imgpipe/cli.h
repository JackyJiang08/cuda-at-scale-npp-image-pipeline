// Command line parsing for the batch edge-detection pipeline.

#ifndef IMGPIPE_INCLUDE_IMGPIPE_CLI_H_
#define IMGPIPE_INCLUDE_IMGPIPE_CLI_H_

#include <string>

namespace imgpipe {

// How the binarization threshold for each image is chosen.
enum class ThresholdMode {
  kOtsu,   // Derived per image from a GPU histogram via Otsu's method.
  kFixed,  // The same user-supplied value for every image.
};

// Runtime configuration assembled from argv.
struct Options {
  std::string input_dir;
  std::string output_dir = "data/output";
  std::string log_path;
  int stream_count = 4;
  int gauss_size = 5;
  float sobel_scale = 0.25f;
  ThresholdMode threshold_mode = ThresholdMode::kOtsu;
  int fixed_threshold = 128;
  bool dilate = false;
  bool save_stages = false;
  int limit = 0;  // 0 means "process every input image".
  int device = 0;
  bool verbose = false;
  bool help = false;
};

// Parses `argc`/`argv` into `*options`. Returns false and fills `*error` when
// an argument is unknown, malformed, or out of range. When `--help` is
// present, parsing stops early with options->help set and returns true.
bool ParseCommandLine(int argc, const char* const* argv, Options* options,
                      std::string* error);

// Returns the multi-line usage message for `program_name`.
std::string UsageText(const std::string& program_name);

// Returns a single-line, human-readable summary of `options` for the log.
std::string DescribeOptions(const Options& options);

}  // namespace imgpipe

#endif  // IMGPIPE_INCLUDE_IMGPIPE_CLI_H_
