#include "imgpipe/cli.h"

#include <cstdlib>
#include <sstream>
#include <string>

namespace imgpipe {
namespace {

// Parses `text` as a base-10 integer. Returns false when the whole string is
// not consumed, so trailing junk such as "4x" is rejected rather than
// silently truncated.
bool ParseInt(const std::string& text, int* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  const long parsed = std::strtol(text.c_str(), &end, 10);
  if (end == nullptr || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseFloat(const std::string& text, float* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  const double parsed = std::strtod(text.c_str(), &end);
  if (end == nullptr || *end != '\0') return false;
  *value = static_cast<float>(parsed);
  return true;
}

// Fetches the value that follows a flag, advancing `*index` past it.
bool TakeValue(int argc, const char* const* argv, int* index,
               const std::string& flag, std::string* value,
               std::string* error) {
  if (*index + 1 >= argc) {
    *error = flag + " requires a value";
    return false;
  }
  ++(*index);
  *value = argv[*index];
  return true;
}

}  // namespace

std::string UsageText(const std::string& program_name) {
  std::ostringstream out;
  out << "GPU batch edge detection with CUDA NPP.\n\n"
      << "Usage: " << program_name << " --input <dir> [options]\n\n"
      << "Required:\n"
      << "  --input <dir>          Directory of input images (png, jpg, bmp,"
         " tga).\n\n"
      << "Options:\n"
      << "  --output <dir>         Output directory (default: data/output).\n"
      << "  --streams <n>          Concurrent CUDA streams and worker"
         " threads,\n"
      << "                         1-32 (default: 4).\n"
      << "  --gauss-size <3|5>     Gaussian smoothing mask size (default:"
         " 5).\n"
      << "  --sobel-scale <f>      Scale applied to gradient magnitude before\n"
      << "                         clamping to 8-bit (default: 0.25).\n"
      << "  --threshold <v|auto>   Binarization threshold 0-255, or 'auto'"
         " for\n"
      << "                         per-image Otsu (default: auto).\n"
      << "  --dilate               Thicken edges with a 3x3 dilation.\n"
      << "  --save-stages          Also write gray, blurred, and gradient"
         " images.\n"
      << "  --limit <n>            Process at most n images (default: all).\n"
      << "  --device <n>           CUDA device index (default: 0).\n"
      << "  --log <file>           Append the run log to this file as well as\n"
      << "                         stdout.\n"
      << "  --verbose              Print one line per image.\n"
      << "  --help                 Show this message and exit.\n";
  return out.str();
}

bool ParseCommandLine(int argc, const char* const* argv, Options* options,
                      std::string* error) {
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    std::string value;

    if (flag == "--help" || flag == "-h") {
      options->help = true;
      return true;
    } else if (flag == "--input") {
      if (!TakeValue(argc, argv, &i, flag, &value, error)) return false;
      options->input_dir = value;
    } else if (flag == "--output") {
      if (!TakeValue(argc, argv, &i, flag, &value, error)) return false;
      options->output_dir = value;
    } else if (flag == "--log") {
      if (!TakeValue(argc, argv, &i, flag, &value, error)) return false;
      options->log_path = value;
    } else if (flag == "--streams") {
      if (!TakeValue(argc, argv, &i, flag, &value, error)) return false;
      if (!ParseInt(value, &options->stream_count) ||
          options->stream_count < 1 || options->stream_count > 32) {
        *error = "--streams expects an integer in [1, 32], got '" + value + "'";
        return false;
      }
    } else if (flag == "--gauss-size") {
      if (!TakeValue(argc, argv, &i, flag, &value, error)) return false;
      if (!ParseInt(value, &options->gauss_size) ||
          (options->gauss_size != 3 && options->gauss_size != 5)) {
        *error = "--gauss-size expects 3 or 5, got '" + value + "'";
        return false;
      }
    } else if (flag == "--sobel-scale") {
      if (!TakeValue(argc, argv, &i, flag, &value, error)) return false;
      if (!ParseFloat(value, &options->sobel_scale) ||
          options->sobel_scale <= 0.0f || options->sobel_scale > 16.0f) {
        *error = "--sobel-scale expects a value in (0, 16], got '" + value +
                 "'";
        return false;
      }
    } else if (flag == "--threshold") {
      if (!TakeValue(argc, argv, &i, flag, &value, error)) return false;
      if (value == "auto") {
        options->threshold_mode = ThresholdMode::kOtsu;
      } else if (ParseInt(value, &options->fixed_threshold) &&
                 options->fixed_threshold >= 0 &&
                 options->fixed_threshold <= 255) {
        options->threshold_mode = ThresholdMode::kFixed;
      } else {
        *error = "--threshold expects 'auto' or an integer in [0, 255], got '" +
                 value + "'";
        return false;
      }
    } else if (flag == "--limit") {
      if (!TakeValue(argc, argv, &i, flag, &value, error)) return false;
      if (!ParseInt(value, &options->limit) || options->limit < 0) {
        *error = "--limit expects a non-negative integer, got '" + value + "'";
        return false;
      }
    } else if (flag == "--device") {
      if (!TakeValue(argc, argv, &i, flag, &value, error)) return false;
      if (!ParseInt(value, &options->device) || options->device < 0) {
        *error = "--device expects a non-negative integer, got '" + value + "'";
        return false;
      }
    } else if (flag == "--dilate") {
      options->dilate = true;
    } else if (flag == "--save-stages") {
      options->save_stages = true;
    } else if (flag == "--verbose") {
      options->verbose = true;
    } else {
      *error = "unknown argument '" + flag + "'";
      return false;
    }
  }

  if (options->input_dir.empty()) {
    *error = "--input is required";
    return false;
  }
  return true;
}

std::string DescribeOptions(const Options& options) {
  std::ostringstream out;
  out << "input=" << options.input_dir << " output=" << options.output_dir
      << " streams=" << options.stream_count
      << " gauss=" << options.gauss_size << "x" << options.gauss_size
      << " sobel_scale=" << options.sobel_scale << " threshold=";
  if (options.threshold_mode == ThresholdMode::kOtsu) {
    out << "auto(otsu)";
  } else {
    out << options.fixed_threshold;
  }
  out << " dilate=" << (options.dilate ? "on" : "off")
      << " save_stages=" << (options.save_stages ? "on" : "off")
      << " device=" << options.device;
  if (options.limit > 0) out << " limit=" << options.limit;
  return out.str();
}

}  // namespace imgpipe
