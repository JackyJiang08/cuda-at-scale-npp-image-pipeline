// Tests for the parts of the pipeline that do not need a GPU: argument
// parsing, Otsu thresholding, and image/directory I/O. These build and run
// with a plain C++ compiler (`make test`) so the logic can be checked
// without a CUDA toolkit.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "imgpipe/cli.h"
#include "imgpipe/image.h"
#include "imgpipe/otsu.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& what, int line) {
  ++g_checks;
  if (condition) return;
  ++g_failures;
  std::cerr << "FAIL (line " << line << "): " << what << "\n";
}

#define CHECK(condition) Check((condition), #condition, __LINE__)
#define CHECK_EQ(actual, expected)                                        \
  Check((actual) == (expected),                                           \
        std::string(#actual) + " == " + #expected + " (got " +            \
            std::to_string(actual) + ")",                                 \
        __LINE__)

void TestOtsu() {
  // Two well-separated modes: the split belongs between them.
  int histogram[imgpipe::kHistogramBins] = {0};
  histogram[10] = 500;
  histogram[200] = 500;
  const int threshold =
      imgpipe::ComputeOtsuThreshold(histogram, imgpipe::kHistogramBins);
  CHECK(threshold >= 10);
  CHECK(threshold < 200);

  // A single populated bin has no meaningful split; the result must still be
  // a valid, in-range threshold rather than a crash or garbage.
  int single[imgpipe::kHistogramBins] = {0};
  single[128] = 42;
  const int single_threshold =
      imgpipe::ComputeOtsuThreshold(single, imgpipe::kHistogramBins);
  CHECK(single_threshold >= 0);
  CHECK(single_threshold < imgpipe::kHistogramBins);

  // Degenerate inputs.
  int empty[imgpipe::kHistogramBins] = {0};
  CHECK_EQ(imgpipe::ComputeOtsuThreshold(empty, imgpipe::kHistogramBins), 0);
  CHECK_EQ(imgpipe::ComputeOtsuThreshold(nullptr, 256), 0);

  // Two narrow modes with very different masses, which is what a gradient
  // magnitude image looks like: a large dark background and a small bright
  // edge population. The split must fall between the modes.
  int skewed[imgpipe::kHistogramBins] = {0};
  for (int bin = 20; bin < 25; ++bin) skewed[bin] = 1000;
  for (int bin = 200; bin < 205; ++bin) skewed[bin] = 50;
  const int skewed_threshold =
      imgpipe::ComputeOtsuThreshold(skewed, imgpipe::kHistogramBins);
  CHECK(skewed_threshold >= 24);
  CHECK(skewed_threshold < 200);
}

void TestCommandLine() {
  {
    const char* argv[] = {"edge_pipeline", "--input", "in",   "--output",
                          "out",           "--streams", "8",  "--gauss-size",
                          "3",             "--threshold", "77", "--dilate",
                          "--save-stages", "--limit",   "5",  "--verbose"};
    imgpipe::Options options;
    std::string error;
    CHECK(imgpipe::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv,
                                    &options, &error));
    CHECK(options.input_dir == "in");
    CHECK(options.output_dir == "out");
    CHECK_EQ(options.stream_count, 8);
    CHECK_EQ(options.gauss_size, 3);
    CHECK(options.threshold_mode == imgpipe::ThresholdMode::kFixed);
    CHECK_EQ(options.fixed_threshold, 77);
    CHECK(options.dilate);
    CHECK(options.save_stages);
    CHECK_EQ(options.limit, 5);
    CHECK(options.verbose);
  }
  {
    // Defaults, and 'auto' selecting Otsu.
    const char* argv[] = {"edge_pipeline", "--input", "in", "--threshold",
                          "auto"};
    imgpipe::Options options;
    std::string error;
    CHECK(imgpipe::ParseCommandLine(5, argv, &options, &error));
    CHECK(options.threshold_mode == imgpipe::ThresholdMode::kOtsu);
    CHECK_EQ(options.stream_count, 4);
    CHECK_EQ(options.gauss_size, 5);
  }
  {
    // --help short-circuits before the --input requirement.
    const char* argv[] = {"edge_pipeline", "--help"};
    imgpipe::Options options;
    std::string error;
    CHECK(imgpipe::ParseCommandLine(2, argv, &options, &error));
    CHECK(options.help);
  }

  // Each of these must be rejected with a message rather than silently
  // accepted; a bad --streams value used to be the easiest way to hang a run.
  const std::vector<std::vector<const char*>> bad_cases = {
      {"edge_pipeline"},
      {"edge_pipeline", "--input"},
      {"edge_pipeline", "--input", "in", "--streams", "0"},
      {"edge_pipeline", "--input", "in", "--streams", "99"},
      {"edge_pipeline", "--input", "in", "--streams", "4x"},
      {"edge_pipeline", "--input", "in", "--gauss-size", "7"},
      {"edge_pipeline", "--input", "in", "--threshold", "300"},
      {"edge_pipeline", "--input", "in", "--threshold", "nope"},
      {"edge_pipeline", "--input", "in", "--sobel-scale", "0"},
      {"edge_pipeline", "--input", "in", "--limit", "-1"},
      {"edge_pipeline", "--input", "in", "--bogus"},
  };
  for (const std::vector<const char*>& argv : bad_cases) {
    imgpipe::Options options;
    std::string error;
    const bool parsed = imgpipe::ParseCommandLine(
        static_cast<int>(argv.size()), argv.data(), &options, &error);
    Check(!parsed && !error.empty(),
          std::string("expected rejection of '") + argv.back() + "'",
          __LINE__);
  }
}

void TestPathHelpers() {
  CHECK(imgpipe::Stem("/a/b/c.png") == "c");
  CHECK(imgpipe::Stem("c.tar.png") == "c.tar");
  CHECK(imgpipe::Stem("noext") == "noext");
  CHECK(imgpipe::JoinPath("a", "b") == "a/b");
  CHECK(imgpipe::JoinPath("a/", "b") == "a/b");
  CHECK(imgpipe::JoinPath("", "b") == "b");
  CHECK(imgpipe::HasSupportedExtension("x.PNG"));
  CHECK(imgpipe::HasSupportedExtension("x.jpeg"));
  CHECK(!imgpipe::HasSupportedExtension("x.txt"));
  // A dot in a parent directory must not be mistaken for an extension.
  CHECK(!imgpipe::HasSupportedExtension("dir.png/file"));
}

void TestImageRoundTrip() {
  const std::string directory = "build/test_scratch/nested";
  std::string error;
  CHECK(imgpipe::MakeDirectories(directory, &error));
  // Creating an existing directory must succeed rather than error out.
  CHECK(imgpipe::MakeDirectories(directory, &error));

  imgpipe::Image gray;
  gray.Allocate(7, 5, 1);
  CHECK_EQ(static_cast<int>(gray.pixels.size()), 35);
  for (size_t i = 0; i < gray.pixels.size(); ++i) {
    gray.pixels[i] = static_cast<unsigned char>(i * 3);
  }
  const std::string gray_path = directory + "/gray.png";
  CHECK(imgpipe::WritePng(gray_path, gray, &error));

  imgpipe::Image reloaded;
  CHECK(imgpipe::ReadImage(gray_path, &reloaded, &error));
  CHECK_EQ(reloaded.width, 7);
  CHECK_EQ(reloaded.height, 5);
  CHECK_EQ(reloaded.channels, 1);
  CHECK(reloaded.pixels == gray.pixels);

  imgpipe::Image color;
  color.Allocate(4, 3, 3);
  for (size_t i = 0; i < color.pixels.size(); ++i) {
    color.pixels[i] = static_cast<unsigned char>(255 - i);
  }
  const std::string color_path = directory + "/color.png";
  CHECK(imgpipe::WritePng(color_path, color, &error));
  imgpipe::Image color_reloaded;
  CHECK(imgpipe::ReadImage(color_path, &color_reloaded, &error));
  CHECK_EQ(color_reloaded.channels, 3);
  CHECK(color_reloaded.pixels == color.pixels);

  std::vector<std::string> listed;
  CHECK(imgpipe::ListImageFiles(directory, &listed, &error));
  CHECK_EQ(static_cast<int>(listed.size()), 2);
  // ListImageFiles sorts so that repeated runs are reproducible.
  CHECK(listed[0] < listed[1]);

  imgpipe::Image missing;
  CHECK(!imgpipe::ReadImage(directory + "/absent.png", &missing, &error));
  CHECK(!error.empty());

  std::vector<std::string> nothing;
  CHECK(!imgpipe::ListImageFiles("no/such/directory", &nothing, &error));

  std::remove(gray_path.c_str());
  std::remove(color_path.c_str());
}

}  // namespace

int main() {
  TestOtsu();
  TestCommandLine();
  TestPathHelpers();
  TestImageRoundTrip();

  std::cout << (g_checks - g_failures) << "/" << g_checks
            << " host checks passed\n";
  if (g_failures > 0) {
    std::cout << g_failures << " check(s) FAILED\n";
    return 1;
  }
  std::cout << "all host tests passed\n";
  return 0;
}
