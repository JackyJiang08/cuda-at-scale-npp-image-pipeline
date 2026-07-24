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
#include "imgpipe/cpu_reference.h"
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

#define REQUIRE(condition) Check((condition), #condition, __LINE__)
#define REQUIRE_EQ(actual, expected)                                        \
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
  REQUIRE(threshold >= 10);
  REQUIRE(threshold < 200);

  // A single populated bin has no meaningful split; the result must still be
  // a valid, in-range threshold rather than a crash or garbage.
  int single[imgpipe::kHistogramBins] = {0};
  single[128] = 42;
  const int single_threshold =
      imgpipe::ComputeOtsuThreshold(single, imgpipe::kHistogramBins);
  REQUIRE(single_threshold >= 0);
  REQUIRE(single_threshold < imgpipe::kHistogramBins);

  // Degenerate inputs.
  int empty[imgpipe::kHistogramBins] = {0};
  REQUIRE_EQ(imgpipe::ComputeOtsuThreshold(empty, imgpipe::kHistogramBins), 0);
  REQUIRE_EQ(imgpipe::ComputeOtsuThreshold(nullptr, 256), 0);

  // Two narrow modes with very different masses, which is what a gradient
  // magnitude image looks like: a large dark background and a small bright
  // edge population. The split must fall between the modes.
  int skewed[imgpipe::kHistogramBins] = {0};
  for (int bin = 20; bin < 25; ++bin) skewed[bin] = 1000;
  for (int bin = 200; bin < 205; ++bin) skewed[bin] = 50;
  const int skewed_threshold =
      imgpipe::ComputeOtsuThreshold(skewed, imgpipe::kHistogramBins);
  REQUIRE(skewed_threshold >= 24);
  REQUIRE(skewed_threshold < 200);
}

void TestCommandLine() {
  {
    const char* argv[] = {"edge_pipeline", "--input", "in",   "--output",
                          "out",           "--streams", "8",  "--gauss-size",
                          "3",             "--threshold", "77", "--dilate",
                          "--save-stages", "--limit",   "5",  "--verbose"};
    imgpipe::Options options;
    std::string error;
    REQUIRE(imgpipe::ParseCommandLine(sizeof(argv) / sizeof(argv[0]), argv,
                                    &options, &error));
    REQUIRE(options.input_dir == "in");
    REQUIRE(options.output_dir == "out");
    REQUIRE_EQ(options.stream_count, 8);
    REQUIRE_EQ(options.gauss_size, 3);
    REQUIRE(options.threshold_mode == imgpipe::ThresholdMode::kFixed);
    REQUIRE_EQ(options.fixed_threshold, 77);
    REQUIRE(options.dilate);
    REQUIRE(options.save_stages);
    REQUIRE_EQ(options.limit, 5);
    REQUIRE(options.verbose);
  }
  {
    // Defaults, and 'auto' selecting Otsu.
    const char* argv[] = {"edge_pipeline", "--input", "in", "--threshold",
                          "auto"};
    imgpipe::Options options;
    std::string error;
    REQUIRE(imgpipe::ParseCommandLine(5, argv, &options, &error));
    REQUIRE(options.threshold_mode == imgpipe::ThresholdMode::kOtsu);
    REQUIRE_EQ(options.stream_count, 4);
    REQUIRE_EQ(options.gauss_size, 5);
  }
  {
    // --help short-circuits before the --input requirement.
    const char* argv[] = {"edge_pipeline", "--help"};
    imgpipe::Options options;
    std::string error;
    REQUIRE(imgpipe::ParseCommandLine(2, argv, &options, &error));
    REQUIRE(options.help);
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
      {"edge_pipeline", "--input", "in", "--engine", "tpu"},
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
  REQUIRE(imgpipe::Stem("/a/b/c.png") == "c");
  REQUIRE(imgpipe::Stem("c.tar.png") == "c.tar");
  REQUIRE(imgpipe::Stem("noext") == "noext");
  REQUIRE(imgpipe::JoinPath("a", "b") == "a/b");
  REQUIRE(imgpipe::JoinPath("a/", "b") == "a/b");
  REQUIRE(imgpipe::JoinPath("", "b") == "b");
  REQUIRE(imgpipe::HasSupportedExtension("x.PNG"));
  REQUIRE(imgpipe::HasSupportedExtension("x.jpeg"));
  REQUIRE(!imgpipe::HasSupportedExtension("x.txt"));
  // A dot in a parent directory must not be mistaken for an extension.
  REQUIRE(!imgpipe::HasSupportedExtension("dir.png/file"));
}

void TestImageRoundTrip() {
  const std::string directory = "build/test_scratch/nested";
  std::string error;
  REQUIRE(imgpipe::MakeDirectories(directory, &error));
  // Creating an existing directory must succeed rather than error out.
  REQUIRE(imgpipe::MakeDirectories(directory, &error));

  imgpipe::Image gray;
  gray.Allocate(7, 5, 1);
  REQUIRE_EQ(static_cast<int>(gray.pixels.size()), 35);
  for (size_t i = 0; i < gray.pixels.size(); ++i) {
    gray.pixels[i] = static_cast<unsigned char>(i * 3);
  }
  const std::string gray_path = directory + "/gray.png";
  REQUIRE(imgpipe::WritePng(gray_path, gray, &error));

  imgpipe::Image reloaded;
  REQUIRE(imgpipe::ReadImage(gray_path, &reloaded, &error));
  REQUIRE_EQ(reloaded.width, 7);
  REQUIRE_EQ(reloaded.height, 5);
  REQUIRE_EQ(reloaded.channels, 1);
  REQUIRE(reloaded.pixels == gray.pixels);

  imgpipe::Image color;
  color.Allocate(4, 3, 3);
  for (size_t i = 0; i < color.pixels.size(); ++i) {
    color.pixels[i] = static_cast<unsigned char>(255 - i);
  }
  const std::string color_path = directory + "/color.png";
  REQUIRE(imgpipe::WritePng(color_path, color, &error));
  imgpipe::Image color_reloaded;
  REQUIRE(imgpipe::ReadImage(color_path, &color_reloaded, &error));
  REQUIRE_EQ(color_reloaded.channels, 3);
  REQUIRE(color_reloaded.pixels == color.pixels);

  std::vector<std::string> listed;
  REQUIRE(imgpipe::ListImageFiles(directory, &listed, &error));
  REQUIRE_EQ(static_cast<int>(listed.size()), 2);
  // ListImageFiles sorts so that repeated runs are reproducible.
  REQUIRE(listed[0] < listed[1]);

  imgpipe::Image missing;
  REQUIRE(!imgpipe::ReadImage(directory + "/absent.png", &missing, &error));
  REQUIRE(!error.empty());

  std::vector<std::string> nothing;
  REQUIRE(!imgpipe::ListImageFiles("no/such/directory", &nothing, &error));

  std::remove(gray_path.c_str());
  std::remove(color_path.c_str());
}

// Builds a single-channel image whose left half is `left` and right half is
// `right`, giving one vertical step edge down the middle.
imgpipe::Image MakeStepImage(int width, int height, int left, int right) {
  imgpipe::Image image;
  image.Allocate(width, height, 1);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image.pixels[static_cast<size_t>(y) * width + x] =
          static_cast<unsigned char>(x < width / 2 ? left : right);
    }
  }
  return image;
}

void TestCpuStages() {
  // BT.601 luma: pure red is 0.299 * 255 = 76.245, so 76 after rounding.
  imgpipe::Image rgb;
  rgb.Allocate(2, 1, 3);
  rgb.pixels = {255, 0, 0, 0, 255, 0};
  imgpipe::Image gray;
  imgpipe::ConvertRgbToGray(rgb, &gray);
  REQUIRE_EQ(gray.channels, 1);
  REQUIRE_EQ(static_cast<int>(gray.pixels[0]), 76);
  REQUIRE_EQ(static_cast<int>(gray.pixels[1]), 150);

  // Blurring a constant image must give back a constant image, edges
  // included. Uniformity is the real assertion: an unnormalised kernel or a
  // border that reads zeros outside the image would both show up as the
  // margin differing from the interior.
  //
  // The result is allowed to sit one grey level below the input rather than
  // exactly on it. NPP truncates instead of rounding, this reference matches
  // that, and normalised floating-point weights sum to 1 only to within a
  // ULP, so a flat 200 can legitimately come back as 199. Asserting equality
  // here would be asserting rounding behaviour the device does not have.
  imgpipe::Image flat = MakeStepImage(8, 8, 200, 200);
  for (const int mask_size : {3, 5}) {
    imgpipe::Image blurred;
    imgpipe::FilterGaussian(flat, mask_size, &blurred);
    REQUIRE_EQ(static_cast<int>(blurred.pixels.size()), 64);
    const int first = blurred.pixels[0];
    bool uniform = true;
    for (unsigned char value : blurred.pixels) uniform &= value == first;
    Check(uniform, "constant image stays constant through the blur", __LINE__);
    Check(first == 200 || first == 199,
          "constant image keeps its level through the blur", __LINE__);
  }

  // A constant image has no gradient anywhere, borders included.
  std::vector<int> gx;
  std::vector<int> gy;
  imgpipe::FilterSobel(flat, &gx, &gy);
  bool flat_gradient = true;
  for (size_t i = 0; i < gx.size(); ++i) {
    flat_gradient &= gx[i] == 0 && gy[i] == 0;
  }
  Check(flat_gradient, "constant image has a zero gradient", __LINE__);

  // A vertical step responds in x only.
  const imgpipe::Image step = MakeStepImage(8, 8, 0, 255);
  imgpipe::FilterSobel(step, &gx, &gy);
  const size_t middle = 4 * 8 + 3;  // Just left of the step, mid-height.
  REQUIRE(gx[middle] != 0);
  bool no_vertical_response = true;
  for (size_t i = 0; i < gy.size(); ++i) no_vertical_response &= gy[i] == 0;
  Check(no_vertical_response, "vertical step has no y gradient", __LINE__);

  // The magnitude saturates rather than wrapping around.
  imgpipe::Image gradient;
  const std::vector<int> big_x(4, 30000);
  const std::vector<int> big_y(4, 30000);
  imgpipe::GradientMagnitude(big_x, big_y, 2, 2, 1.0f, &gradient);
  REQUIRE_EQ(static_cast<int>(gradient.pixels[0]), 255);

  int histogram[imgpipe::kHistogramBins];
  imgpipe::ComputeHistogram(step, histogram);
  REQUIRE_EQ(histogram[0], 32);
  REQUIRE_EQ(histogram[255], 32);
  REQUIRE_EQ(histogram[128], 0);

  imgpipe::Image binary;
  size_t edge_count = 0;
  imgpipe::Binarize(step, 128, &binary, &edge_count);
  REQUIRE_EQ(static_cast<int>(edge_count), 32);
  REQUIRE_EQ(static_cast<int>(binary.pixels[0]), 0);
  REQUIRE_EQ(static_cast<int>(binary.pixels[7]), 255);

  // One lit pixel dilates into the 3x3 block around it.
  imgpipe::Image dot;
  dot.Allocate(5, 5, 1);
  dot.pixels.assign(25, 0);
  dot.pixels[12] = 255;
  imgpipe::Image dilated;
  imgpipe::Dilate3x3(dot, &dilated);
  int lit = 0;
  for (unsigned char value : dilated.pixels) lit += value == 255 ? 1 : 0;
  REQUIRE_EQ(lit, 9);

  const imgpipe::AgreementStats same = imgpipe::CompareImages(dot, dot);
  REQUIRE_EQ(static_cast<int>(same.matching), 25);
  REQUIRE(same.MatchFraction() == 1.0);
  REQUIRE(same.max_difference == 0.0);
  // Mismatched dimensions are reported as no comparison rather than a crash.
  const imgpipe::AgreementStats mismatched = imgpipe::CompareImages(dot, flat);
  REQUIRE_EQ(static_cast<int>(mismatched.pixels), 0);
  REQUIRE(mismatched.MatchFraction() == 0.0);
}

void TestCpuPipeline() {
  imgpipe::Options options;
  options.input_dir = "unused";
  options.gauss_size = 3;
  options.sobel_scale = 0.25f;
  options.threshold_mode = imgpipe::ThresholdMode::kOtsu;

  const imgpipe::Image step = MakeStepImage(32, 32, 0, 255);
  imgpipe::StageImages stages;
  imgpipe::FrameTiming timing;
  std::string error;
  REQUIRE(imgpipe::RunCpuPipeline(options, step, &stages, &timing, &error));
  REQUIRE_EQ(stages.edges.width, 32);
  REQUIRE_EQ(stages.edges.height, 32);
  REQUIRE_EQ(stages.edges.channels, 1);
  REQUIRE(timing.compute_ms >= 0.0);

  // The only edge is the step down the middle, so the flat left and right
  // margins must come back black and the columns beside the step must not.
  REQUIRE_EQ(static_cast<int>(stages.edges.pixels[16 * 32 + 1]), 0);
  REQUIRE_EQ(static_cast<int>(stages.edges.pixels[16 * 32 + 30]), 0);
  REQUIRE(stages.edges.pixels[16 * 32 + 15] == 255 ||
          stages.edges.pixels[16 * 32 + 16] == 255);
  REQUIRE(timing.edge_fraction > 0.0);
  REQUIRE(timing.edge_fraction < 0.5);

  // Dilation can only add edge pixels, never remove them.
  imgpipe::Options dilating = options;
  dilating.dilate = true;
  imgpipe::StageImages thick;
  imgpipe::FrameTiming thick_timing;
  REQUIRE(imgpipe::RunCpuPipeline(dilating, step, &thick, &thick_timing,
                                  &error));
  int thin_lit = 0;
  int thick_lit = 0;
  for (unsigned char value : stages.edges.pixels) thin_lit += value ? 1 : 0;
  for (unsigned char value : thick.edges.pixels) thick_lit += value ? 1 : 0;
  REQUIRE(thick_lit >= thin_lit);

  // A three-channel input goes through the grayscale conversion first.
  imgpipe::Image color;
  color.Allocate(16, 16, 3);
  for (size_t i = 0; i < color.pixels.size(); ++i) {
    color.pixels[i] = static_cast<unsigned char>((i / 3) % 2 ? 240 : 10);
  }
  imgpipe::StageImages color_stages;
  imgpipe::FrameTiming color_timing;
  REQUIRE(imgpipe::RunCpuPipeline(options, color, &color_stages, &color_timing,
                                  &error));
  REQUIRE_EQ(color_stages.edges.channels, 1);
  REQUIRE_EQ(color_stages.edges.width, 16);

  // Unsupported channel counts are rejected instead of read out of bounds.
  imgpipe::Image two_channel;
  two_channel.Allocate(4, 4, 2);
  imgpipe::StageImages ignored;
  imgpipe::FrameTiming ignored_timing;
  REQUIRE(!imgpipe::RunCpuPipeline(options, two_channel, &ignored,
                                   &ignored_timing, &error));
  REQUIRE(!error.empty());
}

}  // namespace

int main() {
  TestOtsu();
  TestCommandLine();
  TestPathHelpers();
  TestImageRoundTrip();
  TestCpuStages();
  TestCpuPipeline();

  std::cout << (g_checks - g_failures) << "/" << g_checks
            << " host checks passed\n";
  if (g_failures > 0) {
    std::cout << g_failures << " check(s) FAILED\n";
    return 1;
  }
  std::cout << "all host tests passed\n";
  return 0;
}
