// Batch GPU edge detection over a directory of images.
//
// Each worker thread owns one CUDA stream and one set of device buffers, so
// host-side PNG decoding overlaps with GPU filtering on other images. Work is
// handed out by a shared atomic cursor, which keeps the device busy even
// though the dataset mixes 256x256 and 1024x1024 inputs.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "imgpipe/cli.h"
#include "imgpipe/cpu_reference.h"
#include "imgpipe/gpu_pipeline.h"
#include "imgpipe/image.h"

namespace imgpipe {
namespace {

// What the run recorded for a single input image.
struct ImageRecord {
  std::string name;
  int width = 0;
  int height = 0;
  int channels = 0;
  // Timing of the engine whose output was written to disk: the GPU whenever
  // it ran, otherwise the host reference.
  FrameTiming timing;
  // Only filled when the host reference also ran, that is under --engine cpu
  // or --engine both.
  double cpu_ms = 0.0;
  int cpu_threshold = 0;
  bool compared = false;
  AgreementStats agreement;
  bool ok = false;
  std::string error;
};

// Writes each line to stdout and, when a log path was given, to that file.
class Logger {
 public:
  bool Open(const std::string& path, std::string* error) {
    if (path.empty()) return true;
    file_.open(path.c_str(), std::ios::out | std::ios::app);
    if (!file_.is_open()) {
      *error = "cannot open log file " + path;
      return false;
    }
    return true;
  }

  void Write(const std::string& line) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << line << "\n";
    if (file_.is_open()) file_ << line << "\n";
  }

  void Flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout.flush();
    if (file_.is_open()) file_.flush();
  }

 private:
  std::mutex mutex_;
  std::ofstream file_;
};

std::string FormatFixed(double value, int precision) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

std::string CurrentTimestamp() {
  const std::time_t now = std::time(nullptr);
  char buffer[32];
  if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S",
                    std::localtime(&now)) == 0) {
    return "unknown-time";
  }
  return buffer;
}

// Writes the edge map, plus the intermediate stages when requested.
bool WriteOutputs(const Options& options, const std::string& stem,
                  const StageImages& stages, std::string* error) {
  if (!WritePng(JoinPath(options.output_dir, stem + "_edges.png"),
                stages.edges, error)) {
    return false;
  }
  if (!options.save_stages) return true;
  return WritePng(JoinPath(options.output_dir, stem + "_gray.png"),
                  stages.gray, error) &&
         WritePng(JoinPath(options.output_dir, stem + "_blur.png"),
                  stages.blurred, error) &&
         WritePng(JoinPath(options.output_dir, stem + "_grad.png"),
                  stages.gradient, error);
}

// Pulls indices off `cursor` until the input list is exhausted.
void RunWorker(const Options& options, const std::vector<std::string>& paths,
               GpuPipeline* pipeline, std::atomic<size_t>* cursor,
               std::vector<ImageRecord>* records, Logger* logger) {
  for (;;) {
    const size_t index = cursor->fetch_add(1);
    if (index >= paths.size()) return;

    // Each worker writes only to its own slot, so no lock is needed here.
    ImageRecord& record = (*records)[index];
    const std::string& path = paths[index];
    record.name = Stem(path);

    Image input;
    if (!ReadImage(path, &input, &record.error)) continue;
    record.width = input.width;
    record.height = input.height;
    record.channels = input.channels;

    StageImages stages;
    if (options.engine == Engine::kCpu) {
      if (!RunCpuPipeline(options, input, &stages, &record.timing,
                          &record.error)) {
        continue;
      }
      record.cpu_ms = record.timing.compute_ms;
    } else if (!pipeline->Process(input, &stages, &record.timing,
                                  &record.error)) {
      continue;
    }

    // In --engine both the same image goes through the host reference as
    // well. The GPU result is the one written to disk; the host result is
    // only used to time the baseline and to check the edge maps agree.
    if (options.engine == Engine::kBoth) {
      StageImages cpu_stages;
      FrameTiming cpu_timing;
      if (!RunCpuPipeline(options, input, &cpu_stages, &cpu_timing,
                          &record.error)) {
        continue;
      }
      record.cpu_ms = cpu_timing.compute_ms;
      record.cpu_threshold = cpu_timing.threshold;
      record.agreement = CompareImages(stages.edges, cpu_stages.edges);
      record.compared = true;
    }

    if (!WriteOutputs(options, record.name, stages, &record.error)) continue;
    record.ok = true;

    if (options.verbose) {
      std::ostringstream line;
      line << "  [" << index + 1 << "/" << paths.size() << "] " << record.name
           << " " << input.width << "x" << input.height << "x"
           << input.channels << " threshold=" << record.timing.threshold
           << " edges=" << FormatFixed(100.0 * record.timing.edge_fraction, 2)
           << "% " << (options.engine == Engine::kCpu ? "cpu=" : "gpu=")
           << FormatFixed(record.timing.compute_ms, 3) << "ms";
      if (record.compared) {
        line << " cpu=" << FormatFixed(record.cpu_ms, 3) << "ms match="
             << FormatFixed(100.0 * record.agreement.MatchFraction(), 3) << "%";
      }
      logger->Write(line.str());
    }
  }
}

int Run(int argc, const char* const* argv) {
  Options options;
  std::string error;
  if (!ParseCommandLine(argc, argv, &options, &error)) {
    std::cerr << "error: " << error << "\n\n"
              << UsageText(argc > 0 ? argv[0] : "edge_pipeline");
    return 2;
  }
  if (options.help) {
    std::cout << UsageText(argc > 0 ? argv[0] : "edge_pipeline");
    return 0;
  }

  Logger logger;
  if (!logger.Open(options.log_path, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }

  logger.Write("=== CUDA NPP batch edge detection ===");
  logger.Write("started " + CurrentTimestamp());

  // --engine cpu is the one mode that must work on a machine with no CUDA
  // device at all, so the device is only claimed when it will be used.
  if (options.engine == Engine::kCpu) {
    logger.Write("device: none (host reference engine)");
  } else {
    std::string device_description;
    if (!SelectDevice(options.device, &device_description, &error)) {
      std::cerr << "error: " << error << "\n";
      return 1;
    }
    logger.Write(device_description);
  }
  logger.Write(DescribeOptions(options));

  std::vector<std::string> paths;
  if (!ListImageFiles(options.input_dir, &paths, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  if (paths.empty()) {
    std::cerr << "error: no supported images found in " << options.input_dir
              << "\n";
    return 1;
  }
  if (options.limit > 0 && static_cast<size_t>(options.limit) < paths.size()) {
    paths.resize(options.limit);
  }
  if (!MakeDirectories(options.output_dir, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }

  std::ostringstream header;
  header << "found " << paths.size() << " image(s) in " << options.input_dir;
  logger.Write(header.str());

  const int worker_count =
      std::min<int>(options.stream_count, static_cast<int>(paths.size()));
  std::vector<std::unique_ptr<GpuPipeline>> pipelines;
  if (options.engine != Engine::kCpu) {
    pipelines.reserve(worker_count);
    for (int i = 0; i < worker_count; ++i) {
      pipelines.emplace_back(new GpuPipeline);
      if (!pipelines[i]->Initialize(options, &error)) {
        std::cerr << "error: worker " << i << ": " << error << "\n";
        return 1;
      }
    }
  } else {
    pipelines.resize(worker_count);
  }

  std::vector<ImageRecord> records(paths.size());
  std::atomic<size_t> cursor(0);
  const auto start = std::chrono::steady_clock::now();

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (int i = 0; i < worker_count; ++i) {
    workers.emplace_back(RunWorker, std::cref(options), std::cref(paths),
                         pipelines[i].get(), &cursor, &records, &logger);
  }
  for (std::thread& worker : workers) worker.join();

  const auto finish = std::chrono::steady_clock::now();
  const double wall_seconds =
      std::chrono::duration<double>(finish - start).count();

  const bool comparing = options.engine == Engine::kBoth;
  logger.Write("");
  {
    std::string header =
        "image                              size      ch  thr  edge%   up(ms) "
        " gpu(ms)  down(ms)";
    std::string rule =
        "-------------------------------------------------------------------"
        "-------------------";
    if (comparing) {
      header += "   cpu(ms)   match%";
      rule += "--------------------";
    }
    logger.Write(header);
    logger.Write(rule);
  }

  size_t succeeded = 0;
  double total_megapixels = 0.0;
  double total_compute_ms = 0.0;
  double total_upload_ms = 0.0;
  double total_download_ms = 0.0;
  double total_cpu_ms = 0.0;
  double matching_pixels = 0.0;
  double compared_pixels = 0.0;
  double worst_match_fraction = 1.0;
  std::string worst_match_name;
  size_t threshold_mismatches = 0;
  for (const ImageRecord& record : records) {
    if (!record.ok) {
      logger.Write("FAILED " + record.name + ": " + record.error);
      continue;
    }
    ++succeeded;
    total_megapixels +=
        static_cast<double>(record.width) * record.height / 1.0e6;
    total_compute_ms += record.timing.compute_ms;
    total_upload_ms += record.timing.upload_ms;
    total_download_ms += record.timing.download_ms;
    total_cpu_ms += record.cpu_ms;
    if (record.compared) {
      matching_pixels += static_cast<double>(record.agreement.matching);
      compared_pixels += static_cast<double>(record.agreement.pixels);
      if (record.agreement.MatchFraction() < worst_match_fraction) {
        worst_match_fraction = record.agreement.MatchFraction();
        worst_match_name = record.name;
      }
      if (record.cpu_threshold != record.timing.threshold) {
        ++threshold_mismatches;
      }
    }

    std::ostringstream size_text;
    size_text << record.width << "x" << record.height;
    std::ostringstream line;
    line << std::left << std::setw(34) << record.name << std::setw(10)
         << size_text.str() << std::setw(4) << record.channels << std::setw(5)
         << record.timing.threshold << std::right << std::setw(7)
         << FormatFixed(100.0 * record.timing.edge_fraction, 2)
         << std::setw(9) << FormatFixed(record.timing.upload_ms, 3)
         << std::setw(9) << FormatFixed(record.timing.compute_ms, 3)
         << std::setw(10) << FormatFixed(record.timing.download_ms, 3);
    if (comparing) {
      line << std::setw(10) << FormatFixed(record.cpu_ms, 3) << std::setw(10)
           << FormatFixed(100.0 * record.agreement.MatchFraction(), 3);
    }
    logger.Write(line.str());
  }

  const size_t failed = records.size() - succeeded;
  logger.Write("");
  logger.Write("=== summary ===");
  {
    std::ostringstream out;
    out << "images processed : " << succeeded << " of " << records.size()
        << " (" << failed << " failed)";
    logger.Write(out.str());
  }
  {
    std::ostringstream out;
    out << "pixels processed : " << FormatFixed(total_megapixels, 2)
        << " megapixels";
    logger.Write(out.str());
  }
  {
    std::ostringstream out;
    out << "wall clock       : " << FormatFixed(wall_seconds, 3) << " s using "
        << worker_count
        << (options.engine == Engine::kCpu ? " worker thread(s)"
                                           : " stream(s)");
    logger.Write(out.str());
  }
  {
    std::ostringstream out;
    out << "throughput       : "
        << FormatFixed(succeeded / std::max(wall_seconds, 1e-9), 2)
        << " images/s, "
        << FormatFixed(total_megapixels / std::max(wall_seconds, 1e-9), 2)
        << " megapixels/s";
    logger.Write(out.str());
  }
  {
    std::ostringstream out;
    out << "gpu time (sum)   : upload " << FormatFixed(total_upload_ms, 1)
        << " ms, compute " << FormatFixed(total_compute_ms, 1)
        << " ms, download " << FormatFixed(total_download_ms, 1) << " ms";
    logger.Write(out.str());
  }
  if (succeeded > 0) {
    std::ostringstream out;
    out << "gpu time (mean)  : "
        << FormatFixed(total_compute_ms / static_cast<double>(succeeded), 3)
        << " ms of kernel time per image";
    logger.Write(out.str());
  }
  if (comparing && succeeded > 0) {
    logger.Write("");
    logger.Write("=== gpu versus host reference ===");
    {
      std::ostringstream out;
      out << "kernel time      : gpu " << FormatFixed(total_compute_ms, 1)
          << " ms, cpu " << FormatFixed(total_cpu_ms, 1) << " ms";
      logger.Write(out.str());
    }
    {
      std::ostringstream out;
      out << "speedup          : "
          << FormatFixed(total_cpu_ms / std::max(total_compute_ms, 1e-9), 1)
          << "x on summed per-image compute time";
      logger.Write(out.str());
    }
    {
      std::ostringstream out;
      out << "edge map match   : "
          << FormatFixed(100.0 * matching_pixels /
                             std::max(compared_pixels, 1.0),
                         4)
          << "% of " << FormatFixed(compared_pixels / 1.0e6, 2)
          << " megapixels identical";
      logger.Write(out.str());
    }
    {
      std::ostringstream out;
      out << "worst image      : " << worst_match_name << " at "
          << FormatFixed(100.0 * worst_match_fraction, 3) << "%";
      logger.Write(out.str());
    }
    {
      std::ostringstream out;
      out << "otsu threshold   : " << succeeded - threshold_mismatches << " of "
          << succeeded << " images chose the same threshold on both engines";
      logger.Write(out.str());
    }
  }
  logger.Write("outputs written to " + options.output_dir);
  logger.Write("finished " + CurrentTimestamp());
  logger.Flush();

  return failed == 0 ? 0 : 1;
}

}  // namespace
}  // namespace imgpipe

int main(int argc, char** argv) {
  return imgpipe::Run(argc, argv);
}
