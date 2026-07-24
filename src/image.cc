#include "imgpipe/image.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

// stb is vendored verbatim, so its warnings are not actionable here; keep
// them out of this project's build output without lowering -Wall elsewhere.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#pragma GCC diagnostic pop

namespace imgpipe {
namespace {

// Extensions the stb decoders are configured to handle, lower-cased.
const char* const kSupportedExtensions[] = {".png", ".jpg", ".jpeg", ".bmp",
                                            ".tga"};

std::string ToLower(const std::string& text) {
  std::string lowered = text;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return lowered;
}

// Returns the extension of `path` including the leading dot, lower-cased, or
// an empty string when there is none.
std::string LowerExtension(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const size_t dot = path.find_last_of('.');
  if (dot == std::string::npos) return std::string();
  if (slash != std::string::npos && dot < slash) return std::string();
  return ToLower(path.substr(dot));
}

bool DirectoryExists(const std::string& path) {
  struct stat info;
  if (stat(path.c_str(), &info) != 0) return false;
  return S_ISDIR(info.st_mode);
}

}  // namespace

void Image::Allocate(int new_width, int new_height, int new_channels) {
  width = new_width;
  height = new_height;
  channels = new_channels;
  pixels.assign(SizeBytes(), 0);
}

bool HasSupportedExtension(const std::string& path) {
  const std::string extension = LowerExtension(path);
  for (const char* candidate : kSupportedExtensions) {
    if (extension == candidate) return true;
  }
  return false;
}

std::string Stem(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const std::string name =
      slash == std::string::npos ? path : path.substr(slash + 1);
  const size_t dot = name.find_last_of('.');
  return dot == std::string::npos ? name : name.substr(0, dot);
}

std::string JoinPath(const std::string& directory, const std::string& name) {
  if (directory.empty()) return name;
  if (directory.back() == '/') return directory + name;
  return directory + "/" + name;
}

bool ReadImage(const std::string& path, Image* image, std::string* error) {
  int width = 0;
  int height = 0;
  int source_channels = 0;
  if (stbi_info(path.c_str(), &width, &height, &source_channels) == 0) {
    *error = "cannot read image header from " + path + ": " +
             stbi_failure_reason();
    return false;
  }

  // Grayscale and grayscale+alpha sources stay single channel; everything
  // else is normalized to packed RGB so the pipeline sees only two layouts.
  const int desired_channels = source_channels <= 2 ? 1 : 3;
  unsigned char* data =
      stbi_load(path.c_str(), &width, &height, &source_channels,
                desired_channels);
  if (data == nullptr) {
    *error = "cannot decode " + path + ": " + stbi_failure_reason();
    return false;
  }

  image->width = width;
  image->height = height;
  image->channels = desired_channels;
  image->pixels.assign(data, data + image->SizeBytes());
  stbi_image_free(data);
  return true;
}

bool WritePng(const std::string& path, const Image& image, std::string* error) {
  if (image.IsEmpty() || image.pixels.size() < image.SizeBytes()) {
    *error = "refusing to write malformed image to " + path;
    return false;
  }
  const int stride = static_cast<int>(image.RowBytes());
  if (stbi_write_png(path.c_str(), image.width, image.height, image.channels,
                     image.pixels.data(), stride) == 0) {
    *error = "cannot write " + path;
    return false;
  }
  return true;
}

bool ListImageFiles(const std::string& directory,
                    std::vector<std::string>* paths, std::string* error) {
  DIR* handle = opendir(directory.c_str());
  if (handle == nullptr) {
    *error = "cannot open directory " + directory + ": " +
             std::strerror(errno);
    return false;
  }

  std::vector<std::string> found;
  for (const dirent* entry = readdir(handle); entry != nullptr;
       entry = readdir(handle)) {
    const std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    if (!HasSupportedExtension(name)) continue;
    found.push_back(JoinPath(directory, name));
  }
  closedir(handle);

  // readdir order is filesystem-dependent; sort so that repeated runs
  // process images in the same order and logs stay comparable.
  std::sort(found.begin(), found.end());
  paths->insert(paths->end(), found.begin(), found.end());
  return true;
}

bool MakeDirectories(const std::string& path, std::string* error) {
  if (path.empty() || DirectoryExists(path)) return true;

  std::string partial;
  size_t start = 0;
  if (path[0] == '/') {
    partial = "/";
    start = 1;
  }
  while (start <= path.size()) {
    const size_t slash = path.find('/', start);
    const size_t end = slash == std::string::npos ? path.size() : slash;
    const std::string component = path.substr(start, end - start);
    if (!component.empty()) {
      if (!partial.empty() && partial.back() != '/') partial += "/";
      partial += component;
      if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
        *error = "cannot create directory " + partial + ": " +
                 std::strerror(errno);
        return false;
      }
    }
    if (slash == std::string::npos) break;
    start = slash + 1;
  }
  return true;
}

}  // namespace imgpipe
