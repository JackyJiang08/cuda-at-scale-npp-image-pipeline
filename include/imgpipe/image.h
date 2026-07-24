// Host-side image container and 8-bit image file I/O helpers.
//
// Images are stored row-major with interleaved channels and no padding.
// Only 8-bit-per-channel data is supported, with either one channel
// (grayscale) or three channels (RGB).

#ifndef INCLUDE_IMGPIPE_IMAGE_H_
#define INCLUDE_IMGPIPE_IMAGE_H_

#include <cstddef>
#include <string>
#include <vector>

namespace imgpipe {

// A decoded 8-bit image held in host memory.
struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<unsigned char> pixels;

  bool IsEmpty() const { return width <= 0 || height <= 0 || channels <= 0; }

  // Number of bytes in one row of pixel data.
  size_t RowBytes() const {
    return static_cast<size_t>(width) * static_cast<size_t>(channels);
  }

  // Total number of bytes of pixel data.
  size_t SizeBytes() const { return RowBytes() * static_cast<size_t>(height); }

  // Resizes the pixel buffer to hold a new_width x new_height image with
  // new_channels channels. Existing contents are not preserved.
  void Allocate(int new_width, int new_height, int new_channels);
};

// Decodes the image at `path` into `*image`. Grayscale and grayscale+alpha
// sources are loaded as one channel; every other source is loaded as RGB.
// Returns false and fills `*error` if the file cannot be decoded.
bool ReadImage(const std::string& path, Image* image, std::string* error);

// Encodes `image` as a PNG file at `path`. Returns false and fills `*error`
// on failure.
bool WritePng(const std::string& path, const Image& image, std::string* error);

// Appends the paths of all files in `directory` whose extension is a
// supported image format to `*paths`, sorted lexicographically so that runs
// are reproducible. Returns false and fills `*error` if the directory cannot
// be opened.
bool ListImageFiles(const std::string& directory,
                    std::vector<std::string>* paths, std::string* error);

// Returns true if `path` has an extension this program can decode.
bool HasSupportedExtension(const std::string& path);

// Returns the file name of `path` without its directory or extension.
std::string Stem(const std::string& path);

// Creates `path` if it does not already exist, including parent directories.
// Returns false and fills `*error` on failure.
bool MakeDirectories(const std::string& path, std::string* error);

// Joins two path components with a single separator.
std::string JoinPath(const std::string& directory, const std::string& name);

}  // namespace imgpipe

#endif  // INCLUDE_IMGPIPE_IMAGE_H_
