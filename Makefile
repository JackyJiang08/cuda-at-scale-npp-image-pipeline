# Build rules for the CUDA NPP batch edge-detection pipeline.
#
#   make                 build bin/edge_pipeline (needs the CUDA toolkit)
#   make test            build and run the host-only tests (no GPU needed)
#   make run             build, then process data/input into data/output
#   make clean           remove build artifacts
#
# Override CUDA_PATH if the toolkit is not in /usr/local/cuda, and GENCODE to
# target a specific architecture, for example:
#   make CUDA_PATH=/opt/cuda GENCODE="-arch=sm_75"

CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc
CXX       ?= g++
STD       ?= c++14

# SASS for the common datacenter and workstation parts, plus compute_86 PTX so
# the binary still runs on newer architectures through JIT.
GENCODE ?= -gencode arch=compute_70,code=sm_70 \
           -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_86,code=compute_86

BUILD_DIR := build
BIN_DIR   := bin
TARGET    := $(BIN_DIR)/edge_pipeline
TEST_BIN  := $(BUILD_DIR)/host_tests

INCLUDES  := -Iinclude -Isrc -Ithird_party -I$(CUDA_PATH)/include
WARNINGS  := -Wall -Wextra
NVCCFLAGS := -std=$(STD) -O3 $(INCLUDES) $(GENCODE) -Xcompiler "$(WARNINGS)"
LDFLAGS   := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib

# nppc: core and stream contexts, nppisu: pitched allocation, nppicc: color
# conversion, nppif: Gaussian and Sobel filters, nppim: morphology,
# nppist: histograms.
LDLIBS    := -lnppc -lnppisu -lnppicc -lnppif -lnppim -lnppist -lcudart

HOST_SRCS := src/cli.cc src/image.cc src/otsu.cc
CUDA_SRCS := src/gpu_pipeline.cu src/kernels.cu
OBJS      := $(BUILD_DIR)/main.o \
             $(patsubst src/%.cc,$(BUILD_DIR)/%.o,$(HOST_SRCS)) \
             $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SRCS))

INPUT_DIR  ?= data/input
OUTPUT_DIR ?= data/output
STREAMS    ?= 4

.PHONY: all test run clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/%.o: src/%.cc | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# The host tests deliberately avoid CUDA so they can be run anywhere.
test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): tests/host_tests.cc $(HOST_SRCS) | $(BUILD_DIR)
	$(CXX) -std=$(STD) -O2 $(WARNINGS) -Iinclude -Ithird_party $^ -o $@

run: $(TARGET)
	./$(TARGET) --input $(INPUT_DIR) --output $(OUTPUT_DIR) \
	            --streams $(STREAMS) --verbose

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
