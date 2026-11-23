# squid5090 — 100% Native C++/CUDA Build, Model Download, OpenAI Server & Benchmark Makefile
SHELL := /bin/bash

CUDA_BIN   ?= /usr/local/cuda/bin
PATH       := $(CUDA_BIN):$(PATH)
export PATH

BUILD_DIR  ?= build
CUDA_ARCH  ?= 120a
JOBS       ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)

MODEL_REPO ?= thinkingmachines/Inkling-2-Mini
MODEL_DIR  ?= ./models/inkling-2-mini

HOST       ?= 0.0.0.0
PORT       ?= 8000
MODE       ?= nvfp4
KV         ?= auto
CTX        ?= 65536
MTP        ?= 3
MAX_TOKENS ?= 512
RUNS       ?= 6

.PHONY: all build build-zig download serve serve-1m serve-zig bench bench-zig longctx longctx-zig clean help

all: build

help:
	@echo "squid5090 targets (100% native C++/CUDA & TigerStyle Zig — zero Python required):"
	@echo "  make build                 Configure & compile C++20 build/squid5090 (CUDA_ARCH=$(CUDA_ARCH))"
	@echo "  make build-zig             Compile TigerStyle Zig build/squid5090-zig (linked with sm_$(CUDA_ARCH) kernels)"
	@echo "  make download              Download $(MODEL_REPO) into MODEL_DIR=$(MODEL_DIR) (via huggingface-cli)"
	@echo "  make serve                 Start native C++ OpenAI API server on http://$(HOST):$(PORT)/v1"
	@echo "  make serve-zig             Start TigerStyle Zig OpenAI API server on http://$(HOST):$(PORT)/v1"
	@echo "  make serve-1m              Start native C++ OpenAI API server with 1M (1,048,576) Q4 KV context"
	@echo "  make bench                 Run $(RUNS)-run median benchmark (C++20, $(MAX_TOKENS) tokens)"
	@echo "  make bench-zig             Run $(RUNS)-run median benchmark (TigerStyle Zig, $(MAX_TOKENS) tokens)"
	@echo "  make longctx               Run long-context scaling sweep up to CTX=$(CTX) (C++20)"
	@echo "  make longctx-zig           Run long-context scaling sweep up to CTX=$(CTX) (TigerStyle Zig)"
	@echo "  make clean                 Remove $(BUILD_DIR)/"
	@echo ""
	@echo "Common overrides: MODEL_DIR=/path/to/model PORT=8000 MODE=nvfp4|fp8|exact KV=auto|bf16|fp8|q4 CTX=65536 MTP=3"

build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=$(CUDA_ARCH)
	cmake --build $(BUILD_DIR) -j $(JOBS)

$(BUILD_DIR)/squid5090:
	$(MAKE) build

NVCC_FLAGS := -O3 --use_fast_math -std=c++20 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -Isrc --expt-relaxed-constexpr
CUDA_LIB   ?= /usr/local/cuda/lib64

build-zig:
	@mkdir -p $(BUILD_DIR)/zig_obj
	nvcc $(NVCC_FLAGS) -c src/kernels_gemv.cu    -o $(BUILD_DIR)/zig_obj/kernels_gemv.o & \
	nvcc $(NVCC_FLAGS) -c src/kernels_inkling.cu -o $(BUILD_DIR)/zig_obj/kernels_inkling.o & \
	nvcc $(NVCC_FLAGS) -c src/kernels_dspark.cu  -o $(BUILD_DIR)/zig_obj/kernels_dspark.o & \
	nvcc $(NVCC_FLAGS) -c zig/src/cuda_abi.cu    -o $(BUILD_DIR)/zig_obj/cuda_abi.o & \
	wait
	ar rcs $(BUILD_DIR)/libsquid_cuda_kernels.a \
		$(BUILD_DIR)/zig_obj/kernels_gemv.o \
		$(BUILD_DIR)/zig_obj/kernels_inkling.o \
		$(BUILD_DIR)/zig_obj/kernels_dspark.o \
		$(BUILD_DIR)/zig_obj/cuda_abi.o
	zig build-exe zig/src/main.zig $(BUILD_DIR)/libsquid_cuda_kernels.a \
		-O ReleaseFast -lc -lstdc++ -lm \
		-L$(CUDA_LIB) -lcudart -lcublas -lpthread \
		-femit-bin=$(BUILD_DIR)/squid5090-zig

$(BUILD_DIR)/squid5090-zig:
	$(MAKE) build-zig

download:
	mkdir -p "$(MODEL_DIR)"
	huggingface-cli download $(MODEL_REPO) --local-dir "$(MODEL_DIR)"
	@echo "Model downloaded to $(MODEL_DIR)"

serve: $(BUILD_DIR)/squid5090
	./$(BUILD_DIR)/squid5090 serve "$(MODEL_DIR)" \
		--host $(HOST) --port $(PORT) \
		--mode $(MODE) --kv $(KV) --ctx $(CTX) --mtp $(MTP)

serve-zig: $(BUILD_DIR)/squid5090-zig
	./$(BUILD_DIR)/squid5090-zig serve "$(MODEL_DIR)" \
		--host $(HOST) --port $(PORT) \
		--mode $(MODE) --kv $(KV) --ctx $(CTX) --mtp $(MTP)

serve-1m: $(BUILD_DIR)/squid5090
	./$(BUILD_DIR)/squid5090 serve "$(MODEL_DIR)" \
		--host $(HOST) --port $(PORT) \
		--mode nvfp4 --kv q4 --ctx 1048576 --mtp $(MTP)

bench: $(BUILD_DIR)/squid5090
	./$(BUILD_DIR)/squid5090 bench "$(MODEL_DIR)" \
		--prompt "Write a Python binary search function with type hints, docstring, and unit tests." \
		--mode $(MODE) --kv $(KV) --ctx $(CTX) --mtp $(MTP) \
		--max-tokens $(MAX_TOKENS) --runs $(RUNS)

bench-zig: $(BUILD_DIR)/squid5090-zig
	./$(BUILD_DIR)/squid5090-zig bench "$(MODEL_DIR)" \
		--prompt "Write a Python binary search function with type hints, docstring, and unit tests." \
		--mode $(MODE) --kv $(KV) --ctx $(CTX) --mtp $(MTP) \
		--max-tokens $(MAX_TOKENS) --runs $(RUNS)

longctx: $(BUILD_DIR)/squid5090
	./$(BUILD_DIR)/squid5090 longctx "$(MODEL_DIR)" \
		--mode $(MODE) --kv $(KV) --ctx $(CTX)

longctx-zig: $(BUILD_DIR)/squid5090-zig
	./$(BUILD_DIR)/squid5090-zig longctx "$(MODEL_DIR)" \
		--mode $(MODE) --kv $(KV) --ctx $(CTX)

clean:
	rm -rf $(BUILD_DIR)
