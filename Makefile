# Thin wrapper around `zig build` and the squid5090 CLI. Override any variable:
#   make serve PORT=9000 CTX=524288
#   make serve ARGS="--ablation /path/to/ablation.safetensors"

# CUDA toolkit root: derived from nvcc on PATH, else the documented default.
NVCC      := $(shell command -v nvcc 2>/dev/null)
CUDA_PATH ?= $(if $(NVCC),$(patsubst %/bin/nvcc,%,$(realpath $(NVCC))),/usr/local/cuda)
ZIG       ?= zig

MODEL          ?= $(HOME)/models/inkling-2-mini
HOST           ?= 0.0.0.0
PORT           ?= 8000
MODE           ?= nvfp4
KV             ?= q4
CTX            ?= 1048576
MTP            ?= 7
SPEC_THRESHOLD ?= 0.80
EXACT_CTX      ?= 65536
PROMPT         ?= Write a Python LRU cache with tests.
RUNS           ?= 3
ARGS           ?=

BIN := ./zig-out/bin/squid5090
RUN := LD_LIBRARY_PATH=$(CUDA_PATH)/lib64 $(BIN)

.DEFAULT_GOAL := help
.PHONY: help build serve serve-exact bench health omp clean

help: ## Show targets
	@awk -F ':.*## ' '/^[a-z-]+:.*## /{printf "  %-8s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build zig-out/bin/squid5090
	$(ZIG) build -Dcuda-path=$(CUDA_PATH)

serve: ## Start the OpenAI-compatible API server
	$(RUN) serve $(MODEL) --host $(HOST) --port $(PORT) --mode $(MODE) --kv $(KV) --ctx $(CTX) --mtp $(MTP) --spec-threshold $(SPEC_THRESHOLD) $(ARGS)

serve-exact: ## Start the full-precision (exact BF16, mtp=0) API server
	$(RUN) serve $(MODEL) --host $(HOST) --port $(PORT) --mode exact --kv bf16 --ctx $(EXACT_CTX) --mtp 0 $(ARGS)

bench: ## Run the decode benchmark
	$(RUN) bench $(MODEL) --prompt "$(PROMPT)" --mode $(MODE) --kv $(KV) --mtp $(MTP) --max-tokens 1024 --runs $(RUNS) $(ARGS)

health: ## Check the running server
	@curl -fsS http://127.0.0.1:$(PORT)/health && echo

omp: ## Open omp on the served model (needs the squid5090 provider in omp's models.yml)
	omp --model squid5090/inkling-2-mini $(ARGS)

clean: ## Remove build output and cache
	rm -rf zig-out .zig-cache
