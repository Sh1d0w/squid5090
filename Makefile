# squid5090 — Build, Model Download, Benchmark & OpenAI Server Makefile
SHELL := /bin/bash

CUDA_BIN   ?= /usr/local/cuda/bin
PATH       := $(CUDA_BIN):$(PATH)
export PATH

BUILD_DIR  ?= build
VENV_DIR   ?= .venv
PYTHON     ?= $(VENV_DIR)/bin/python
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

.PHONY: all build venv download serve serve-1m bench longctx clean help

all: build

help:
	@echo "squid5090 targets:"
	@echo "  make build                 Configure & compile build/squid5090 (CUDA_ARCH=$(CUDA_ARCH))"
	@echo "  make venv                  Create .venv and install Python deps (tokenizers, jinja2, huggingface_hub)"
	@echo "  make download              Download $(MODEL_REPO) into MODEL_DIR=$(MODEL_DIR)"
	@echo "  make serve                 Start OpenAI API server on http://$(HOST):$(PORT)/v1"
	@echo "  make serve-1m              Start OpenAI API server with 1M (1,048,576) Q4 KV context"
	@echo "  make bench                 Run $(RUNS)-run median benchmark ($(MAX_TOKENS) tokens)"
	@echo "  make longctx               Run long-context scaling sweep up to CTX=$(CTX)"
	@echo "  make clean                 Remove $(BUILD_DIR)/"
	@echo ""
	@echo "Common overrides: MODEL_DIR=/path/to/model PORT=8000 MODE=nvfp4|fp8|exact KV=auto|bf16|fp8|q4 CTX=65536 MTP=3"

build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=$(CUDA_ARCH)
	cmake --build $(BUILD_DIR) -j $(JOBS)

$(BUILD_DIR)/squid5090:
	$(MAKE) build

venv:
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

$(PYTHON):
	$(MAKE) venv

download: $(PYTHON)
	mkdir -p "$(MODEL_DIR)"
	$(PYTHON) -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$(MODEL_REPO)', local_dir='$(MODEL_DIR)')"
	@echo "Model downloaded to $(MODEL_DIR)"

serve: $(BUILD_DIR)/squid5090 $(PYTHON)
	$(PYTHON) tools/server.py "$(MODEL_DIR)" \
		--bin $(BUILD_DIR)/squid5090 \
		--host $(HOST) --port $(PORT) \
		--mode $(MODE) --kv $(KV) --ctx $(CTX) --mtp $(MTP)

serve-1m: $(BUILD_DIR)/squid5090 $(PYTHON)
	$(PYTHON) tools/server.py "$(MODEL_DIR)" \
		--bin $(BUILD_DIR)/squid5090 \
		--host $(HOST) --port $(PORT) \
		--mode nvfp4 --kv q4 --ctx 1048576 --mtp $(MTP)

bench: $(BUILD_DIR)/squid5090 $(PYTHON)
	$(PYTHON) tools/chat.py "$(MODEL_DIR)" \
		"Write a Python binary search function with type hints, docstring, and unit tests." \
		--bin $(BUILD_DIR)/squid5090 \
		--cmd bench --mode $(MODE) --kv $(KV) --ctx $(CTX) --mtp $(MTP) \
		--max-tokens $(MAX_TOKENS) --runs $(RUNS)

longctx: $(BUILD_DIR)/squid5090
	./$(BUILD_DIR)/squid5090 longctx "$(MODEL_DIR)" \
		--mode $(MODE) --kv $(KV) --ctx $(CTX)

clean:
	rm -rf $(BUILD_DIR)
