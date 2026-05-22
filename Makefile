CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = q4.o q4_metal.o
CPU_CORE_OBJS = q4_cpu.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CORE_OBJS = q4.o q4_cuda.o
CPU_CORE_OBJS = q4_cpu.o
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic

ifeq ($(UNAME_S),Darwin)
all: q4 q4-server q4-bench

help:
	@echo "Q4 build targets:"
	@echo "  make              Build Metal ./q4, ./q4-server, and ./q4-bench"
	@echo "  make cpu          Build CPU-only binaries (diagnostics only)"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

q4: q4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ q4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

q4-server: q4_server.o q4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ q4_server.o q4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

q4-bench: q4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ q4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: q4_cli_cpu.o q4_server_cpu.o q4_bench_cpu.o q4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o q4 q4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o q4-server q4_server_cpu.o q4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o q4-bench q4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "Q4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only binaries"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) q4 q4-server q4-bench CUDA_ARCH=

cuda-generic:
	$(MAKE) q4 q4-server q4-bench CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) q4 q4-server q4-bench CUDA_ARCH="$(CUDA_ARCH)"

q4: q4_cli.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

q4-server: q4_server.o q4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

q4-bench: q4_bench.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cuda-regression:
	@echo "No regression test yet"
endif

q4.o: q4.c q4.h q4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ q4.c

q4_cli.o: q4_cli.c q4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ q4_cli.c

q4_server.o: q4_server.c q4.h q4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ q4_server.c

q4_bench.o: q4_bench.c q4.h
	$(CC) $(CFLAGS) -c -o $@ q4_bench.c

q4_kvstore.o: q4_kvstore.c q4_kvstore.h q4.h
	$(CC) $(CFLAGS) -c -o $@ q4_kvstore.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

q4_cpu.o: q4.c q4.h q4_gpu.h
	$(CC) $(CFLAGS) -DQ4_NO_GPU -c -o $@ q4.c

q4_cli_cpu.o: q4_cli.c q4.h linenoise.h
	$(CC) $(CFLAGS) -DQ4_NO_GPU -c -o $@ q4_cli.c

q4_server_cpu.o: q4_server.c q4.h q4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DQ4_NO_GPU -c -o $@ q4_server.c

q4_bench_cpu.o: q4_bench.c q4.h
	$(CC) $(CFLAGS) -DQ4_NO_GPU -c -o $@ q4_bench.c

q4_metal.o: q4_metal.m q4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ q4_metal.m

q4_cuda.o: q4_cuda.cu q4_gpu.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ q4_cuda.cu

q4_test.o: tests/q4_test.c q4.c q4.h q4_gpu.h q4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/q4_test.c

q4_test: q4_test.o q4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ q4_test.o q4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ q4_test.o q4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: q4_test
	./q4_test

clean:
	rm -f q4 q4-server q4-bench q4_test *.o tests/*.o
