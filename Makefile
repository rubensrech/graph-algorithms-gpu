GPU=0

TARGETS=bfs floyd

ifeq ($(GPU), 1)
	TARGETS+= bfs_gpu floyd_gpu
endif

ARCH= -gencode arch=compute_60,code=sm_60 \
      -gencode arch=compute_61,code=sm_61 \
      -gencode arch=compute_62,code=[sm_62,compute_62] \
      -gencode arch=compute_70,code=[sm_70,compute_70]

CFLAGS=-O2

all: $(TARGETS)

bfs: bfs.cpp
	g++ $^ $(CFLAGS) -o $@

floyd: floyd.cpp
	g++ $^ $(CFLAGS) -o $@

bfs_gpu: bfs_gpu.cu
	nvcc $(ARCH) --compiler-options "$(CFLAGS)" $^ -o $@

floyd_gpu: floyd_gpu.cu
	nvcc $(ARCH) --compiler-options "$(CFLAGS)" $^ -o $@

clean:
	rm -f bfs floyd bfs_gpu floyd_gpu