
#NVCC specific flags
CUDA_FLAGS += --std=c++11 -m64 

#adapt to your architecture
ARCH_FLAGS ?= -gencode arch=compute_20,code=sm_20

NVCC ?= $(shell which nvcc)
SRC_FILES=$(wildcard *.cu)
DST_FILES=$(SRC_FILES:%.cu=%)

all : $(DST_FILES)

% : %.cu
	$(NVCC) $(CUDA_FLAGS) $(ARCH_FLAGS) $< -o $@

