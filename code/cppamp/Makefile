
CLANG_AMP_BUILD_DIR ?= $(shell dirname `which clang++`/..) # where you build
CLANG_AMP_HOME ?= $(CLANG_AMP_BUILD_DIR) # where you checkout source

OPENCL_INC ?= /opt/AMDAPPSDK-3.0-0-Beta/include

CLANG_AMP ?= $(shell which clang++)
CLAMP_CONFIG ?= $(shell which clamp-config)
CLAMP_CXXFLAGS=$(shell clamp-config --build --cxxflags)
CLAMP_LDFLAGS =$(shell clamp-config --build --ldflags)

SHARED_CXXFLAGS += $(CLAMP_GPU_CXXFLAGS) -I./include -I$(OPENCL_INC)

CPP_FILES=$(wildcard *.cpp)
TARGETS=$(CPP_FILES:%.cpp=%)

all : $(TARGETS)

% : %.cpp
	$(CLANG_AMP) $(SHARED_CXXFLAGS) $(CLAMP_CXXFLAGS) $(CLAMP_LDFLAGS) $< -o $@ -g

print :
	@echo "CLANG_AMP      \t" $(CLANG_AMP)
	@echo "CLANG_CONFIG   \t" $(CLAMP_CONFIG)
	@echo "CLAMP_CXXFLAGS \t" $(CLAMP_CXXFLAGS)
	@echo "CLAMP_LDFLAGS  \t" $(CLAMP_LDFLAGS)

