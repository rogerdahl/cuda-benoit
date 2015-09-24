BIN := editor

CUDA_INSTALL_PATH ?= /usr/local/cuda-7.5
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc

INCLUDE = -I"$(CUDA_INSTALL_PATH)/include"
INCLUDE += -I"/usr/include/GL/"
INCLUDE += -I"/usr/local/cuda-7.5/samples/common/inc/"
INCLUDE += -I"/usr/lib/x86_64-linux-gnu/wx/include/gtk2-unicode-release-2.8"
INCLUDE += -I"../../lib"
INCLUDE += -I"../../track"

LIBS += -L"$(CUDA_INSTALL_PATH)/lib64"


#-I"./"
LIBS += -lboost_program_options
LIBS += -lboost_system
LIBS += -lboost_filesystem
LIBS += -lboost_serialization
LIBS += -lGLEW
LIBS += -lglut
LIBS += -lGL
LIBS += -lcuda
LIBS += -lcudart

NVCCFLAGS := -O3 -arch=sm_30
# Device debug
#NVCCFLAGS += -G
# Host debug
#NVCCFLAGS += -g

CXXFLAGS := -O3 -g

CPP_SOURCES := $(wildcard *.cpp)
CPP_SOURCES += ../../editor/track/track.cpp
CU_SOURCES := $(wildcard *.cu)
HEADERS := $(wildcard *.h)
CPP_OBJS := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCLUDE) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDE) -o $@ $< `wx-config --cxxflags` `wx-config --libs`

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCLUDE) $(LIBS) `wx-config --cxxflags` `wx-config --libs`
	strip $(BIN)

clean:
	rm -f $(BIN) *.o *.cu_o
