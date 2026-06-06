CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXXFLAGS := -arch=sm_75 -std=c++11 -O3
LDFLAGS := -lcufft -lcudart

# Se stai usando CUDA 11+, usa -arch=sm_86
# Per Colab, sm_75 è generalmente corretto (Tesla T4)

SOURCES := edge_detection_kernel.cu main.cu
OBJECTS := $(SOURCES:.cu=.o)
TARGET := edge_detection

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

# Target opzionale per debug
debug: CXXFLAGS := -arch=sm_75 -std=c++11 -g -G
debug: clean $(TARGET)

# Target per controllare dispositivo
check_device:
	$(NVCC) --version
	$(CUDA_PATH)/extras/demo_suite/deviceQuery
