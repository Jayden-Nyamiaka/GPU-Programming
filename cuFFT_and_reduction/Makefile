# CS 179 Lab 3 Unix Makefile
# Written by Loko Kung, 2018
# Edited by Tyler Port, 2018

# Product Names
CUDA_OBJ = cuda.o

# Input Names
CUDA_FILES = src/fft_convolve.cu
CPP_FILES = src/fft_convolve.cpp src/ta_utilities.cpp

# ------------------------------------------------------------------------------

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_INCLUDE =
NVCC_LIBS = 
NVCC_GENCODES = --gpu-architecture=sm_52

# CUDA Object Files
CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(CUDA_FILES)))

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++0x -pthread
INCLUDE = -I$(CUDA_INC_PATH)
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lsndfile

# ------------------------------------------------------------------------------
# Make Rules (Lab 3 specific)
# ------------------------------------------------------------------------------

# C++ Object Files
OBJ_AUDIO = $(addprefix audio-, $(notdir $(addsuffix .o, $(CPP_FILES))))
OBJ_NOAUDIO = $(addprefix noaudio-, $(notdir $(addsuffix .o, $(CPP_FILES))))

# Top level rules
all: audio noaudio

audio: $(OBJ_AUDIO) $(CUDA_OBJ) $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o audio-fft $(INCLUDE) $^ $(LIBS) 

noaudio: $(OBJ_NOAUDIO) $(CUDA_OBJ) $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o noaudio-fft $(INCLUDE) $^ $(LIBS) 


# Compile C++ Source Files
audio-%.cpp.o: src/%.cpp
	$(GPP) $(FLAGS) -D AUDIO_ON=1 -c -o $@ $(INCLUDE) $< 

noaudio-%.cpp.o: src/%.cpp
	$(GPP) $(FLAGS) -D AUDIO_ON=0 -c -o $@ $(INCLUDE) $< 


# Compile CUDA Source Files
%.cu.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

cuda: $(CUDA_OBJ_FILES) $(CUDA_OBJ)

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^


# Clean everything including temporary Emacs files
clean:
	rm -f audio-fft noaudio-fft *.o *~
	rm -f src/*~

.PHONY: clean
