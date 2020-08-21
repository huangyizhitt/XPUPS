NVCC = nvcc
CXX = g++
RM = rm -rf
CU_FLAGS = -gencode arch=compute_70,code=compute_70 -Xcompiler -fopenmp
CU_CFLAGS = -std=c++11 -g -O3 -lrt -lpthread
CFLAGS = -std=c++11 -g -O3 -lrt -lpthread -DUSEOMP -fopenmp 
ARCH_FLAGS = -march=native
AVX2_FLAGS = -mavx2 -DUSE_AVX2
AVX512F_FLAGS = -mavx512f -DUSE_AVX512
CU_PIC_FLAGS = -Xcompiler -fPIC
CU_LIB_FLAGS = -Xcompiler -shared $(CU_PIC_FLAGS)
PIC_FLAGS = -fPIC

SRC_DIR = ./src
INC_DIR = ./inc
OBJ_DIR = ./obj
LIB_DIR = ./lib
LIB = $(LIB_DIR)/libxpu.so
PS_LIB_DIR = ./ps-lite/build/
PS_INC = ./ps-lite/include

CFLAGS += -I$(INC_DIR) -I$(PS_INC)
CU_CFLAGS += -I$(INC_DIR) -I$(PS_INC)

SRCS = $(wildcard $(SRC_DIR)/*cpp)
OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(SRCS)))
#OBJS = $(SRCS:%.cpp=%.o)
CU_SRCS = $(wildcard $(SRC_DIR)/*cu)
CU_OBJS = $(patsubst %.cu,$(OBJ_DIR)/%.o,$(notdir $(CU_SRCS)))
#CU_OBJS = $(CU_SRCS:%.cu=%.o)

ifeq ($(arch), avx2)
	ARCH_FLAGS += $(AVX2_FLAGS)
else ifeq ($(arch), avx512f)
	ARCH_FLAGS += $(AVX512F_FLAGS)
endif

ifeq ($(use_lock), 1)
	CFLAGS += -DUSE_LOCK
endif

ifeq ($(debug), 1)
	CFLAGS += -DDEBUG
	CU_CFLAGS += -DDEBUG
endif

ifeq ($(check), 1)
	CFLAGS += -DCAL_RMSE
	CU_CFLAGS += -DCAL_RMSE
endif

$(LIB): $(OBJS) $(CU_OBJS)
	$(NVCC) $^ -o $@ $(CU_CFLAGS) $(CU_FLAGS) $(CU_LIB_FLAGS) -L$(PS_LIB_DIR) -lps

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp 
	$(CXX) -c $< -o $@ $(CFLAGS) $(ARCH_FLAGS) $(PIC_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) -c $< -o $@ $(CU_CFLAGS) $(CU_FLAGS) $(CU_PIC_FLAGS) 
.PHONY:
clean:
	$(RM) $(OBJS) $(CU_OBJS) $(LIB)

