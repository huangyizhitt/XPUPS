NVCC = nvcc
CXX = g++
RM = rm -rf
CU_FLAGS = -gencode arch=compute_70,code=compute_70 -Xcompiler -fopenmp
CU_CFLAGS = -std=c++11 -g -O3 -lrt -lpthread
CFLAGS = -std=c++11 -g -O3 -lrt -lpthread -DUSEOMP -fopenmp -lnuma 
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
PS_LIB = ./ps-lite/build/libps.a -Wl,-rpath,./ps-lite/deps/lib -L./ps-lite/deps/lib -lprotobuf-lite -lzmq
PS_INC = -I./ps-lite/include -I./ps-lite/src -I./ps-lite/deps/include 

CFLAGS += -I$(INC_DIR) 
CU_CFLAGS += -I$(INC_DIR) 

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
else ifeq ($(check), 2)
	CFLAGS += -DCAL_PORTION_RMSE 
	CU_CFLAGS += -DCAL_PORTION_RMSE
endif

ifeq ($(test), 1)
	CFLAGS += -DEXPLORE
endif

comm_op=0
ifeq ($(comm_op), 0)
	CFLAGS += -DSEND_ALL_FEATURE
else ifeq ($(comm_op), 1)
	CFLAGS += -DSEND_Q_FEATURE
else ifeq ($(comm_op), 2)
	CFLAGS += -DSEND_COMPRESS_Q_FEATURE -fpermissive
endif



mf: main.cpp $(OBJS) $(CU_OBJS)
	$(CXX) $^ -o $@ $(CFLAGS) $(ARCH_FLAGS) $(PS_INC) $(PS_LIB)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp 
	$(CXX) -c $< -o $@ $(CFLAGS) $(ARCH_FLAGS) $(PS_INC)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) -c $< -o $@ $(CU_CFLAGS) $(CU_FLAGS) $(PS_INC)
.PHONY:
clean:
	$(RM) $(OBJS) $(CU_OBJS) mf

