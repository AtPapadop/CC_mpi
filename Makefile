CC = mpicc
CFLAGS = -O3 -fopenmp -Iinclude -Wall -Wextra -Wpedantic
LDFLAGS = -fopenmp
LDLIBS = -lmatio

SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin

# Find all source files
SRCS = $(wildcard $(SRC_DIR)/*.c)
# Common objects (everything except files with main)
COMMON_SRCS = $(filter-out $(SRC_DIR)/test_loaders.c $(SRC_DIR)/mpi_cc_benchmark.c, $(SRCS))
COMMON_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(COMMON_SRCS))

# Executables
TARGETS = test_loaders mpi_cc_benchmark

# Default target
all: $(TARGETS)

# Link test_loaders
test_loaders: $(OBJ_DIR)/test_loaders.o $(COMMON_OBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) -o $(BIN_DIR)/$@ $^ $(LDLIBS)

# Link mpi_cc_benchmark
mpi_cc_benchmark: $(OBJ_DIR)/mpi_cc_benchmark.o $(COMMON_OBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) -o $(BIN_DIR)/$@ $^ $(LDLIBS)

# Compile source files
$(OBJ_DIR)/mmio.o: $(SRC_DIR)/mmio.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -w -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create build directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
