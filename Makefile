# Compiler and base flags
CC       := gcc
STD      := -std=c11
WARN     := -Wall -Wextra -Wpedantic
OPT      := -O3 -march=native -flto -ffast-math -funroll-loops
SIMD     := -mavx2 -mfma
LDFLAGS  := -pthread -lm

CFLAGS_SCALAR := $(STD) $(WARN) $(OPT)
CFLAGS_AVX2   := $(STD) $(WARN) $(OPT) $(SIMD)

# Auto-detect whether source files use hyphens or underscores
SRC_SCALAR := $(firstword $(wildcard terminal-marcher.c terminal_marcher.c))
SRC_AVX2   := $(firstword $(wildcard terminal-marcher-avx2.c terminal_marcher_avx2.c))

# Default fallback names if files haven't been created yet
ifeq ($(SRC_SCALAR),)
    SRC_SCALAR := terminal-marcher.c
endif
ifeq ($(SRC_AVX2),)
    SRC_AVX2 := terminal-marcher-avx2.c
endif

BIN_SCALAR := terminal-marcher
BIN_AVX2   := terminal-marcher-avx2

.PHONY: all scalar avx2 run-scalar run-avx2 asm debug clean

# Build both binaries by default
all: $(BIN_SCALAR) $(BIN_AVX2)

scalar: $(BIN_SCALAR)
avx2: $(BIN_AVX2)

$(BIN_SCALAR): $(SRC_SCALAR)
	$(CC) $(CFLAGS_SCALAR) $< -o $@ $(LDFLAGS)

$(BIN_AVX2): $(SRC_AVX2)
	$(CC) $(CFLAGS_AVX2) $< -o $@ $(LDFLAGS)

# Quick launch targets
run-scalar: $(BIN_SCALAR)
	./$(BIN_SCALAR)

run-avx2: $(BIN_AVX2)
	./$(BIN_AVX2)

# Dump annotated Intel-syntax assembly to inspect YMM register usage
asm: $(SRC_AVX2)
	$(CC) $(CFLAGS_AVX2) -S -fverbose-asm -masm=intel $< -o $(BIN_AVX2).s
	@echo "Generated Intel-syntax assembly in $(BIN_AVX2).s"

# Build with debug symbols and AddressSanitizer
debug: $(SRC_SCALAR) $(SRC_AVX2)
	$(CC) $(STD) $(WARN) -O0 -g3 -fsanitize=address,undefined $(SRC_SCALAR) -o $(BIN_SCALAR)-dbg $(LDFLAGS)
	$(CC) $(STD) $(WARN) $(SIMD) -O0 -g3 -fsanitize=address,undefined $(SRC_AVX2) -o $(BIN_AVX2)-dbg $(LDFLAGS)

clean:
	rm -f $(BIN_SCALAR) $(BIN_AVX2) $(BIN_SCALAR)-dbg $(BIN_AVX2)-dbg $(BIN_AVX2).s