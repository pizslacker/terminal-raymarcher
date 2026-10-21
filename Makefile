CC       := gcc
STD      := -std=c11
WARN     := -Wall -Wextra -Wpedantic
OPT      := -O3 -march=native -flto -ffast-math -funroll-loops
SIMD     := -mavx2 -mfma
LDFLAGS  := -pthread -lm -lxmp -lasound

CFLAGS_SCALAR := $(STD) $(WARN) $(OPT)
CFLAGS_AVX2   := $(STD) $(WARN) $(OPT) $(SIMD)

SRC_SCALAR := $(firstword $(wildcard terminal-raymarcher.c terminal_raymarcher.c))
SRC_AVX2   := $(firstword $(wildcard terminal-raymarcher-avx2.c terminal_raymarcher_avx2.c))

ifeq ($(SRC_SCALAR),)
    SRC_SCALAR := terminal-raymarcher.c
endif
ifeq ($(SRC_AVX2),)
    SRC_AVX2 := terminal-raymarcher-avx2.c
endif

# Auto-detect any .mod, .xm, .s3m, or .it file in the directory to embed
TRACK ?= $(firstword $(wildcard *.mod *.xm *.s3m *.it))
EMBED_HDR := embedded_mod.h

BIN_SCALAR := terminal-raymarcher
BIN_AVX2   := terminal-raymarcher-avx2

.PHONY: all scalar avx2 embed run-scalar run-avx2 clean

all: $(BIN_SCALAR) $(BIN_AVX2)

scalar: $(BIN_SCALAR)
avx2: $(BIN_AVX2)

# Generate embedded_mod.h from $(TRACK) or create a safe empty fallback
$(EMBED_HDR): $(TRACK)
ifneq ($(TRACK),)
	@echo "Embedding tracker module: $(TRACK) -> $(EMBED_HDR)"
	xxd -i -n embedded_mod_data "$(TRACK)" | sed 's/unsigned char/static const unsigned char/; s/unsigned int/static const unsigned int/' > $(EMBED_HDR)
else
	@echo "No .mod/.xm found; generating empty $(EMBED_HDR) fallback"
	@printf "static const unsigned char embedded_mod_data[] = {0};\nstatic const unsigned int embedded_mod_data_len = 0;\n" > $(EMBED_HDR)
endif

# Force re-embedding if you pass make embed TRACK=new_song.xm
embed:
	rm -f $(EMBED_HDR)
	$(MAKE) $(EMBED_HDR)

$(BIN_SCALAR): $(SRC_SCALAR) $(EMBED_HDR)
	$(CC) $(CFLAGS_SCALAR) $< -o $@ $(LDFLAGS)

$(BIN_AVX2): $(SRC_AVX2) $(EMBED_HDR)
	$(CC) $(CFLAGS_AVX2) $< -o $@ $(LDFLAGS)

run-avx2: $(BIN_AVX2)
	./$(BIN_AVX2)

run-scalar: $(BIN_SCALAR)
	./$(BIN_SCALAR)

clean:
	rm -f $(BIN_SCALAR) $(BIN_AVX2) $(EMBED_HDR)