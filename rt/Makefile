CXX ?= clang++
# Portable by default. `make NATIVE=1` adds -march=native (x86) / -mcpu=native (arm64) so the
# compiler can use the host's full SIMD width; the mat-vecs are simple loops it auto-vectorises.
# No -ffast-math: the parity test against Python expects IEEE float32 arithmetic.
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
ifeq ($(NATIVE),1)
  ifeq ($(shell uname -m),arm64)
    CXXFLAGS += -mcpu=native
  else
    CXXFLAGS += -march=native
  endif
endif
BUILD := build

all: $(BUILD)/bench

$(BUILD)/bench: bench.cpp decoders.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ bench.cpp

clean:
	rm -rf $(BUILD)

.PHONY: all clean
