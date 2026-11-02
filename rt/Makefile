CXX ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
BUILD := build

all: $(BUILD)/bench

$(BUILD)/bench: bench.cpp decoders.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ bench.cpp

clean:
	rm -rf $(BUILD)

.PHONY: all clean
