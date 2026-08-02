CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
TARGET := held_karp_solver

.PHONY: all clean

all: $(TARGET)

$(TARGET): held_karp.cpp main.cpp held_karp.hpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) held_karp.cpp main.cpp

clean:
	rm -f $(TARGET)
