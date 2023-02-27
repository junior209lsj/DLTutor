CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra
LIBS = -pthread

SRC_DIR = ../src
TEST_DIR = ../test
BUILD_DIR = .

SRCS := $(wildcard $(TEST_DIR)/*.cc)
OBJS := $(patsubst $(TEST_DIR)/%.cc,$(BUILD_DIR)/%,$(SRCS))

.PHONY: all clean

all: $(OBJS)

$(BUILD_DIR)/%: $(TEST_DIR)/%.cc
	$(CXX) $(CXXFLAGS) $(LIBS) -I$(SRC_DIR) $< -o $@

clean:
	@rm -f $(OBJS)