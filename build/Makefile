CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
LIBS = -pthread

SRC_DIR = ../src
TEST_DIR = ../test
BUILD_DIR = .

SRCS := $(wildcard $(SRC_DIR)/*.cc)
OBJS := $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(SRCS))

TEST_SRCS := $(wildcard $(TEST_DIR)/*.cc)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cc,$(BUILD_DIR)/%.o,$(TEST_SRCS))
TEST_TARGET := $(patsubst $(TEST_DIR)/%.cc,$(BUILD_DIR)/%,$(TEST_SRCS))

all: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJS)
$(TEST_TARGET): $(OBJS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@ 

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $< -c -o $@

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(CXX) $< $(OBJS) -o $@ $(LIBS)

clean:
	@rm -f $(OBJS) $(TEST_OBJS)