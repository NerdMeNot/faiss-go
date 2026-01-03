# Makefile for FAISS Go Extensions
#
# This builds the custom C API extensions that are missing from
# the standard FAISS C API.

# Detect platform
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
    OS := linux
    CXX ?= g++
    AR ?= ar
endif
ifeq ($(UNAME_S),Darwin)
    OS := darwin
    CXX ?= clang++
    AR ?= ar
endif

ifeq ($(UNAME_M),x86_64)
    ARCH := amd64
endif
ifeq ($(UNAME_M),amd64)
    ARCH := amd64
endif
ifeq ($(UNAME_M),arm64)
    ARCH := arm64
endif
ifeq ($(UNAME_M),aarch64)
    ARCH := arm64
endif

PLATFORM := $(OS)_$(ARCH)

# Paths
PROJECT_ROOT := $(shell cd .. && pwd)
LIBS_DIR := $(PROJECT_ROOT)/libs/$(PLATFORM)
FAISS_INCLUDE := $(LIBS_DIR)/include

# Compiler flags
CXXFLAGS := -std=c++17 -O3 -fPIC -I$(FAISS_INCLUDE)

ifeq ($(UNAME_S),Darwin)
    # macOS - use Accelerate framework
    CXXFLAGS += -stdlib=libc++
endif

# Source files
SOURCES := faiss_go_ext.cpp
OBJECTS := $(SOURCES:.cpp=.o)
TARGET := libfaiss_go_ext.a

.PHONY: all clean install merge

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(AR) rcs $@ $^
	ranlib $@
	@echo "Built $(TARGET) for $(PLATFORM)"

%.o: %.cpp faiss_go_ext.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

# Install: merge with existing libfaiss_c.a
install: $(TARGET)
	@echo "Merging $(TARGET) into $(LIBS_DIR)/libfaiss_c.a..."
	@mkdir -p merge_temp
	@cd merge_temp && $(AR) x $(LIBS_DIR)/libfaiss_c.a
	@cd merge_temp && $(AR) x ../$(TARGET)
	@$(AR) rcs $(LIBS_DIR)/libfaiss_c.a merge_temp/*.o
	@ranlib $(LIBS_DIR)/libfaiss_c.a
	@rm -rf merge_temp
	@cp faiss_go_ext.h $(LIBS_DIR)/include/
	@echo "Successfully merged extensions into libfaiss_c.a"
	@echo "Header installed to $(LIBS_DIR)/include/faiss_go_ext.h"

# Just merge without rebuilding
merge:
	@if [ ! -f $(TARGET) ]; then \
		echo "Error: $(TARGET) not found. Run 'make' first."; \
		exit 1; \
	fi
	@echo "Merging $(TARGET) into $(LIBS_DIR)/libfaiss_c.a..."
	@mkdir -p merge_temp
	@cd merge_temp && $(AR) x $(LIBS_DIR)/libfaiss_c.a
	@cd merge_temp && $(AR) x ../$(TARGET)
	@$(AR) rcs $(LIBS_DIR)/libfaiss_c.a merge_temp/*.o
	@ranlib $(LIBS_DIR)/libfaiss_c.a
	@rm -rf merge_temp
	@cp faiss_go_ext.h $(LIBS_DIR)/include/
	@echo "Done!"

# Verify the merged library has the new symbols
verify:
	@echo "Checking for new symbols in libfaiss_c.a..."
	@nm $(LIBS_DIR)/libfaiss_c.a 2>/dev/null | grep -E "faiss_RangeSearchResult_distances|faiss_IndexBinaryFlat_new|faiss_serialize_index" || echo "Symbols not found!"
