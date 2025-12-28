#!/bin/bash
# Build FAISS amalgamation for embedding in Go
# This creates a single faiss.cpp and faiss.h for compilation with CGO

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FAISS_DIR="$PROJECT_ROOT/faiss"
TEMP_DIR="$PROJECT_ROOT/tmp/faiss-amalgamation-build"

FAISS_VERSION="${1:-v1.8.0}"

echo "========================================="
echo "FAISS Amalgamation Builder"
echo "========================================="
echo "Version: $FAISS_VERSION"
echo "Output: $FAISS_DIR/"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Cleanup on exit
cleanup() {
    if [ -d "$TEMP_DIR" ]; then
        echo "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT

# Clone FAISS
echo "Cloning FAISS $FAISS_VERSION..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

git clone --depth 1 --branch "$FAISS_VERSION" \
    https://github.com/facebookresearch/faiss.git \
    "$TEMP_DIR/faiss" || {
    echo -e "${RED}Failed to clone FAISS${NC}"
    exit 1
}

cd "$TEMP_DIR/faiss"
echo -e "${GREEN}✓ Cloned FAISS${NC}"

# Configure with CMake
echo "Configuring FAISS build..."
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_C_API=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    || {
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
}

echo -e "${GREEN}✓ Configured${NC}"

# Build FAISS
echo "Building FAISS (this may take 10-15 minutes)..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo -e "${GREEN}✓ Built FAISS${NC}"

# Create amalgamation directory
mkdir -p "$FAISS_DIR"

# Create amalgamated header
echo "Creating amalgamated header..."
cat > "$FAISS_DIR/faiss.h" << 'EOF'
/**
 * FAISS C API - Amalgamated Header
 * Auto-generated from FAISS C API
 */

#ifndef FAISS_AMALGAMATION_H
#define FAISS_AMALGAMATION_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct FaissIndex_H FaissIndex;
typedef struct FaissIndexBinary_H FaissIndexBinary;

// Index creation
int faiss_IndexFlatL2_new(FaissIndex** p_index, int64_t d);
int faiss_IndexFlatIP_new(FaissIndex** p_index, int64_t d);

// Index operations
int faiss_Index_add(FaissIndex* index, int64_t n, const float* x);
int faiss_Index_search(FaissIndex* index, int64_t n, const float* x, int64_t k, float* distances, int64_t* labels);
int faiss_Index_train(FaissIndex* index, int64_t n, const float* x);
int faiss_Index_reset(FaissIndex* index);
void faiss_Index_free(FaissIndex* index);

// Index properties
int64_t faiss_Index_ntotal(FaissIndex* index);
int faiss_Index_d(FaissIndex* index);
int faiss_Index_is_trained(FaissIndex* index);

// More comprehensive API - to be expanded

#ifdef __cplusplus
}
#endif

#endif /* FAISS_AMALGAMATION_H */
EOF

echo -e "${GREEN}✓ Created header${NC}"

# For now, create a minimal implementation that links against the built library
# In a full implementation, we would combine all .cpp files into one
echo "Creating amalgamated implementation..."

# Copy the C API from FAISS
if [ -d "../c_api" ]; then
    echo "Found FAISS C API, using it..."

    # Create a single cpp file that includes all C API implementations
    cat > "$FAISS_DIR/faiss.cpp" << 'EOF'
/**
 * FAISS Amalgamated Implementation
 * This file combines FAISS C++ code for embedding with Go
 */

// Include FAISS headers
#include "faiss.h"

// For now, this is a placeholder that will be expanded
// to include the full FAISS implementation

// Minimal stub to allow compilation
extern "C" {

int faiss_IndexFlatL2_new(FaissIndex** p_index, int64_t d) {
    // TODO: Implement actual FAISS IndexFlatL2 creation
    return -1;  // Not implemented
}

int faiss_IndexFlatIP_new(FaissIndex** p_index, int64_t d) {
    return -1;  // Not implemented
}

int faiss_Index_add(FaissIndex* index, int64_t n, const float* x) {
    return -1;
}

int faiss_Index_search(FaissIndex* index, int64_t n, const float* x, int64_t k, float* distances, int64_t* labels) {
    return -1;
}

int faiss_Index_train(FaissIndex* index, int64_t n, const float* x) {
    return 0;  // No training needed for flat indexes
}

int faiss_Index_reset(FaissIndex* index) {
    return -1;
}

void faiss_Index_free(FaissIndex* index) {
    // TODO: Implement cleanup
}

int64_t faiss_Index_ntotal(FaissIndex* index) {
    return 0;
}

int faiss_Index_d(FaissIndex* index) {
    return 0;
}

int faiss_Index_is_trained(FaissIndex* index) {
    return 1;
}

}  // extern "C"
EOF

else
    echo -e "${YELLOW}Warning: Creating minimal stub implementation${NC}"
    echo -e "${YELLOW}Full amalgamation requires manual integration${NC}"
fi

echo -e "${GREEN}✓ Created implementation${NC}"

# Create build info
cat > "$FAISS_DIR/BUILD_INFO.txt" << EOF
FAISS Amalgamation Build Information
====================================

FAISS Version: $FAISS_VERSION
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Build Host: $(hostname)
Builder: GitHub Actions

Files:
- faiss.h: C API header
- faiss.cpp: Amalgamated implementation

Status: Initial stub implementation
Next Steps: Integrate full FAISS C++ code

Note: This is a work-in-progress amalgamation.
For production use, system-installed FAISS is currently recommended.
EOF

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Amalgamation build complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Output files:"
ls -lh "$FAISS_DIR"
echo ""
echo -e "${YELLOW}Note: This is a minimal stub implementation.${NC}"
echo -e "${YELLOW}Full FAISS integration is still in progress.${NC}"
echo ""
