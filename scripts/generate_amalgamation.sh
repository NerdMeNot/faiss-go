#!/bin/bash
# Generate FAISS amalgamation for embedding in Go
# This script creates a single faiss.cpp and faiss.h file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FAISS_DIR="$PROJECT_ROOT/faiss"
TEMP_DIR="$PROJECT_ROOT/tmp/faiss-build"

FAISS_VERSION="${1:-v1.8.0}"

echo "========================================="
echo "FAISS Amalgamation Generator"
echo "========================================="
echo "Version: $FAISS_VERSION"
echo "Output: $FAISS_DIR/"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dependencies
check_dependencies() {
    echo "Checking dependencies..."

    local missing=0

    if ! command -v git &> /dev/null; then
        echo -e "${RED}✗ git not found${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ git${NC}"
    fi

    if ! command -v cmake &> /dev/null; then
        echo -e "${RED}✗ cmake not found${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ cmake${NC}"
    fi

    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        echo -e "${RED}✗ C++ compiler (g++ or clang++) not found${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ C++ compiler${NC}"
    fi

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ python3 not found${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ python3${NC}"
    fi

    if [ $missing -eq 1 ]; then
        echo ""
        echo -e "${RED}Missing dependencies! Please install them and try again.${NC}"
        exit 1
    fi

    echo ""
}

# Clone FAISS repository
clone_faiss() {
    echo "Cloning FAISS repository..."

    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"

    git clone --depth 1 --branch "$FAISS_VERSION" \
        https://github.com/facebookresearch/faiss.git \
        "$TEMP_DIR/faiss" || {
        echo -e "${RED}Failed to clone FAISS${NC}"
        echo "Note: Make sure the version tag exists. Try 'main' for latest."
        exit 1
    }

    cd "$TEMP_DIR/faiss"
    ACTUAL_VERSION=$(git describe --tags --always)
    echo -e "${GREEN}✓ Cloned FAISS $ACTUAL_VERSION${NC}"
    echo ""
}

# Generate amalgamation
generate_amalgamation() {
    echo "Generating amalgamation..."
    echo "This may take 10-20 minutes..."
    echo ""

    cd "$TEMP_DIR/faiss"

    # Configure with CMake for CPU-only build
    mkdir -p build
    cd build

    echo "Configuring CMake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DFAISS_ENABLE_C_API=ON \
        || {
        echo -e "${RED}CMake configuration failed${NC}"
        exit 1
    }

    echo -e "${GREEN}✓ CMake configured${NC}"
    echo ""

    # Note: FAISS doesn't have built-in amalgamation like DuckDB
    # We'll need to create our own amalgamation script
    # For now, this script will prepare for manual amalgamation

    echo -e "${YELLOW}Note: FAISS doesn't provide automatic amalgamation.${NC}"
    echo -e "${YELLOW}We'll create a wrapper approach instead.${NC}"
    echo ""
}

# Create wrapper files
create_wrapper() {
    echo "Creating C API wrapper files..."

    WRAPPER_DIR="$FAISS_DIR"
    mkdir -p "$WRAPPER_DIR"

    # Create a minimal C API wrapper header
    cat > "$WRAPPER_DIR/faiss_c.h" << 'EOF'
/**
 * FAISS C API - Minimal wrapper for Go bindings
 * This is a simplified C API for the most common FAISS operations
 */

#ifndef FAISS_C_H
#define FAISS_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque index type */
typedef void* FaissIndex;

/* Index creation */
int faiss_IndexFlatL2_new(FaissIndex* p_index, int64_t d);
int faiss_IndexFlatIP_new(FaissIndex* p_index, int64_t d);

/* Index operations */
int faiss_Index_add(FaissIndex index, int64_t n, const float* x);
int faiss_Index_search(FaissIndex index, int64_t n, const float* x,
                       int64_t k, float* distances, int64_t* labels);
int faiss_Index_reset(FaissIndex index);
void faiss_Index_free(FaissIndex index);

/* Index properties */
int64_t faiss_Index_ntotal(FaissIndex index);
int faiss_Index_is_trained(FaissIndex index);
int faiss_Index_d(FaissIndex index);

#ifdef __cplusplus
}
#endif

#endif /* FAISS_C_H */
EOF

    # Create implementation placeholder
    cat > "$WRAPPER_DIR/faiss_c_stub.cpp" << 'EOF'
/**
 * FAISS C API Implementation - Stub for development
 *
 * This is a placeholder implementation that will be replaced with
 * actual FAISS integration once the full amalgamation is ready.
 *
 * For now, this allows the Go code to compile and provides
 * a clear path for integration.
 */

#include "faiss_c.h"
#include <cstring>
#include <vector>
#include <cmath>

// Stub implementation - returns errors for now
// This will be replaced with actual FAISS C++ API calls

extern "C" {

int faiss_IndexFlatL2_new(FaissIndex* p_index, int64_t d) {
    // TODO: Implement with actual FAISS IndexFlatL2
    // *p_index = new faiss::IndexFlatL2(d);
    *p_index = nullptr;
    return -1; // Not implemented
}

int faiss_IndexFlatIP_new(FaissIndex* p_index, int64_t d) {
    // TODO: Implement with actual FAISS IndexFlatIP
    *p_index = nullptr;
    return -1; // Not implemented
}

int faiss_Index_add(FaissIndex index, int64_t n, const float* x) {
    // TODO: Implement
    return -1;
}

int faiss_Index_search(FaissIndex index, int64_t n, const float* x,
                      int64_t k, float* distances, int64_t* labels) {
    // TODO: Implement
    return -1;
}

int faiss_Index_reset(FaissIndex index) {
    // TODO: Implement
    return -1;
}

void faiss_Index_free(FaissIndex index) {
    // TODO: Implement
}

int64_t faiss_Index_ntotal(FaissIndex index) {
    return 0;
}

int faiss_Index_is_trained(FaissIndex index) {
    return 0;
}

int faiss_Index_d(FaissIndex index) {
    return 0;
}

} // extern "C"
EOF

    echo -e "${GREEN}✓ Created wrapper files${NC}"
    echo ""
}

# Create build instructions
create_instructions() {
    cat > "$FAISS_DIR/BUILD_NOTES.md" << EOF
# FAISS Amalgamation Build Notes

## Status

⚠️ **Work in Progress**

The FAISS amalgamation is not yet complete. Currently, we have:

1. ✅ Cloned FAISS source ($ACTUAL_VERSION)
2. ✅ Created C API wrapper headers
3. ⏳ Need to integrate actual FAISS implementation

## Next Steps

### Option 1: Manual Amalgamation (Recommended)

FAISS doesn't provide automatic amalgamation like DuckDB. We need to:

1. Identify minimal required source files for CPU-only operation
2. Combine them into a single compilation unit
3. Resolve dependencies and include order

Required files (approximate):
- \`faiss/Index*.cpp\`
- \`faiss/utils/*.cpp\`
- \`faiss/impl/*.cpp\`
- Exclude: GPU files, Python bindings, tests

### Option 2: Direct Source Integration

Instead of amalgamation, we could:

1. Include all necessary FAISS source files in \`faiss/\` directory
2. List them in CGO directives
3. Let CGO compile them individually

This is simpler but results in longer compile times.

### Option 3: Use FAISS C API Directly

1. Build FAISS normally with C API enabled
2. Extract the C API wrapper
3. Create amalgamation of just the wrapper + compiled library

## Current Workaround

The stub implementation in \`faiss_c_stub.cpp\` allows the Go code to compile
but doesn't provide actual functionality yet.

## Generated on

$(date)

FAISS Version: $ACTUAL_VERSION
EOF

    echo -e "${GREEN}✓ Created build notes${NC}"
    echo ""
}

# Main execution
main() {
    check_dependencies
    clone_faiss
    generate_amalgamation
    create_wrapper
    create_instructions

    echo "========================================="
    echo -e "${GREEN}Amalgamation preparation complete!${NC}"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "1. Review $FAISS_DIR/BUILD_NOTES.md"
    echo "2. Choose an integration approach"
    echo "3. Implement actual FAISS integration"
    echo ""
    echo "FAISS source available at: $TEMP_DIR/faiss"
    echo ""
}

main "$@"
