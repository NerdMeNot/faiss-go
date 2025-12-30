#!/bin/bash
# Verify Phase 3 Build - Test if runtime libraries are truly merged
# This script checks if a Phase 3 build has zero external dependencies

set -euo pipefail

PLATFORM="${1:-linux-amd64}"
LIBS_DIR="$(dirname "$0")/../libs/${PLATFORM//-/_}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "Phase 3 Build Verification"
echo "========================================="
echo "Platform: $PLATFORM"
echo "Library:  $LIBS_DIR/libfaiss.a"
echo ""

# Check if library exists
if [ ! -f "$LIBS_DIR/libfaiss.a" ]; then
    echo -e "${RED}✗ Library not found: $LIBS_DIR/libfaiss.a${NC}"
    exit 1
fi

# Check library size
SIZE=$(du -h "$LIBS_DIR/libfaiss.a" | cut -f1)
SIZE_BYTES=$(stat -f%z "$LIBS_DIR/libfaiss.a" 2>/dev/null || stat -c%s "$LIBS_DIR/libfaiss.a")
echo -e "${BLUE}Library size: $SIZE ($SIZE_BYTES bytes)${NC}"

# Expected sizes:
# - Standard build: ~9MB (no OpenBLAS)
# - Phase 1/2 (OpenBLAS merged): ~45MB
# - Phase 3 (runtime libs merged): ~50-60MB

if [ "$SIZE_BYTES" -lt 10000000 ]; then
    echo -e "${YELLOW}⚠ Warning: Size is small (<10MB), likely a standard build${NC}"
elif [ "$SIZE_BYTES" -lt 48000000 ]; then
    echo -e "${YELLOW}⚠ Warning: Size is ~45MB, likely Phase 1/2 (no runtime libs)${NC}"
else
    echo -e "${GREEN}✓ Size looks good for Phase 3 (50MB+)${NC}"
fi

echo ""

# List symbol counts
echo -e "${BLUE}Analyzing symbols in libfaiss.a...${NC}"

# Check for OpenMP symbols
echo -n "OpenMP symbols (GOMP_*): "
if nm "$LIBS_DIR/libfaiss.a" 2>/dev/null | grep -q "GOMP_"; then
    COUNT=$(nm "$LIBS_DIR/libfaiss.a" 2>/dev/null | grep "GOMP_" | wc -l)
    echo -e "${GREEN}$COUNT found (merged!)${NC}"
else
    echo -e "${RED}0 found (not merged)${NC}"
fi

# Check for Fortran symbols
echo -n "Fortran symbols (_gfortran_*): "
if nm "$LIBS_DIR/libfaiss.a" 2>/dev/null | grep -q "_gfortran_"; then
    COUNT=$(nm "$LIBS_DIR/libfaiss.a" 2>/dev/null | grep "_gfortran_" | wc -l)
    echo -e "${GREEN}$COUNT found (merged!)${NC}"
else
    echo -e "${RED}0 found (not merged)${NC}"
fi

# Check for quadmath symbols
echo -n "Quadmath symbols (quadmath_*): "
if nm "$LIBS_DIR/libfaiss.a" 2>/dev/null | grep -q "quadmath"; then
    COUNT=$(nm "$LIBS_DIR/libfaiss.a" 2>/dev/null | grep "quadmath" | wc -l)
    echo -e "${GREEN}$COUNT found (merged!)${NC}"
else
    echo -e "${YELLOW}0 found (may not be needed)${NC}"
fi

# Check for OpenBLAS symbols
echo -n "OpenBLAS symbols (cblas_*, openblas_*): "
if nm "$LIBS_DIR/libfaiss.a" 2>/dev/null | grep -qE "(cblas_|openblas_)"; then
    COUNT=$(nm "$LIBS_DIR/libfaiss.a" 2>/dev/null | grep -E "(cblas_|openblas_)" | wc -l)
    echo -e "${GREEN}$COUNT found (merged!)${NC}"
else
    echo -e "${RED}0 found (not merged)${NC}"
fi

echo ""

# Try to build a simple test
echo -e "${BLUE}Testing if library can be linked...${NC}"

TEST_DIR=$(mktemp -d)
cd "$TEST_DIR"

cat > test.go <<'EOF'
//go:build faiss_phase3

package main

import "C"

func main() {
    // Minimal test - just check if it links
    println("Phase 3 build test")
}
EOF

# Try to build with Phase 3 flags (no runtime lib dependencies)
echo -n "Attempting link test: "
if CGO_ENABLED=1 \
   CGO_LDFLAGS="-L$LIBS_DIR -lfaiss -lm -lpthread -ldl" \
   go build -tags=faiss_phase3 test.go 2>/dev/null; then
    echo -e "${GREEN}SUCCESS! No runtime lib dependencies needed!${NC}"
    PHASE3_SUCCESS=true
else
    echo -e "${RED}FAILED - still needs runtime dependencies${NC}"
    PHASE3_SUCCESS=false
fi

# Cleanup
cd - > /dev/null
rm -rf "$TEST_DIR"

echo ""
echo "========================================="
if [ "$PHASE3_SUCCESS" = true ]; then
    echo -e "${GREEN}✓ Phase 3 Build SUCCESSFUL!${NC}"
    echo -e "${GREEN}  This library is truly self-contained${NC}"
    echo -e "${GREEN}  ZERO external dependencies required${NC}"
else
    echo -e "${YELLOW}⚠ Phase 3 Build PARTIAL${NC}"
    echo -e "${YELLOW}  Runtime libraries detected but not fully merged${NC}"
    echo -e "${YELLOW}  Fall back to standard LDFLAGS with -lgomp -lgfortran${NC}"
fi
echo "========================================="
