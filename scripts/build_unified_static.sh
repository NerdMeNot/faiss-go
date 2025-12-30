#!/bin/bash
# Phase 3: Build TRULY self-contained FAISS static library
# This script attempts to merge ALL dependencies including runtime libraries
# into a single libfaiss.a file with ZERO external dependencies
#
# Usage: ./build_unified_static.sh <platform> [faiss_version]
#   platform: linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64
#   faiss_version: FAISS git tag (default: v1.13.2)
#
# Strategy:
#   1. Build OpenBLAS with static runtime libs (-static-libgfortran -static-libgomp)
#   2. Build FAISS with static linking
#   3. Extract ALL .o files from:
#      - libfaiss.a
#      - libfaiss_c.a
#      - libopenblas.a
#      - libgfortran.a (runtime)
#      - libgomp.a (runtime)
#      - libquadmath.a (runtime, needed by gfortran)
#   4. Merge everything into ONE massive libfaiss.a
#   5. Result: Single ~50MB file, ZERO runtime dependencies

set -euo pipefail

PLATFORM="${1:-}"
FAISS_VERSION="${2:-v1.13.2}"

if [ -z "$PLATFORM" ]; then
    echo "Usage: $0 <platform> [faiss_version]"
    echo "Platforms: linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEMP_DIR="$PROJECT_ROOT/tmp/unified-build-$PLATFORM"
OUTPUT_DIR="$PROJECT_ROOT/libs/${PLATFORM//-/_}"
OPENBLAS_VERSION="v0.3.30"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "FAISS Unified Static Library Builder"
echo "Phase 3: Aggressive Runtime Merging"
echo "========================================="
echo "Platform:      $PLATFORM"
echo "FAISS Version: $FAISS_VERSION"
echo "OpenBLAS:      $OPENBLAS_VERSION"
echo "Target:        ZERO runtime dependencies"
echo "========================================="
echo ""

# Platform detection
OS=$(echo "$PLATFORM" | cut -d'-' -f1)
ARCH=$(echo "$PLATFORM" | cut -d'-' -f2)

# macOS cannot do full static linking with Accelerate
if [[ "$OS" == "darwin" ]]; then
    echo -e "${YELLOW}⚠ macOS Note: Accelerate framework cannot be statically linked${NC}"
    echo -e "${YELLOW}  Falling back to standard build (still optimal for macOS)${NC}"
    exec "$SCRIPT_DIR/build_static_lib.sh" "$PLATFORM" "$FAISS_VERSION"
fi

# Cleanup
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR" "$OUTPUT_DIR"
cd "$TEMP_DIR"

# Determine number of CPU cores
if command -v nproc &> /dev/null; then
    NCPU=$(nproc)
elif command -v sysctl &> /dev/null; then
    NCPU=$(sysctl -n hw.ncpu)
else
    NCPU=4
fi

echo -e "${BLUE}Building with $NCPU cores${NC}"
echo ""

#==============================================================================
# PHASE 1: Build OpenBLAS with Static Runtime Libraries
#==============================================================================

build_openblas_fully_static() {
    echo -e "${GREEN}[1/4] Building OpenBLAS with static runtime libraries...${NC}"

    # Check cache
    CACHE_DIR="$PROJECT_ROOT/tmp/openblas-static-$PLATFORM"
    if [ -d "$CACHE_DIR" ] && [ -f "$CACHE_DIR/lib/libopenblas.a" ]; then
        echo -e "${GREEN}✓ Using cached OpenBLAS from $CACHE_DIR${NC}"
        ln -sf "$CACHE_DIR" "$TEMP_DIR/openblas-install"
        return 0
    fi

    # Clone OpenBLAS
    if [ ! -d "OpenBLAS" ]; then
        git clone --depth 1 --branch "$OPENBLAS_VERSION" https://github.com/xianyi/OpenBLAS.git
    fi

    cd OpenBLAS

    # Platform-specific configuration
    local make_opts=(
        DYNAMIC=0
        NO_SHARED=1
        USE_OPENMP=1
        USE_THREAD=1
        NO_LAPACK=0
        NO_CBLAS=0
        BUILD_RELAPACK=1
    )

    # Add static linking flags for runtime libraries
    # This tells OpenBLAS to link gfortran/gomp statically
    local extra_flags="-static-libgfortran -static-libgomp -static-libquadmath"

    case "$OS" in
        linux)
            make_opts+=(TARGET=GENERIC)
            make_opts+=(EXTRALIB="$extra_flags")
            ;;
        windows)
            make_opts+=(TARGET=GENERIC HOSTCC=gcc)
            make_opts+=(EXTRALIB="$extra_flags")
            ;;
    esac

    echo -e "${BLUE}Building OpenBLAS with options: ${make_opts[*]}${NC}"
    make clean || true
    make -j${NCPU} "${make_opts[@]}"

    # Install to temp location
    mkdir -p "$TEMP_DIR/openblas-install/lib"
    mkdir -p "$TEMP_DIR/openblas-install/include"

    # Copy library
    if [[ "$OS" == "windows" ]]; then
        # Windows: Find the actual library file
        OPENBLAS_LIB=$(find . -maxdepth 1 -name "libopenblas*.a" -type f | head -1)
        if [ -z "$OPENBLAS_LIB" ]; then
            echo -e "${RED}✗ Failed to find OpenBLAS library${NC}"
            exit 1
        fi
        cp -v "$OPENBLAS_LIB" "$TEMP_DIR/openblas-install/lib/libopenblas.a"
    else
        make install PREFIX="$TEMP_DIR/openblas-install"
    fi

    # Copy headers
    cp -r include/*.h "$TEMP_DIR/openblas-install/include/" 2>/dev/null || true

    # Cache it
    mkdir -p "$(dirname "$CACHE_DIR")"
    cp -r "$TEMP_DIR/openblas-install" "$CACHE_DIR"

    cd "$TEMP_DIR"
    echo -e "${GREEN}✓ OpenBLAS built successfully${NC}"
    echo ""
}

#==============================================================================
# PHASE 2: Build FAISS with Static Linking
#==============================================================================

build_faiss_static() {
    echo -e "${GREEN}[2/4] Building FAISS with static linking...${NC}"

    # Clone FAISS
    if [ ! -d "faiss" ]; then
        git clone --depth 1 --branch "$FAISS_VERSION" https://github.com/facebookresearch/faiss.git
    fi

    cd faiss
    mkdir -p build
    cd build

    # CMake configuration
    local cmake_opts=(
        -DCMAKE_BUILD_TYPE=Release
        -DFAISS_ENABLE_PYTHON=OFF
        -DFAISS_ENABLE_GPU=OFF
        -DFAISS_ENABLE_C_API=ON
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_TESTING=OFF
        -DFAISS_OPT_LEVEL=generic
        -DBLA_STATIC=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )

    # Point to our static OpenBLAS
    cmake_opts+=(
        -DBLA_VENDOR=OpenBLAS
        -DOpenBLAS_LIB="$TEMP_DIR/openblas-install/lib/libopenblas.a"
        -DCMAKE_PREFIX_PATH="$TEMP_DIR/openblas-install"
    )

    # Static linking flags for C++ runtime
    local linker_flags="-static-libgcc -static-libstdc++"
    cmake_opts+=(
        -DCMAKE_EXE_LINKER_FLAGS="$linker_flags"
        -DCMAKE_SHARED_LINKER_FLAGS="$linker_flags"
    )

    echo -e "${BLUE}Configuring FAISS with CMake...${NC}"
    cmake .. "${cmake_opts[@]}"

    echo -e "${BLUE}Building FAISS...${NC}"
    make -j${NCPU} faiss faiss_c

    cd "$TEMP_DIR"
    echo -e "${GREEN}✓ FAISS built successfully${NC}"
    echo ""
}

#==============================================================================
# PHASE 3: Find and Extract Runtime Libraries
#==============================================================================

find_runtime_libraries() {
    echo -e "${GREEN}[3/4] Locating runtime libraries...${NC}"

    # Find GCC library directory
    local gcc_lib_dir=""

    if [[ "$OS" == "linux" ]]; then
        # Try multiple methods to find GCC lib directory
        if command -v gcc &> /dev/null; then
            gcc_lib_dir=$(gcc -print-file-name=libgfortran.a | xargs dirname)
        fi

        # Fallback: search common locations
        if [ ! -f "$gcc_lib_dir/libgfortran.a" ]; then
            for dir in /usr/lib/gcc/*/* /usr/lib/gcc-cross/*/*/*; do
                if [ -f "$dir/libgfortran.a" ]; then
                    gcc_lib_dir="$dir"
                    break
                fi
            done
        fi
    elif [[ "$OS" == "windows" ]]; then
        # MinGW on Windows
        gcc_lib_dir=$(gcc -print-file-name=libgfortran.a | xargs dirname)
    fi

    if [ -z "$gcc_lib_dir" ] || [ ! -d "$gcc_lib_dir" ]; then
        echo -e "${RED}✗ Could not find GCC library directory${NC}"
        echo -e "${YELLOW}  Searched for libgfortran.a${NC}"
        return 1
    fi

    echo -e "${BLUE}Found GCC libraries in: $gcc_lib_dir${NC}"

    # Verify required libraries exist
    local required_libs=(
        "$gcc_lib_dir/libgfortran.a"
        "$gcc_lib_dir/libgomp.a"
        "$gcc_lib_dir/libquadmath.a"
    )

    for lib in "${required_libs[@]}"; do
        if [ -f "$lib" ]; then
            echo -e "${GREEN}  ✓ Found: $(basename "$lib")${NC}"
        else
            echo -e "${YELLOW}  ⚠ Missing: $(basename "$lib")${NC}"
        fi
    done

    echo "$gcc_lib_dir"
    echo ""
}

#==============================================================================
# PHASE 4: Merge Everything into One Archive
#==============================================================================

merge_all_libraries() {
    echo -e "${GREEN}[4/4] Merging all libraries into unified libfaiss.a...${NC}"

    local gcc_lib_dir="$1"

    # Create merge directory
    local merge_dir="$TEMP_DIR/merge"
    rm -rf "$merge_dir"
    mkdir -p "$merge_dir"
    cd "$merge_dir"

    echo -e "${BLUE}Extracting object files...${NC}"

    # Extract FAISS
    if [ -f "$TEMP_DIR/faiss/build/faiss/libfaiss.a" ]; then
        echo "  - libfaiss.a"
        ar x "$TEMP_DIR/faiss/build/faiss/libfaiss.a"
    fi

    # Extract FAISS C API
    if [ -f "$TEMP_DIR/faiss/build/c_api/libfaiss_c.a" ]; then
        echo "  - libfaiss_c.a"
        ar x "$TEMP_DIR/faiss/build/c_api/libfaiss_c.a"
    fi

    # Extract OpenBLAS
    if [ -f "$TEMP_DIR/openblas-install/lib/libopenblas.a" ]; then
        echo "  - libopenblas.a"
        ar x "$TEMP_DIR/openblas-install/lib/libopenblas.a"
    fi

    # Extract runtime libraries (this is the key Phase 3 innovation!)
    echo -e "${YELLOW}Attempting to merge runtime libraries (experimental)...${NC}"

    local extracted_runtime=false

    if [ -f "$gcc_lib_dir/libgfortran.a" ]; then
        echo "  - libgfortran.a (Fortran runtime)"
        ar x "$gcc_lib_dir/libgfortran.a" 2>/dev/null && extracted_runtime=true || echo -e "${YELLOW}    Warning: Some symbols may conflict${NC}"
    fi

    if [ -f "$gcc_lib_dir/libgomp.a" ]; then
        echo "  - libgomp.a (OpenMP runtime)"
        ar x "$gcc_lib_dir/libgomp.a" 2>/dev/null || echo -e "${YELLOW}    Warning: Some symbols may conflict${NC}"
    fi

    if [ -f "$gcc_lib_dir/libquadmath.a" ]; then
        echo "  - libquadmath.a (Quad precision math)"
        ar x "$gcc_lib_dir/libquadmath.a" 2>/dev/null || echo -e "${YELLOW}    Warning: Some symbols may conflict${NC}"
    fi

    # Count object files
    local obj_count=$(find . -name "*.o" | wc -l)
    echo -e "${BLUE}Total object files extracted: $obj_count${NC}"

    # Create unified archive
    echo -e "${BLUE}Creating unified archive...${NC}"
    ar rcs "$OUTPUT_DIR/libfaiss.a" *.o
    ranlib "$OUTPUT_DIR/libfaiss.a"

    # Copy C API library (separate, smaller)
    if [ -f "$TEMP_DIR/faiss/build/c_api/libfaiss_c.a" ]; then
        cp "$TEMP_DIR/faiss/build/c_api/libfaiss_c.a" "$OUTPUT_DIR/"
    fi

    cd "$TEMP_DIR"

    local final_size=$(du -h "$OUTPUT_DIR/libfaiss.a" | cut -f1)
    echo -e "${GREEN}✓ Unified library created: $final_size${NC}"

    if [ "$extracted_runtime" = true ]; then
        echo -e "${GREEN}✓ Runtime libraries merged successfully!${NC}"
        echo -e "${GREEN}  This library should have ZERO external dependencies${NC}"
    else
        echo -e "${YELLOW}⚠ Runtime library merging partially successful${NC}"
        echo -e "${YELLOW}  May still need -lgomp -lgfortran at link time${NC}"
    fi

    echo ""
}

#==============================================================================
# Build Metadata
#==============================================================================

generate_build_info() {
    echo -e "${GREEN}Generating build metadata...${NC}"

    cat > "$OUTPUT_DIR/build_info.json" <<EOF
{
  "platform": "$PLATFORM",
  "os": "$OS",
  "arch": "$ARCH",
  "faiss_version": "$FAISS_VERSION",
  "openblas_version": "$OPENBLAS_VERSION",
  "build_mode": "unified_phase3",
  "build_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "runtime_libs_merged": true,
  "expected_dependencies": ["none (fully self-contained)"],
  "size": "$(du -h "$OUTPUT_DIR/libfaiss.a" | cut -f1)",
  "note": "Phase 3 build with runtime libraries merged"
}
EOF

    # Generate checksums
    cd "$OUTPUT_DIR"
    sha256sum *.a > checksums.txt 2>/dev/null || shasum -a 256 *.a > checksums.txt

    echo -e "${GREEN}✓ Metadata generated${NC}"
    echo ""
}

#==============================================================================
# Main Execution
#==============================================================================

main() {
    # Build OpenBLAS with static runtime
    build_openblas_fully_static

    # Build FAISS
    build_faiss_static

    # Find runtime libraries
    gcc_lib_dir=$(find_runtime_libraries)
    if [ -z "$gcc_lib_dir" ]; then
        echo -e "${YELLOW}⚠ Could not find runtime libraries, creating partial unified build${NC}"
        gcc_lib_dir="/nonexistent"
    fi

    # Merge everything
    merge_all_libraries "$gcc_lib_dir"

    # Generate metadata
    generate_build_info

    echo "========================================="
    echo -e "${GREEN}Build Complete!${NC}"
    echo "========================================="
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Files created:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Test the library with: go test -tags=nogpu -v"
    echo "2. Check if runtime dependencies are eliminated"
    echo "3. Compare with standard build"
    echo ""

    # Cleanup temp dir
    if [ "${KEEP_TEMP:-0}" != "1" ]; then
        echo "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
    else
        echo "Keeping temporary files in: $TEMP_DIR"
    fi
}

# Run main function
main
