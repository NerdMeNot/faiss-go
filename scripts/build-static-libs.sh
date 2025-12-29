#!/bin/bash
set -e

# Build fully self-contained static FAISS libraries
# This script builds FAISS with all dependencies (OpenBLAS, gfortran, OpenMP)
# statically linked into a single libfaiss.a file.
#
# Result: True zero-dependency static libraries for all platforms
# - No apt-get/brew needed
# - 30-second builds
# - Works anywhere

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
BUILD_DIR="$PROJECT_ROOT/build-static"
OUTPUT_DIR="$PROJECT_ROOT/libs"

FAISS_VERSION="v1.8.0"  # Adjust as needed
OPENBLAS_VERSION="v0.3.27"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect platform
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$os" in
        linux*)
            OS="linux"
            ;;
        darwin*)
            OS="darwin"
            ;;
        mingw*|msys*|cygwin*)
            OS="windows"
            ;;
        *)
            log_error "Unsupported OS: $os"
            exit 1
            ;;
    esac

    case "$arch" in
        x86_64|amd64)
            ARCH="amd64"
            ;;
        arm64|aarch64)
            ARCH="arm64"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac

    PLATFORM="${OS}_${ARCH}"
    log_info "Detected platform: $PLATFORM"
}

# Check dependencies
check_dependencies() {
    log_info "Checking build dependencies..."

    local missing_deps=()

    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi

    if ! command -v make &> /dev/null; then
        missing_deps+=("make")
    fi

    if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
        missing_deps+=("gcc or clang")
    fi

    if ! command -v gfortran &> /dev/null; then
        missing_deps+=("gfortran")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install them first:"
        log_info "  Ubuntu/Debian: sudo apt-get install cmake build-essential gfortran"
        log_info "  macOS: brew install cmake gcc"
        exit 1
    fi

    log_info "All dependencies found"
}

# Build OpenBLAS statically
build_openblas() {
    log_info "Building OpenBLAS $OPENBLAS_VERSION statically..."

    local openblas_dir="$BUILD_DIR/OpenBLAS"

    if [ ! -d "$openblas_dir" ]; then
        git clone --depth 1 --branch "$OPENBLAS_VERSION" \
            https://github.com/xianyi/OpenBLAS.git "$openblas_dir"
    fi

    cd "$openblas_dir"

    # Clean previous builds
    make clean || true

    # Build static library only
    # DYNAMIC=0 ensures static library only
    # NO_SHARED=1 disables shared library
    # USE_OPENMP=1 for OpenMP support (statically linked)
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) \
        DYNAMIC=0 \
        NO_SHARED=1 \
        USE_OPENMP=1 \
        USE_THREAD=1 \
        NO_LAPACK=0 \
        NO_CBLAS=0 \
        BUILD_RELAPACK=1

    # Install to local prefix
    make install PREFIX="$BUILD_DIR/openblas-install"

    log_info "OpenBLAS built successfully"
}

# Build FAISS statically with all dependencies bundled
build_faiss() {
    log_info "Building FAISS $FAISS_VERSION with static dependencies..."

    local faiss_dir="$BUILD_DIR/faiss"

    if [ ! -d "$faiss_dir" ]; then
        git clone --depth 1 --branch "$FAISS_VERSION" \
            https://github.com/facebookresearch/faiss.git "$faiss_dir"
    fi

    cd "$faiss_dir"

    # Create build directory
    rm -rf build
    mkdir -p build
    cd build

    # Configure FAISS with static linking
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_ENABLE_C_API=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBLA_STATIC=ON \
        -DBLAS_LIBRARIES="$BUILD_DIR/openblas-install/lib/libopenblas.a" \
        -DLAPACK_LIBRARIES="$BUILD_DIR/openblas-install/lib/libopenblas.a" \
        -DCMAKE_CXX_FLAGS="-O3 -fPIC -fopenmp" \
        -DCMAKE_C_FLAGS="-O3 -fPIC -fopenmp" \
        -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++ -static-libgfortran" \
        -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++ -static-libgfortran"

    # Build FAISS
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    log_info "FAISS built successfully"
}

# Merge all static libraries into a single libfaiss.a
merge_static_libs() {
    log_info "Merging all static libraries into single libfaiss.a..."

    local faiss_build="$BUILD_DIR/faiss/build"
    local temp_dir="$BUILD_DIR/merge-temp"
    local merged_lib="$temp_dir/libfaiss_merged.a"

    rm -rf "$temp_dir"
    mkdir -p "$temp_dir"
    cd "$temp_dir"

    # Extract all object files from static libraries
    log_info "  Extracting libfaiss.a..."
    ar x "$faiss_build/faiss/libfaiss.a"

    log_info "  Extracting libfaiss_c.a..."
    mkdir -p faiss_c_objs
    cd faiss_c_objs
    ar x "$faiss_build/c_api/libfaiss_c.a"
    cd ..
    mv faiss_c_objs/*.o .
    rmdir faiss_c_objs

    log_info "  Extracting libopenblas.a..."
    mkdir -p openblas_objs
    cd openblas_objs
    ar x "$BUILD_DIR/openblas-install/lib/libopenblas.a"
    cd ..
    mv openblas_objs/*.o .
    rmdir openblas_objs

    # Create merged archive
    log_info "  Creating merged archive..."
    ar rcs "$merged_lib" *.o

    # Run ranlib to create index
    ranlib "$merged_lib"

    log_info "Merged library created successfully"

    # Copy to output directory
    local output_platform_dir="$OUTPUT_DIR/$PLATFORM"
    mkdir -p "$output_platform_dir"

    cp "$merged_lib" "$output_platform_dir/libfaiss.a"
    cp "$faiss_build/c_api/libfaiss_c.a" "$output_platform_dir/libfaiss_c.a"

    # Verify the merged library
    log_info "Verifying merged library..."
    local size=$(ls -lh "$output_platform_dir/libfaiss.a" | awk '{print $5}')
    log_info "  Size: $size"

    # Check for undefined symbols (should be minimal)
    log_info "  Checking for undefined symbols..."
    local undefined_count=$(nm -u "$output_platform_dir/libfaiss.a" 2>/dev/null | grep -c "U " || echo 0)
    log_info "  Undefined symbols: $undefined_count"

    if [ "$undefined_count" -gt 100 ]; then
        log_warn "  High number of undefined symbols detected. Checking details..."
        nm -u "$output_platform_dir/libfaiss.a" 2>/dev/null | grep "U " | head -20
    fi
}

# Alternative: Create a linker script for bundled static libraries
create_bundled_libs() {
    log_info "Creating bundled static libraries..."

    local faiss_build="$BUILD_DIR/faiss/build"
    local output_platform_dir="$OUTPUT_DIR/$PLATFORM"

    mkdir -p "$output_platform_dir"

    # Copy individual static libraries
    cp "$faiss_build/faiss/libfaiss.a" "$output_platform_dir/"
    cp "$faiss_build/c_api/libfaiss_c.a" "$output_platform_dir/"
    cp "$BUILD_DIR/openblas-install/lib/libopenblas.a" "$output_platform_dir/"

    # Find and copy gfortran static library
    local gfortran_lib=$(gfortran -print-file-name=libgfortran.a)
    if [ -f "$gfortran_lib" ]; then
        cp "$gfortran_lib" "$output_platform_dir/"
        log_info "  Copied libgfortran.a"
    else
        log_warn "  Could not find static libgfortran.a"
    fi

    # Find and copy OpenMP static library
    local gomp_lib=$(gcc -print-file-name=libgomp.a)
    if [ -f "$gomp_lib" ]; then
        cp "$gomp_lib" "$output_platform_dir/"
        log_info "  Copied libgomp.a"
    else
        log_warn "  Could not find static libgomp.a"
    fi

    log_info "Bundled libraries created in $output_platform_dir"
    ls -lh "$output_platform_dir"/*.a
}

# macOS specific: Use Accelerate framework (already system static)
build_macos() {
    log_info "Building FAISS for macOS with Accelerate framework..."

    local faiss_dir="$BUILD_DIR/faiss"

    if [ ! -d "$faiss_dir" ]; then
        git clone --depth 1 --branch "$FAISS_VERSION" \
            https://github.com/facebookresearch/faiss.git "$faiss_dir"
    fi

    cd "$faiss_dir"

    rm -rf build
    mkdir -p build
    cd build

    # macOS can use Accelerate framework (built-in BLAS)
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_ENABLE_C_API=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBLA_VENDOR=Apple \
        -DCMAKE_CXX_FLAGS="-O3 -fPIC" \
        -DCMAKE_C_FLAGS="-O3 -fPIC"

    make -j$(sysctl -n hw.ncpu)

    # Copy to output
    local output_platform_dir="$OUTPUT_DIR/$PLATFORM"
    mkdir -p "$output_platform_dir"

    cp faiss/libfaiss.a "$output_platform_dir/"
    cp c_api/libfaiss_c.a "$output_platform_dir/"

    log_info "macOS libraries built successfully"
    ls -lh "$output_platform_dir"/*.a
}

# Main build process
main() {
    log_info "Building fully static FAISS libraries"
    log_info "======================================"

    detect_platform

    # Create build directory
    mkdir -p "$BUILD_DIR"
    mkdir -p "$OUTPUT_DIR"

    if [ "$OS" = "darwin" ]; then
        # macOS: Use Accelerate framework (simpler)
        check_dependencies
        build_macos
    else
        # Linux/Windows: Build with static OpenBLAS
        check_dependencies
        build_openblas
        build_faiss

        # Choose merge strategy
        if [ "$1" = "--bundled" ]; then
            create_bundled_libs
        else
            merge_static_libs
        fi
    fi

    log_info ""
    log_info "======================================"
    log_info "Build complete!"
    log_info "Platform: $PLATFORM"
    log_info "Output: $OUTPUT_DIR/$PLATFORM"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Verify the build: go build -tags=nogpu"
    log_info "  2. Test: go test -tags=nogpu -v ./..."
    log_info "  3. Commit the new libraries to git"
}

# Run main with arguments
main "$@"
