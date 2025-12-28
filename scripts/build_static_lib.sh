#!/bin/bash
# Build FAISS static library for a specific platform
# Usage: ./build_static_lib.sh <platform> [faiss_version]
#   platform: linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64
#   faiss_version: FAISS git tag (default: v1.8.0)

set -euo pipefail

PLATFORM="${1:-}"
FAISS_VERSION="${2:-v1.8.0}"

if [ -z "$PLATFORM" ]; then
    echo "Usage: $0 <platform> [faiss_version]"
    echo "Platforms: linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEMP_DIR="$PROJECT_ROOT/tmp/faiss-lib-build-$PLATFORM"
OUTPUT_DIR="$PROJECT_ROOT/libs/${PLATFORM//-/_}"

echo "========================================="
echo "FAISS Static Library Builder"
echo "========================================="
echo "Platform: $PLATFORM"
echo "FAISS Version: $FAISS_VERSION"
echo "Output: $OUTPUT_DIR"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Platform-specific settings
case "$PLATFORM" in
    linux-amd64)
        CMAKE_SYSTEM_PROCESSOR="x86_64"
        ;;
    linux-arm64)
        CMAKE_SYSTEM_PROCESSOR="aarch64"
        ;;
    darwin-amd64)
        CMAKE_SYSTEM_PROCESSOR="x86_64"
        CMAKE_OSX_ARCHITECTURES="x86_64"
        ;;
    darwin-arm64)
        CMAKE_SYSTEM_PROCESSOR="arm64"
        CMAKE_OSX_ARCHITECTURES="arm64"
        ;;
    windows-amd64)
        CMAKE_SYSTEM_PROCESSOR="AMD64"
        ;;
    *)
        echo -e "${RED}Unknown platform: $PLATFORM${NC}"
        exit 1
        ;;
esac

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

# Configure
echo "Configuring FAISS build for $PLATFORM..."
mkdir -p build
cd build

CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DFAISS_ENABLE_GPU=OFF
    -DFAISS_ENABLE_PYTHON=OFF
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DFAISS_ENABLE_C_API=ON
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DFAISS_OPT_LEVEL=generic  # Use generic optimizations for faster builds
)

# Platform-specific flags
if [ -n "${CMAKE_SYSTEM_PROCESSOR:-}" ]; then
    CMAKE_FLAGS+=(-DCMAKE_SYSTEM_PROCESSOR="$CMAKE_SYSTEM_PROCESSOR")
fi

if [ -n "${CMAKE_OSX_ARCHITECTURES:-}" ]; then
    CMAKE_FLAGS+=(-DCMAKE_OSX_ARCHITECTURES="$CMAKE_OSX_ARCHITECTURES")
fi

# macOS OpenMP detection
if [[ "$PLATFORM" == darwin-* ]]; then
    # Help CMake find OpenMP on macOS (installed via Homebrew)
    if [ -d "/opt/homebrew/opt/libomp" ]; then
        # Apple Silicon (M1/M2)
        CMAKE_FLAGS+=(
            -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
            -DOpenMP_C_LIB_NAMES="omp"
            -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
            -DOpenMP_CXX_LIB_NAMES="omp"
            -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib"
        )
    elif [ -d "/usr/local/opt/libomp" ]; then
        # Intel Mac
        CMAKE_FLAGS+=(
            -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
            -DOpenMP_C_LIB_NAMES="omp"
            -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
            -DOpenMP_CXX_LIB_NAMES="omp"
            -DOpenMP_omp_LIBRARY="/usr/local/opt/libomp/lib/libomp.dylib"
        )
    fi
fi

# Windows vcpkg toolchain
if [[ "$PLATFORM" == windows-* ]]; then
    # Check for vcpkg toolchain file
    VCPKG_ROOT="${VCPKG_INSTALLATION_ROOT:-C:/vcpkg}"
    if [ -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]; then
        CMAKE_FLAGS+=(-DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")
    fi
fi

# Build
cmake .. "${CMAKE_FLAGS[@]}" || {
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
}

echo -e "${GREEN}✓ Configured${NC}"

# Build
echo "Building FAISS (this may take 10-20 minutes)..."

# Determine number of parallel jobs
if [[ "$PLATFORM" == "linux-arm64" ]] && [ "$(uname -m)" != "aarch64" ]; then
    # Cross-compiling ARM64 via QEMU - use fewer jobs to avoid QEMU overhead
    JOBS=2
    echo "Note: Using reduced parallelism (j=${JOBS}) for QEMU emulation"
else
    # Native builds can use more parallelism
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

if [[ "$PLATFORM" == windows-* ]]; then
    # Windows requires --config for multi-config generators
    cmake --build . --config Release -j${JOBS}
else
    cmake --build . -j${JOBS}
fi
echo -e "${GREEN}✓ Built FAISS${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy static libraries
echo "Copying static libraries..."
if [ -f "faiss/libfaiss.a" ]; then
    # Unix-like systems (Linux, macOS)
    cp "faiss/libfaiss.a" "$OUTPUT_DIR/"
    echo -e "${GREEN}✓ Copied libfaiss.a${NC}"

    # Copy C API library if it exists
    if [ -f "c_api/libfaiss_c.a" ]; then
        cp "c_api/libfaiss_c.a" "$OUTPUT_DIR/"
        echo -e "${GREEN}✓ Copied libfaiss_c.a${NC}"
    fi
elif [ -f "faiss/Release/faiss.lib" ]; then
    # Windows Release build
    cp "faiss/Release/faiss.lib" "$OUTPUT_DIR/"
    echo -e "${GREEN}✓ Copied faiss.lib${NC}"

    # Copy C API library if it exists
    if [ -f "c_api/Release/faiss_c.lib" ]; then
        cp "c_api/Release/faiss_c.lib" "$OUTPUT_DIR/"
        echo -e "${GREEN}✓ Copied faiss_c.lib${NC}"
    fi
else
    echo -e "${RED}Failed to find built library${NC}"
    echo "Searching for libraries..."
    find . -name "libfaiss.a" -o -name "faiss.lib" -o -name "libfaiss_c.a" -o -name "faiss_c.lib"
    exit 1
fi

# Copy headers if needed
if [ -d "../c_api" ]; then
    mkdir -p "$OUTPUT_DIR/include"
    cp -r ../c_api/*.h "$OUTPUT_DIR/include/" 2>/dev/null || true
fi

# Generate build info
cat > "$OUTPUT_DIR/build_info.json" << EOF
{
  "platform": "$PLATFORM",
  "faiss_version": "$FAISS_VERSION",
  "build_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "cmake_system_processor": "${CMAKE_SYSTEM_PROCESSOR:-}",
  "cmake_osx_architectures": "${CMAKE_OSX_ARCHITECTURES:-}",
  "builder": "GitHub Actions"
}
EOF

# Generate checksums
cd "$OUTPUT_DIR"
if command -v sha256sum >/dev/null 2>&1; then
    find . -maxdepth 1 -type f -exec sha256sum {} \; > checksums.txt
else
    find . -maxdepth 1 -type f -exec shasum -a 256 {} \; > checksums.txt
fi

# Show results
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "Library size: $(du -h "$OUTPUT_DIR"/libfaiss.a "$OUTPUT_DIR"/faiss.lib 2>/dev/null | awk '{print $1}' || echo 'N/A')"
echo ""

# Cleanup
echo "Cleaning up build directory..."
rm -rf "$TEMP_DIR"
echo -e "${GREEN}✓ Done${NC}"
