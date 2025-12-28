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
)

# Platform-specific flags
if [ -n "${CMAKE_SYSTEM_PROCESSOR:-}" ]; then
    CMAKE_FLAGS+=(-DCMAKE_SYSTEM_PROCESSOR="$CMAKE_SYSTEM_PROCESSOR")
fi

if [ -n "${CMAKE_OSX_ARCHITECTURES:-}" ]; then
    CMAKE_FLAGS+=(-DCMAKE_OSX_ARCHITECTURES="$CMAKE_OSX_ARCHITECTURES")
fi

# Build
cmake .. "${CMAKE_FLAGS[@]}" || {
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
}

echo -e "${GREEN}✓ Configured${NC}"

# Build
echo "Building FAISS (this may take 10-20 minutes)..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo -e "${GREEN}✓ Built FAISS${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy static library
echo "Copying static library..."
if [ -f "faiss/libfaiss.a" ]; then
    cp "faiss/libfaiss.a" "$OUTPUT_DIR/"
    echo -e "${GREEN}✓ Copied libfaiss.a${NC}"
elif [ -f "faiss/Release/faiss.lib" ]; then
    cp "faiss/Release/faiss.lib" "$OUTPUT_DIR/"
    echo -e "${GREEN}✓ Copied faiss.lib${NC}"
else
    echo -e "${RED}Failed to find built library${NC}"
    find . -name "libfaiss.a" -o -name "faiss.lib"
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
sha256sum * > checksums.txt 2>/dev/null || shasum -a 256 * > checksums.txt

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
