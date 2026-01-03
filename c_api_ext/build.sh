#!/bin/bash
#
# Build FAISS Go Extensions
#
# This script builds the custom C API extensions and merges them
# into the existing libfaiss_c.a static library.
#
# Usage: ./build.sh [--install]
#   --install: Also merge the extensions into libfaiss_c.a

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
    linux*) OS="linux" ;;
    darwin*) OS="darwin" ;;
    *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64|amd64) ARCH="amd64" ;;
    arm64|aarch64) ARCH="arm64" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

PLATFORM="${OS}_${ARCH}"
LIBS_DIR="$PROJECT_ROOT/libs/$PLATFORM"
FAISS_VERSION="v1.8.0"

echo "Building FAISS Go Extensions for $PLATFORM"
echo "============================================"

# Check if FAISS headers are available
FAISS_HEADERS_DIR="$SCRIPT_DIR/faiss_headers"
if [ ! -d "$FAISS_HEADERS_DIR/faiss" ]; then
    echo "Downloading FAISS headers..."
    mkdir -p "$FAISS_HEADERS_DIR"
    cd "$FAISS_HEADERS_DIR"

    # Download just the header files from FAISS
    curl -sL "https://github.com/facebookresearch/faiss/archive/refs/tags/${FAISS_VERSION}.tar.gz" | \
        tar xz --strip-components=1 "faiss-${FAISS_VERSION#v}/faiss"

    echo "FAISS headers downloaded"
    cd "$SCRIPT_DIR"
fi

# Set compiler flags
if [ "$OS" = "darwin" ]; then
    CXX="${CXX:-clang++}"
    # Find OpenMP include path (from Homebrew)
    OMP_INCLUDE=""
    if [ -d "/opt/homebrew/opt/libomp/include" ]; then
        OMP_INCLUDE="-I/opt/homebrew/opt/libomp/include"
    elif [ -d "/usr/local/opt/libomp/include" ]; then
        OMP_INCLUDE="-I/usr/local/opt/libomp/include"
    fi
    CXXFLAGS="-std=c++17 -O3 -fPIC -stdlib=libc++ -I$FAISS_HEADERS_DIR -I$LIBS_DIR/include $OMP_INCLUDE"
else
    CXX="${CXX:-g++}"
    CXXFLAGS="-std=c++17 -O3 -fPIC -I$FAISS_HEADERS_DIR -I$LIBS_DIR/include"
fi

# Compile
echo "Compiling faiss_go_ext.cpp..."
$CXX $CXXFLAGS -c faiss_go_ext.cpp -o faiss_go_ext.o

# Create static library
echo "Creating libfaiss_go_ext.a..."
ar rcs libfaiss_go_ext.a faiss_go_ext.o
ranlib libfaiss_go_ext.a

echo "Built libfaiss_go_ext.a successfully"

# Install if requested
if [ "$1" = "--install" ]; then
    echo ""
    echo "Merging into $LIBS_DIR/libfaiss_c.a..."

    # Create temp directory for merging
    MERGE_DIR=$(mktemp -d)
    trap "rm -rf $MERGE_DIR" EXIT

    # Extract both libraries
    cd "$MERGE_DIR"
    ar x "$LIBS_DIR/libfaiss_c.a"
    ar x "$SCRIPT_DIR/libfaiss_go_ext.a"

    # Create merged library
    ar rcs "$LIBS_DIR/libfaiss_c.a" *.o
    ranlib "$LIBS_DIR/libfaiss_c.a"

    # Copy header
    cp "$SCRIPT_DIR/faiss_go_ext.h" "$LIBS_DIR/include/"

    echo "Merged successfully!"
    echo ""
    echo "Verifying new symbols..."
    nm "$LIBS_DIR/libfaiss_c.a" 2>/dev/null | grep -E "T _faiss_RangeSearchResult_distances|T _faiss_IndexBinaryFlat_new|T _faiss_serialize_index" | head -5 || echo "Warning: New symbols not found"
fi

echo ""
echo "Done!"
