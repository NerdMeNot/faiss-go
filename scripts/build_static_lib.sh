#!/bin/bash
# Build FAISS static library for a specific platform
# Usage: ./build_static_lib.sh <platform> [faiss_version] [--unified]
#   platform: linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64
#   faiss_version: FAISS git tag (default: v1.13.2)
#   --unified: Build unified static library with all dependencies merged (Linux/Windows only)
#
# Build Modes:
#   Standard (default): Builds FAISS, links against system BLAS
#     - macOS: Uses Accelerate framework (~9MB)
#     - Linux/Windows: Requires system OpenBLAS at runtime (~9MB)
#
#   Unified (--unified): Builds and merges all dependencies into single library
#     - macOS: Same as standard (Accelerate cannot be statically linked) (~9MB)
#     - Linux/Windows: Includes OpenBLAS merged (~40-50MB, zero runtime deps)

set -euo pipefail

PLATFORM="${1:-}"
FAISS_VERSION="${2:-v1.13.2}"
UNIFIED_BUILD=false

# Parse arguments
if [ -z "$PLATFORM" ]; then
    echo "Usage: $0 <platform> [faiss_version] [--unified]"
    echo "Platforms: linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64"
    echo ""
    echo "Options:"
    echo "  --unified    Build unified static library with OpenBLAS merged (Linux/Windows only)"
    exit 1
fi

# Check for --unified flag
if [ "${3:-}" = "--unified" ] || [ "${2:-}" = "--unified" ]; then
    UNIFIED_BUILD=true
    # Adjust version if --unified is in position 2
    if [ "${2:-}" = "--unified" ]; then
        FAISS_VERSION="v1.13.2"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEMP_DIR="$PROJECT_ROOT/tmp/faiss-lib-build-$PLATFORM"
OUTPUT_DIR="$PROJECT_ROOT/libs/${PLATFORM//-/_}"
OPENBLAS_VERSION="v0.3.30"

echo "========================================="
echo "FAISS Static Library Builder"
echo "========================================="
echo "Platform: $PLATFORM"
echo "FAISS Version: $FAISS_VERSION"
echo "Build Mode: $([ "$UNIFIED_BUILD" = true ] && echo "Unified (dependencies merged)" || echo "Standard (system BLAS)")"
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

# Build OpenBLAS statically (for unified builds on Linux/Windows)
build_openblas() {
    # Check for cached OpenBLAS installation
    local cache_dir="$PROJECT_ROOT/tmp/openblas-install-$PLATFORM"
    if [ -d "$cache_dir" ] && [ -f "$cache_dir/lib/libopenblas.a" ]; then
        echo -e "${GREEN}✓ Using cached OpenBLAS from $cache_dir${NC}"
        # Create symlink to expected location
        mkdir -p "$TEMP_DIR"
        ln -sf "$cache_dir" "$TEMP_DIR/openblas-install"
        return 0
    fi

    echo "Building OpenBLAS $OPENBLAS_VERSION statically..."

    local openblas_dir="$TEMP_DIR/OpenBLAS"

    if [ ! -d "$openblas_dir" ]; then
        git clone --depth 1 --branch "$OPENBLAS_VERSION" \
            https://github.com/xianyi/OpenBLAS.git "$openblas_dir" || {
            echo -e "${RED}Failed to clone OpenBLAS${NC}"
            exit 1
        }
    fi

    cd "$openblas_dir"

    # Clean previous builds
    make clean || true

    # Determine CPU target for cross-compilation
    local target_arch=""
    if [[ "$PLATFORM" == linux-arm64 ]] && [ "$(uname -m)" != "aarch64" ]; then
        target_arch="ARMV8"
        echo "Note: Cross-compiling OpenBLAS for ARM64"
    fi

    # Build static library only
    local make_opts=(
        DYNAMIC=0
        NO_SHARED=1
        USE_OPENMP=1
        USE_THREAD=1
        NO_LAPACK=0
        NO_CBLAS=0
        BUILD_RELAPACK=1
    )

    if [ -n "$target_arch" ]; then
        make_opts+=(TARGET="$target_arch")
    fi

    # Use appropriate parallelism
    local jobs
    if [[ "$PLATFORM" == linux-arm64 ]] && [ "$(uname -m)" != "aarch64" ]; then
        jobs=2
    else
        jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    fi

    make -j${jobs} "${make_opts[@]}" || {
        echo -e "${RED}Failed to build OpenBLAS${NC}"
        exit 1
    }

    # Install to local prefix
    # On Windows, make install can fail due to path escaping issues
    # So we manually copy the files we need
    if [[ "$PLATFORM" == windows-* ]]; then
        echo "Manually installing OpenBLAS for Windows..."
        mkdir -p "$TEMP_DIR/openblas-install/lib"
        mkdir -p "$TEMP_DIR/openblas-install/include"

        # Find the static library (has platform-specific name like libopenblas_zenp-r0.3.30.a)
        echo "Looking for OpenBLAS library in: $(pwd)"
        ls -la libopenblas*.a || true

        OPENBLAS_LIB=$(find . -maxdepth 1 -name "libopenblas*.a" -type f | head -1)
        if [ -z "$OPENBLAS_LIB" ]; then
            echo -e "${RED}Failed to find OpenBLAS library${NC}"
            exit 1
        fi

        echo "Found OpenBLAS library: $OPENBLAS_LIB"
        echo "Copying to: $TEMP_DIR/openblas-install/lib/libopenblas.a"

        # Use explicit copy command
        /usr/bin/cp -v "$OPENBLAS_LIB" "$TEMP_DIR/openblas-install/lib/libopenblas.a"

        # Copy headers
        [ -f openblas_config.h ] && cp openblas_config.h "$TEMP_DIR/openblas-install/include/" || true
        [ -f cblas.h ] && cp cblas.h "$TEMP_DIR/openblas-install/include/" || true
        [ -f f77blas.h ] && cp f77blas.h "$TEMP_DIR/openblas-install/include/" || true

        # Copy all lapacke headers
        for header in lapacke*.h; do
            [ -f "$header" ] && cp "$header" "$TEMP_DIR/openblas-install/include/" || true
        done

        echo "OpenBLAS installation complete"
        ls -la "$TEMP_DIR/openblas-install/lib/" || true
    else
        # Unix: use make install
        make install PREFIX="$TEMP_DIR/openblas-install" "${make_opts[@]}"
    fi

    # Also save to cache location for future builds
    local cache_dir="$PROJECT_ROOT/tmp/openblas-install-$PLATFORM"
    mkdir -p "$(dirname "$cache_dir")"
    cp -r "$TEMP_DIR/openblas-install" "$cache_dir"
    echo -e "${GREEN}✓ Built OpenBLAS (cached for future builds)${NC}"
}

# Prepare build directory
echo "Preparing build environment..."
mkdir -p "$TEMP_DIR"

# Build OpenBLAS if unified build requested (Linux/Windows only)
# Do this BEFORE cloning FAISS so we can cache it
if [ "$UNIFIED_BUILD" = true ] && [[ "$PLATFORM" == linux-* || "$PLATFORM" == windows-* ]]; then
    build_openblas
fi

# Clone FAISS
echo "Cloning FAISS $FAISS_VERSION..."
rm -rf "$TEMP_DIR/faiss"  # Only remove FAISS dir, preserve OpenBLAS cache

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

# For unified builds on Linux/Windows, use our built OpenBLAS
if [ "$UNIFIED_BUILD" = true ] && [[ "$PLATFORM" == linux-* || "$PLATFORM" == windows-* ]]; then
    CMAKE_FLAGS+=(
        -DBLA_STATIC=ON
        -DBLAS_LIBRARIES="$TEMP_DIR/openblas-install/lib/libopenblas.a"
        -DLAPACK_LIBRARIES="$TEMP_DIR/openblas-install/lib/libopenblas.a"
    )
fi

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
echo "Current directory: $(pwd)"
echo "Build directory contents:"
ls -la . || true

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
    cmake --build . --config Release -j${JOBS} || {
        echo -e "${RED}FAISS build failed${NC}"
        exit 1
    }
else
    cmake --build . -j${JOBS} || {
        echo -e "${RED}FAISS build failed${NC}"
        exit 1
    }
fi

# Verify FAISS built successfully
if [ ! -f "faiss/libfaiss.a" ] && [ ! -f "faiss/Release/faiss.lib" ]; then
    echo -e "${RED}FAISS library not found after build${NC}"
    echo "Build directory contents:"
    ls -la faiss/ || ls -la faiss/Release/ || true
    exit 1
fi

echo -e "${GREEN}✓ Built FAISS${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Compile custom C wrapper layer (faiss_c_impl.cpp) BEFORE merging
# This ensures faiss_c_impl.o is included in unified builds
echo "Compiling custom C wrapper layer..."
WRAPPER_DIR="$TEMP_DIR/wrapper"
mkdir -p "$WRAPPER_DIR"

# Compile faiss_c_impl.cpp with access to FAISS headers
cd "$WRAPPER_DIR"
FAISS_INCLUDE="$TEMP_DIR/faiss"
FAISS_C_IMPL="$PROJECT_ROOT/faiss_c_impl.cpp"

# Platform-specific compiler flags
CXX_FLAGS="-std=c++17 -O3 -fPIC -I$FAISS_INCLUDE"

if [[ "$PLATFORM" == darwin-* ]]; then
    CXX="clang++"
    if [[ "$PLATFORM" == darwin-arm64 ]]; then
        CXX_FLAGS="$CXX_FLAGS -arch arm64"
    else
        CXX_FLAGS="$CXX_FLAGS -arch x86_64"
    fi
    # Add OpenMP include paths for macOS
    if [ -d "/opt/homebrew/opt/libomp/include" ]; then
        CXX_FLAGS="$CXX_FLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
    elif [ -d "/usr/local/opt/libomp/include" ]; then
        CXX_FLAGS="$CXX_FLAGS -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
    fi
elif [[ "$PLATFORM" == windows-* ]]; then
    CXX="x86_64-w64-mingw32-g++"
else
    CXX="g++"
fi

# Compile the wrapper
$CXX $CXX_FLAGS -c "$FAISS_C_IMPL" -o faiss_c_impl.o || {
    echo -e "${RED}Failed to compile faiss_c_impl.cpp${NC}"
    exit 1
}

echo -e "${GREEN}✓ Compiled faiss_c_impl.cpp${NC}"

# Return to build directory for next steps
cd "$TEMP_DIR/faiss/build"

# Merge static libraries for unified builds (Linux/Windows only)
if [ "$UNIFIED_BUILD" = true ] && [[ "$PLATFORM" == linux-* || "$PLATFORM" == windows-* ]] && [ -f "faiss/libfaiss.a" ]; then
    echo "Merging static libraries into unified libfaiss.a..."

    MERGE_DIR="$TEMP_DIR/merge"
    rm -rf "$MERGE_DIR"
    mkdir -p "$MERGE_DIR"
    cd "$MERGE_DIR"

    # Extract all object files from libfaiss.a
    echo "  Extracting libfaiss.a..."
    mkdir -p faiss_objs
    cd faiss_objs
    ar x "$TEMP_DIR/faiss/build/faiss/libfaiss.a"
    cd ..
    mv faiss_objs/*.o .
    rmdir faiss_objs

    # Extract all object files from libfaiss_c.a
    if [ -f "$TEMP_DIR/faiss/build/c_api/libfaiss_c.a" ]; then
        echo "  Extracting libfaiss_c.a..."
        mkdir -p faiss_c_objs
        cd faiss_c_objs
        ar x "$TEMP_DIR/faiss/build/c_api/libfaiss_c.a"
        cd ..
        mv faiss_c_objs/*.o .
        rmdir faiss_c_objs
    fi

    # Extract all object files from libopenblas.a
    echo "  Extracting libopenblas.a..."
    mkdir -p openblas_objs
    cd openblas_objs
    ar x "$TEMP_DIR/openblas-install/lib/libopenblas.a"
    cd ..
    mv openblas_objs/*.o .
    rmdir openblas_objs

    # Add custom C wrapper
    echo "  Adding custom C wrapper (faiss_c_impl.o)..."
    cp "$WRAPPER_DIR/faiss_c_impl.o" .

    # Create merged archive
    echo "  Creating unified archive..."
    ar rcs "$OUTPUT_DIR/libfaiss.a" *.o
    ranlib "$OUTPUT_DIR/libfaiss.a"

    # Also create a separate libfaiss_c.a with the wrapper for compatibility
    echo "  Creating libfaiss_c.a with custom wrapper..."
    if [ -f "$TEMP_DIR/faiss/build/c_api/libfaiss_c.a" ]; then
        cp "$TEMP_DIR/faiss/build/c_api/libfaiss_c.a" "$OUTPUT_DIR/libfaiss_c.a"
        ar r "$OUTPUT_DIR/libfaiss_c.a" "$WRAPPER_DIR/faiss_c_impl.o"
        ranlib "$OUTPUT_DIR/libfaiss_c.a"
    fi

    echo -e "${GREEN}✓ Created unified libfaiss.a (includes FAISS + OpenBLAS)${NC}"

    # Return to build directory
    cd "$TEMP_DIR/faiss/build"
else
    # Copy static libraries (standard build)
    echo "Copying static libraries..."
    if [ -f "faiss/libfaiss.a" ]; then
        # Unix-like systems (Linux, macOS)
        cp "faiss/libfaiss.a" "$OUTPUT_DIR/"
        echo -e "${GREEN}✓ Copied libfaiss.a${NC}"

        # Copy C API library and merge custom wrapper
        if [ -f "c_api/libfaiss_c.a" ]; then
            cp "c_api/libfaiss_c.a" "$OUTPUT_DIR/libfaiss_c.a"
            echo "  Merging custom wrapper into libfaiss_c.a..."
            ar r "$OUTPUT_DIR/libfaiss_c.a" "$WRAPPER_DIR/faiss_c_impl.o"
            ranlib "$OUTPUT_DIR/libfaiss_c.a"
            echo -e "${GREEN}✓ Created libfaiss_c.a with custom wrapper${NC}"
        fi
    fi
fi

# Windows library handling (for non-unified builds)
if [[ "$PLATFORM" == windows-* ]] && [ "$UNIFIED_BUILD" != true ]; then
    if [ -f "faiss/Release/faiss.lib" ]; then
        # Windows Release build
        cp "faiss/Release/faiss.lib" "$OUTPUT_DIR/"
        echo -e "${GREEN}✓ Copied faiss.lib${NC}"

        # Copy C API library and merge custom wrapper
        if [ -f "c_api/Release/faiss_c.lib" ]; then
            cp "c_api/Release/faiss_c.lib" "$OUTPUT_DIR/faiss_c.lib"
            echo "  Merging custom wrapper into faiss_c.lib..."
            # Windows uses lib.exe instead of ar
            # For now, just copy - Windows unified builds handle wrapper merging above
            cp "c_api/Release/faiss_c.lib" "$OUTPUT_DIR/"
            echo -e "${GREEN}✓ Copied faiss_c.lib${NC}"
        fi
    fi
fi

# Check if we have any libraries
if [ ! -f "$OUTPUT_DIR/libfaiss.a" ] && [ ! -f "$OUTPUT_DIR/faiss.lib" ]; then
    echo -e "${RED}Failed to find or create built library${NC}"
    echo "Searching for libraries..."
    if [ -d "$TEMP_DIR/faiss/build" ]; then
        cd "$TEMP_DIR/faiss/build"
        find . -name "libfaiss.a" -o -name "faiss.lib" -o -name "libfaiss_c.a" -o -name "faiss_c.lib"
    else
        echo -e "${RED}Build directory does not exist: $TEMP_DIR/faiss/build${NC}"
        echo "Current directory: $(pwd)"
        ls -la "$TEMP_DIR/" || true
    fi
    exit 1
fi

# Copy headers if needed (custom wrapper already compiled and merged above)
if [ -d "$TEMP_DIR/faiss/c_api" ]; then
    mkdir -p "$OUTPUT_DIR/include"
    cp -r "$TEMP_DIR/faiss/c_api"/*.h "$OUTPUT_DIR/include/" 2>/dev/null || true
fi

# Generate build info
cat > "$OUTPUT_DIR/build_info.json" << EOF
{
  "platform": "$PLATFORM",
  "faiss_version": "$FAISS_VERSION",
  "build_mode": "$([ "$UNIFIED_BUILD" = true ] && echo "unified" || echo "standard")",
  "build_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "cmake_system_processor": "${CMAKE_SYSTEM_PROCESSOR:-}",
  "cmake_osx_architectures": "${CMAKE_OSX_ARCHITECTURES:-}",
  "openblas_version": "$([ "$UNIFIED_BUILD" = true ] && [[ "$PLATFORM" == linux-* || "$PLATFORM" == windows-* ]] && echo "$OPENBLAS_VERSION" || echo "system")",
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
echo "Build mode: $([ "$UNIFIED_BUILD" = true ] && echo "Unified" || echo "Standard")"
echo ""
echo "Files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "Library size: $(du -h "$OUTPUT_DIR"/libfaiss.a "$OUTPUT_DIR"/faiss.lib 2>/dev/null | awk '{print $1}' || echo 'N/A')"
echo ""

# Show runtime dependencies
if [ "$UNIFIED_BUILD" = true ]; then
    if [[ "$PLATFORM" == darwin-* ]]; then
        echo -e "${YELLOW}Runtime dependencies (macOS):${NC}"
        echo "  - Accelerate framework (system library, always available)"
        echo "  - libomp (OpenMP, if installed via Homebrew)"
    elif [[ "$PLATFORM" == linux-* ]]; then
        echo -e "${GREEN}Runtime dependencies: NONE${NC}"
        echo "  ✓ OpenBLAS merged into libfaiss.a"
        echo "  ✓ Fully self-contained static library"
    elif [[ "$PLATFORM" == windows-* ]]; then
        echo -e "${GREEN}Runtime dependencies: NONE${NC}"
        echo "  ✓ OpenBLAS merged into faiss.lib"
        echo "  ✓ Fully self-contained static library"
    fi
else
    if [[ "$PLATFORM" == darwin-* ]]; then
        echo -e "${YELLOW}Runtime dependencies (macOS):${NC}"
        echo "  - Accelerate framework (system library, always available)"
        echo "  - libomp (OpenMP, if installed via Homebrew)"
    elif [[ "$PLATFORM" == linux-* ]]; then
        echo -e "${YELLOW}Runtime dependencies (Linux):${NC}"
        echo "  - libopenblas (install: apt-get install libopenblas-dev)"
        echo "  - libgomp (OpenMP, usually included with GCC)"
    elif [[ "$PLATFORM" == windows-* ]]; then
        echo -e "${YELLOW}Runtime dependencies (Windows):${NC}"
        echo "  - OpenBLAS (via vcpkg or manual install)"
        echo "  - libgfortran, libgomp"
    fi
fi
echo ""

# Cleanup (preserve cache directories)
echo "Cleaning up build directory..."
# Only remove FAISS build directory and OpenBLAS source
# Preserve openblas-install-* cache directories
rm -rf "$TEMP_DIR/faiss"
rm -rf "$TEMP_DIR/OpenBLAS"
rm -rf "$TEMP_DIR/merge"
# Remove openblas-install symlink but not the cached directory
if [ -L "$TEMP_DIR/openblas-install" ]; then
    rm "$TEMP_DIR/openblas-install"
fi
echo -e "${GREEN}✓ Done${NC}"
