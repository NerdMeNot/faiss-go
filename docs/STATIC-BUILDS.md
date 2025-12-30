# Static Library Builds

> **Quick Links:**
> - [BUILD-MODES.md](BUILD-MODES.md) - Complete comparison of all build modes
> - [PHASE3-BUILDS.md](PHASE3-BUILDS.md) - Experimental zero-dependency builds
> - [libs/README.md](../libs/README.md) - Pre-built library information

This document explains the static library build system for faiss-go, including the differences between standard and unified builds across platforms.

## Overview

faiss-go provides pre-built static libraries for all major platforms. These libraries can be built in two modes:

1. **Standard Mode**: FAISS built as static library, links against system BLAS
2. **Unified Mode**: FAISS + OpenBLAS merged into single static library (Linux/Windows only)

## Build Modes Explained

### Standard Mode (Default)

```bash
./scripts/build_static_lib.sh linux-amd64 v1.13.2
```

**What it does:**
- Builds FAISS as a static library (`libfaiss.a`)
- Links against system BLAS libraries at runtime
- Smaller file size (~9MB)
- Requires BLAS libraries installed on target system

**Runtime dependencies:**
- **macOS**: Accelerate framework (system library, always available)
- **Linux**: `libopenblas`, `libgomp`
- **Windows**: OpenBLAS, libgfortran, libgomp

**Use cases:**
- Development environments where dependencies are already installed
- macOS builds (Accelerate is always available)
- Smaller download size preferred

### Unified Mode

```bash
./scripts/build_static_lib.sh linux-amd64 v1.13.2 --unified
```

**What it does:**
- Builds OpenBLAS v0.3.30 from source
- Builds FAISS with static OpenBLAS linkage
- Merges all object files into single `libfaiss.a`
- Minimal runtime dependencies (only gomp/gfortran)

**File sizes:**
- **Linux/Windows**: ~40-50MB (includes OpenBLAS)
- **macOS**: ~9MB (same as standard, Accelerate cannot be merged)

**Runtime dependencies:**
- **macOS**: Accelerate framework, libomp (same as standard mode)
- **Linux**: gomp (OpenMP), gfortran (Fortran runtime)
- **Windows**: gomp (OpenMP), gfortran (Fortran runtime)

**Use cases:**
- Production deployments
- Docker containers (minimal dependencies: gomp + gfortran)
- CI/CD environments
- Distribution to end users
- Minimal-dependency builds (no separate BLAS library needed)

## Platform-Specific Details

### macOS (darwin-amd64, darwin-arm64)

macOS builds **always use Apple's Accelerate framework** for BLAS operations, regardless of build mode.

**Why Accelerate?**
- Pre-installed on all macOS systems
- Highly optimized for Apple Silicon and Intel
- Cannot be statically linked (system framework)
- Provides BLAS and LAPACK implementations

**Build command:**
```bash
./scripts/build_static_lib.sh darwin-arm64 v1.13.2
# --unified flag has no effect on macOS
```

**Expected size:** ~9MB

**Dependencies:**
- Accelerate framework (system, always available)
- libomp (OpenMP, via Homebrew)

### Linux (linux-amd64, linux-arm64)

Linux builds benefit most from unified mode.

**Standard build:**
```bash
./scripts/build_static_lib.sh linux-amd64 v1.13.2
```
- Size: ~9MB
- Requires: `libopenblas`, `libgomp` at runtime
- Install: `apt-get install libopenblas-dev`

**Unified build (recommended):**
```bash
./scripts/build_static_lib.sh linux-amd64 v1.13.2 --unified
```
- Size: ~40-50MB
- Requires: gomp (OpenMP), gfortran (Fortran runtime)
- No separate BLAS library needed

**Build process:**
1. Clones OpenBLAS v0.3.30
2. Builds OpenBLAS as static library
3. Builds FAISS with static OpenBLAS
4. Extracts all `.o` files from both libraries
5. Merges into single `libfaiss.a` archive

### Windows (windows-amd64)

Windows unified builds work similarly to Linux.

**Standard build:**
```bash
./scripts/build_static_lib.sh windows-amd64 v1.13.2
```
- Size: ~9MB
- Requires: OpenBLAS, libgfortran, libgomp via vcpkg

**Unified build (recommended):**
```bash
./scripts/build_static_lib.sh windows-amd64 v1.13.2 --unified
```
- Size: ~40-50MB
- Requires: gomp (OpenMP), gfortran (Fortran runtime)
- No separate BLAS library needed

## GitHub Actions Workflow

The `.github/workflows/build-static-libs.yml` workflow builds libraries for all platforms.

**Current configuration:**
- **Linux AMD64**: Unified build (requires gomp + gfortran)
- **Linux ARM64**: Unified build (requires gomp + gfortran)
- **macOS Intel**: Standard build (uses Accelerate)
- **macOS ARM64**: Standard build (uses Accelerate)
- **Windows AMD64**: Unified build (requires gomp + gfortran)

**Trigger workflow:**
```bash
# Via GitHub UI: Actions → Build Static Libraries → Run workflow
# Select FAISS version and platforms
```

**Build times:**
- macOS: ~10-15 minutes
- Linux (standard): ~10-15 minutes
- Linux (unified): ~30-40 minutes (builds OpenBLAS)
- Windows (unified): ~35-45 minutes

## Custom C Wrapper Layer

The build process includes a custom C wrapper layer (`faiss_c_impl.cpp`) that provides additional C API functions beyond the official FAISS C API.

**Compilation process:**
```bash
# After building libfaiss.a and libfaiss_c.a:
1. Compile faiss_c_impl.cpp with FAISS headers
   → produces faiss_c_impl.o

2. Merge into libfaiss_c.a
   ar r libfaiss_c.a faiss_c_impl.o
   ranlib libfaiss_c.a
```

**Why this matters:**
- Provides Go-friendly wrapper functions (e.g., `faiss_IndexBinaryFlat_new`, `faiss_Kmeans_new`)
- These symbols MUST be included in final linking
- Direct `.a` file linking ensures faiss_c_impl.o is always linked
- Using `-lfaiss_c` flag could cause linker to skip these symbols

**Verification:**
```bash
# Check wrapper symbols are present
nm libs/darwin_arm64/libfaiss_c.a | grep "faiss_IndexBinaryFlat_new"
# Should show: faiss_c_impl.o: T _faiss_IndexBinaryFlat_new
```

## Build Output

After a successful build, you'll see:

```
=========================================
Build complete!
=========================================

Output directory: libs/linux_amd64
Build mode: Unified

Files:
-rw-r--r-- libfaiss.a       45M
-rw-r--r-- libfaiss_c.a    373K
-rw-r--r-- build_info.json  209B
-rw-r--r-- checksums.txt    326B

Library size: 45M

Runtime dependencies:
  - gomp (OpenMP runtime)
  - gfortran (Fortran runtime)
  ✓ OpenBLAS object code merged into libfaiss.a
  ✓ No separate BLAS library needed
```

### Build Metadata

Each build generates `build_info.json`:

```json
{
  "platform": "linux-amd64",
  "faiss_version": "v1.13.2",
  "build_mode": "unified",
  "build_date": "2025-12-29T19:08:00Z",
  "openblas_version": "v0.3.30",
  "builder": "GitHub Actions"
}
```

## Using the Static Libraries

### Standard Build Usage

**All platforms (default):**
```bash
go build -tags=nogpu
```

**Platform-specific requirements:**

**Linux:**
```bash
go build -tags=nogpu
# Requires: gomp and gfortran runtime libraries
# Install: apt-get install libgomp1 libgfortran5
```

**macOS:**
```bash
go build -tags=nogpu
# Requires: brew install libomp
# Accelerate framework automatically available
```

**Windows:**
```bash
go build -tags=nogpu
# Requires: gomp and gfortran runtime libraries (provided by MinGW)
```

### Implementation Details

All platform-specific `prebuilt_*.go` files use **direct `.a` file linking** instead of `-l` flags:

```go
// Example: prebuilt_linux_amd64.go
/*
#cgo LDFLAGS: ${SRCDIR}/libs/linux_amd64/libfaiss_c.a ${SRCDIR}/libs/linux_amd64/libfaiss.a -lgomp -lgfortran -lm -lstdc++ -lpthread -ldl
*/
```

**Why direct linking?**
- Ensures all object files from archives are included
- Guarantees custom C wrapper symbols (faiss_c_impl.o) are linked
- Avoids selective linking issues with `-l` flags

## Building Locally

### Prerequisites

**All platforms:**
- CMake 3.20+
- Git
- C++ compiler (GCC 11+, Clang 13+, or MSVC 2022)

**Unified builds (Linux/Windows):**
- gfortran (for OpenBLAS)
- make

**macOS:**
```bash
brew install cmake libomp
```

**Linux:**
```bash
apt-get install build-essential cmake git gfortran libomp-dev
```

### Building Standard

```bash
# Build for current platform
./scripts/build_static_lib.sh linux-amd64 v1.13.2

# Output: libs/linux_amd64/libfaiss.a (~9MB)
```

### Building Unified

```bash
# Build unified for Linux/Windows
./scripts/build_static_lib.sh linux-amd64 v1.13.2 --unified

# Output: libs/linux_amd64/libfaiss.a (~45MB)
```

### Cross-Compilation

**ARM64 on AMD64 (via QEMU):**
```bash
docker run --rm --platform linux/arm64 \
  -v $PWD:/workspace \
  -w /workspace \
  arm64v8/ubuntu:24.04 \
  bash -c "
    apt-get update && \
    apt-get install -y build-essential cmake git gfortran libomp-dev && \
    chmod +x scripts/build_static_lib.sh && \
    ./scripts/build_static_lib.sh linux-arm64 v1.13.2 --unified
  "
```

## Size Comparison

| Platform | Standard | Unified | Notes |
|----------|----------|---------|-------|
| linux-amd64 | ~9MB | ~45MB | Unified includes OpenBLAS |
| linux-arm64 | ~9MB | ~45MB | Unified includes OpenBLAS |
| darwin-amd64 | ~9MB | ~9MB | Both use Accelerate |
| darwin-arm64 | ~9MB | ~9MB | Both use Accelerate |
| windows-amd64 | ~9MB | ~45MB | Unified includes OpenBLAS |

## Troubleshooting

### "Failed to build OpenBLAS"

**Solution:** Ensure gfortran is installed:
```bash
# Linux
apt-get install gfortran

# macOS
brew install gcc
```

### "Cannot find libopenblas" (runtime)

**For standard builds**, install OpenBLAS:
```bash
# Linux
apt-get install libopenblas-dev

# macOS
brew install openblas  # Not needed, use unified or Accelerate
```

**Better solution:** Use unified builds to avoid runtime dependencies.

### macOS unified build same size as standard

**This is expected!** macOS cannot merge Accelerate framework into static library. Both modes produce ~9MB libraries that use Accelerate at runtime.

### Large library size on Linux/Windows

**This is expected for unified builds!** The library includes:
- FAISS (~9MB)
- OpenBLAS (~30-35MB)
- Total: ~40-50MB

This is the tradeoff for zero runtime dependencies.

## Architecture Details

### How Unified Builds Work

1. **Build OpenBLAS:**
   ```bash
   git clone https://github.com/xianyi/OpenBLAS.git
   make DYNAMIC=0 NO_SHARED=1 USE_OPENMP=1
   # Output: libopenblas.a (~35MB)
   ```

2. **Build FAISS with static OpenBLAS:**
   ```cmake
   cmake -DBLA_STATIC=ON \
         -DBLAS_LIBRARIES=/path/to/libopenblas.a \
         -DLAPACK_LIBRARIES=/path/to/libopenblas.a
   # Output: libfaiss.a (~9MB)
   ```

3. **Merge static libraries:**
   ```bash
   # Extract all object files
   ar x libfaiss.a
   ar x libfaiss_c.a
   ar x libopenblas.a

   # Create merged archive
   ar rcs libfaiss_merged.a *.o
   ranlib libfaiss_merged.a
   # Output: libfaiss_merged.a (~45MB)
   ```

### Why macOS Can't Be Unified

Apple's Accelerate framework is a **dynamic system framework**, not a static library:
- Located at: `/System/Library/Frameworks/Accelerate.framework`
- Contains: BLAS, LAPACK, and vector math operations
- Cannot be statically linked (macOS system policy)
- Always available on macOS (no installation needed)

Attempting unified build on macOS:
- OpenBLAS would be merged ✓
- But apps would still load Accelerate at runtime ✗
- Result: Wasted space, no benefit
- **Solution**: Use standard mode, rely on Accelerate

## Best Practices

### For Development
- **macOS**: Use standard build (Accelerate is always there)
- **Linux**: Use standard build (smaller, faster iteration)
- **Windows**: Use standard build if dependencies available

### For Production
- **macOS**: Use standard build (smallest, fastest)
- **Linux**: Use **unified build** (minimal dependencies)
- **Windows**: Use **unified build** (minimal dependencies)

### For Distribution
- **Always use unified builds for Linux/Windows**
- Reduces dependency complexity (no separate BLAS library)
- Users only need gomp + gfortran (commonly available)
- Easier installation than managing multiple BLAS libraries

### For CI/CD
- **Use unified builds**
- Minimal dependencies (gomp + gfortran usually pre-installed)
- No need to install libopenblas in CI environment
- Faster pipeline setup
- Reproducible builds

## FAQ

**Q: Why is macOS library same size in both modes?**
A: Accelerate framework cannot be statically linked. Both modes use Accelerate at runtime.

**Q: Should I use unified or standard for Linux?**
A: Unified for production/distribution, standard for development.

**Q: Can I use MKL instead of OpenBLAS?**
A: Yes, but requires manual modification of build script. OpenBLAS is open-source and works everywhere.

**Q: What's the performance difference?**
A: Negligible. OpenBLAS and Accelerate are both highly optimized. macOS Accelerate may be slightly faster on Apple Silicon.

**Q: Can I reduce unified build size?**
A: Not significantly. OpenBLAS is already built with `NO_LAPACK=0` (includes only essential LAPACK). Further size reduction would require removing FAISS features.

**Q: Why do unified builds still need gomp and gfortran?**
A: OpenBLAS is compiled with OpenMP parallelization and includes Fortran LAPACK code. While the object code is merged into libfaiss.a, the runtime libraries (gomp for OpenMP, gfortran for Fortran) are still required at link time.

## References

- **FAISS**: https://github.com/facebookresearch/faiss
- **OpenBLAS**: https://github.com/xianyi/OpenBLAS
- **Accelerate**: https://developer.apple.com/documentation/accelerate
- **Build Script**: `/scripts/build_static_lib.sh`
- **Workflow**: `.github/workflows/build-static-libs.yml`

## Support

For issues with static builds:
1. Check build logs for errors
2. Verify dependencies are installed
3. Review this documentation
4. Open issue at https://github.com/NerdMeNot/faiss-go/issues

Include:
- Platform (linux-amd64, darwin-arm64, etc.)
- Build mode (standard or unified)
- Full build log
- `build_info.json` contents
