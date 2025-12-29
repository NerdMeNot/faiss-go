# Building Fully Static FAISS Libraries

This document explains how to build fully self-contained static FAISS libraries for faiss-go.

## Overview

**Goal**: Create static libraries where all dependencies (OpenBLAS, gfortran, OpenMP) are bundled into `libfaiss.a`, achieving true zero-dependency builds.

**Current State** (as of Dec 2025):
- ⚠️ Existing static libraries still require runtime dependencies (libopenblas, libgfortran, libgomp)
- 🎯 Target: Fully self-contained libraries with everything bundled

**Benefits of Fully Static Libraries**:
- ✅ True zero dependencies - no apt-get/brew needed
- ✅ 30-second builds anywhere
- ✅ Simplified CI (no dependency installation steps)
- ✅ Single source of truth for all dependencies
- ✅ Consistent BLAS implementation across platforms

## Quick Start

```bash
# Build for your current platform
cd faiss-go
./scripts/build-static-libs.sh

# Output goes to libs/<platform>/
# Example: libs/linux_amd64/libfaiss.a
```

## Platform-Specific Instructions

### Linux (AMD64 / ARM64)

**Dependencies**:
```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential gfortran git

# Fedora/RHEL
sudo dnf install cmake gcc gcc-c++ gcc-gfortran git

# Arch
sudo pacman -S cmake gcc gcc-fortran git
```

**Build**:
```bash
./scripts/build-static-libs.sh
```

**What happens**:
1. Clones OpenBLAS v0.3.27
2. Builds OpenBLAS as static library with OpenMP support
3. Clones FAISS v1.8.0
4. Builds FAISS with static OpenBLAS linking
5. Merges libfaiss.a + libfaiss_c.a + libopenblas.a + libgfortran.a + libgomp.a
6. Output: Single `libfaiss.a` with everything included

**Output**:
```
libs/linux_amd64/libfaiss.a      (~50-70MB, fully self-contained)
libs/linux_amd64/libfaiss_c.a    (~300-700KB, C API wrapper)
```

### macOS (Intel / Apple Silicon)

**Dependencies**:
```bash
brew install cmake gcc git
```

**Build**:
```bash
./scripts/build-static-libs.sh
```

**What happens**:
1. Clones FAISS v1.8.0
2. Builds FAISS using system Accelerate framework (already static)
3. No need to build OpenBLAS - Accelerate provides BLAS/LAPACK
4. Output: `libfaiss.a` linked against Accelerate framework

**Note**: macOS Accelerate framework is system-provided and always available, so no bundling needed.

**Output**:
```
libs/darwin_amd64/libfaiss.a     (~20-30MB)
libs/darwin_amd64/libfaiss_c.a   (~300-700KB)
```

### Windows (AMD64)

**Dependencies**:
```bash
# Using MSYS2 MinGW64
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc \
          mingw-w64-x86_64-gcc-fortran git
```

**Build**:
```bash
./scripts/build-static-libs.sh
```

**Output**:
```
libs/windows_amd64/libfaiss.a
libs/windows_amd64/libfaiss_c.a
```

## Build Strategies

The build script supports two strategies:

### Strategy 1: Merged Libraries (Default, Recommended)

Merges all dependencies into a single `libfaiss.a`:

```bash
./scripts/build-static-libs.sh
```

**Pros**:
- ✅ Single library to manage
- ✅ Cleanest CGO flags
- ✅ Truly self-contained

**Cons**:
- ⚠️ Larger file size (~50-70MB vs ~15-20MB)
- ⚠️ Longer build time (~10-15 minutes)

**CGO flags become**:
```go
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss_c -lfaiss -lpthread
```

### Strategy 2: Bundled Libraries (Alternative)

Keeps dependencies as separate `.a` files in `libs/`:

```bash
./scripts/build-static-libs.sh --bundled
```

**Pros**:
- ✅ Easier to rebuild individual components
- ✅ Smaller individual files
- ✅ Still zero runtime dependencies

**Cons**:
- ⚠️ Multiple files to manage
- ⚠️ Slightly more complex CGO flags

**Output**:
```
libs/linux_amd64/libfaiss.a
libs/linux_amd64/libfaiss_c.a
libs/linux_amd64/libopenblas.a
libs/linux_amd64/libgfortran.a
libs/linux_amd64/libgomp.a
```

**CGO flags**:
```go
// Linker finds bundled libs first due to -L path
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss_c -lfaiss \
                           -lopenblas -lgfortran -lgomp -lpthread
```

## Build All Platforms (Release Process)

To build libraries for all platforms for a release:

```bash
# On Linux AMD64 machine
./scripts/build-static-libs.sh
# Output: libs/linux_amd64/

# On Linux ARM64 machine (or cross-compile)
./scripts/build-static-libs.sh
# Output: libs/linux_arm64/

# On macOS Intel machine
./scripts/build-static-libs.sh
# Output: libs/darwin_amd64/

# On macOS Apple Silicon machine
./scripts/build-static-libs.sh
# Output: libs/darwin_arm64/

# On Windows AMD64 machine (MSYS2)
./scripts/build-static-libs.sh
# Output: libs/windows_amd64/
```

**Then commit all libraries**:
```bash
git add libs/
git commit -m "chore: Update static libraries to v1.8.0 (fully self-contained)"
git push
```

## Verifying the Build

### 1. Check Library Size

Fully static libraries should be larger:

```bash
ls -lh libs/*/libfaiss.a

# Expected sizes:
# Linux AMD64:   50-70MB (merged) or 15-20MB (non-merged)
# Linux ARM64:   50-70MB (merged) or 15-20MB (non-merged)
# macOS AMD64:   20-30MB
# macOS ARM64:   20-30MB
# Windows AMD64: 50-70MB (merged) or 15-20MB (non-merged)
```

### 2. Check for Undefined Symbols

Fully static libraries should have minimal undefined symbols (only system calls):

```bash
nm -u libs/linux_amd64/libfaiss.a | grep "U " | wc -l

# Expected: < 50 symbols
# Should NOT contain: sgemm_, dgemm_, GOMP_*, omp_*
```

### 3. Test Build Without Runtime Dependencies

The ultimate test - build without installing any BLAS libraries:

```bash
# On a clean Ubuntu container (no libopenblas-dev installed)
docker run -it --rm -v $(pwd):/workspace ubuntu:24.04
cd /workspace
apt-get update && apt-get install -y golang git
go build -tags=nogpu ./...  # Should succeed!
```

### 4. Run Tests

```bash
go test -tags=nogpu -v ./...
```

## Customization

### Using Different FAISS Version

Edit `scripts/build-static-libs.sh`:

```bash
FAISS_VERSION="v1.9.0"  # Change this
```

### Using Different OpenBLAS Version

Edit `scripts/build-static-libs.sh`:

```bash
OPENBLAS_VERSION="v0.3.28"  # Change this
```

### Optimization Flags

For maximum performance, edit CMAKE flags in the script:

```bash
-DCMAKE_CXX_FLAGS="-O3 -march=native -fPIC -fopenmp"
```

**Warning**: `-march=native` optimizes for the build machine CPU. Don't use for distribution.

### GPU Support

To build with GPU support:

1. Install CUDA toolkit
2. Edit `scripts/build-static-libs.sh`:
   ```bash
   -DFAISS_ENABLE_GPU=ON \
   -DCUDAToolkit_ROOT=/usr/local/cuda
   ```

## Troubleshooting

### "Cannot find -lgfortran"

**Problem**: gfortran not installed or not in standard location.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install gfortran

# macOS
brew install gcc

# Check location
gfortran -print-file-name=libgfortran.a
```

### "Undefined symbols for architecture"

**Problem**: Libraries not properly merged or missing dependencies.

**Solution**: Use bundled strategy instead:
```bash
./scripts/build-static-libs.sh --bundled
```

### "Library too large for git"

**Problem**: Merged libraries can be 50-70MB, near GitHub's file size limit.

**Solutions**:
1. Use Git LFS:
   ```bash
   git lfs install
   git lfs track "libs/**/*.a"
   git add .gitattributes
   ```

2. Or use bundled strategy (smaller individual files)

### Build fails with "No space left on device"

**Problem**: Build requires 5-10GB temporary space.

**Solution**: Clean build artifacts:
```bash
rm -rf build-static/
```

## CI Integration

Once fully static libraries are built, update `.github/workflows/ci.yml`:

**Before** (current):
```yaml
- name: Install runtime dependencies (Ubuntu)
  if: runner.os == 'Linux'
  run: |
    sudo apt-get update
    sudo apt-get install -y libopenblas-dev libgomp1 libomp-dev
```

**After** (with fully static libs):
```yaml
# No dependency installation needed! 🎉
# Static libraries are fully self-contained
```

**Build step stays the same**:
```yaml
- name: Build
  run: go build -tags=nogpu -v ./...
```

## Migration Plan

### Phase 1: Build New Libraries (Current)

```bash
# Build fully static libraries for all platforms
./scripts/build-static-libs.sh
```

### Phase 2: Update CGO Flags

Edit `faiss_lib.go`:

```go
// Before (needs runtime dependencies)
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss_c -lfaiss \
                          -lopenblas -lgfortran -lgomp -lpthread

// After (fully self-contained)
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss_c -lfaiss -lpthread
```

### Phase 3: Update CI

Remove dependency installation steps from `.github/workflows/ci.yml`.

### Phase 4: Update Documentation

Update installation docs to emphasize true zero dependencies.

### Phase 5: Test & Release

```bash
# Test on clean system
go build -tags=nogpu ./...
go test -tags=nogpu -v ./...

# Tag release
git tag v0.1.0
git push origin v0.1.0
```

## Future Enhancements

### Cross-Compilation

Build for all platforms from a single machine using:
- Docker for Linux ARM64
- Cross-compilation toolchains
- GitHub Actions matrix

### Automated Builds

GitHub Actions workflow to build libraries:
```yaml
name: Build Static Libraries
on:
  workflow_dispatch:
    inputs:
      faiss_version:
        description: 'FAISS version to build'
        required: true
        default: 'v1.8.0'
```

### Size Optimization

Reduce library size by:
- Stripping debug symbols: `strip -S libfaiss.a`
- Removing unused FAISS index types
- Using compiler flags: `-Os` instead of `-O3`

## References

- FAISS: https://github.com/facebookresearch/faiss
- OpenBLAS: https://github.com/xianyi/OpenBLAS
- Static Linking Guide: https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html
- CMake Static Libraries: https://cmake.org/cmake/help/latest/prop_tgt/BUILD_SHARED_LIBS.html

## Support

For questions about building static libraries:
- Open an issue: https://github.com/NerdMeNot/faiss-go/issues
- Discussions: https://github.com/NerdMeNot/faiss-go/discussions
