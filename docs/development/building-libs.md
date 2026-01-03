# Building Static Libraries

Guide for building the pre-built static FAISS libraries included with faiss-go.

## Overview

faiss-go includes pre-built static libraries for major platforms. This guide is for maintainers who need to rebuild these libraries.

**Most users don't need this** - just use the pre-built libraries included in the repository.

## Runtime Dependencies

| Platform | Current State | Target State |
|----------|---------------|--------------|
| macOS | ✅ Zero deps (uses Accelerate) | ✅ Zero deps |
| Linux | ⚠️ Needs libopenblas, libgfortran | ✅ Fully static (OpenBLAS bundled) |
| Windows | ⚠️ Needs libopenblas, libgfortran | ✅ Fully static (OpenBLAS bundled) |

The `build-static-libs.sh` script can build **fully static** libraries with OpenBLAS bundled, eliminating all runtime dependencies on Linux/Windows.

## Quick Build

```bash
# Build fully static libraries (OpenBLAS bundled, no runtime deps)
./scripts/build-static-libs.sh

# Or use GitHub Actions
# Go to Actions > Build Static Libraries > Run workflow
# Select "Build fully static libs" option

# Output goes to libs/<platform>/
```

## Platform Requirements

### Linux (AMD64 / ARM64)

```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential gfortran git

# Fedora/RHEL
sudo dnf install cmake gcc gcc-c++ gcc-gfortran git
```

### macOS (Intel / Apple Silicon)

```bash
brew install cmake gcc git
```

### Windows (AMD64)

Using MSYS2 MinGW64:
```bash
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc \
          mingw-w64-x86_64-gcc-fortran git
```

## Build Process

The script:
1. Clones OpenBLAS (v0.3.27) and builds it as static library
2. Clones FAISS (v1.8.0) and builds with static OpenBLAS
3. Creates combined static libraries
4. Outputs to `libs/<platform>/`

Output files:
- `libfaiss.a` - Main FAISS library
- `libfaiss_c.a` - C API wrapper

## Building All Platforms

For a release, build on each platform:

```bash
# Linux AMD64
./scripts/build-static-libs.sh
# Output: libs/linux_amd64/

# Linux ARM64
./scripts/build-static-libs.sh
# Output: libs/linux_arm64/

# macOS Intel
./scripts/build-static-libs.sh
# Output: libs/darwin_amd64/

# macOS Apple Silicon
./scripts/build-static-libs.sh
# Output: libs/darwin_arm64/

# Windows (MSYS2)
./scripts/build-static-libs.sh
# Output: libs/windows_amd64/
```

Then commit:
```bash
git add libs/
git commit -m "chore: Update static libraries"
git push
```

## Verification

```bash
# Check sizes
ls -lh libs/*/libfaiss.a

# Test build
go build ./...
go test -v ./...
```

## Customization

Edit `scripts/build-static-libs.sh`:

```bash
FAISS_VERSION="v1.8.0"      # FAISS version
OPENBLAS_VERSION="v0.3.27"  # OpenBLAS version
```

## GPU Support

GPU builds require CUDA toolkit and aren't included in pre-built libraries. See [GPU Setup](../getting-started/gpu-setup.md).

## Troubleshooting

### "Cannot find -lgfortran"

Install gfortran:
```bash
# Ubuntu
sudo apt-get install gfortran

# macOS
brew install gcc
```

### Build runs out of space

Clean build artifacts:
```bash
rm -rf build-static/
```

### Library too large for git

Use Git LFS:
```bash
git lfs install
git lfs track "libs/**/*.a"
```
