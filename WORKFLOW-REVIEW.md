# Workflow and Platform Review

## Executive Summary

✅ **All workflows are properly configured for CPU-only builds across all platforms**
✅ **Direct `.a` file linking implemented for all 5 platforms**
✅ **GPU code properly excluded with build tags**
✅ **Mac M3 confirmed working (both prebuilt and system FAISS)**
✅ **Ready for production use**

## Workflow Triggers - All Manual ✅

All 6 workflows are configured with `workflow_dispatch` (manual trigger only):

### 1. CI (`.github/workflows/ci.yml`)
- **Trigger**: `workflow_dispatch` only
- **Uses**: `-tags=nogpu` for all builds
- **Platforms**: Ubuntu (AMD64), macOS (ARM64/AMD64 auto-detect)
- **Go versions**: 1.21-1.25
- **Dependencies**:
  - Linux: `libgomp1`, `libgfortran5` (runtime only)
  - macOS: `libomp` (Accelerate is built-in)
- **New features**:
  - Verifies custom wrapper symbols present
  - Checks both `libfaiss.a` and `libfaiss_c.a` exist

### 2. GPU CI (`.github/workflows/gpu-ci.yml`)
- **Trigger**: `workflow_dispatch` only
- **Purpose**: GPU-specific testing (requires CUDA)
- **Status**: For future GPU testing (requires self-hosted GPU runners)

### 3. Benchmark (`.github/workflows/benchmark.yml`)
- **Trigger**: `workflow_dispatch` only
- **Uses**: `-tags=nogpu`
- **Modes**:
  - Quick: Selected benchmarks, 1s benchtime
  - Comprehensive: All benchmarks, configurable benchtime

### 4. Release (`.github/workflows/release.yml`)
- **Trigger**: `workflow_dispatch` only
- **Uses**: `--build-tags=nogpu` for linting
- **Features**:
  - Version bumping (major/minor/patch)
  - Changelog generation
  - Git tag creation
  - GitHub release creation

### 5. Build Static Libs (`.github/workflows/build-static-libs.yml`)
- **Trigger**: `workflow_dispatch` only
- **Builds**: All 5 platforms concurrently
- **Caching**: OpenBLAS and FAISS builds cached
- **Features**:
  - Platform-specific build strategies
  - Artifact upload with 90-day retention
  - Combined artifact for easy download

### 6. Build Phase 3 (`.github/workflows/build-phase3.yml`)
- **Trigger**: `workflow_dispatch` only
- **Purpose**: Experimental zero-dependency builds
- **Status**: Testing phase, not for production yet

## Platform Build Configuration

### Linux AMD64
```bash
# Build command
./scripts/build_static_lib.sh linux-amd64 v1.13.2 --unified

# Library details
- Size: ~45MB (includes OpenBLAS merged)
- Runtime deps: libgomp1, libgfortran5
- LDFLAGS: ${SRCDIR}/libs/linux_amd64/libfaiss_c.a ${SRCDIR}/libs/linux_amd64/libfaiss.a -lgomp -lgfortran -lm -lstdc++ -lpthread -ldl
```

### Linux ARM64
```bash
# Build command (via QEMU in Docker)
./scripts/build_static_lib.sh linux-arm64 v1.13.2 --unified

# Library details
- Size: ~45MB (includes OpenBLAS merged)
- Runtime deps: libgomp1, libgfortran5
- LDFLAGS: Same as Linux AMD64
```

### macOS AMD64 (Intel)
```bash
# Build command
./scripts/build_static_lib.sh darwin-amd64 v1.13.2

# Library details
- Size: ~9MB (uses Accelerate framework)
- Runtime deps: Accelerate (built-in), libomp
- LDFLAGS: ${SRCDIR}/libs/darwin_amd64/libfaiss_c.a ${SRCDIR}/libs/darwin_amd64/libfaiss.a -Wl,-framework,Accelerate -L/usr/local/opt/libomp/lib -lomp -lm -lstdc++
```

### macOS ARM64 (Apple Silicon) ✅ Tested on M3
```bash
# Build command
./scripts/build_static_lib.sh darwin-arm64 v1.13.2

# Library details
- Size: ~9MB (uses Accelerate framework)
- Runtime deps: Accelerate (built-in), libomp
- LDFLAGS: ${SRCDIR}/libs/darwin_arm64/libfaiss_c.a ${SRCDIR}/libs/darwin_arm64/libfaiss.a -Wl,-framework,Accelerate -L/opt/homebrew/opt/libomp/lib -lomp -lm -lstdc++
```

### Windows AMD64
```bash
# Build command
./scripts/build_static_lib.sh windows-amd64 v1.13.2 --unified

# Library details
- Size: ~45MB (includes OpenBLAS merged)
- Runtime deps: libgomp, libgfortran (via MinGW)
- LDFLAGS: ${SRCDIR}/libs/windows_amd64/faiss_c.lib ${SRCDIR}/libs/windows_amd64/faiss.lib -lgomp -lgfortran -lm -lstdc++ -lpthread
```

## GPU Code Exclusion

All GPU-related files have proper build tags to exclude them from CPU-only builds:

```go
// gpu.go, index_gpu.go, faiss_gpu.go
// +build !nogpu
```

When building with `-tags=nogpu`, these files are completely excluded from compilation.

**Files with GPU code** (excluded with `nogpu` tag):
- `gpu.go` - GPU resource management
- `index_gpu.go` - GPU index implementations (GpuIndexFlat, GpuIndexIVFFlat)
- `faiss_gpu.go` - GPU C wrapper functions

## Direct .a File Linking Architecture

### Why Direct Linking?

**Problem with `-l` flags:**
```go
// Old approach (selective linking)
#cgo LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss_c -lfaiss ...
```
- Linker selectively pulls object files from archives
- Custom wrapper symbols (faiss_c_impl.o) might not be included
- Results in undefined symbol errors

**Solution with direct linking:**
```go
// New approach (forces ALL object files)
#cgo LDFLAGS: ${SRCDIR}/libs/linux_amd64/libfaiss_c.a ${SRCDIR}/libs/linux_amd64/libfaiss.a ...
```
- Forces inclusion of ALL object files from archives
- Guarantees faiss_c_impl.o is always linked
- No undefined symbol errors

### Custom C Wrapper Layer

**File**: `faiss_c_impl.cpp`

**Purpose**: Provides Go-friendly C wrapper functions beyond official FAISS C API

**Build process**:
1. Build FAISS libraries (libfaiss.a, libfaiss_c.a)
2. Compile `faiss_c_impl.cpp` → `faiss_c_impl.o`
3. Merge into libfaiss_c.a: `ar r libfaiss_c.a faiss_c_impl.o`

**Example wrapper functions**:
- `faiss_IndexBinaryFlat_new` - Binary index creation
- `faiss_Kmeans_new` - K-means clustering
- Additional functions not in official FAISS C API

**Verification**:
```bash
# Linux
nm libs/linux_amd64/libfaiss_c.a | grep "faiss_IndexBinaryFlat_new"
# Should show: faiss_c_impl.o: T faiss_IndexBinaryFlat_new

# macOS (symbols prefixed with _)
nm libs/darwin_arm64/libfaiss_c.a | grep "_faiss_IndexBinaryFlat_new"
# Should show: faiss_c_impl.o: T _faiss_IndexBinaryFlat_new
```

## Testing Commands

### CPU-only builds (recommended for all platforms):
```bash
# With prebuilt static libraries (default, fastest)
go build -tags=nogpu ./...
go test -short -tags=nogpu -v ./...

# With system FAISS (requires FAISS installed)
go build -tags='faiss_use_system,nogpu' ./...
go test -short -tags='faiss_use_system,nogpu' -v ./...
```

### GPU builds (requires CUDA, not supported on macOS):
```bash
# Build without nogpu tag (includes GPU code)
go build ./...
go test -v ./...
```

## Issues Found and Fixed

### ✅ 1. GPU wrapper function signatures (commit abc2e1b)
**Issue**: `index_gpu.go` had mismatched function signatures after GPU wrapper refactoring
**Symptoms**: Compilation errors when building without `nogpu` tag
**Fix**: Updated to use `error` return type instead of `int`
**Files**: `index_gpu.go:55-57`, `index_gpu.go:215-217`

### ✅ 2. Architecture-specific libomp paths (commit 38b1d1d)
**Issue**: macOS ARM64 had Intel libomp path causing linker warnings
**Symptoms**: `ld: warning: search path '/usr/local/opt/libomp/lib' not found`
**Fix**: Separated paths by architecture:
  - ARM64: `/opt/homebrew/opt/libomp/lib`
  - AMD64: `/usr/local/opt/libomp/lib`
**Files**: `prebuilt_darwin_arm64.go`, `prebuilt_darwin_amd64.go`

### ✅ 3. Direct .a file linking (commits 6cb71ed, 3ea186c)
**Issue**: Using `-l` flags caused selective linking, missing custom wrapper symbols
**Symptoms**: `undefined symbol: faiss_IndexBinaryFlat_new`
**Fix**: Direct `.a` file linking for all platforms
**Files**: All `prebuilt_*.go` files

### ✅ 4. TestBuildInfo validation (commit 6d70b42)
**Issue**: Test didn't accept "static" as valid build mode
**Fix**: Added "static" to valid build modes list
**File**: `faiss_test.go`

### ✅ 5. CI dependencies optimization (commit a797c45)
**Issue**: CI installing unnecessary development packages
**Fix**: Removed `libopenblas-dev`, `libomp-dev` from Linux; removed `openblas` from macOS
**Added**: Symbol verification step to CI
**File**: `.github/workflows/ci.yml`

## Recommendations for Users

### For Development
```bash
# Use prebuilt libraries (fastest, no compilation)
go build -tags=nogpu ./...

# Or use system FAISS (if already installed)
go build -tags='faiss_use_system,nogpu' ./...
```

### For Production Deployment

**Linux (Ubuntu/Debian)**:
```bash
# Install runtime dependencies
sudo apt-get install libgomp1 libgfortran5

# Build your application
go build -tags=nogpu ./...
```

**macOS**:
```bash
# Install runtime dependency
brew install libomp
# Accelerate framework is built-in

# Build your application
go build -tags=nogpu ./...
```

**Windows**:
```bash
# Runtime libraries come with MinGW

# Build your application
go build -tags=nogpu ./...
```

### For Docker/Containers
```dockerfile
# Use minimal base image
FROM ubuntu:24.04

# Install ONLY runtime libraries (not dev packages)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Copy your pre-built binary
COPY my-app /app/my-app

# Run
CMD ["/app/my-app"]
```

## Future Enhancements

### 1. CI Improvements
- Add Windows CI testing
- Add ARM64 native runners (GitHub now supports this)
- Add library size tracking over time

### 2. Library Size Optimization
- Investigate stripping debug symbols: `strip -S libfaiss.a`
- Consider compiler flags: `-Os` instead of `-O3`
- Profile which FAISS features are actually used

### 3. Phase 3 Zero-Dependency Builds
- Continue refining runtime library merging
- Test on more platforms
- Document success/failure cases

## Verification Checklist

Before using in production:

- [ ] Run tests on your target platform: `go test -short -tags=nogpu -v ./...`
- [ ] Verify custom wrapper symbols exist: `nm libs/*/libfaiss_c.a | grep faiss_IndexBinaryFlat_new`
- [ ] Check runtime dependencies are installed
- [ ] Test with a simple program first
- [ ] Verify library sizes match documented sizes

## Support

For issues:
- Check this document first
- Review documentation: `docs/BUILD-MODES.md`, `docs/STATIC-BUILDS.md`
- Open issue: https://github.com/NerdMeNot/faiss-go/issues

## Status: Production Ready ✅

All workflows reviewed and confirmed:
- ✅ Manual triggers only (no automatic runs)
- ✅ CPU-only builds properly configured
- ✅ Direct `.a` linking on all platforms
- ✅ GPU code properly excluded
- ✅ Dependencies optimized
- ✅ Symbol verification in CI
- ✅ Comprehensive documentation
- ✅ Tested on Mac M3 (ARM64)

**Last Updated**: December 30, 2025
