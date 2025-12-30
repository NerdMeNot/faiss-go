# FAISS-Go Build Modes

This document explains all available build modes for faiss-go, from simplest to most advanced.

## Quick Reference

| Build Mode | Binary Size | Dependencies | Build Time | Use Case |
|------------|-------------|--------------|------------|----------|
| **System** | Small | System FAISS | Fast (~30s) | Development |
| **Unified Static** | ~45MB | gomp, gfortran | Medium (~20min) | Production ⭐ |

## Mode 1: System Build

**Uses system-installed FAISS library**

```bash
# Install FAISS first
sudo apt-get install libfaiss-dev  # Linux
brew install faiss                  # macOS

# Build
go build -tags=faiss_use_system
```

**Pros:**
- ✅ Fastest build (~30 seconds)
- ✅ Smallest binary size
- ✅ Easy to update FAISS (apt-get upgrade)

**Cons:**
- ❌ Requires FAISS pre-installed
- ❌ Version dependency hell
- ❌ Not portable

**When to use:** Local development, testing

## Mode 2: Unified Static Build (Default)

**Everything merged into libfaiss.a** ⭐ **Recommended for production**

```bash
# Automatically used with pre-built libs (CPU-only is default)
go build
```

**Includes:**
- FAISS static library
- OpenBLAS merged in (~45MB total)

**Runtime dependencies:**
- Linux/Windows: libgomp, libgfortran
- macOS: Accelerate framework

**Pros:**
- ✅ No separate BLAS library needed
- ✅ Consistent behavior across systems
- ✅ Easy deployment
- ✅ Minimal runtime deps (usually pre-installed)

**Cons:**
- ❌ Larger size (~45MB)
- ❌ Still needs gomp + gfortran

**When to use:** Production deployments, Docker containers, CI/CD

**Installation:**
```bash
# Linux - install runtime libs (if not already present)
sudo apt-get install libgomp1 libgfortran5

# macOS - Accelerate is built-in, may need libomp
brew install libomp

# Windows - MinGW provides these
```

## Platform-Specific Notes

### Linux

| Mode | amd64 | arm64 | Dependencies |
|------|-------|-------|--------------|
| System | ✅ | ✅ | libfaiss-dev |
| Unified | ✅ | ✅ | gomp, gfortran |

### macOS

| Mode | Intel | ARM64 | Dependencies |
|------|-------|-------|--------------|
| System | ✅ | ✅ | brew faiss |
| Unified | ✅ | ✅ | Accelerate (built-in), libomp |

**Note:** macOS uses Accelerate framework (built-in) for BLAS operations

### Windows

| Mode | amd64 | Dependencies |
|------|-------|--------------|
| System | ⚠️ | Manual install |
| Unified | ✅ | gomp, gfortran (MinGW) |

## How Build Mode is Selected

The build mode is determined automatically by build tags and platform:

```go
// Platform-specific prebuilt files handle this
// prebuilt_linux_amd64.go - Linux AMD64 config
// prebuilt_darwin_arm64.go - macOS ARM64 config
// etc.
```

**Selection logic:**
1. If `-tags=faiss_use_system` → System build
2. If `-tags=faiss_phase3` → Phase 3 build
3. Else → Static build (default, best for most)

**Note:** All static builds use direct `.a` file linking to ensure custom C wrapper symbols are included.

## Building Your Own Static Libraries

### Unified Build

```bash
# Build unified static lib for your platform
./scripts/build_static_lib.sh linux-amd64 v1.13.2 --unified

# Output: libs/linux_amd64/libfaiss.a (~45MB)
```

### Phase 3 Build

```bash
# Build Phase 3 with runtime libs merged
./scripts/build_unified_static.sh linux-amd64 v1.13.2

# Verify it worked
./scripts/verify_phase3.sh linux-amd64

# Output: libs/linux_amd64/libfaiss.a (~55MB)
```

## Recommendations

### For Development
```bash
# Use system build for fastest iteration
go build -tags=faiss_use_system
```

### For Production (Most Projects)
```bash
# CPU-only build (default) - best balance
go build

# Dependencies: gomp + gfortran (usually pre-installed)
```

### For GPU Acceleration
```bash
# Enable GPU support (opt-in)
go build -tags=gpu

# Requires: CUDA runtime, cuBLAS
```

### For Distribution
```bash
# Use default CPU build
go build

# Document requirements:
# - Linux: apt-get install libgomp1 libgfortran5
# - macOS: Built-in (no action needed)
# - Windows: Comes with MinGW
```

## Debugging Build Issues

### Check which mode is active

```bash
go build -v 2>&1 | grep faiss
# Should show which .go files are being compiled
```

### Verify library dependencies

```bash
# On Linux
ldd ./your_binary | grep -E "(gomp|gfortran|openblas)"

# Expected for static build: libgomp, libgfortran
# Expected for system build: libfaiss.so, libopenblas.so
```

### Check library symbols

```bash
# See what's in libfaiss.a
nm -g libs/linux_amd64/libfaiss.a | grep -E "(GOMP|gfortran|openblas)" | head

# Phase 3: All symbols present in libfaiss.a
# Unified: Only OpenBLAS symbols present
```

## Migration Guide

### From System Build → Static Build

```bash
# Before
go build -tags=faiss_use_system

# After (CPU-only is now default)
go build

# No code changes needed!
```

### Enabling GPU Support

```bash
# CPU-only (default)
go build

# With GPU acceleration (opt-in)
go build -tags=gpu

# Requires CUDA runtime and cuBLAS installed
```

## FAQ

**Q: Which mode should I use?**
A: Static build (default) for production. It's the sweet spot of size, dependencies, and reliability.

**Q: Do I need to specify build tags for CPU-only builds?**
A: No! CPU-only is now the default. Just use `go build` with no tags.

**Q: How do I enable GPU support?**
A: Add `-tags=gpu` when building. This requires CUDA runtime and cuBLAS installed.

**Q: Can I use different modes for different platforms?**
A: Yes! The platform-specific prebuilt files handle this automatically.

**Q: What are the runtime dependencies?**
A: Linux/Windows: libgomp, libgfortran (usually pre-installed). macOS: Accelerate framework (built-in).

## See Also

- [STATIC-BUILDS.md](STATIC-BUILDS.md) - Technical details on static builds
- [PHASE3-BUILDS.md](PHASE3-BUILDS.md) - Deep dive into Phase 3
- [libs/README.md](../libs/README.md) - Pre-built library information
