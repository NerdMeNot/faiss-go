# FAISS-Go Build Modes

This document explains all available build modes for faiss-go, from simplest to most advanced.

## Quick Reference

| Build Mode | Binary Size | Dependencies | Build Time | Use Case |
|------------|-------------|--------------|------------|----------|
| **System** | Small | System FAISS | Fast (~30s) | Development |
| **Standard Static** | ~9MB | libopenblas | Medium (~5min) | Basic use |
| **Unified Static** | ~45MB | gomp, gfortran | Slow (~20min) | Production |
| **Phase 3** (Exp) | ~55MB | NONE! | Slowest (~30min) | Ultimate |

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

## Mode 2: Standard Static Build

**Uses pre-built static libraries from libs/ directory**

```bash
# Default mode - just build
go build

# Or explicitly
go build -tags=nogpu
```

**Includes:**
- FAISS static library (~9MB)
- Links against system OpenBLAS

**Pros:**
- ✅ Fast build (~30 seconds)
- ✅ Moderate size (~9MB)
- ✅ Works out of the box

**Cons:**
- ❌ Needs system OpenBLAS/Accelerate
- ❌ Different behavior on different systems

**When to use:** Quick prototyping, development

## Mode 3: Unified Static Build

**Everything merged into libfaiss.a** ⭐ **Recommended for production**

```bash
# Automatically used with pre-built libs
go build -tags=nogpu
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

## Mode 4: Phase 3 Build (Experimental)

**ZERO external dependencies** ⚡ **Experimental**

```bash
# Build your project with Phase 3
go build -tags="nogpu,faiss_phase3"
```

**Includes:**
- FAISS static library
- OpenBLAS merged
- **libgomp merged** (NEW!)
- **libgfortran merged** (NEW!)
- **libquadmath merged** (NEW!)

**Runtime dependencies:**
- **NONE!** (if successful)

**Pros:**
- ✅ Truly self-contained
- ✅ Zero dependency deployment
- ✅ Perfect for minimal containers
- ✅ Ultimate portability

**Cons:**
- ❌ Experimental (may not work everywhere)
- ❌ Largest size (~55-60MB)
- ❌ Complex build process
- ❌ Longer build time (~30min)

**When to use:**
- Deploying to unknown/minimal environments
- Embedded systems
- When you absolutely cannot have dependencies
- Alpine/distroless containers

**See:** [PHASE3-BUILDS.md](PHASE3-BUILDS.md) for complete details

## Platform-Specific Notes

### Linux

| Mode | amd64 | arm64 | Dependencies |
|------|-------|-------|--------------|
| System | ✅ | ✅ | libfaiss-dev |
| Standard | ✅ | ✅ | libopenblas |
| Unified | ✅ | ✅ | gomp, gfortran |
| Phase 3 | ✅ | ✅ | NONE |

### macOS

| Mode | Intel | ARM64 | Dependencies |
|------|-------|-------|--------------|
| System | ✅ | ✅ | brew faiss |
| Standard | ✅ | ✅ | Accelerate (built-in) |
| Unified | ✅ | ✅ | Accelerate (built-in) |
| Phase 3 | ❌ | ❌ | Not supported |

**Note:** macOS unified is identical to standard (Accelerate framework cannot be statically linked)

### Windows

| Mode | amd64 | Dependencies |
|------|-------|--------------|
| System | ⚠️ | Manual install |
| Standard | ✅ | OpenBLAS via vcpkg |
| Unified | ✅ | gomp, gfortran (MinGW) |
| Phase 3 | ⚠️ | Experimental |

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
3. Else → Unified build (default, best for most)

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
# Use unified build - best balance
go build -tags=nogpu

# Dependencies: gomp + gfortran (usually pre-installed)
```

### For Minimal Containers
```bash
# Try Phase 3 if you need zero-dep
go build -tags="nogpu,faiss_phase3"

# Fallback to unified if Phase 3 doesn't work
```

### For Distribution
```bash
# Use unified build
go build -tags=nogpu

# Document requirements:
# - Linux: apt-get install libgomp1 libgfortran5
# - macOS: Built-in (no action needed)
# - Windows: Comes with MinGW
```

## Debugging Build Issues

### Check which mode is active

```bash
go build -v -tags=nogpu 2>&1 | grep faiss
# Should show which .go files are being compiled
```

### Verify library dependencies

```bash
# On Linux
ldd ./your_binary | grep -E "(gomp|gfortran|openblas)"

# Expected for unified: libgomp, libgfortran
# Expected for Phase 3: (nothing!)
```

### Check library symbols

```bash
# See what's in libfaiss.a
nm -g libs/linux_amd64/libfaiss.a | grep -E "(GOMP|gfortran|openblas)" | head

# Phase 3: All symbols present in libfaiss.a
# Unified: Only OpenBLAS symbols present
```

## Migration Guide

### From System Build → Unified Build

```bash
# Before
go build -tags=faiss_use_system

# After
go build -tags=nogpu

# No code changes needed!
```

### From Standard → Phase 3

```bash
# Before
go build -tags=nogpu

# After
go build -tags="nogpu,faiss_phase3"

# Rebuild libs first:
./scripts/build_unified_static.sh linux-amd64 v1.13.2
```

## FAQ

**Q: Which mode should I use?**
A: Unified build for production. It's the sweet spot of size, dependencies, and reliability.

**Q: Why is Phase 3 experimental?**
A: Merging runtime libraries is complex and may not work on all platforms/toolchains. We're pushing the boundaries!

**Q: Can I use different modes for different platforms?**
A: Yes! The platform-specific prebuilt files handle this automatically.

**Q: How do I know if Phase 3 worked?**
A: Run `./scripts/verify_phase3.sh linux-amd64` after building.

**Q: What if Phase 3 fails?**
A: Fall back to unified build. It's still excellent with minimal deps.

## See Also

- [STATIC-BUILDS.md](STATIC-BUILDS.md) - Technical details on static builds
- [PHASE3-BUILDS.md](PHASE3-BUILDS.md) - Deep dive into Phase 3
- [libs/README.md](../libs/README.md) - Pre-built library information
