# faiss-go Roadmap

## Current State (v0.1.0-alpha)

### ✅ Implemented
- Complete Go bindings for FAISS C++ API via CGO
- Support for 20+ index types (Flat, IVF, HNSW, PQ, GPU, OnDisk, Binary)
- All core operations (add, search, train, serialize, range search)
- Comprehensive test suite with recall validation
- Cross-platform support (Linux, macOS, Windows)

### ❌ Not Yet Implemented
- **Embedded FAISS** - Currently requires system-installed FAISS
- Pre-built static libraries
- Amalgamated source distribution
- Single-binary deployment

---

## Unique Selling Proposition (USP)

**The goal**: Make faiss-go the easiest FAISS bindings to use by **eliminating the need to install FAISS separately**.

**Vision**: `go get github.com/NerdMeNot/faiss-go` should be all you need — no system dependencies, no manual installation, just pure Go simplicity with FAISS power.

---

## Implementation Plan

### Phase 1: System FAISS (Current - v0.1.0-alpha)

**Status**: ✅ Complete

**Approach**: Traditional CGO bindings requiring system-installed FAISS
- Users install FAISS via `apt`, `brew`, or from source
- Go code links against system libraries
- Works but defeats the USP

**Limitations**:
- Requires manual FAISS installation
- Complex setup for users
- Version compatibility issues
- Not truly "go get and go"

---

### Phase 2: Embedded FAISS (Target - v0.2.0)

**Status**: 🚧 In Progress

**Goal**: Eliminate the need for system-installed FAISS

#### Option A: Amalgamated Source (Preferred)

**How it works**:
1. Include FAISS C++ source as a single file (`faiss.cpp`)
2. CGO compiles it alongside Go code
3. Everything embeds in final binary
4. Similar to how go-duckdb works

**Advantages**:
- ✅ True "go get" experience
- ✅ No external dependencies (except BLAS)
- ✅ Version locked to faiss-go release
- ✅ Cross-platform compatible
- ✅ Users control optimization flags

**Challenges**:
- FAISS doesn't provide official amalgamation
- Need to manually create amalgamation (~10-15MB source)
- First build takes 5-10 minutes (but cached after)
- Still requires BLAS library (unavoidable)

**Implementation Steps**:
1. ✅ Create amalgamation generation script
2. ⏳ Identify minimal FAISS source files for CPU-only
3. ⏳ Combine into single compilation unit
4. ⏳ Create proper C API wrapper
5. ⏳ Update CGO directives to compile amalgamation
6. ⏳ Test across all platforms

**Files Needed**:
```
faiss/
├── faiss.cpp          # ~10-15MB amalgamated source
├── faiss.h            # C API header
└── faiss_c_impl.cpp   # C++ to C bridge
```

#### Option B: Pre-built Static Libraries

**How it works**:
1. Pre-compile FAISS for each platform
2. Include static libraries in repo
3. CGO links against pre-built libraries
4. Fast builds (<30 seconds)

**Advantages**:
- ✅ Fastest build times
- ✅ No C++ compilation needed
- ✅ Smaller repo (no source code)

**Challenges**:
- Need to maintain libraries for all platforms
- Larger repository size (~50-100MB per platform)
- Less flexibility for users
- Platform-specific binaries

**Implementation Steps**:
1. ✅ Create build matrix for platforms
2. ⏳ Set up CI to build libraries
3. ⏳ Host libraries in repo or CDN
4. ⏳ Update CGO to use pre-built libs with build tag
5. ⏳ Create `faiss_use_lib` build mode

**Platforms Needed**:
```
libs/
├── linux_amd64/    # libfaiss.a + libopenblas.a
├── linux_arm64/    # libfaiss.a + libopenblas.a
├── darwin_amd64/   # libfaiss.a
├── darwin_arm64/   # libfaiss.a (M1/M2)
└── windows_amd64/  # faiss.lib + openblas.lib
```

#### Hybrid Approach (Recommended)

**Both options available**:
- Default: Amalgamated source (best for most users)
- Tag `faiss_use_lib`: Pre-built libraries (for fast iteration)

```bash
# Build from source (5-10 min first time)
go build

# Use pre-built libraries (<30 sec)
go build -tags=faiss_use_lib
```

---

### Phase 3: Enhanced Features (v0.3.0+)

After achieving the embedded FAISS goal:

**Planned Features**:
- [ ] Better error messages and debugging
- [ ] Index parameter auto-tuning helpers
- [ ] Built-in benchmarking tools
- [ ] Index selection wizard/CLI
- [ ] Performance profiling integration
- [ ] Memory usage monitoring
- [ ] Distributed search examples
- [ ] Vector database patterns
- [ ] Integration examples (gRPC, REST, etc.)

---

## Technical Challenges

### 1. BLAS Dependency

**Problem**: FAISS requires BLAS (Basic Linear Algebra Subprograms)

**Options**:
- **Option A**: Require system BLAS (OpenBLAS, MKL, Accelerate)
  - Simplest approach
  - Users install via package manager
  - Platform-specific

- **Option B**: Bundle static BLAS library
  - Include libopenblas.a in pre-built libraries
  - Increases size significantly (~40MB)
  - More complex but better UX

- **Option C**: Pure Go BLAS implementation
  - Slower than optimized BLAS
  - Not feasible for production use
  - Not recommended

**Decision**: Start with Option A (system BLAS), consider Option B for pre-built libraries

### 2. Amalgamation Complexity

**Problem**: FAISS has ~100+ source files and complex dependencies

**Approach**:
1. Start with minimal feature set (Flat, IVF, HNSW)
2. Gradually add more index types
3. Use conditional compilation for optional features
4. Create automated amalgamation script

### 3. Build Times

**Problem**: Compiling FAISS from source takes 5-10 minutes

**Solutions**:
- Go build cache (subsequent builds are fast)
- Pre-built libraries option for development
- Parallel compilation flags
- Compiler optimization flags

### 4. Cross-Platform Support

**Challenges**:
- Different compilers (GCC, Clang, MSVC)
- Platform-specific BLAS libraries
- Architecture differences (x64, ARM64)
- Endianness considerations

**Approach**:
- Extensive CI/CD testing
- Platform-specific CGO flags
- Clear documentation per platform
- Community testing and feedback

---

## Timeline Estimate

### v0.2.0 - Embedded FAISS (Target: 2-3 months)

**Month 1**: Amalgamation Research & Development
- Week 1-2: Analyze FAISS source structure
- Week 3-4: Create minimal amalgamation

**Month 2**: Implementation & Testing
- Week 1-2: Implement C API wrapper
- Week 3-4: Cross-platform testing and fixes

**Month 3**: Pre-built Libraries & Polish
- Week 1-2: Build library matrix
- Week 3-4: Documentation, examples, release

### v0.3.0 - Enhanced Features (Target: +2 months)

Focus on developer experience and production features

---

## How to Contribute

### Help with Embedded FAISS

1. **FAISS Amalgamation**:
   - Help identify minimal required source files
   - Test amalgamation on different platforms
   - Optimize compilation process

2. **Pre-built Libraries**:
   - Build and test libraries for your platform
   - Validate across different OS versions
   - Help with CI/CD automation

3. **Documentation**:
   - Platform-specific setup guides
   - Troubleshooting common issues
   - Performance optimization tips

### Current Priority Tasks

1. **Urgent**: Create working FAISS amalgamation
2. **High**: Build pre-built libraries for major platforms
3. **Medium**: Improve build scripts and automation
4. **Low**: Enhanced features and integrations

---

## Success Criteria

We'll know we've achieved the USP when:

1. ✅ New users can `go get github.com/NerdMeNot/faiss-go` and start coding immediately
2. ✅ No "install FAISS first" step in README
3. ✅ Works out-of-the-box on all major platforms
4. ✅ Build times are acceptable (< 1 min with pre-built libs)
5. ✅ Documentation focuses on use cases, not installation

---

## Questions & Discussions

Have ideas or want to help? Join the discussion:
- [Embedded FAISS Discussion](https://github.com/NerdMeNot/faiss-go/discussions)
- [GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues)

---

**Last Updated**: 2025-12-28
**Next Review**: 2025-01-28
