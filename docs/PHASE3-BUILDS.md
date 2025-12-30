# Phase 3: Zero-Dependency Static Builds (Experimental)

## Overview

Phase 3 is an experimental build mode that attempts to create **truly self-contained** static libraries with **ZERO external dependencies** by aggressively merging runtime libraries directly into `libfaiss.a`.

## The Problem We're Solving

**Standard unified builds** (Phase 1/2):
```
libfaiss.a (~45MB) = FAISS + OpenBLAS merged
Runtime dependencies: libgomp, libgfortran
```

**Phase 3 goal**:
```
libfaiss.a (~50-60MB) = FAISS + OpenBLAS + libgomp + libgfortran + libquadmath
Runtime dependencies: NONE! (truly zero-dep)
```

## How It Works

### Step 1: Build with Static Runtime Flags

```bash
# Build OpenBLAS with static runtime linking
make EXTRALIB="-static-libgfortran -static-libgomp -static-libquadmath"
```

This tells the compiler to link runtime libraries statically within OpenBLAS.

### Step 2: Extract Runtime Library Objects

```bash
# Find GCC runtime libraries
gcc_lib_dir=$(gcc -print-file-name=libgfortran.a | xargs dirname)

# Extract .o files from runtime libraries
ar x $gcc_lib_dir/libgfortran.a    # Fortran runtime
ar x $gcc_lib_dir/libgomp.a        # OpenMP runtime
ar x $gcc_lib_dir/libquadmath.a    # Quad precision math
```

### Step 3: Merge Everything

```bash
# Extract all .o files
ar x libfaiss.a
ar x libfaiss_c.a
ar x libopenblas.a
ar x libgfortran.a
ar x libgomp.a
ar x libquadmath.a

# Create one massive archive with ALL symbols
ar rcs libfaiss.a *.o
ranlib libfaiss.a
```

### Step 4: Link Without Runtime Dependencies

```go
// Phase 3 CGO flags (NO runtime libs!)
#cgo LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss -lm -lpthread -ldl

// Notice: No -lgomp, no -lgfortran, no -lstdc++
// Everything is in libfaiss.a!
```

## Usage

### Building Phase 3 Libraries

```bash
# Build for Linux AMD64
./scripts/build_unified_static.sh linux-amd64 v1.13.2

# Build for Linux ARM64
./scripts/build_unified_static.sh linux-arm64 v1.13.2

# Build for Windows
./scripts/build_unified_static.sh windows-amd64 v1.13.2
```

### Verifying Phase 3 Build

```bash
# Check if runtime libraries were successfully merged
./scripts/verify_phase3.sh linux-amd64
```

This will:
- Check library size (should be 50-60MB for Phase 3)
- Verify GOMP symbols are present
- Verify gfortran symbols are present
- Attempt to link without runtime dependencies

### Using Phase 3 in Your Project

```bash
# Build your Go project with Phase 3 libraries
go build -tags="nogpu,faiss_phase3"
```

**CGO configuration is automatic** - the `faiss_lib_phase3.go` file has the right LDFLAGS.

## Expected Results

### Success Indicators

✅ **Library size**: 50-60MB (vs 45MB for standard unified)
✅ **Symbol check**: `GOMP_*` symbols found in libfaiss.a
✅ **Symbol check**: `_gfortran_*` symbols found in libfaiss.a
✅ **Link test**: Can link with only `-lfaiss -lm -lpthread -ldl`
✅ **Runtime**: `ldd ./your_binary` shows no libgomp or libgfortran

### Failure Indicators

❌ **Library size**: Still ~45MB (runtime libs not merged)
❌ **Symbol check**: No GOMP/gfortran symbols
❌ **Link test**: Fails with undefined references
❌ **Needs**: Still requires `-lgomp -lgfortran`

If Phase 3 fails, the build system automatically falls back to standard unified builds.

## Potential Issues & Solutions

### Issue 1: Global Constructor Conflicts

**Problem**: Runtime libraries have global constructors that may conflict

**Solution**: The build script uses `ranlib` to reindex symbols and resolve conflicts

### Issue 2: Thread-Local Storage (TLS)

**Problem**: gomp uses TLS which needs special initialization

**Solution**: Modern linkers handle this automatically, but on older systems you might see warnings

### Issue 3: Symbol Duplication

**Problem**: Same symbol in multiple .o files

**Solution**: `ar rcs` replaces duplicates automatically

### Issue 4: Platform-Specific ABI

**Problem**: Runtime libraries compiled for different ABI versions

**Solution**: Build on same system that will use the library, or use manylinux/docker containers

## Comparison with Python (faiss-cpu)

**Python's approach** (using manylinux wheels):
```
1. Build FAISS as .so (shared library)
2. Build OpenBLAS as .so
3. Use auditwheel to bundle .so files into wheel
4. Patch RPATH so they find each other
5. Result: Self-contained wheel with shared libs
```

**Our Phase 3 approach**:
```
1. Build FAISS as .a (static library)
2. Build OpenBLAS as .a
3. Merge .a files at object level
4. Merge runtime .a files too!
5. Result: Single .a file with everything
```

**Advantage of Phase 3**:
- Simpler: One file instead of multiple .so files
- Faster: No runtime dynamic linking overhead
- Go-esque: Native Go approach (cgo with static libs)

## Platform Support

| Platform | Phase 3 Status | Notes |
|----------|----------------|-------|
| `linux-amd64` | ✅ Supported | Primary target |
| `linux-arm64` | ✅ Supported | Works with QEMU |
| `windows-amd64` | ⚠️ Experimental | MinGW required |
| `darwin-*` | ❌ Not supported | Accelerate can't be statically linked |

## Troubleshooting

### "undefined reference to GOMP_parallel"

Phase 3 failed - runtime libs weren't merged. Fall back to standard build:
```bash
go build -tags=nogpu  # Uses standard unified build
```

### Library size is still 45MB

Runtime libs didn't merge. Check:
```bash
./scripts/verify_phase3.sh linux-amd64
```

If verification fails, the build script should have automatically fallen back.

### Link errors about TLS

Your system doesn't support TLS in static archives. Use standard unified build instead.

## When to Use Phase 3

**✅ Use Phase 3 when:**
- Distributing to unknown environments
- Want absolute zero dependencies
- Building for embedded/minimal systems
- Docker containers with minimal base images

**❌ Don't use Phase 3 when:**
- macOS (not supported)
- You're okay with gomp/gfortran deps (standard unified is fine)
- Cross-compiling (complex)

## Future Work

### Potential Improvements

1. **Link-Time Optimization (LTO)**
   ```bash
   cmake -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
   ```
   Could reduce final size by eliminating dead code

2. **Strip Unused Symbols**
   ```bash
   strip --strip-unneeded libfaiss.a
   ```
   Might reduce size by 10-20%

3. **Custom OpenBLAS Build**
   ```bash
   NO_LAPACKE=1  # Remove LAPACK extras
   NO_CBLAS=1    # Keep only BLAS core
   ```
   Could reduce by another 5-10MB

4. **Static libstdc++**
   Try merging libstdc++.a too for complete independence

## Conclusion

Phase 3 is an ambitious experiment to achieve true zero-dependency static linking in Go/CGO.

**If it works**: You get truly self-contained binaries, simpler deployment, and the satisfaction of pushing the limits of static linking.

**If it doesn't**: Standard unified builds with minimal runtime deps (gomp + gfortran) are still excellent and follow best practices.

Either way, we're learning and pushing the boundaries of what's possible with Go + C++ integration!

## References

- [DuckDB Go Bindings](https://github.com/duckdb/duckdb-go-bindings) - Similar approach
- [GCC Static Linking Guide](https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html)
- [OpenBLAS Static Builds](https://github.com/xianyi/OpenBLAS/wiki)
- [CGO Documentation](https://pkg.go.dev/cmd/cgo)
