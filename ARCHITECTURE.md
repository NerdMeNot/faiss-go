# FAISS-Go Architecture

## Overview

This project uses a **DuckDB-inspired architecture** for managing static library builds across multiple platforms with different dependency strategies.

## Architecture Principles

1. **Platform-specific configuration** - Each platform has its own CGO configuration file
2. **Automatic selection** - Build tags automatically select the right configuration
3. **Multiple build modes** - Support standard, unified, and experimental zero-dep builds
4. **Clean separation** - Implementation separate from platform configuration

## File Structure

```
faiss-go/
├── faiss.go, index.go, etc.        # Public Go API
├── faiss_lib.go                    # Default build (Phase 2 unified)
├── faiss_lib_phase3.go             # Experimental build (Phase 3 zero-dep)
├── prebuilt_linux_amd64.go         # Linux AMD64 CGO config
├── prebuilt_linux_arm64.go         # Linux ARM64 CGO config
├── prebuilt_darwin_amd64.go        # macOS Intel CGO config
├── prebuilt_darwin_arm64.go        # macOS ARM64 CGO config
├── prebuilt_windows_amd64.go       # Windows AMD64 CGO config
├── libs/                           # Pre-built static libraries
│   ├── linux_amd64/libfaiss.a     # ~45MB unified
│   ├── linux_arm64/libfaiss.a     # ~45MB unified
│   ├── darwin_amd64/libfaiss.a    # ~9MB standard
│   ├── darwin_arm64/libfaiss.a    # ~9MB standard
│   └── windows_amd64/libfaiss.a   # ~45MB unified
├── scripts/
│   ├── build_static_lib.sh         # Standard & unified builds
│   ├── build_unified_static.sh     # Phase 3 aggressive builds
│   └── verify_phase3.sh            # Verify Phase 3 success
└── docs/
    ├── BUILD-MODES.md              # Comparison of all modes
    ├── STATIC-BUILDS.md            # Standard/unified details
    └── PHASE3-BUILDS.md            # Experimental zero-dep

```

## Build Tag Logic

### File Selection

```
User runs: go build -tags=nogpu

Build tag resolution:
1. Check for faiss_use_system? No → skip faiss_system.go
2. Check for faiss_phase3? No → skip faiss_lib_phase3.go
3. Use faiss_lib.go (default Phase 2 unified)
4. Select platform-specific config:
   - On Linux AMD64 → prebuilt_linux_amd64.go
   - On macOS ARM64 → prebuilt_darwin_arm64.go
   - etc.

Result: Platform-appropriate unified build
```

### Build Modes

| Command | Mode | Files Used |
|---------|------|------------|
| `go build -tags=faiss_use_system` | System | faiss_system.go |
| `go build -tags=nogpu` | Phase 2 Unified ⭐ | faiss_lib.go + prebuilt_*.go |
| `go build -tags="nogpu,faiss_phase3"` | Phase 3 Zero-dep | faiss_lib_phase3.go + prebuilt_*.go |

## Platform-Specific Configuration

Each `prebuilt_*.go` file contains:

```go
//go:build !faiss_use_system && !faiss_phase3 && linux && amd64

package faiss

/*
#cgo LDFLAGS: ${SRCDIR}/libs/linux_amd64/libfaiss_c.a ${SRCDIR}/libs/linux_amd64/libfaiss.a -lgomp -lgfortran -lm -lstdc++ -lpthread -ldl
*/
import "C"
```

**Key points:**
- Build constraints ensure only one config file is selected
- LDFLAGS link directly to `.a` files (ensures all symbols included, including custom C wrapper)
- Import "C" makes CGO directives active

## Custom C Wrapper Layer

The project includes a custom C wrapper (`faiss_c_impl.cpp`) that provides additional C API functions beyond the official FAISS C API. These wrapper functions are compiled and merged into `libfaiss_c.a` during the build process.

**Build process:**
1. Build FAISS libraries (libfaiss.a, libfaiss_c.a)
2. Compile faiss_c_impl.cpp → faiss_c_impl.o
3. Merge faiss_c_impl.o into libfaiss_c.a using `ar r`

**Why direct linking matters:**
- Using `-lfaiss_c` allows the linker to selectively pull object files
- Custom wrapper symbols might not be pulled if linker doesn't see them as "needed"
- Direct linking `${SRCDIR}/libs/*/libfaiss_c.a` forces inclusion of ALL object files
- This guarantees faiss_c_impl.o is always linked

**Example wrapper functions:**
- `faiss_IndexBinaryFlat_new` - Binary index creation
- `faiss_Kmeans_new` - K-means clustering
- Additional functions not in official FAISS C API

## Static Library Builds

### Phase 1: Standard Static
```
libfaiss.a (~9MB)
└── FAISS code only
Dependencies: System OpenBLAS
```

### Phase 2: Unified Static (Current Production) ⭐
```
libfaiss.a (~45MB)
├── FAISS code
└── OpenBLAS code (merged)
Dependencies: libgomp, libgfortran
```

### Phase 3: Zero-Dependency (Experimental) 🚀
```
libfaiss.a (~55MB)
├── FAISS code
├── OpenBLAS code
├── libgomp code (merged!)
├── libgfortran code (merged!)
└── libquadmath code (merged!)
Dependencies: NONE!
```

## How DuckDB Does It

DuckDB uses a similar approach:

```
duckdb-go/
├── duckdb.go                        # Main implementation
├── prebuilt_darwin_amd64.go        # Blank import of platform module
└── ...

duckdb-go-bindings/                  # Separate repository
├── lib/
│   ├── darwin-amd64/               # Separate Go module
│   │   ├── go.mod
│   │   ├── cgo.go                  # LDFLAGS here
│   │   └── libduckdb.a
│   └── linux-amd64/
│       ├── go.mod
│       ├── cgo.go
│       └── libduckdb.a
```

## Our Approach vs DuckDB

### DuckDB:
- ✅ Separate repository for binaries
- ✅ Each platform is separate Go module
- ✅ Blank imports select platform
- ❌ More complex (2 repos, module dependencies)

### Ours:
- ✅ Single repository
- ✅ Platform-specific files with build tags
- ✅ Simpler module structure
- ✅ Same automatic selection
- ✅ **Plus**: Experimental Phase 3 zero-dep builds!

## Future: Full DuckDB-Style Split (Optional)

If we wanted to fully match DuckDB's architecture:

```
Create faiss-go-libs repository:
├── lib/
│   ├── linux-amd64/
│   │   ├── go.mod  # github.com/NerdMeNot/faiss-go-libs/lib/linux-amd64
│   │   ├── cgo.go
│   │   └── libfaiss.a
│   └── ...

Update faiss-go/prebuilt_linux_amd64.go:
import _ "github.com/NerdMeNot/faiss-go-libs/lib/linux-amd64"
```

**Benefits:**
- Separate binary artifacts from source code
- Can version libraries independently
- Cleaner git history (no large binary commits)

**Tradeoffs:**
- More complex setup
- Two repositories to manage
- Module dependency overhead

**Decision:** Current single-repo approach is simpler and works well. Can split later if needed.

## Build Process

### Local Development

```bash
# Use existing pre-built libraries
go build -tags=nogpu

# Build your own unified library
./scripts/build_static_lib.sh linux-amd64 v1.13.2 --unified

# Try experimental Phase 3
./scripts/build_unified_static.sh linux-amd64 v1.13.2
./scripts/verify_phase3.sh linux-amd64
```

### CI/CD (GitHub Actions)

```yaml
# Standard unified builds (production)
.github/workflows/build-static-libs.yml

# Experimental Phase 3 builds
.github/workflows/build-phase3.yml
```

## Cross-Compilation

Thanks to platform-specific prebuilt files, cross-compilation "just works":

```bash
# Cross-compile for Linux ARM64 from macOS
GOOS=linux GOARCH=arm64 go build -tags=nogpu

# Build process:
# 1. GOOS=linux GOARCH=arm64 → selects prebuilt_linux_arm64.go
# 2. Links against libs/linux_arm64/libfaiss.a
# 3. Outputs ARM64 Linux binary
```

## Symbol Resolution

How symbols are resolved:

```
Go code calls:
  index.Add(vectors)
    ↓
  faissIndexAdd(ptr, vectors, n)  [in faiss_lib.go]
    ↓
  C.faiss_Index_add(idx, ...)      [CGO call]
    ↓
  [Linker looks for faiss_Index_add symbol]
    ↓
  [Found in libfaiss.a via LDFLAGS from prebuilt_*.go]
    ↓
  [faiss_Index_add calls cblas_* functions]
    ↓
  Phase 2: Found in libfaiss.a (OpenBLAS merged)
           but calls GOMP_*, _gfortran_* from system libs

  Phase 3: All found in libfaiss.a (everything merged!)
```

## Troubleshooting

### "undefined reference to GOMP_parallel"

**Cause:** Phase 3 failed to merge runtime libraries

**Solution:** Use Phase 2 unified build:
```bash
go build -tags=nogpu  # Don't use faiss_phase3 tag
```

### "multiple definition of symbol"

**Cause:** Multiple prebuilt files being compiled

**Check:**
```bash
go build -v -tags=nogpu 2>&1 | grep prebuilt
# Should show ONLY ONE prebuilt_*.go file
```

### "platform not supported"

**Cause:** No prebuilt file for your GOOS/GOARCH

**Solution:** Either:
1. Create prebuilt file for your platform
2. Use system build: `go build -tags=faiss_use_system`

## Design Decisions

### Why platform-specific files instead of one file with all platforms?

**One file approach:**
```go
/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/libs/linux_amd64 ...
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/libs/linux_arm64 ...
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/libs/darwin_amd64 ...
// ... 20+ more lines ...
*/
```

**Separate files approach (current):**
```go
// prebuilt_linux_amd64.go
#cgo LDFLAGS: -L${SRCDIR}/libs/linux_amd64 ...
```

**Benefits:**
- ✅ Easier to read and maintain
- ✅ Clear separation of concerns
- ✅ Can document platform-specific quirks
- ✅ Matches DuckDB's proven pattern
- ✅ Easier to add new platforms

### Why keep libs/ in main repo instead of separate repo?

**Current (libs/ in main repo):**
- ✅ Simpler for users (`go get` just works)
- ✅ Single repository to clone
- ✅ Libraries versioned with code
- ❌ Large files in git history

**Separate repo approach:**
- ✅ Cleaner git history
- ✅ Can update libraries independently
- ❌ Two repos to manage
- ❌ Module dependency complexity

**Decision:** Keep it simple for now. Can split later if git size becomes a problem.

## Summary

This architecture provides:

1. **Clean separation** - Platform config vs implementation
2. **Automatic selection** - Build tags handle everything
3. **Multiple strategies** - Standard, unified, and experimental builds
4. **DuckDB-inspired** - Proven patterns from similar projects
5. **Extensible** - Easy to add new platforms or build modes

It's **Go-esque** (simple, explicit, uses standard tools) while achieving sophisticated static linking goals.
