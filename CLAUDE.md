# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

faiss-go provides Go bindings for FAISS (Facebook AI Similarity Search), enabling billion-scale vector similarity search. The project emphasizes:
- **Fast builds** via pre-built static libraries (~30 seconds vs 15-30 minutes)
- **Zero dependencies** on supported platforms
- **Production quality** with comprehensive testing across Go 1.21-1.25
- **Type-safe CGO** bindings with explicit memory management

## Build Commands

### Building

```bash
# Default: Uses pre-built static libraries (fast, recommended)
go build -v ./...

# Alternative: Use Makefile
make build              # Same as above
make build-prebuilt     # Explicit static lib build
```

### Testing

```bash
# Quick tests during development
go test -short ./...

# Full test suite (recommended before commits)
go test -v ./...

# Specific test package
go test -v ./test/recall          # Recall validation tests
go test -v ./test/scenarios       # Real-world scenario tests
go test -v ./test/integration     # Integration tests

# Single test
go test -v -run TestIndexFlatL2_Search

# With coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Benchmarks

```bash
# All benchmarks
go test -bench=. -benchmem ./...

# Specific benchmark
go test -bench=BenchmarkIndexFlatL2 -benchtime=5s

# Quick smoke test
go test -bench=. -benchtime=100ms ./...
```

### Linting

```bash
# Format code
make fmt

# Run linters
make lint

# Full golangci-lint (as in CI)
golangci-lint run --timeout=5m
```

### Examples

```bash
make example-basic      # Run basic_search.go
make example-ip         # Run inner_product.go
```

## Architecture

### Build System Architecture

faiss-go uses **build tags** to support two build modes:

1. **Static Libraries (Default)** - `//go:build !faiss_use_system`
   - File: `faiss_lib.go` (imports `faiss-go-bindings` module)
   - Uses pre-built static libraries from `github.com/NerdMeNot/faiss-go-bindings`
   - Platform-specific LDFLAGS in bindings module's `cgo_*.go` files
   - Fast builds (~30s), no dependencies

2. **System FAISS (Fallback)** - `//go:build faiss_use_system`
   - File: `faiss_system.go`
   - Links against system-installed FAISS
   - For custom FAISS builds or unsupported platforms

**Key insight**: Build tags automatically select the right `.go` files for compilation.

### Core Index Architecture

All indexes implement the `Index` interface defined in `index.go`:

```go
type Index interface {
    D() int                    // Vector dimension
    Ntotal() int64             // Total vectors indexed
    IsTrained() bool           // Training status
    MetricType() MetricType    // L2 or InnerProduct
    Train([]float32) error     // Train (no-op for flat indexes)
    Add([]float32) error       // Add vectors
    Search([]float32, int) (distances, indices, error)
    Reset() error              // Remove all vectors
    Close() error              // Free C++ resources
}
```

**Index implementations** (one file per index type):
- `faiss.go` - IndexFlat (base flat index)
- `index_ivf.go` - IndexIVFFlat (inverted file index)
- `index_hnsw.go` - IndexHNSW (hierarchical navigable small world)
- `index_pq.go` - IndexPQ (product quantization)
- `index_sq.go` - IndexSQ (scalar quantization)
- `index_idmap.go` - IndexIDMap (custom ID mapping wrapper)
- `index_composite.go` - Composite indexes (IndexShards, etc.)
- `index_binary.go` - Binary indexes
- `index_gpu.go` - GPU indexes

### Critical Memory Management Pattern

**FAISS manages C++ objects**. Go wrappers MUST:

1. **Keep quantizer references alive** for composite indexes:
   ```go
   type IndexIVFFlat struct {
       quantizer Index  // CRITICAL: prevents GC of quantizer
       // ...
   }
   ```

2. **Set `own_fields=0`** to prevent double-free:
   ```go
   // IVF indexes
   faiss_IndexIVF_set_own_fields(ptr, 0)

   // IDMap indexes
   faiss_IndexIDMap_set_own_fields(ptr, 0)

   // Sharded indexes
   faiss_IndexShards_set_own_indices(ptr, 0)
   ```

3. **Use finalizers** as safety net:
   ```go
   runtime.SetFinalizer(idx, func(i *IndexFlat) {
       if i.ptr != 0 {
           _ = i.Close()
       }
   })
   ```

**Recent critical fixes** (see recent commits):
- Fixed signature mismatches in Kmeans wrappers (exception safety)
- Fixed use-after-free in composite indexes via `own_fields=0`
- Restored custom C wrapper for FAISS 1.13.2 missing functions

### CGO Function Organization

CGO bridge functions in `faiss_lib.go`:

```go
// Pattern: faiss<Type><Operation>
func faissIndexFlatL2New(d int) (uintptr, error)
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error
func faissIndexSearch(ptr uintptr, queries []float32, nq, k int, distances []float32, indices []int64) error
```

**ALWAYS**:
- Check return codes from C functions
- Validate inputs before CGO calls
- Free C memory with `defer C.free()`
- Convert errors to descriptive Go errors

### Test Organization

```
test/
├── datasets/        # Test data generators and loaders
│   ├── generators.go      # Random vector generation
│   ├── loader.go          # Dataset loading utilities
│   └── groundtruth/       # Ground truth caching
├── helpers/         # Test utilities
│   ├── test_helpers.go    # Common test setup/teardown
│   ├── test_utils.go      # Assertion helpers
│   └── recall_calculator.go  # Recall computation
├── recall/          # Recall validation (critical for quality)
│   ├── framework.go       # Recall testing framework
│   ├── ivf_recall_test.go
│   ├── hnsw_recall_test.go
│   ├── pq_recall_test.go
│   └── combined_recall_test.go
├── scenarios/       # Real-world use cases
│   ├── semantic_search_test.go
│   ├── image_similarity_test.go
│   ├── recommendations_test.go
│   └── streaming_test.go
└── integration/     # End-to-end tests
    └── integration_test.go
```

**Recall tests** validate search quality - these are essential and must pass.

### Static Library Structure

Static libraries are provided by the `faiss-go-bindings` module:

```
github.com/NerdMeNot/faiss-go-bindings/
├── lib/
│   ├── linux_amd64/
│   │   ├── libfaiss.a        # Unified FAISS library
│   │   ├── libfaiss_c.a      # FAISS C API wrapper
│   │   └── libfaiss_go_ext.a # Go-specific extensions
│   ├── linux_arm64/
│   ├── darwin_amd64/
│   └── darwin_arm64/
└── include/                   # C API headers
```

The bindings module is automatically downloaded and cached by Go modules.

## Development Patterns

### Adding a New Index Type

1. **Create new file** `index_<type>.go`
2. **Define struct** with `ptr`, `d`, `metric`, `ntotal`, `isTrained` fields
3. **Implement Index interface** (all methods required)
4. **Add CGO functions** in `faiss_lib.go` (after C function declarations)
5. **Set `own_fields=0`** if index wraps another index
6. **Add tests** in `*_test.go` (recall validation essential)
7. **Update serialization** in `serialization.go` (ReadIndex/WriteIndex)

### CGO Best Practices (Critical)

**From programming-guide.md - MUST READ**:

1. **Minimize CGO calls** (50ns overhead each)
   - Cache index properties (dimension, ntotal)
   - Batch operations when possible

2. **Safe pointer conversion**:
   ```go
   idx := C.FaissIndex(unsafe.Pointer(ptr))  // Explicit types
   vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
   ```

3. **Always check empty slices**:
   ```go
   if len(vectors) == 0 {
       return nil  // Avoid &vectors[0] panic
   }
   ```

4. **Free C memory**:
   ```go
   cStr := C.CString(filename)
   defer C.free(unsafe.Pointer(cStr))
   ```

5. **Check return codes**:
   ```go
   ret := C.faiss_Index_add(idx, n, vecPtr)
   if ret != 0 {
       return fmt.Errorf("FAISS error code: %d", ret)
   }
   ```

### Error Handling

- **Wrap errors** with context: `fmt.Errorf("failed to add vectors: %w", err)`
- **Pre-defined errors** in `faiss.go`: `ErrInvalidDimension`, `ErrInvalidVectors`, `ErrIndexNotTrained`, `ErrNullPointer`
- **Validate inputs** before CGO calls to provide clear Go errors
- **Return descriptive errors** (not just "error code: 1")

### Testing Guidelines

1. **Table-driven tests** for multiple scenarios
2. **Always `defer index.Close()`** to prevent leaks
3. **Use `-short` flag** for slow tests:
   ```go
   if testing.Short() {
       t.Skip("skipping slow test in short mode")
   }
   ```
4. **Validate recall** for approximate indexes (see `test/recall/`)
5. **Benchmark performance-critical code**

### CRITICAL: Use Structured Test Data, NOT Random Data

**NEVER** write tests like this (flaky, meaningless):
```go
// BAD: Random data with arbitrary thresholds
vectors := generateRandomVectors(10000, 128)
recall := runSearch(index, vectors)
if recall < 0.15 {  // Why 0.15? Just matches whatever random gives us
    t.Error("recall too low")
}
```

**ALWAYS** use deterministic, structured data where you KNOW the expected results:
```go
// GOOD: Clustered data where we KNOW neighbors should be in same cluster
vectors, clusterIDs := generateClusteredData(nClusters, pointsPerCluster, dim)
// Query cluster 5 → neighbors MUST be from cluster 5
// This actually tests that the algorithm works!
```

**Structured test patterns** (see `test/scenarios/structured_test.go`):
- **Clustered data**: Create well-separated clusters. Query a point → verify neighbors are from same cluster
- **Known duplicates**: Insert exact copies → verify they match with distance 0
- **Ordered data**: Create vectors at known distances → verify correct ordering
- **Hierarchical data**: Categories + subcategories → verify recommendations stay in category

This approach:
- Tests actual algorithm correctness, not random variance
- Is deterministic and reproducible
- Eliminates flaky tests from system load or random seeds
- Provides meaningful pass/fail criteria

### Code Style

- **Follow Effective Go** and idiomatic patterns
- **Descriptive test names**: `TestIndexFlatL2_SearchReturnsNearestNeighbors`
- **Comment WHY, not WHAT**:
   ```go
   // Set own_fields=0 to prevent FAISS from freeing the quantizer
   // (Go manages the quantizer lifecycle via GC and finalizers)
   faiss_IndexIVF_set_own_fields(ptr, 0)
   ```
- **Public APIs need godoc** with examples
- **Use `golangci-lint`** (CI enforces this)

## Critical Gotchas

### 1. Composite Index Ownership
**ALWAYS** set `own_fields=0` for indexes that wrap other indexes:
- IndexIVF* (wraps quantizer)
- IndexIDMap (wraps base index)
- IndexShards (wraps sub-indexes)

**Why**: FAISS would double-free the wrapped index (Go also frees it).

### 2. Vector Slice Layout
Vectors are **flattened**: `[v1_d1, v1_d2, ..., v1_dN, v2_d1, v2_d2, ...]`
- Length must be `n_vectors * dimension`
- Check with `len(vectors) % d != 0` → error

### 3. Training Required
Some indexes need training before adding vectors:
- IndexIVF* (needs training data for clustering)
- IndexPQ (needs training for quantization)
- IndexHNSW does NOT need training

### 4. CGO Overhead
- Don't call CGO in hot loops
- Cache index properties (dimension, ntotal)
- Batch operations when possible

### 5. Platform-Specific LDFLAGS
Static library builds use platform-specific linker flags from the bindings module:
- Linux: needs `-lgomp -lgfortran` for OpenBLAS
- macOS: uses Accelerate framework

## CI/CD

### Test Matrix
- **Go versions**: 1.21, 1.22, 1.23, 1.24, 1.25
- **OS**: Ubuntu, macOS
- **Architectures**: AMD64, ARM64
- **Total**: 10 parallel jobs (5 Go versions × 2 OSes)

### Workflows
- `.github/workflows/ci.yml` - Main CI (runs on every push)
- `.github/workflows/benchmark.yml` - Comprehensive benchmarks (manual)
- `.github/workflows/build-static-libs.yml` - Rebuild static libraries
- `.github/workflows/gpu-ci.yml` - GPU tests (requires GPU runner)

### Pre-commit Checklist
```bash
go build -v ./...                    # Build succeeds
go test -v ./...                     # All tests pass
go test -bench=. -benchtime=100ms    # Benchmarks work
golangci-lint run --timeout=5m       # Linting passes
```

## Common Tasks

### Running a Single Test
```bash
go test -v -run TestIndexIVFFlat_SearchAfterTrain
```

### Debugging a Segfault
```bash
# Build with debug info
go build -gcflags="all=-N -l" -v ./...

# Run with race detector
go test -race -v ./...

# Check for null pointers, missing own_fields=0, double-free
```

### Updating Static Libraries
```bash
# Build for current platform
./scripts/build-static-libs.sh

# Test new libraries
go clean -cache
go build -v ./...
go test -v ./...
```

### Adding Documentation
- Update godoc comments in source files
- Add examples in `examples/`
- Update `docs/*.md` if needed
- **Do NOT** create new markdown files unless necessary

## Key References

- **API patterns**: See `index.go` for interface definitions
- **CGO patterns**: Read `docs/programming-guide.md` (comprehensive guide)
- **Memory safety**: See recent commits on `own_fields` fixes
- **Build system**: Read `docs/BUILD-MODES.md`
- **Testing**: See `CONTRIBUTING.md` testing section
