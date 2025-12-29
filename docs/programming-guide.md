# Programming Guide & Best Practices

This document explains the programming practices, patterns, and design decisions used in faiss-go. It serves as both a guide for contributors and a showcase of the quality standards we maintain.

---

## Table of Contents

- [Philosophy](#philosophy)
- [CGO Best Practices](#cgo-best-practices)
- [Memory Management](#memory-management)
- [Error Handling](#error-handling)
- [API Design Principles](#api-design-principles)
- [Build System Architecture](#build-system-architecture)
- [Testing Philosophy](#testing-philosophy)
- [Performance Considerations](#performance-considerations)
- [Code Organization](#code-organization)

---

## Philosophy

### Core Principles

1. **Safety First** - Memory safety, type safety, no undefined behavior
2. **Developer Experience** - Fast builds, clear errors, good documentation
3. **Quality Over Speed** - Comprehensive testing before shipping
4. **Idiomatic Go** - Feel natural to Go developers, not a Python port
5. **Zero Surprises** - Explicit is better than implicit

### Why These Matter

faiss-go bridges two very different worlds: Go's safety and simplicity with C++'s raw performance. Our practices ensure this bridge is solid, safe, and pleasant to cross.

---

## CGO Best Practices

CGO is powerful but dangerous. We follow strict patterns to avoid the common pitfalls.

### Pattern 1: Safe Pointer Conversion

**❌ WRONG:**
```go
// Dangerous: No type safety, easy to get wrong
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error {
    idx := unsafe.Pointer(ptr)
    vecPtr := unsafe.Pointer(&vectors[0])
    C.faiss_Index_add(idx, C.int64_t(n), vecPtr)  // Type errors possible
    return nil
}
```

**✅ CORRECT:**
```go
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error {
    // Explicit type conversion with clear intent
    idx := C.FaissIndex(unsafe.Pointer(ptr))
    vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))

    // Always check return codes
    ret := C.faiss_Index_add(idx, C.int64_t(n), vecPtr)
    if ret != 0 {
        return fmt.Errorf("FAISS error code: %d", ret)
    }
    return nil
}
```

**Why:**
- Explicit types catch errors at compile time
- Return code checking prevents silent failures
- Clear error messages help debugging

### Pattern 2: C Memory Management

**❌ WRONG:**
```go
func example() error {
    cStr := C.CString("filename.faiss")
    // Forgot to free - memory leak!
    return C.faiss_write_index(idx, cStr)
}
```

**✅ CORRECT:**
```go
func faissWriteIndex(ptr uintptr, filename string) error {
    idx := C.FaissIndex(unsafe.Pointer(ptr))
    cFilename := C.CString(filename)
    defer C.free(unsafe.Pointer(cFilename))  // Always free!

    ret := C.faiss_write_index(idx, cFilename)
    if ret != 0 {
        return fmt.Errorf("FAISS error code: %d", ret)
    }
    return nil
}
```

**Why:**
- `defer` ensures cleanup even on early returns
- Prevents memory leaks in long-running applications
- Idiomatic Go pattern

### Pattern 3: Minimize CGO Calls

**❌ WRONG:**
```go
// Calling C function in hot loop
for i := 0; i < 1000000; i++ {
    C.faiss_get_dimension(idx)  // CGO overhead on every iteration!
}
```

**✅ CORRECT:**
```go
// Call once, cache the result
dimension := C.faiss_get_dimension(idx)
for i := 0; i < 1000000; i++ {
    // Use cached dimension
    processVector(dimension)
}
```

**Why:**
- CGO calls have ~50ns overhead (100x slower than Go function call)
- Batching operations improves performance significantly
- Cache invariant properties

### Pattern 4: Slice to C Array Conversion

**❌ WRONG:**
```go
// Unsafe: Doesn't check for empty slices
func addVectors(vectors []float32) error {
    ptr := (*C.float)(unsafe.Pointer(&vectors[0]))  // Panic if vectors is empty!
    C.faiss_Index_add(idx, C.int64_t(len(vectors)), ptr)
    return nil
}
```

**✅ CORRECT:**
```go
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error {
    // Validate input
    if len(vectors) == 0 {
        return nil  // or error, depending on semantics
    }

    idx := C.FaissIndex(unsafe.Pointer(ptr))
    vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))

    ret := C.faiss_Index_add(idx, C.int64_t(n), vecPtr)
    if ret != 0 {
        return fmt.Errorf("FAISS error code: %d", ret)
    }
    return nil
}
```

**Why:**
- Prevents panics from accessing empty slices
- Validates preconditions before unsafe operations
- Graceful handling of edge cases

---

## Memory Management

### Resource Lifecycle

Every FAISS object must be explicitly closed to prevent leaks.

**Public API Pattern:**
```go
type Index struct {
    ptr    uintptr
    d      int
    closed bool
    mu     sync.Mutex  // Protect against concurrent Close()
}

func (idx *Index) Close() error {
    idx.mu.Lock()
    defer idx.mu.Unlock()

    if idx.closed {
        return nil  // Idempotent
    }

    if idx.ptr != 0 {
        if err := faissIndexFree(idx.ptr); err != nil {
            return err
        }
        idx.ptr = 0
    }

    idx.closed = true
    return nil
}
```

**Why:**
- Idempotent Close() prevents double-free bugs
- Mutex protects against concurrent Close() calls
- Zero-ing ptr prevents use-after-free

**Usage Pattern:**
```go
func Example() error {
    index, err := faiss.NewIndexFlatL2(128)
    if err != nil {
        return err
    }
    defer index.Close()  // Always defer Close!

    // Use index...
    return nil
}
```

### Finalizers: When NOT to Use Them

**We deliberately don't use finalizers** for FAISS objects:

```go
// ❌ WRONG: Don't do this!
func NewIndex(d int) (*Index, error) {
    idx := &Index{...}
    runtime.SetFinalizer(idx, func(idx *Index) {
        idx.Close()  // Unreliable!
    })
    return idx, nil
}
```

**Why finalizers are bad here:**
1. **Unpredictable timing** - GC runs when it wants, not when resources are tight
2. **CGO can't call from finalizers** - Causes crashes
3. **False sense of security** - Developers forget to Close()
4. **Better to fail fast** - Explicit Close() makes leaks obvious

**Our approach:**
- Require explicit `Close()` (use `defer`)
- Document resource management clearly
- Add tests that verify cleanup

---

## Error Handling

### Pattern: Wrapping FAISS Errors

**Low-level wrapper (internal):**
```go
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error {
    idx := C.FaissIndex(unsafe.Pointer(ptr))
    vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))

    ret := C.faiss_Index_add(idx, C.int64_t(n), vecPtr)
    if ret != 0 {
        return fmt.Errorf("FAISS error code: %d", ret)
    }
    return nil
}
```

**High-level API (public):**
```go
func (idx *Index) Add(vectors []float32) error {
    // Validate preconditions
    if idx.closed {
        return errors.New("index is closed")
    }

    expectedLen := idx.Ntotal() * idx.D()
    if len(vectors)%idx.d != 0 {
        return fmt.Errorf("vector dimensions mismatch: got %d, expected multiple of %d",
            len(vectors), idx.d)
    }

    n := len(vectors) / idx.d

    // Call low-level wrapper
    if err := faissIndexAdd(idx.ptr, vectors, n); err != nil {
        return fmt.Errorf("failed to add vectors: %w", err)
    }

    return nil
}
```

**Why:**
- Low-level: Simple, translates C errors to Go errors
- High-level: Validates inputs, provides context
- Error wrapping (`%w`) enables `errors.Is()` and `errors.As()`
- Clear error messages help debugging

### Pattern: Sentinel Errors

For errors that callers might want to check:

```go
var (
    ErrIndexClosed     = errors.New("index is closed")
    ErrNotTrained      = errors.New("index not trained")
    ErrDimensionMismatch = errors.New("vector dimension mismatch")
)

func (idx *Index) Search(query []float32, k int) ([]float32, []int64, error) {
    if idx.closed {
        return nil, nil, ErrIndexClosed  // Can be checked with errors.Is()
    }
    // ...
}
```

**Usage:**
```go
_, _, err := index.Search(query, 5)
if errors.Is(err, faiss.ErrIndexClosed) {
    // Handle closed index specifically
}
```

---

## API Design Principles

### Principle 1: Idiomatic Go

**❌ Python-style (WRONG):**
```go
// Mimics Python too closely
index.search(query, k=5, nprobe=10)  // Named args don't exist in Go
```

**✅ Go-style (CORRECT):**
```go
// Idiomatic Go: simple method, explicit configuration
index.SetNprobe(10)
distances, labels, err := index.Search(query, 5)
```

### Principle 2: Fail Fast with Good Errors

```go
func (idx *Index) Add(vectors []float32) error {
    // Check preconditions BEFORE calling C
    if idx.closed {
        return ErrIndexClosed
    }

    if len(vectors) == 0 {
        return nil  // Early return for degenerate case
    }

    if len(vectors)%idx.d != 0 {
        return fmt.Errorf("invalid vector data: length %d is not a multiple of dimension %d",
            len(vectors), idx.d)
    }

    // Now safe to call C
    return faissIndexAdd(idx.ptr, vectors, len(vectors)/idx.d)
}
```

**Why:**
- Catch errors in Go before crossing CGO boundary
- Better error messages (Go errors > C error codes)
- Fail fast prevents undefined behavior

### Principle 3: Zero Values Are Useful

```go
type SearchOptions struct {
    K       int     // Number of neighbors (required)
    Nprobe  int     // IVF cells to search (0 = use default)
    EfSearch int    // HNSW search depth (0 = use default)
}

func (idx *Index) SearchWithOptions(query []float32, opts SearchOptions) error {
    // Zero values have sensible defaults
    if opts.K == 0 {
        return errors.New("K must be > 0")
    }

    if opts.Nprobe == 0 {
        opts.Nprobe = idx.defaultNprobe  // Use default
    }

    // ...
}
```

**Why:**
- Follows Go convention that zero values are useful
- Makes optional parameters easy
- Backwards compatible (add fields without breaking API)

---

## Build System Architecture

### Two-Mode Build: The Innovation

```go
//go:build !faiss_use_system
// faiss_lib.go - DEFAULT: Uses pre-built static libraries

//go:build faiss_use_system
// faiss_system.go - FALLBACK: Uses system FAISS

//go:build faiss_use_system
// faiss_c_impl.cpp - C++ bridge (only for system mode)
```

**Why this design?**

1. **Default mode is fastest** - Pre-built binaries = 30-second builds
2. **Fallback for flexibility** - System FAISS for custom builds/platforms
3. **Auto-detection** - No user configuration needed
4. **No runtime switching** - Build-time decision = zero overhead

### Build Tag Strategy

```go
// Both modes implement the same functions:
func faissIndexFlatL2New(d int) (uintptr, error)
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error
// ... etc

// But different implementations:
// - faiss_lib.go: Links against libs/*/libfaiss.a
// - faiss_system.go: Compiles faiss_c_impl.cpp, links -lfaiss
```

**Benefits:**
- Public API is identical (mode is invisible to users)
- Easy to test both modes
- Can switch modes with just build tags

---

## Testing Philosophy

### Recall Validation

**Why it matters:**
Approximate indexes trade accuracy for speed. We MUST verify they actually work.

```go
func TestIndexIVFFlat_RecallValidation(t *testing.T) {
    // 1. Generate ground truth with exact search
    exactIndex, _ := faiss.NewIndexFlatL2(128)
    exactIndex.Add(vectors)
    exactDist, exactLabels, _ := exactIndex.Search(queries, 10)

    // 2. Test approximate index
    ivfIndex, _ := faiss.NewIndexIVFFlat(quantizer, 128, 100, faiss.MetricL2)
    ivfIndex.Train(vectors)
    ivfIndex.Add(vectors)
    ivfDist, ivfLabels, _ := ivfIndex.Search(queries, 10)

    // 3. Calculate recall
    recall := calculateRecall(exactLabels, ivfLabels)

    // 4. Verify acceptable recall
    require.Greater(t, recall, 0.95, "Recall too low")
}
```

**Why this pattern:**
- Ensures indexes actually find similar vectors
- Catches subtle bugs (wrong distance metric, improper training)
- Documents expected accuracy

### Benchmark Patterns

```go
func BenchmarkIndexFlatL2_Search(b *testing.B) {
    index, _ := faiss.NewIndexFlatL2(128)
    vectors := generateRandomVectors(10000, 128)
    index.Add(vectors)
    query := vectors[:128]

    b.ResetTimer()  // Don't measure setup time

    for i := 0; i < b.N; i++ {
        b.StopTimer()   // Pause timing
        // Reset state if needed
        b.StartTimer()  // Resume timing

        index.Search(query, 10)
    }
}
```

**Common mistakes we avoid:**
- ❌ Measuring setup time
- ❌ Not resetting state between iterations
- ❌ Forgetting `b.ReportAllocs()` for memory benchmarks

### Cross-Platform Testing Strategy

Our CI tests:
- **5 Go versions**: 1.21, 1.22, 1.23, 1.24, 1.25
- **2 OSes**: Ubuntu, macOS
- **2 architectures**: AMD64, ARM64 (via runners)
- **11 parallel jobs**: Fast feedback

**Why this coverage?**
- Go version compatibility (APIs change)
- Platform differences (BLAS libraries, calling conventions)
- Architecture differences (ARM vs x86 SIMD)

---

## Performance Considerations

### Hot Path Optimization

```go
// ❌ WRONG: Creating error in hot path
func (idx *Index) Search(query []float32, k int) ([]float32, []int64, error) {
    for i := 0; i < 1000000; i++ {
        if idx.closed {
            return nil, nil, fmt.Errorf("index is closed")  // Allocates every time!
        }
    }
}

// ✅ CORRECT: Check once, use sentinel error
var ErrIndexClosed = errors.New("index is closed")  // Allocated once

func (idx *Index) Search(query []float32, k int) ([]float32, []int64, error) {
    if idx.closed {
        return nil, nil, ErrIndexClosed  // No allocation
    }
    // ... hot path ...
}
```

### Slice Preallocation

```go
// ✅ CORRECT: Preallocate result slices
func (idx *Index) Search(query []float32, k int) ([]float32, []int64, error) {
    nq := len(query) / idx.d

    // Preallocate exact size
    distances := make([]float32, nq*k)
    labels := make([]int64, nq*k)

    // Fill in-place (no reallocations)
    if err := faissIndexSearch(idx.ptr, query, nq, k, distances, labels); err != nil {
        return nil, nil, err
    }

    return distances, labels, nil
}
```

**Why:**
- Avoids slice growth reallocations
- Better cache locality
- Predictable memory usage

### Batch Operations

```go
// ❌ WRONG: One vector at a time
for _, vec := range vectors {
    index.Add(vec)  // CGO overhead for each vector!
}

// ✅ CORRECT: Batch add
allVectors := flattenVectors(vectors)
index.Add(allVectors)  // Single CGO call
```

---

## Code Organization

### File Structure Philosophy

```
index.go          # Core Index interface, common operations
index_flat.go     # Flat indexes (exact search)
index_ivf.go      # IVF indexes (partitioned search)
index_hnsw.go     # HNSW indexes (graph-based)
index_pq.go       # Product quantization indexes
index_gpu.go      # GPU indexes

faiss_lib.go      # Static library build mode (default)
faiss_system.go   # System FAISS build mode (fallback)
faiss_c_impl.cpp  # C++ bridge (system mode only)
```

**Principles:**
- One file per index family
- Build modes in separate files (build tags)
- Test files alongside implementation (`*_test.go`)

### Internal vs Public API

```go
// Public API (index.go)
func (idx *Index) Add(vectors []float32) error {
    // Validation, type checking, error wrapping
    return faissIndexAdd(idx.ptr, vectors, n)
}

// Internal wrapper (faiss_lib.go / faiss_system.go)
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error {
    // Direct CGO call, minimal logic
    idx := C.FaissIndex(unsafe.Pointer(ptr))
    vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
    ret := C.faiss_Index_add(idx, C.int64_t(n), vecPtr)
    if ret != 0 {
        return fmt.Errorf("FAISS error code: %d", ret)
    }
    return nil
}
```

**Why this separation?**
- Public API can change without changing CGO layer
- Internal functions are testable independently
- Build modes share public API, differ only in implementation

---

## Summary: Why These Practices Matter

1. **Safety** - CGO is dangerous; our patterns prevent common bugs
2. **Performance** - Minimize CGO overhead, preallocate, batch operations
3. **Testability** - Recall validation, cross-platform CI, benchmarks
4. **Maintainability** - Clear patterns, good organization, comprehensive docs
5. **Developer Experience** - Fast builds, clear errors, idiomatic Go

These aren't just best practices—they're what makes faiss-go **production-ready despite being new**. Quality from day one.

---

## References

- [Effective Go](https://go.dev/doc/effective_go)
- [CGO Best Practices](https://github.com/golang/go/wiki/cgo)
- [Go Testing Best Practices](https://go.dev/blog/table-driven-tests)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)

---

**Contributing?** See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to apply these practices in your PRs.
