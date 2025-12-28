# Frequently Asked Questions

Common questions about faiss-go.

---

## General Questions

### What is faiss-go?

faiss-go provides Go bindings for FAISS (Facebook AI Similarity Search) with FAISS embedded directly in the library. No separate FAISS installation required.

### How does it compare to other Go bindings?

faiss-go embeds FAISS similar to how go-duckdb embeds DuckDB:
- ✅ No separate FAISS installation needed
- ✅ Two build modes: source compilation or pre-built libraries
- ✅ Single `go get` command to install
- ✅ Complete feature coverage

---

## Build Questions

### Why does the first build take 5-10 minutes?

When building from source (default mode), CGO compiles the entire FAISS library from amalgamated C++ code (~10-15 MB). This happens once - subsequent builds use Go's cache and complete in seconds.

### Can I speed up builds?

Yes! Use pre-built libraries:

```bash
go build -tags=faiss_use_lib
```

This uses pre-compiled FAISS libraries and builds in ~30 seconds.

### What dependencies do I need?

**For pre-built libraries mode:**
- Go 1.21+ only

**For source build mode:**
- Go 1.21+
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2019+)
- BLAS library (OpenBLAS or MKL)

---

## Usage Questions

### What index types are supported?

All major FAISS index types:
- Flat indexes (exact search)
- IVF indexes (fast approximate)
- HNSW indexes (best recall/speed balance)
- PQ indexes (compression)
- GPU indexes (acceleration)
- OnDisk indexes (billion-scale)

See [Index Types Guide](guides/index-types.md) for details.

### Is it thread-safe?

Index operations are not thread-safe by default. For concurrent access:

1. **Use synchronization:**
   ```go
   var mu sync.Mutex
   mu.Lock()
   defer mu.Unlock()
   index.Add(vectors)
   ```

2. **Use separate indexes per goroutine (for read-heavy workloads)**

### How do I normalize vectors for cosine similarity?

```go
func normalize(vectors []float32, dimension int) []float32 {
    numVectors := len(vectors) / dimension
    normalized := make([]float32, len(vectors))
    copy(normalized, vectors)

    for i := 0; i < numVectors; i++ {
        start := i * dimension
        end := start + dimension
        vec := normalized[start:end]

        var norm float32
        for _, v := range vec {
            norm += v * v
        }
        norm = float32(math.Sqrt(float64(norm)))

        if norm > 0 {
            for j := range vec {
                vec[j] /= norm
            }
        }
    }
    return normalized
}
```

### Can I save and load indexes?

Yes! Serialization is fully supported:

```go
// Save
err := faiss.WriteIndex(index, "my_index.faiss")

// Load
index, err := faiss.ReadIndex("my_index.faiss")
defer index.Close()
```

---

## Performance Questions

### Should I use L2 or Inner Product?

**Use L2 distance when:**
- Vector magnitude matters
- Traditional nearest neighbor search
- Lower distance = more similar

**Use Inner Product when:**
- Vectors are normalized (cosine similarity)
- Maximum similarity search
- Higher score = more similar

### What about GPU acceleration?

GPU indexes are supported for massive throughput:

```go
gpuIndex, _ := faiss.NewGpuIndexFlatL2(0, dimension) // GPU 0
```

Requires CUDA-capable GPU.

---

## Platform Questions

### Does it work on Windows?

Yes! Both build modes work on Windows:
- Pre-built libraries included
- Source build requires MSYS2 or Visual Studio

### Does it work on Apple Silicon (M1/M2)?

Yes! Fully supported on ARM64 macOS:
- Source build uses Accelerate framework
- Pre-built ARM64 libraries included

### Does it work on ARM64 Linux?

Yes, including Raspberry Pi and cloud ARM instances.

---

## Troubleshooting

### Build fails with "BLAS not found"

Install OpenBLAS or use pre-built libraries:
```bash
# Install OpenBLAS
sudo apt-get install libopenblas-dev  # Ubuntu/Debian
brew install openblas                  # macOS

# Or use pre-built
go build -tags=faiss_use_lib
```

### Runtime error: dimension mismatch

Ensure vector slice length is exactly `numVectors × dimension`:

```go
vectors := make([]float32, numVectors * dimension)
```

### Memory leaks

Always call `Close()`:

```go
index, _ := faiss.NewIndexFlatL2(dimension)
defer index.Close()  // Essential!
```

---

## Development Questions

### How do I contribute?

See [Contributing Guide](development/contributing.md).

### Where do I report issues?

[GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues)

---

**Still have questions?** [Open a discussion](https://github.com/NerdMeNot/faiss-go/discussions)
