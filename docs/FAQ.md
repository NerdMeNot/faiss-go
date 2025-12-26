# Frequently Asked Questions (FAQ)

## General Questions

### What is faiss-go?

faiss-go provides Go bindings for FAISS (Facebook AI Similarity Search) with FAISS embedded directly in the library. You don't need to install or compile FAISS separately.

### How does it compare to other FAISS Go bindings?

Most existing FAISS Go bindings require you to:
1. Install FAISS separately
2. Manage system dependencies
3. Link against pre-installed FAISS libraries

faiss-go embeds FAISS, similar to how go-duckdb embeds DuckDB:
- ✅ No separate FAISS installation needed
- ✅ Two build modes: source compilation or pre-built libraries
- ✅ Single `go get` command to install

## Build Questions

### Why does the first build take 5-10 minutes?

When building from source (default mode), CGO compiles the entire FAISS library from the amalgamated source file (~10-15 MB of C++ code). This only happens once - subsequent builds use Go's build cache and complete in seconds.

### Can I speed up the build?

Yes! Use pre-built libraries:

```bash
go build -tags=faiss_use_lib
```

This uses pre-compiled FAISS libraries and builds in ~30 seconds.

### Do I need to rebuild FAISS every time?

No. Go caches compiled C/C++ code. You only recompile FAISS when:
- First time building the project
- Updating to a new faiss-go version
- Running `go clean -cache` (which clears the cache)

### What dependencies do I need?

**For pre-built libraries mode:**
- None! Just Go 1.21+

**For source build mode:**
- Go 1.21+
- C++17 compiler (GCC 7+, Clang 5+, or MSVC 2019+)
- BLAS library (OpenBLAS or MKL)

### How do I install BLAS?

**Ubuntu/Debian:**
```bash
sudo apt-get install libopenblas-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install openblas-devel
```

**macOS:**
```bash
brew install openblas
```

**Windows:**
Use MSYS2 or vcpkg to install OpenBLAS.

## Usage Questions

### What index types are supported?

Currently supported:
- ✅ `IndexFlatL2` - Exact L2 distance search
- ✅ `IndexFlatIP` - Exact inner product search

Coming soon:
- 🚧 `IndexIVFFlat` - Inverted file index
- 🚧 `IndexIVFPQ` - Product quantization
- 🚧 `IndexHNSW` - Hierarchical navigable small world

### Is it thread-safe?

Index operations are **not thread-safe** by default. If you need concurrent access:

1. **Use mutex synchronization:**
   ```go
   var mu sync.Mutex
   mu.Lock()
   defer mu.Unlock()
   index.Add(vectors)
   ```

2. **Use separate indexes per goroutine:**
   ```go
   // Better for read-heavy workloads
   indexPerWorker := make([]*faiss.IndexFlat, numWorkers)
   for i := range indexPerWorker {
       indexPerWorker[i], _ = faiss.NewIndexFlatL2(dim)
       indexPerWorker[i].Add(vectors) // Same data
   }
   ```

### How do I normalize vectors for inner product search?

```go
import "math"

func normalize(vector []float32) []float32 {
    var norm float32
    for _, v := range vector {
        norm += v * v
    }
    norm = float32(math.Sqrt(float64(norm)))

    normalized := make([]float32, len(vector))
    for i, v := range vector {
        normalized[i] = v / norm
    }
    return normalized
}
```

### Can I save and load indexes?

Not yet implemented, but coming soon! The API will look like:

```go
// Save (future)
index.Save("my_index.faiss")

// Load (future)
index, err := faiss.LoadIndex("my_index.faiss")
```

## Performance Questions

### How does performance compare to Python FAISS?

The underlying FAISS library is the same, so search performance should be identical. Go overhead is minimal since we use CGO to call directly into FAISS C++ code.

Benchmark comparisons (coming soon).

### Should I use L2 or Inner Product?

**Use L2 distance when:**
- Vectors represent features/embeddings where magnitude matters
- You want traditional nearest neighbor search
- Lower distance = more similar

**Use Inner Product when:**
- Vectors are normalized (cosine similarity)
- You're doing maximum similarity search
- Higher score = more similar

**Cosine similarity:**
Normalize vectors, then use Inner Product:
```go
index, _ := faiss.NewIndexFlatIP(dim)
normalizedVectors := normalize(vectors)
index.Add(normalizedVectors)
```

### Can I use GPU acceleration?

Not yet. CPU-only indexes are currently supported. GPU support is planned for a future release.

## Development Questions

### How do I update the FAISS version?

```bash
make update-faiss VERSION=v1.8.0
```

Or manually:
```bash
cd scripts
./generate_amalgamation.sh v1.8.0
```

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed development setup and guidelines.

### How do I build pre-compiled libraries?

For maintainers:
```bash
cd scripts
./build_static_libs.sh [platform]
```

Requires Docker for cross-compilation.

### Where can I get help?

- 📖 Read the [documentation](https://pkg.go.dev/github.com/NerdMeNot/faiss-go)
- 🐛 [Report issues](https://github.com/NerdMeNot/faiss-go/issues)
- 💬 [Start a discussion](https://github.com/NerdMeNot/faiss-go/discussions)

## Platform-Specific Questions

### Does it work on Windows?

Yes, but requires:
- MSYS2 or Visual Studio Build Tools
- OpenBLAS (via vcpkg or manual install)

Pre-built libraries for Windows are included for the `faiss_use_lib` build mode.

### Does it work on Apple Silicon (M1/M2)?

Yes! Both build modes work on ARM64 macOS:
- Source build: Uses Accelerate framework + OpenBLAS
- Pre-built: Includes arm64 static libraries

### Does it work on ARM64 Linux?

Yes, including Raspberry Pi 4 and cloud ARM instances. Both build modes supported.

## Troubleshooting

### Build fails with "BLAS not found"

Install OpenBLAS (see "How do I install BLAS?" above) or use pre-built libraries:
```bash
go build -tags=faiss_use_lib
```

### Build fails with "C++ compiler not found"

Install a C++ compiler:
- Linux: `sudo apt-get install build-essential`
- macOS: `xcode-select --install`
- Windows: Install MSYS2 or Visual Studio

### Tests fail with "not implemented" error

The stub implementation is active. Run the amalgamation generator:
```bash
make generate-amalgamation
```

Then rebuild:
```bash
make build
```

### Linker errors on macOS

Make sure OpenBLAS is installed:
```bash
brew install openblas
```

If still failing, try pre-built libraries:
```bash
go build -tags=faiss_use_lib
```

---

**Still have questions?** [Open an issue](https://github.com/NerdMeNot/faiss-go/issues/new)!
