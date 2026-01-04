# Frequently Asked Questions

Common questions about faiss-go.

## General

### What is faiss-go?

faiss-go provides Go bindings for FAISS (Facebook AI Similarity Search). It includes pre-built static libraries, so no separate FAISS installation is required.

### How do I install it?

```bash
go get github.com/NerdMeNot/faiss-go
```

That's it. Pre-built binaries are included for Linux and macOS.

### What index types are supported?

All major FAISS index types via `IndexFactory()`:
- Flat (exact search)
- IVF (inverted file)
- HNSW (graph-based)
- PQ (product quantization)
- LSH, Scalar Quantizer
- Composite types (IVF+PQ, PCA+IVF, etc.)

## Usage

### Which index should I use?

| Dataset Size | Recommended | Factory String |
|--------------|-------------|----------------|
| < 10K | Flat | `"Flat"` |
| 10K - 1M | HNSW | `"HNSW32"` |
| 1M+ | IVF or IVF+PQ | `"IVF1000,Flat"` |

See [Choosing an Index](../getting-started/choosing-an-index.md).

### Is it thread-safe?

Index operations are not thread-safe. For concurrent access, use synchronization:

```go
var mu sync.Mutex
mu.Lock()
defer mu.Unlock()
index.Search(query, k)
```

### How do I use cosine similarity?

Use inner product metric with normalized vectors:

```go
index, _ := faiss.IndexFactory(128, "Flat", faiss.MetricInnerProduct)
// Normalize vectors before adding
```

### Can I save and load indexes?

Yes:

```go
// Save
faiss.WriteIndex(index, "index.faiss")

// Load
loaded, _ := faiss.ReadIndex("index.faiss")
defer loaded.Close()
```

### Do I need to train indexes?

Some indexes require training before adding vectors:

| Index | Training Required |
|-------|------------------|
| Flat | No |
| HNSW | No |
| IVF | Yes |
| PQ | Yes |

```go
index, _ := faiss.IndexFactory(128, "IVF100,Flat", faiss.MetricL2)
index.Train(trainingData)  // Required
index.Add(vectors)
```

## Platform Support

### Does it work on Apple Silicon?

Yes, ARM64 macOS is fully supported with pre-built binaries.

### Does it support GPU?

GPU support requires building with CUDA. See [GPU Setup](../getting-started/gpu-setup.md).

## Troubleshooting

### "dimension mismatch" error

Vector slice length must be `numVectors * dimension`:

```go
dim := 128
numVectors := 100
vectors := make([]float32, numVectors * dim) // Correct
```

### Memory leaks

Always close indexes:

```go
index, _ := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
defer index.Close() // Required!
```

### Build issues

Make sure you have Go 1.21+ and a C compiler:

```bash
# Check Go version
go version

# Ubuntu: install build tools
sudo apt-get install build-essential
```

### Search returns unexpected results

1. Check your metric type (L2 vs InnerProduct)
2. For cosine similarity, normalize vectors
3. For approximate indexes, increase search parameters

## Getting Help

- [GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues)
- [GitHub Discussions](https://github.com/NerdMeNot/faiss-go/discussions)
