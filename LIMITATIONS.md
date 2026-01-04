# Limitations

This document describes current limitations of faiss-go and recommended workarounds.

## Summary

faiss-go provides comprehensive Go bindings for FAISS. The **IndexFactory API** is the recommended way to create indexes and supports all index types.

**Recommendation**: Use `IndexFactory()` for all index creation - it's the most flexible and well-tested approach.

```go
// Recommended approach - works for all index types
index, err := faiss.IndexFactory(128, "IVF100,Flat", faiss.MetricL2)
```

---

## Index Creation

### IndexFactory (Recommended)

The `IndexFactory()` function works correctly for all index types:

| Index Type | Factory String | Status |
|------------|----------------|--------|
| Flat (exact) | `"Flat"` | Works |
| IVF | `"IVF100,Flat"` | Works |
| HNSW | `"HNSW32"` | Works |
| PQ | `"PQ8"` | Works |
| IVF+PQ | `"IVF100,PQ8"` | Works |
| LSH | `"LSH"` | Works |
| Scalar Quantizer | `"SQ8"` | Works |
| With transforms | `"PCA64,IVF100,Flat"` | Works |

### Direct Constructors

All major index types have direct constructors:

| Index Type | Constructor |
|------------|-------------|
| Flat | `NewIndexFlat(d, metric)`, `NewIndexFlatL2(d)`, `NewIndexFlatIP(d)` |
| IVF | `NewIndexIVFFlat(quantizer, d, nlist, metric)` |
| HNSW | `NewIndexHNSW(d, M, metric)`, `NewIndexHNSWFlat(d, M, metric)` |
| LSH | `NewIndexLSH(d, nbits)`, `NewIndexLSHWithRotation(d, nbits)` |
| PQ | `NewIndexPQ(d, M, nbits, metric)`, `NewIndexIVFPQ(quantizer, d, nlist, M, nbits)` |
| SQ | `NewIndexScalarQuantizer(d, qtype, metric)`, `NewIndexIVFScalarQuantizer(...)` |

All constructors use the factory pattern internally for reliability.

---

## HNSW Parameter Tuning

HNSW indexes work correctly:

**Available**:
- Graph connectivity (M parameter) via factory string: `"HNSW32"`, `"HNSW64"`
- Dynamic `efSearch` adjustment via `SetEfSearch()` method

**Example**:
```go
index, _ := faiss.IndexFactory(128, "HNSW32", faiss.MetricL2)
idx := index.(*faiss.GenericIndex)
idx.SetEfSearch(64)  // Increase search quality
```

**Note**: `efConstruction` is set at index creation time via the M parameter and cannot be changed dynamically.

---

## Transform APIs

Direct transform APIs work correctly:

| Transform | Status |
|-----------|--------|
| PCAMatrix | Works |
| OPQMatrix | Works (requires sufficient training data) |
| RandomRotationMatrix | Works |

**Note**: OPQ requires substantial training data (at least 10,000 vectors for good results).

For composite indexes with transforms, use the factory approach:

```go
// Dimension reduction + IVF via factory
index, err := faiss.IndexFactory(128, "PCA64,IVF100,Flat", faiss.MetricL2)

// OPQ + PQ via factory
index, err := faiss.IndexFactory(128, "OPQ32,IVF100,PQ32", faiss.MetricL2)
```

---

## Composite Indexes

All composite index constructors work correctly:

| Type | Constructor | Notes |
|------|-------------|-------|
| IndexPreTransform | `NewIndexPreTransform()` | Combines transform + index |
| IndexShards | `NewIndexShards()` | Distributes across sub-indexes |
| IndexRefine | `NewIndexRefine()` | Two-stage search refinement |
| IndexIDMap | `NewIndexIDMap()` | Custom ID mapping |

For simpler cases, the factory string approach also works: `"PCA64,Flat"`

---

## Platform Support

### Pre-built Binaries

Pre-built static libraries are available for:

| Platform | Architecture | Status |
|----------|--------------|--------|
| Linux | AMD64 | Supported |
| Linux | ARM64 | Supported |
| macOS | Intel (AMD64) | Supported |
| macOS | Apple Silicon (ARM64) | Supported |

### Other Platforms

For platforms without pre-built binaries, use system FAISS mode:

```bash
go build -tags=faiss_use_system ./...
```

Requires FAISS to be installed on the system.

---

## GPU Support

GPU support requires:
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- GPU-enabled FAISS libraries

GPU indexes are not included in the standard pre-built binaries. See [GPU Setup](docs/getting-started/gpu-setup.md) for details.

---

## Binary Indexes

Binary index support (IndexBinaryFlat, IndexBinaryIVF) is **not available**. Use float32 indexes with appropriate quantization (e.g., LSH) for binary-like use cases.

---

## Recommendations

1. **Always use IndexFactory** - It's the most reliable and flexible approach
2. **Use defer for cleanup** - Always `defer index.Close()` to prevent memory leaks
3. **Validate dimensions** - Ensure vector dimensions match index dimensions
4. **Train before adding** - IVF and PQ indexes require training before adding vectors

---

## Reporting Issues

If you encounter issues not covered here:

1. Check the [FAQ](docs/reference/faq.md)
2. Search [existing issues](https://github.com/NerdMeNot/faiss-go/issues)
3. Open a new issue with:
   - Go version (`go version`)
   - Platform and architecture
   - Minimal reproduction code
   - Error message or unexpected behavior
