# FAISS-Go Examples

This directory contains examples demonstrating various FAISS-Go features for vector similarity search.

## Quick Start

```bash
# Run any example
cd examples/01_basic_search
go run main.go
```

## Examples Overview

| Example | Description | Use Case |
|---------|-------------|----------|
| [01_basic_search](./01_basic_search) | L2 and inner product search with flat indexes | Learning FAISS basics |
| [02_ivf_clustering](./02_ivf_clustering) | IVF indexes with training and nprobe tuning | Medium to large datasets (100K-10M) |
| [03_hnsw_graph](./03_hnsw_graph) | HNSW graph-based approximate search | High-recall requirements |
| [04_pq_compression](./04_pq_compression) | Product quantization for memory efficiency | Billion-scale datasets |
| [05_gpu_acceleration](./05_gpu_acceleration) | GPU-accelerated search (requires CUDA) | Maximum throughput |
| [06_pretransform](./06_pretransform) | PCA and OPQ pre-processing | High-dimensional vectors |

## Choosing an Index Type

```
Dataset Size        Recommended Index       Factory String
─────────────────────────────────────────────────────────────
< 10,000           Flat                    "Flat"
10K - 100K         HNSW or IVF             "HNSW32" or "IVF100,Flat"
100K - 1M          IVF + tuning            "IVF1000,Flat"
1M - 10M           IVF + PQ                "IVF4096,PQ16"
> 10M              IVF + PQ + OPQ          "OPQ16,IVF65536,PQ16"
```

## Key Concepts

### Flat Index (Exact Search)
- Brute-force comparison with all vectors
- 100% accuracy, O(n) complexity
- Best for small datasets or ground truth validation

### IVF (Inverted File Index)
- Clusters vectors into cells for faster search
- **Requires training** before adding vectors
- Tune `nprobe` for speed/accuracy tradeoff

### HNSW (Hierarchical Navigable Small World)
- Graph-based approximate search
- **No training required**
- Excellent recall with fast queries
- Tune `M` parameter (16, 32, 64)

### Product Quantization (PQ)
- Compresses vectors for memory efficiency
- 32x-64x memory reduction typical
- **Requires training** to learn codebooks
- Use with IVF for billion-scale search

### Pre-transforms (PCA, OPQ)
- Reduce dimensionality before indexing
- OPQ optimizes for PQ compression
- Improves search quality for high-dimensional vectors

## Common Factory Strings

```go
// Exact search
index, _ := faiss.IndexFactory(128, "Flat", faiss.MetricL2)

// HNSW approximate search
index, _ := faiss.IndexFactory(128, "HNSW32", faiss.MetricL2)

// IVF clustering
index, _ := faiss.IndexFactory(128, "IVF100,Flat", faiss.MetricL2)

// IVF with compression
index, _ := faiss.IndexFactory(128, "IVF100,PQ8", faiss.MetricL2)

// PCA dimension reduction + IVF
index, _ := faiss.IndexFactory(256, "PCA64,IVF100,Flat", faiss.MetricL2)

// Full compression pipeline
index, _ := faiss.IndexFactory(256, "OPQ16,IVF4096,PQ16", faiss.MetricL2)
```

## GPU Examples

GPU examples require:
- NVIDIA GPU with CUDA support
- FAISS built with GPU support
- Build with: `go build -tags=gpu`

```bash
cd examples/05_gpu_acceleration
go build -tags=gpu && ./main
```

## Running All Examples

```bash
# Run all CPU examples
for dir in 01_* 02_* 03_* 04_* 06_*; do
    echo "=== $dir ==="
    (cd "$dir" && go run main.go)
    echo
done
```

## Python Equivalents

These examples mirror common Python FAISS patterns:

| Go | Python |
|----|--------|
| `faiss.IndexFactory(d, "HNSW32", MetricL2)` | `faiss.index_factory(d, "HNSW32", faiss.METRIC_L2)` |
| `faiss.NewIndexFlatL2(d)` | `faiss.IndexFlatL2(d)` |
| `index.Train(vectors)` | `index.train(vectors)` |
| `index.Add(vectors)` | `index.add(vectors)` |
| `index.Search(query, k)` | `index.search(query, k)` |

## Troubleshooting

### "index must be trained"
IVF and PQ indexes require training before adding vectors:
```go
index.Train(trainingVectors)  // Must call before Add()
index.Add(vectors)
```

### Poor recall with IVF
Increase nprobe (default is 1):
```go
genericIndex := index.(*faiss.GenericIndex)
genericIndex.SetNprobe(32)  // Search more clusters
```

### Out of memory with large datasets
Use compression:
```go
// Instead of "IVF100,Flat", use:
index, _ := faiss.IndexFactory(d, "IVF100,PQ8", faiss.MetricL2)
```

### Slow indexing with HNSW
Reduce M parameter:
```go
// Instead of "HNSW64", use:
index, _ := faiss.IndexFactory(d, "HNSW16", faiss.MetricL2)
```
