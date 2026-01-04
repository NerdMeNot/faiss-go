# Choosing an Index

This guide helps you select the right FAISS index type for your use case.

## Quick Decision

| Dataset Size | Recommended Index | Factory String |
|--------------|-------------------|----------------|
| < 10K | Flat (exact) | `"Flat"` |
| 10K - 1M | HNSW (graph-based) | `"HNSW32"` |
| 1M - 10M | IVF (inverted file) | `"IVF1000,Flat"` |
| 10M+ | IVF+PQ (compressed) | `"IVF4096,PQ16"` |

## Index Types Overview

### Flat (Exact Search)

```go
index, _ := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
```

- **Best for**: Small datasets (< 10K vectors)
- **Recall**: 100% (exact)
- **Training**: Not required
- **Speed**: Slow for large datasets (brute force)

### HNSW (Graph-Based)

```go
index, _ := faiss.IndexFactory(128, "HNSW32", faiss.MetricL2)
```

- **Best for**: Medium datasets (10K - 1M vectors)
- **Recall**: 95-99%
- **Training**: Not required
- **Speed**: Very fast

The number after HNSW is M (connections per node):
- `HNSW16` - Lower memory, ~92% recall
- `HNSW32` - Balanced (recommended)
- `HNSW64` - Higher memory, ~98% recall

### IVF (Inverted File)

```go
index, _ := faiss.IndexFactory(128, "IVF100,Flat", faiss.MetricL2)
// Train before adding vectors
index.Train(trainingData)
index.Add(vectors)
```

- **Best for**: Large datasets (1M+ vectors)
- **Recall**: 85-98% (adjustable)
- **Training**: Required
- **Speed**: Fast

The number after IVF is nlist (number of clusters):
- `IVF100` - Good for ~10K vectors
- `IVF1000` - Good for ~1M vectors
- `IVF4096` - Good for ~10M vectors

Rule of thumb: `nlist ≈ sqrt(n_vectors)`

### PQ (Product Quantization)

```go
index, _ := faiss.IndexFactory(128, "PQ8", faiss.MetricL2)
// Train before adding vectors
index.Train(trainingData)
index.Add(vectors)
```

- **Best for**: Memory-constrained scenarios
- **Recall**: 80-90%
- **Training**: Required
- **Compression**: 16-32x memory reduction

The number after PQ is M (sub-quantizers):
- `PQ8` - 16x compression
- `PQ16` - 8x compression
- `PQ32` - 4x compression

### IVF+PQ (Combined)

```go
index, _ := faiss.IndexFactory(128, "IVF1000,PQ16", faiss.MetricL2)
index.Train(trainingData)
index.Add(vectors)
```

- **Best for**: Large-scale + memory constraints
- **Recall**: 85-95%
- **Training**: Required
- **Use case**: Production systems with millions of vectors

### Scalar Quantizer

```go
index, _ := faiss.IndexFactory(128, "SQ8", faiss.MetricL2)
index.Train(trainingData)
index.Add(vectors)
```

- **Best for**: Balance between compression and accuracy
- **Recall**: 90-98%
- **Training**: Required
- **Compression**: 4x (8-bit quantization)

### LSH (Locality-Sensitive Hashing)

```go
index, _ := faiss.IndexFactory(128, "LSH", faiss.MetricL2)
```

- **Best for**: Specific use cases
- **Recall**: Variable
- **Training**: Not required

## With Preprocessing

Add PCA or other transforms before the index:

```go
// PCA to 64 dimensions + IVF
index, _ := faiss.IndexFactory(128, "PCA64,IVF100,Flat", faiss.MetricL2)
index.Train(trainingData)
index.Add(vectors)
```

## Metric Types

**L2 (Euclidean Distance)**:
- Lower distance = more similar
- Use for general embeddings

**Inner Product**:
- Higher value = more similar
- Use for normalized vectors (cosine similarity)

```go
// L2 distance
index, _ := faiss.IndexFactory(128, "Flat", faiss.MetricL2)

// Inner product (cosine similarity with normalized vectors)
index, _ := faiss.IndexFactory(128, "Flat", faiss.MetricInnerProduct)
```

## Common Patterns

### Semantic Search

Document/text embeddings with high recall:

```go
// HNSW for high recall
index, _ := faiss.IndexFactory(768, "HNSW32", faiss.MetricInnerProduct)
```

### Image Similarity

Visual features with memory constraints:

```go
// IVF+PQ for compression
index, _ := faiss.IndexFactory(2048, "IVF4096,PQ32", faiss.MetricL2)
index.Train(features)
```

### Recommendations

Item embeddings with speed priority:

```go
// IVF for fast search
index, _ := faiss.IndexFactory(128, "IVF1000,Flat", faiss.MetricL2)
index.Train(embeddings)
```

## Training Tips

For indexes that require training (IVF, PQ, SQ):

1. **Training data size**: Use 30x to 256x the number of clusters
   - `IVF100` → train on at least 10,000 vectors
   - `IVF1000` → train on at least 100,000 vectors

2. **Representative data**: Training data should be similar to your actual data

3. **Training once**: Train once, then add all vectors

```go
index, _ := faiss.IndexFactory(128, "IVF1000,Flat", faiss.MetricL2)

// Train on a sample (at least 100K for IVF1000)
index.Train(trainingVectors)

// Now add all vectors
index.Add(allVectors)
```

## Memory Estimates

| Index | Memory per vector (128-dim) |
|-------|----------------------------|
| Flat | 512 bytes |
| HNSW32 | ~800 bytes |
| IVF+Flat | 512 bytes |
| PQ16 | 16 bytes |
| IVF+PQ16 | ~20 bytes |
| SQ8 | 128 bytes |

## Next Steps

- [Quickstart](quickstart.md) - Try different index types
- [Benchmarking](../guides/benchmarking.md) - Measure performance
