# Choosing the Right Index

One of the most important decisions when using FAISS is selecting the right index type for your use case. This guide will help you make an informed decision.

---

## Quick Decision Tree

```
START: How many vectors?
│
├─ < 10K vectors
│  └─→ Use IndexFlatL2/IP (exact search, fast enough)
│
├─ 10K - 1M vectors
│  ├─ Need 100% recall?
│  │  └─→ IndexFlatL2/IP
│  │
│  ├─ Need speed + high recall (>95%)?
│  │  └─→ IndexHNSWFlat (best choice for most cases)
│  │
│  └─ Limited RAM?
│      └─→ IndexIVFPQ (8-32x compression)
│
├─ 1M - 100M vectors
│  ├─ Lots of RAM?
│  │  └─→ IndexHNSWFlat
│  │
│  ├─ Balanced RAM/speed?
│  │  └─→ IndexIVFFlat or IndexIVFPQ
│  │
│  └─ Very limited RAM?
│      └─→ IndexPQ or IndexScalarQuantizer
│
└─ > 100M vectors (billion-scale)
   ├─ Fits in RAM (compressed)?
   │  └─→ IndexIVFPQ with high nlist
   │
   └─ Doesn't fit in RAM?
       └─→ IndexIVFPQOnDisk
```

---

## Index Type Comparison

### Overview Table

| Index Type | Search Speed | Memory Usage | Recall | Training Required | Best For |
|------------|--------------|--------------|--------|-------------------|----------|
| **FlatL2/IP** | Baseline | 100% (4 bytes/dim) | 100% | No | <100K vectors, baselines |
| **HNSW** | Very Fast | High (>100%) | 95-99% | No | Production (10K-10M) |
| **IVFFlat** | Fast | 100% | 90-98% | Yes | Balanced speed/accuracy |
| **IVFPQ** | Very Fast | Low (1-5%) | 85-95% | Yes | Large-scale (1M-1B) |
| **PQ** | Very Fast | Very Low (1-5%) | 80-95% | Yes | Memory-constrained |
| **PQFastScan** | Extremely Fast | Very Low | 85-95% | Yes | Modern CPUs (SIMD) |
| **ScalarQuantizer** | Fast | Low (25%) | 90-98% | Yes | Compression + accuracy |
| **OnDisk** | Medium | Minimal RAM | 85-95% | Yes | Billion-scale |
| **GPU** | Extreme | GPU VRAM | Varies | Depends | Massive throughput |

---

## Detailed Index Guide

### 1. Exact Search Indexes

#### IndexFlatL2 / IndexFlatIP

**Use when**:
- Dataset < 100K vectors
- You need 100% recall
- Building a baseline for comparison
- Accuracy is critical

**Characteristics**:
- ✅ Perfect recall (exact search)
- ✅ No training required
- ✅ Simple to use
- ❌ Slow for large datasets (linear scan)
- ❌ High memory usage

**Example**:
```go
index, _ := faiss.NewIndexFlatL2(dimension)
index.Add(vectors)
distances, indices, _ := index.Search(query, k)
```

**Performance**: ~10-20K QPS for 1M vectors (128-dim)

---

### 2. Graph-Based Indexes

#### IndexHNSW (Hierarchical Navigable Small World)

**Use when**:
- Dataset: 10K - 10M vectors
- You need high recall (95-99%)
- You have RAM for the dataset
- Production use case

**Characteristics**:
- ✅ Excellent recall/speed tradeoff
- ✅ No training required
- ✅ Very fast search
- ✅ Best general-purpose index
- ❌ High memory usage (1.5-2x dataset size)
- ❌ Slow to build

**Example**:
```go
// M=32 is good default (higher = better recall, more memory)
index, _ := faiss.NewIndexHNSWFlat(dimension, 32, faiss.MetricL2)
index.SetEfConstruction(40) // Build quality
index.SetEfSearch(16)        // Search quality
index.Add(vectors)
```

**Parameters**:
- `M`: Number of connections per node (16-64, default 32)
- `efConstruction`: Build quality (higher = better graph, slower build)
- `efSearch`: Search quality (higher = better recall, slower search)

**Tuning**:
```go
// Fast search, lower recall (~92%)
index.SetEfSearch(8)

// Balanced (default, ~95% recall)
index.SetEfSearch(16)

// High recall (~98%)
index.SetEfSearch(64)

// Maximum recall (~99%)
index.SetEfSearch(128)
```

**Performance**: ~50-100K QPS for 1M vectors, 95-98% recall

---

### 3. Inverted File Indexes

#### IndexIVFFlat

**Use when**:
- Dataset: 100K - 10M vectors
- You need good recall (90-98%)
- Training data is available
- Balanced speed/memory

**Characteristics**:
- ✅ Fast search
- ✅ Tunable recall/speed
- ✅ Same memory as Flat
- ⚠️ Requires training
- ❌ Lower recall than HNSW

**Example**:
```go
// nlist = sqrt(N) is common heuristic
quantizer, _ := faiss.NewIndexFlatL2(dimension)
nlist := int(math.Sqrt(float64(numVectors)))
index, _ := faiss.NewIndexIVFFlat(quantizer, dimension, nlist, faiss.MetricL2)

// Train on subset of data
trainingData := vectors[:trainingSize]
index.Train(trainingData)

// Add all vectors
index.Add(vectors)

// Search quality: probe more clusters for higher recall
index.SetNprobe(10) // Search 10 of nlist clusters
```

**Parameters**:
- `nlist`: Number of clusters (typical: sqrt(N) to 4*sqrt(N))
- `nprobe`: Clusters to search (higher = better recall, slower)

**Tuning**:
```go
// Fast search, lower recall (~85%)
index.SetNprobe(1)

// Balanced (~92% recall)
index.SetNprobe(10)

// High recall (~97%)
index.SetNprobe(50)

// Very high recall (~99%)
index.SetNprobe(nlist/2) // Search half the clusters
```

**Performance**: ~100-200K QPS for 1M vectors, 90-95% recall

---

### 4. Compressed Indexes

#### IndexPQ (Product Quantization)

**Use when**:
- Limited RAM (need 8-32x compression)
- Dataset: 1M - 1B vectors
- Can tolerate lower recall (85-95%)

**Characteristics**:
- ✅ Extreme memory savings (1-5% of original)
- ✅ Fast search
- ⚠️ Requires training
- ❌ Lower recall than uncompressed
- ❌ Lossy compression

**Example**:
```go
m := 16         // Number of sub-quantizers (divisor of dimension)
nBits := 8      // Bits per sub-quantizer (typically 8)

index, _ := faiss.NewIndexPQ(dimension, m, nBits, faiss.MetricL2)
index.Train(trainingData)
index.Add(vectors)
```

**Memory Calculation**:
- Original: N vectors × D dims × 4 bytes = N × D × 4
- PQ: N vectors × m bytes = N × m
- Compression: (D × 4) / m times

Example: 1M vectors, 128-dim
- Original: 1M × 128 × 4 = 512 MB
- PQ (m=16): 1M × 16 = 16 MB
- Compression: 32x

**Parameters**:
- `m`: Sub-quantizers (8, 16, 32, 64) — must divide dimension
- `nBits`: Bits per code (8 is standard, 16 for better quality)

**Performance**: ~150-300K QPS, 85-92% recall

---

#### IndexIVFPQ (Combined IVF + PQ)

**Use when**:
- Large-scale datasets (1M - 1B)
- Need speed + compression
- Production deployment

**Characteristics**:
- ✅ Combines speed of IVF with compression of PQ
- ✅ Best overall balance
- ✅ Industry standard for billion-scale
- ⚠️ Requires training
- ❌ Complex parameter tuning

**Example**:
```go
quantizer, _ := faiss.NewIndexFlatL2(dimension)
nlist := 4096   // Many clusters for large datasets
m := 16         // Compression
nBits := 8

index, _ := faiss.NewIndexIVFPQ(quantizer, dimension, nlist, m, nBits, faiss.MetricL2)
index.Train(trainingData)
index.Add(vectors)
index.SetNprobe(16) // Search quality
```

**Tuning Guide**:

For 1M vectors:
```go
nlist := 1000
nprobe := 10
// Memory: ~20-30 MB
// Recall: ~92%
```

For 10M vectors:
```go
nlist := 4096
nprobe := 32
// Memory: ~200-300 MB
// Recall: ~90%
```

For 100M vectors:
```go
nlist := 16384
nprobe := 64
// Memory: ~2-3 GB
// Recall: ~88%
```

**Performance**: ~200-500K QPS for 10M vectors

---

#### IndexPQFastScan (SIMD-Optimized PQ)

**Use when**:
- Modern CPU with AVX2/AVX-512
- Need maximum QPS
- Willing to trade some flexibility

**Characteristics**:
- ✅ 2-4x faster than regular PQ
- ✅ SIMD-optimized
- ✅ Same compression as PQ
- ❌ Dimension restrictions
- ❌ Limited to 4-bit codes

**Example**:
```go
// Dimension must be multiple of M
m := 16
bbs := 32 // Block size for SIMD

index, _ := faiss.NewIndexPQFastScan(dimension, m, bbs, faiss.MetricL2)
index.Train(trainingData)
index.Add(vectors)
```

**Performance**: ~500K-1M QPS, 90-95% recall

---

### 5. Billion-Scale Indexes

#### IndexIVFPQOnDisk

**Use when**:
- Dataset doesn't fit in RAM
- Billion+ vectors
- Can tolerate disk I/O latency

**Characteristics**:
- ✅ Minimal RAM usage (only active clusters)
- ✅ Handles billion+ vectors
- ✅ Memory-mapped I/O
- ❌ Slower than in-memory
- ❌ Requires SSD for good performance

**Example**:
```go
indexPath := "/data/billion_vectors.idx"
nlist := 65536  // Many clusters for billion-scale
m := 16
nBits := 8

index, _ := faiss.NewIndexIVFPQOnDisk(
    indexPath,
    dimension,
    nlist,
    m,
    nBits,
)
index.Train(trainingData)
index.Add(vectors) // Writes to disk
index.SetNprobe(32)
```

**RAM Usage**: Typically <10 GB for billion vectors

---

### 6. GPU Indexes

#### GpuIndexFlat / GpuIndexIVFFlat

**Use when**:
- Have CUDA GPU available
- Need extreme throughput
- Batch queries (not single queries)

**Characteristics**:
- ✅ 10-100x faster than CPU
- ✅ Massive throughput
- ❌ Requires CUDA
- ❌ Limited by GPU VRAM
- ❌ Higher latency for small batches

**Example**:
```go
// Use GPU 0
gpuIndex, _ := faiss.NewGpuIndexFlatL2(0, dimension)
gpuIndex.Add(vectors)
distances, indices, _ := gpuIndex.Search(queries, k)
```

**Performance**: ~1-10M QPS with batch queries

---

## Parameter Tuning Guide

### nlist (IVF indexes)

Rule of thumb: `nlist = sqrt(N)` to `nlist = 4*sqrt(N)`

| Dataset Size | nlist | Rationale |
|--------------|-------|-----------|
| 10K | 100 | sqrt(10K) ≈ 100 |
| 100K | 316-1000 | sqrt(100K) ≈ 316 |
| 1M | 1000-4000 | sqrt(1M) = 1000 |
| 10M | 3162-10000 | sqrt(10M) ≈ 3162 |
| 100M | 10000-40000 | sqrt(100M) = 10000 |

### nprobe (IVF search)

| nprobe | Recall | Speed | Use Case |
|--------|--------|-------|----------|
| 1 | ~80% | Fastest | Preliminary search |
| 10 | ~92% | Fast | Balanced |
| nlist/100 | ~95% | Medium | High recall |
| nlist/10 | ~98% | Slow | Very high recall |
| nlist | 100% | Slowest | Exact (defeats purpose) |

### M (HNSW)

| M | Memory | Recall | Build Time |
|---|--------|--------|------------|
| 16 | Low | ~92% | Fast |
| 32 | Medium | ~96% | Medium (recommended) |
| 48 | High | ~97% | Slow |
| 64 | Very High | ~98% | Very Slow |

---

## Real-World Scenarios

### Scenario 1: Semantic Search (100K documents)

**Requirements**:
- 100K documents
- 768-dim BERT embeddings
- Need >95% recall
- Latency <20ms

**Recommendation**: IndexHNSWFlat
```go
index, _ := faiss.NewIndexHNSWFlat(768, 32, faiss.MetricL2)
index.SetEfSearch(16)
index.Add(embeddings)
```

**Why**: HNSW provides excellent recall with fast search. 100K vectors fit comfortably in RAM.

---

### Scenario 2: Image Search (10M images)

**Requirements**:
- 10M images
- 2048-dim ResNet features
- RAM limited to 8GB
- Recall >90% acceptable

**Recommendation**: IndexIVFPQ
```go
quantizer, _ := faiss.NewIndexFlatL2(2048)
index, _ := faiss.NewIndexIVFPQ(quantizer, 2048, 4096, 16, 8, faiss.MetricL2)
index.Train(trainingData)
index.Add(features)
index.SetNprobe(16)
```

**Why**:
- Original size: 10M × 2048 × 4 = 80 GB
- IVFPQ size: 10M × 16 = 160 MB
- 500x compression fits in RAM budget

---

### Scenario 3: Recommendation (1M items)

**Requirements**:
- 1M items
- 128-dim embeddings
- Sub-10ms latency
- High QPS (10K+)

**Recommendation**: IndexPQFastScan
```go
index, _ := faiss.NewIndexPQFastScan(128, 16, 32, faiss.MetricL2)
index.Train(embeddings)
index.Add(embeddings)
```

**Why**: FastScan provides maximum QPS with good recall. 128-dim works well with SIMD.

---

## Common Mistakes

### 1. Not Training IVF/PQ Indexes

```go
// ❌ WRONG
index, _ := faiss.NewIndexIVFFlat(...)
index.Add(vectors) // Will fail!

// ✅ CORRECT
index.Train(trainingData)
index.Add(vectors)
```

### 2. Using Flat Index for Large Datasets

```go
// ❌ WRONG for 10M vectors
index, _ := faiss.NewIndexFlatL2(dimension)

// ✅ CORRECT
index, _ := faiss.NewIndexHNSWFlat(dimension, 32, faiss.MetricL2)
```

### 3. Setting nprobe=1 and Expecting High Recall

```go
// ❌ WRONG - only searches 1 cluster
index.SetNprobe(1)

// ✅ CORRECT - balanced
index.SetNprobe(10)
```

---

## Quick Reference

### Selection Flowchart

1. **Need exact search?** → IndexFlatL2/IP
2. **< 1M vectors + have RAM?** → IndexHNSWFlat
3. **Need extreme speed?** → IndexPQFastScan
4. **Limited RAM?** → IndexIVFPQ or IndexPQ
5. **Billion+ vectors?** → IndexIVFPQOnDisk
6. **Have GPU?** → GpuIndexFlat / GpuIndexIVFFlat

### Default Parameters

For most production use cases:

```go
// Best general-purpose index
index, _ := faiss.NewIndexHNSWFlat(dimension, 32, faiss.MetricL2)
index.SetEfSearch(16)

// Best for large-scale + compression
quantizer, _ := faiss.NewIndexFlatL2(dimension)
nlist := int(4 * math.Sqrt(float64(numVectors)))
index, _ := faiss.NewIndexIVFPQ(quantizer, dimension, nlist, 16, 8, faiss.MetricL2)
index.SetNprobe(nlist / 100)
```

---

## Next Steps

- **[API Reference](../api/)** - Learn all index operations
- **[Performance Tuning](../guides/performance-tuning.md)** - Optimize your index
- **[Examples](../examples/)** - See real-world implementations

---

**Need help choosing?** [Open a discussion](https://github.com/NerdMeNot/faiss-go/discussions)
