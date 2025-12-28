# Migrating from Python FAISS to faiss-go

This guide helps Python FAISS users transition to faiss-go quickly and efficiently.

---

## Overview

faiss-go provides **100% API parity** with Python FAISS. Most Python code translates directly to Go with minimal changes.

**Key Differences**:
- Type safety (compile-time vs runtime)
- Explicit error handling
- Manual memory management (`defer index.Close()`)
- Snake_case → CamelCase naming

---

## Quick Translation Reference

### Index Creation

**Python**:
```python
import faiss

# Flat indexes
index = faiss.IndexFlatL2(128)
index = faiss.IndexFlatIP(128)

# IVF indexes
quantizer = faiss.IndexFlatL2(128)
index = faiss.IndexIVFFlat(quantizer, 128, 100, faiss.METRIC_L2)

# HNSW
index = faiss.IndexHNSWFlat(128, 32)

# PQ
index = faiss.IndexPQ(128, 16, 8)

# IVFPQ
index = faiss.IndexIVFPQ(quantizer, 128, 100, 16, 8)
```

**Go**:
```go
import "github.com/NerdMeNot/faiss-go"

// Flat indexes
index, err := faiss.NewIndexFlatL2(128)
index, err := faiss.NewIndexFlatIP(128)

// IVF indexes
quantizer, _ := faiss.NewIndexFlatL2(128)
index, err := faiss.NewIndexIVFFlat(quantizer, 128, 100, faiss.MetricL2)

// HNSW
index, err := faiss.NewIndexHNSWFlat(128, 32, faiss.MetricL2)

// PQ
index, err := faiss.NewIndexPQ(128, 16, 8, faiss.MetricL2)

// IVFPQ
index, err := faiss.NewIndexIVFPQ(quantizer, 128, 100, 16, 8, faiss.MetricL2)

// Always defer cleanup
defer index.Close()
```

---

### Training

**Python**:
```python
import numpy as np

# Training data
training_data = np.random.rand(10000, 128).astype('float32')

# Train index
if not index.is_trained:
    index.train(training_data)
```

**Go**:
```go
// Training data (flattened)
trainingData := make([]float32, 10000*128)
// ... populate training data ...

// Train index
if !index.IsTrained() {
    err := index.Train(trainingData)
    if err != nil {
        log.Fatal(err)
    }
}
```

---

### Adding Vectors

**Python**:
```python
# Add vectors
vectors = np.random.rand(1000, 128).astype('float32')
index.add(vectors)

# Add with IDs
ids = np.arange(1000)
index.add_with_ids(vectors, ids)

# Check count
print(f"Total vectors: {index.ntotal}")
```

**Go**:
```go
// Add vectors (must flatten 2D → 1D)
vectors := make([]float32, 1000*128)
// ... populate vectors ...
err := index.Add(vectors)
if err != nil {
    log.Fatal(err)
}

// Add with IDs
ids := make([]int64, 1000)
for i := range ids {
    ids[i] = int64(i)
}
err = index.AddWithIDs(vectors, ids)

// Check count
fmt.Printf("Total vectors: %d\n", index.Ntotal())
```

---

### Searching

**Python**:
```python
# Search
query = np.random.rand(1, 128).astype('float32')
k = 10

distances, indices = index.search(query, k)

# Process results
for i in range(k):
    print(f"Neighbor {i}: ID={indices[0,i]}, Distance={distances[0,i]}")
```

**Go**:
```go
// Search (single query)
query := make([]float32, 128)
// ... populate query ...
k := 10

distances, indices, err := index.Search(query, k)
if err != nil {
    log.Fatal(err)
}

// Process results
for i := 0; i < k; i++ {
    fmt.Printf("Neighbor %d: ID=%d, Distance=%.4f\n",
        i, indices[i], distances[i])
}
```

---

### Parameter Configuration

**Python**:
```python
# IVF nprobe
index.nprobe = 10

# HNSW parameters
index.hnsw.efConstruction = 40
index.hnsw.efSearch = 16

# PQ search type
index.pq_search_type = faiss.PQ_SEARCH_TYPE_POLYSEMOUS
```

**Go**:
```go
// IVF nprobe
index.SetNprobe(10)

// HNSW parameters
index.SetEfConstruction(40)
index.SetEfSearch(16)

// PQ search type
index.SetPQSearchType(faiss.PQSearchTypePolysemous)
```

---

### Range Search

**Python**:
```python
# Find all neighbors within radius
radius = 2.0
lims, distances, indices = index.range_search(query, radius)

# Process variable-length results
for i in range(len(query)):
    neighbors_for_query = indices[lims[i]:lims[i+1]]
    distances_for_query = distances[lims[i]:lims[i+1]]
```

**Go**:
```go
// Find all neighbors within radius
radius := float32(2.0)
result, err := index.RangeSearch(query, radius)
if err != nil {
    log.Fatal(err)
}

// Process results
fmt.Printf("Found %d neighbors within radius\n", len(result.Indices))
for i, idx := range result.Indices {
    fmt.Printf("  %d: distance=%.4f\n", idx, result.Distances[i])
}
```

---

### Serialization

**Python**:
```python
# Save index
faiss.write_index(index, "my_index.faiss")

# Load index
index = faiss.read_index("my_index.faiss")
```

**Go**:
```go
// Save index
err := faiss.WriteIndex(index, "my_index.faiss")
if err != nil {
    log.Fatal(err)
}

// Load index
index, err := faiss.ReadIndex("my_index.faiss")
if err != nil {
    log.Fatal(err)
}
defer index.Close()
```

---

### Index Factory

**Python**:
```python
# Create index from string descriptor
index = faiss.index_factory(128, "IVF100,PQ16")

# With metric
index = faiss.index_factory(128, "HNSW32", faiss.METRIC_L2)
```

**Go**:
```go
// Create index from string descriptor
index, err := faiss.IndexFactory(128, "IVF100,PQ16", faiss.MetricL2)
if err != nil {
    log.Fatal(err)
}
defer index.Close()

// With metric
index, err = faiss.IndexFactory(128, "HNSW32", faiss.MetricL2)
```

---

### GPU Indexes

**Python**:
```python
# Create GPU resource
res = faiss.StandardGpuResources()

# Create GPU index
gpu_index = faiss.GpuIndexFlatL2(res, 128)

# Or convert CPU → GPU
cpu_index = faiss.IndexFlatL2(128)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# Search on GPU
distances, indices = gpu_index.search(query, k)
```

**Go**:
```go
// Create GPU index directly
gpuIndex, err := faiss.NewGpuIndexFlatL2(0, 128) // GPU 0
if err != nil {
    log.Fatal(err)
}
defer gpuIndex.Close()

// Or convert CPU → GPU
cpuIndex, _ := faiss.NewIndexFlatL2(128)
gpuIndex, err = faiss.IndexCPUToGPU(0, cpuIndex)
if err != nil {
    log.Fatal(err)
}
defer gpuIndex.Close()

// Search on GPU
distances, indices, err := gpuIndex.Search(query, k)
```

---

### Clustering

**Python**:
```python
# Kmeans clustering
kmeans = faiss.Kmeans(128, 100)  # 128-dim, 100 clusters
kmeans.train(vectors)

# Get centroids
centroids = kmeans.centroids
```

**Go**:
```go
// Kmeans clustering
kmeans, err := faiss.NewKmeans(128, 100, faiss.MetricL2)
if err != nil {
    log.Fatal(err)
}
defer kmeans.Close()

err = kmeans.Train(vectors)
if err != nil {
    log.Fatal(err)
}

// Get centroids
centroids := kmeans.Centroids()
```

---

## Complete Example Migration

### Python Version

```python
import numpy as np
import faiss

# Configuration
dimension = 128
num_vectors = 10000
nlist = 100
nprobe = 10

# Generate data
vectors = np.random.rand(num_vectors, dimension).astype('float32')
queries = np.random.rand(10, dimension).astype('float32')

# Create and train index
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

# Training
index.train(vectors)

# Add vectors
index.add(vectors)

# Configure search
index.nprobe = nprobe

# Search
k = 10
distances, indices = index.search(queries, k)

# Save index
faiss.write_index(index, "index.faiss")

# Results
print(f"Index contains {index.ntotal} vectors")
print(f"Top result for first query: index {indices[0,0]}, distance {distances[0,0]}")
```

### Go Version

```go
package main

import (
    "fmt"
    "log"

    "github.com/NerdMeNot/faiss-go"
)

func main() {
    // Configuration
    dimension := 128
    numVectors := 10000
    nlist := 100
    nprobe := 10

    // Generate data
    vectors := generateRandomVectors(numVectors, dimension)
    queries := generateRandomVectors(10, dimension)

    // Create index
    quantizer, err := faiss.NewIndexFlatL2(dimension)
    if err != nil {
        log.Fatal(err)
    }
    defer quantizer.Close()

    index, err := faiss.NewIndexIVFFlat(quantizer, dimension, nlist, faiss.MetricL2)
    if err != nil {
        log.Fatal(err)
    }
    defer index.Close()

    // Training
    err = index.Train(vectors)
    if err != nil {
        log.Fatal(err)
    }

    // Add vectors
    err = index.Add(vectors)
    if err != nil {
        log.Fatal(err)
    }

    // Configure search
    index.SetNprobe(nprobe)

    // Search
    k := 10
    distances, indices, err := index.Search(queries, k)
    if err != nil {
        log.Fatal(err)
    }

    // Save index
    err = faiss.WriteIndex(index, "index.faiss")
    if err != nil {
        log.Fatal(err)
    }

    // Results
    fmt.Printf("Index contains %d vectors\n", index.Ntotal())
    fmt.Printf("Top result for first query: index %d, distance %.4f\n",
        indices[0], distances[0])
}

func generateRandomVectors(n, d int) []float32 {
    vectors := make([]float32, n*d)
    for i := range vectors {
        vectors[i] = rand.Float32()
    }
    return vectors
}
```

---

## Key Differences

### 1. Error Handling

**Python**: Exceptions
```python
try:
    index.add(vectors)
except RuntimeError as e:
    print(f"Error: {e}")
```

**Go**: Explicit errors
```go
err := index.Add(vectors)
if err != nil {
    log.Printf("Error: %v", err)
}
```

### 2. Memory Management

**Python**: Automatic garbage collection
```python
index = faiss.IndexFlatL2(128)
# Automatically freed
```

**Go**: Manual cleanup
```go
index, _ := faiss.NewIndexFlatL2(128)
defer index.Close() // Must call Close()
```

### 3. Array Handling

**Python**: NumPy 2D arrays
```python
vectors = np.random.rand(1000, 128).astype('float32')
index.add(vectors)  # 2D array
```

**Go**: Flattened 1D slices
```go
vectors := make([]float32, 1000*128)  // 1D slice
index.Add(vectors)
```

### 4. Naming Conventions

| Python | Go |
|--------|-----|
| `snake_case` | `CamelCase` |
| `index.ntotal` | `index.Ntotal()` |
| `index.is_trained` | `index.IsTrained()` |
| `index.add(...)` | `index.Add(...)` |
| `index.search(...)` | `index.Search(...)` |
| `faiss.METRIC_L2` | `faiss.MetricL2` |

---

## Common Patterns

### Pattern 1: Batch Processing

**Python**:
```python
batch_size = 1000
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.add(batch)
```

**Go**:
```go
batchSize := 1000 * dimension
for i := 0; i < len(vectors); i += batchSize {
    end := min(i+batchSize, len(vectors))
    batch := vectors[i:end]
    index.Add(batch)
}
```

### Pattern 2: Vector Normalization

**Python**:
```python
import numpy as np

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

normalized = normalize(vectors)
```

**Go**:
```go
import "math"

func normalize(vectors []float32, dimension int) []float32 {
    numVectors := len(vectors) / dimension
    normalized := make([]float32, len(vectors))
    copy(normalized, vectors)

    for i := 0; i < numVectors; i++ {
        start := i * dimension
        end := start + dimension
        vec := normalized[start:end]

        // Compute norm
        var norm float32
        for _, v := range vec {
            norm += v * v
        }
        norm = float32(math.Sqrt(float64(norm)))

        // Normalize
        if norm > 0 {
            for j := range vec {
                vec[j] /= norm
            }
        }
    }

    return normalized
}
```

### Pattern 3: Index Selection

**Python**:
```python
def get_index_for_dataset_size(n_vectors):
    if n_vectors < 10000:
        return faiss.IndexFlatL2(dimension)
    elif n_vectors < 1000000:
        return faiss.IndexHNSWFlat(dimension, 32)
    else:
        quantizer = faiss.IndexFlatL2(dimension)
        return faiss.IndexIVFPQ(quantizer, dimension, 4096, 16, 8)
```

**Go**:
```go
func getIndexForDatasetSize(nVectors, dimension int) (faiss.Index, error) {
    if nVectors < 10000 {
        return faiss.NewIndexFlatL2(dimension)
    } else if nVectors < 1000000 {
        return faiss.NewIndexHNSWFlat(dimension, 32, faiss.MetricL2)
    } else {
        quantizer, err := faiss.NewIndexFlatL2(dimension)
        if err != nil {
            return nil, err
        }
        return faiss.NewIndexIVFPQ(quantizer, dimension, 4096, 16, 8, faiss.MetricL2)
    }
}
```

---

## Performance Considerations

### 1. CGO Overhead

**Insight**: Go ↔ C boundary has small overhead

**Best Practice**: Batch operations when possible
```go
// ❌ SLOW: Many small calls
for _, vec := range vectors {
    index.Add(vec)
}

// ✅ FAST: Single batched call
index.Add(flattenedVectors)
```

### 2. Concurrency

**Python**: Limited by GIL
```python
# Python struggles with concurrent searches
```

**Go**: Native concurrency
```go
// Go excels at concurrent searches
var wg sync.WaitGroup
for i := 0; i < numWorkers; i++ {
    wg.Add(1)
    go func(queries []float32) {
        defer wg.Done()
        index.Search(queries, k)
    }(workerQueries[i])
}
wg.Wait()
```

**Note**: Wrap with mutex for thread safety or use separate indexes per goroutine.

---

## Migration Checklist

- [ ] Replace `import faiss` with `import "github.com/NerdMeNot/faiss-go"`
- [ ] Convert NumPy arrays to Go slices (2D → 1D flattening)
- [ ] Add `defer index.Close()` for all indexes
- [ ] Convert snake_case to CamelCase
- [ ] Add explicit error handling (`if err != nil`)
- [ ] Update function signatures (return values)
- [ ] Convert metrics: `faiss.METRIC_L2` → `faiss.MetricL2`
- [ ] Test with small dataset first
- [ ] Benchmark performance
- [ ] Deploy!

---

## Troubleshooting

### Issue: Dimension mismatch

**Error**: `faiss: invalid vectors (length must be multiple of dimension)`

**Cause**: Forgot to flatten 2D array to 1D

**Fix**:
```go
// Ensure len(vectors) % dimension == 0
vectors := make([]float32, numVectors * dimension)
```

### Issue: Segfault

**Cause**: Forgot to call `Close()`

**Fix**:
```go
index, _ := faiss.NewIndexFlatL2(128)
defer index.Close() // Always defer!
```

### Issue: Training error

**Cause**: Forgot to train IVF/PQ index

**Fix**:
```go
if !index.IsTrained() {
    index.Train(trainingData)
}
index.Add(vectors)
```

---

## Next Steps

- **[Choosing an Index](choosing-an-index.md)** - Select the right index type
- **[API Reference](../api/)** - Complete API documentation
- **[Examples](../examples/)** - Real-world Go examples

---

**Need migration help?** [Ask on Discussions](https://github.com/NerdMeNot/faiss-go/discussions)
