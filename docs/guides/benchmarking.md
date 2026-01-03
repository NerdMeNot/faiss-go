# Benchmarking Guide

How to measure and optimize faiss-go performance.

## Running Benchmarks

### Built-in Benchmarks

```bash
# Run all benchmarks
go test -bench=. -benchmem ./...

# Run specific benchmark
go test -bench=BenchmarkIndexFlat -benchmem

# Run with longer duration
go test -bench=. -benchtime=5s -benchmem
```

### Quick Performance Test

```go
package main

import (
    "fmt"
    "time"
    faiss "github.com/NerdMeNot/faiss-go"
)

func main() {
    dim := 128
    numVectors := 100000
    numQueries := 1000
    k := 10

    // Create and populate index
    index, _ := faiss.IndexFactory(dim, "HNSW32", faiss.MetricL2)
    defer index.Close()

    vectors := make([]float32, numVectors*dim)
    // ... populate vectors

    start := time.Now()
    index.Add(vectors)
    addTime := time.Since(start)
    fmt.Printf("Add time: %v (%.0f vectors/sec)\n",
        addTime, float64(numVectors)/addTime.Seconds())

    queries := make([]float32, numQueries*dim)
    // ... populate queries

    start = time.Now()
    for i := 0; i < numQueries; i++ {
        index.Search(queries[i*dim:(i+1)*dim], k)
    }
    searchTime := time.Since(start)
    qps := float64(numQueries) / searchTime.Seconds()
    fmt.Printf("Search time: %v (%.0f QPS)\n", searchTime, qps)
}
```

## Key Metrics

### Throughput

- **QPS (Queries Per Second)**: Search operations per second
- **Add rate**: Vectors added per second

### Latency

- **P50**: Median latency
- **P95/P99**: Tail latencies

### Memory

- **Index size**: Memory used by index
- **Peak memory**: Maximum during build

## Index Performance Comparison

Typical results for 100K 128-dimensional vectors:

| Index | Build Time | QPS | Memory | Recall@10 |
|-------|------------|-----|--------|-----------|
| Flat | <1s | ~1,000 | 50 MB | 100% |
| HNSW32 | ~10s | ~10,000 | 75 MB | 95% |
| IVF100 | ~2s | ~20,000 | 50 MB | 85% |
| IVFPQ | ~5s | ~50,000 | 5 MB | 75% |

*Results vary by hardware*

## Measuring Recall

Recall measures search accuracy:

```go
// Ground truth with Flat index
flatIndex, _ := faiss.IndexFactory(dim, "Flat", faiss.MetricL2)
flatIndex.Add(vectors)
_, truthLabels, _ := flatIndex.Search(query, k)

// Approximate index
approxIndex, _ := faiss.IndexFactory(dim, "HNSW32", faiss.MetricL2)
approxIndex.Add(vectors)
_, approxLabels, _ := approxIndex.Search(query, k)

// Calculate recall
matches := 0
for _, t := range truthLabels {
    for _, a := range approxLabels {
        if t == a {
            matches++
            break
        }
    }
}
recall := float64(matches) / float64(k)
fmt.Printf("Recall@%d: %.2f%%\n", k, recall*100)
```

## Optimization Tips

### Index Selection

| Dataset Size | Recommended Index | Expected QPS |
|--------------|-------------------|--------------|
| < 10K | Flat | ~10,000 |
| 10K - 100K | HNSW32 | ~5,000 |
| 100K - 1M | IVF+Flat | ~20,000 |
| 1M+ | IVF+PQ | ~50,000 |

### Parameter Tuning

**HNSW**:
- Higher M → better recall, slower search
- Higher efSearch → better recall, slower search

**IVF**:
- More nlist → faster search, more training
- Higher nprobe → better recall, slower search

### Batch Operations

```go
// Single queries (slower)
for _, query := range queries {
    index.Search(query, k)
}

// Batch queries (faster)
allQueries := flatten(queries)
index.Search(allQueries, k)
```

## Profiling

```bash
# CPU profile
go test -bench=. -cpuprofile=cpu.prof
go tool pprof cpu.prof

# Memory profile
go test -bench=. -memprofile=mem.prof
go tool pprof mem.prof
```

## Reference Results

See `test/recall/` for comprehensive benchmarks:

```bash
go test -v ./test/recall/
```
