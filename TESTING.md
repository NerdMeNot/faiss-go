## Comprehensive Testing Guide

This document describes the testing strategy and infrastructure for faiss-go. Our goal is to make **thorough, real-world testing the USP** of this project.

---

## Testing Philosophy

**faiss-go is the most comprehensively tested FAISS binding in any language.**

### Core Principles

1. **Real-World First** - Every test should reflect actual usage patterns
2. **Data-Driven** - Use actual datasets and realistic embeddings
3. **Measurable Quality** - Quantify recall, precision, and performance
4. **Reproducible** - Deterministic tests with versioned data
5. **Comprehensive** - Cover all code paths and edge cases
6. **Performance-Aware** - Track regressions automatically

---

## Test Organization

```
test/
├── datasets/          # Dataset loading and generation
│   ├── loader.go     # Load SIFT1M, GIST1M, etc.
│   └── generators.go # Generate synthetic data
│
├── helpers/           # Test utilities
│   ├── recall_calculator.go  # Quality metrics
│   └── test_utils.go          # Common test functions
│
├── recall/            # Recall validation tests
│   ├── framework.go           # Reusable framework
│   └── *_recall_test.go       # Per-index tests
│
├── scenarios/         # Real-world use cases
│   ├── semantic_search_test.go
│   ├── image_similarity_test.go
│   └── recommendations_test.go
│
├── stress/            # Scale and robustness
│   ├── scale_test.go
│   ├── concurrent_test.go
│   └── longevity_test.go
│
└── properties/        # Property-based tests
    ├── search_properties_test.go
    └── serialization_properties_test.go

testdata/
├── embeddings/        # Downloaded datasets (SIFT1M, etc.)
└── golden/            # Expected outputs
```

---

## Quick Start

### 1. Run Basic Tests

```bash
# Run all tests (skip slow ones)
go test -v ./...

# Run with coverage
go test -v -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Run including slow tests
go test -v ./... -timeout=30m
```

### 2. Download Test Datasets

```bash
# Download SIFT10K (5MB, fast)
./scripts/download_test_datasets.sh sift10k

# Download SIFT1M (160MB)
./scripts/download_test_datasets.sh sift1m

# Download all datasets
./scripts/download_test_datasets.sh all
```

### 3. Run Recall Validation Tests

```bash
# Run recall tests (requires datasets)
go test -v ./test/recall/...

# Run specific index type
go test -v ./test/recall/ -run TestHNSW_Recall
```

### 4. Run Real-World Scenario Tests

```bash
# Semantic search scenarios
go test -v ./test/scenarios/ -run TestSemanticSearch

# All scenarios
go test -v ./test/scenarios/...
```

### 5. Run Stress Tests

```bash
# Scale tests (can take 10+ minutes)
go test -v ./test/stress/ -run TestScale

# Concurrent access tests
go test -v ./test/stress/ -run TestConcurrent
```

---

## Test Categories

### 1. Recall Validation Tests (`test/recall/`)

**Purpose:** Validate that approximate indexes achieve target recall levels.

**How it works:**
1. Build ground truth index (IndexFlat)
2. Build approximate index (HNSW, IVF, PQ, etc.)
3. Run same queries on both
4. Measure recall@1, recall@10, recall@100
5. Assert recall >= target (e.g., 95%)

**Example:**
```go
func TestHNSW_Recall_SIFT10K(t *testing.T) {
    // Load dataset
    dataset, err := datasets.LoadDataset("SIFT10K", "testdata")
    require.NoError(t, err)

    // Build ground truth
    gtIndex := faiss.NewIndexFlatL2(dataset.D)
    gtIndex.Add(dataset.Vectors)
    defer gtIndex.Delete()

    // Build HNSW
    hnsw := faiss.NewIndexHNSWFlat(dataset.D, 32, faiss.MetricL2)
    hnsw.HnswSetEfSearch(64)
    hnsw.Add(dataset.Vectors)
    defer hnsw.Delete()

    // Search both
    gtResults, _, _ := helpers.SearchWithTiming(gtIndex, dataset.Queries, 10)
    hnswResults, latencies, _ := helpers.SearchWithTiming(hnsw, dataset.Queries, 10)

    // Calculate metrics
    recall := helpers.CalculateRecall(
        helpers.ConvertToSearchResults(gtResults),
        hnswResults,
        10,
    )

    // Assert quality
    assert.GreaterOrEqual(t, recall, 0.95)

    // Log performance
    perf := helpers.MeasureLatencies(latencies)
    t.Logf("HNSW: Recall@10=%.4f, QPS=%.0f", recall, perf.QPS)
}
```

**Tested Indexes:**
- ✅ IndexFlatL2/IP (baseline, 100% recall)
- ✅ IndexIVFFlat (various nprobe settings)
- ✅ IndexHNSW (various M, efSearch settings)
- ✅ IndexPQ (compression vs recall tradeoff)
- ✅ IndexIVFPQ (combined IVF+PQ)
- ✅ IndexPQFastScan (SIMD optimizations)
- ✅ IndexScalarQuantizer (8-bit quantization)
- More...

### 2. Real-World Scenario Tests (`test/scenarios/`)

**Purpose:** Test complete workflows that users would actually run.

**Scenarios:**

#### A. Semantic Search
```go
// 100K document embeddings (768-dim BERT)
// Search for similar documents
// Indexes: Flat, IVF100, HNSW32
// Metrics: Recall@10, latency, memory
```

#### B. Image Similarity
```go
// 1M image features (2048-dim ResNet50)
// Find visually similar images
// Indexes: HNSW48, IVF1000_PQ16
// Metrics: Recall@20, QPS, compression ratio
```

#### C. Recommendations
```go
// 10M user-item embeddings (128-dim)
// Generate item recommendations
// Indexes: IVF4096_PQ8, PQFastScan
// Metrics: Recall@50, throughput
```

#### D. Real-Time Updates
```go
// Streaming vector additions (1K/sec)
// Simultaneous search queries
// Indexes: IndexIDMap(HNSW), IndexShards
// Metrics: Latency stability, throughput degradation
```

### 3. Stress & Scale Tests (`test/stress/`)

**Purpose:** Validate behavior at production scale.

#### Scale Tests
```go
// Test sizes: 1K → 10K → 100K → 1M → 10M → 100M
// Dimensions: 128, 256, 512, 768, 1536, 2048
// Measure: Build time, search QPS, memory usage
// Validate: OnDisk indexes for billion-scale
```

#### Concurrent Access
```go
// N goroutines searching simultaneously
// Add + Search concurrent operations
// Multiple indexes in same process
// Thread safety validation
```

#### Long-Running Stability
```go
// Run for 1+ hour: continuous add/search
// Check for memory leaks
// Monitor performance degradation
// Validate finalizers work correctly
```

#### Memory Stress
```go
// Approach system memory limits
// Test OOM handling
// Validate compression (PQ, SQ)
// Measure actual memory footprint
```

### 4. Property-Based Tests (`test/properties/`)

**Purpose:** Validate fundamental properties that should always hold.

**Properties:**

```go
// Search Properties
- distances[i] <= distances[i+1] (sorted results)
- Exact search: top-1 is the vector itself
- Higher k returns superset of lower k
- Metric symmetry: d(a,b) == d(b,a)

// Training Properties
- Training twice produces same behavior
- More training data → better recall
- IsTrained() == true after Train()

// Serialization Properties
- deserialize(serialize(idx)) ≈ idx
- Search results identical before/after save/load
- File size proportional to compression ratio
```

### 5. Integration Tests (existing)

**Purpose:** Full lifecycle validation for each index type.

```go
// For each index:
1. Create index
2. Configure parameters
3. Train (if needed)
4. Add vectors (in batches)
5. Search with various k
6. Save to file
7. Load from file
8. Search again (verify same results)
9. Add more vectors
10. Reset
11. Cleanup
```

---

## Test Data

### Synthetic Data

Use `test/datasets/generators.go` to create synthetic datasets:

```go
import "github.com/NerdMeNot/faiss-go/test/datasets"

// Uniform random
config := datasets.GeneratorConfig{
    N: 10000,
    D: 128,
    Distribution: datasets.UniformRandom,
}
data := datasets.GenerateSyntheticData(config)

// Clustered (realistic)
config.Distribution = datasets.GaussianClustered
config.NumClusters = 100
data = datasets.GenerateSyntheticData(config)

// Normalized (for cosine similarity)
config.Distribution = datasets.Normalized
data = datasets.GenerateSyntheticData(config)

// Realistic embeddings
data = datasets.GenerateRealisticEmbeddings(100000, 768)
```

### Real Datasets

**SIFT1M** - Classic ANN benchmark
- 1M 128-dim SIFT descriptors
- 10K query vectors
- 100-NN ground truth
- Download: `./scripts/download_test_datasets.sh sift1m`

**SIFT10K** - Smaller subset (fast tests)
- 10K vectors, 100 queries
- Created from SIFT1M
- Download: `./scripts/download_test_datasets.sh sift10k`

**GIST1M** - High-dimensional benchmark
- 1M 960-dim GIST descriptors
- 1K query vectors
- Download: `./scripts/download_test_datasets.sh gist1m`

**Loading datasets:**
```go
import "github.com/NerdMeNot/faiss-go/test/datasets"

dataset, err := datasets.LoadDataset("SIFT10K", "testdata")
if err != nil {
    t.Skip("Dataset not available")
}

// Use dataset
index.Add(dataset.Vectors)
results, _ := index.Search(dataset.Queries, 10)
```

---

## Metrics & Assertions

### Quality Metrics

```go
import "github.com/NerdMeNot/faiss-go/test/helpers"

// Calculate recall@k
recall := helpers.CalculateRecall(groundTruth, results, k)

// Calculate all metrics
metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
// Returns: Recall@1, @10, @100, Precision, MRR, NDCG

// Assert minimum recall
helpers.AssertRecallAbove(t, recall, 0.95, "HNSW should achieve 95% recall")
```

### Performance Metrics

```go
// Measure latencies
results, latencies, _ := helpers.SearchWithTiming(index, queries, k)
perf := helpers.MeasureLatencies(latencies)

// Metrics available:
perf.P50Latency  // Median
perf.P95Latency  // 95th percentile
perf.P99Latency  // 99th percentile
perf.QPS         // Queries per second
perf.AvgLatency  // Average

// Assert latency
helpers.AssertLatencyBelow(t, perf.P99Latency, 10*time.Millisecond, "P99 too high")

// Measure QPS
qps := helpers.MeasureQPS(index, queries, k, 5*time.Second)
```

### Memory Metrics

```go
// Estimate memory usage
mem := helpers.MeasureIndexMemory(index)
t.Logf("Index memory: %d MB", mem/(1024*1024))
```

---

## Best Practices

### 1. Use Table-Driven Tests

```go
func TestIndexTypes(t *testing.T) {
    tests := []struct {
        name string
        builder func(d int) faiss.Index
        minRecall float64
    }{
        {"Flat", func(d int) { return faiss.NewIndexFlatL2(d) }, 1.0},
        {"IVF100", func(d int) { /* create IVF */ }, 0.90},
        {"HNSW32", func(d int) { /* create HNSW */ }, 0.95},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Test logic
        })
    }
}
```

### 2. Skip Expensive Tests in Short Mode

```go
func TestLargeScale(t *testing.T) {
    helpers.SkipIfShort(t, "large-scale test")

    // Test with 1M+ vectors
}
```

### 3. Require Datasets Gracefully

```go
func TestWithRealData(t *testing.T) {
    dataset, err := datasets.LoadDataset("SIFT10K", "testdata")
    if err != nil {
        t.Skip("SIFT10K dataset not available. Run: ./scripts/download_test_datasets.sh sift10k")
    }

    // Test with dataset
}
```

### 4. Clean Up Resources

```go
func TestIndex(t *testing.T) {
    index := faiss.NewIndexFlatL2(128)
    defer index.Delete()  // Always defer Delete()

    // Test logic
}
```

### 5. Log Useful Metrics

```go
func TestPerformance(t *testing.T) {
    // Run test

    // Log results for analysis
    helpers.LogMetrics(t, "HNSW", recallMetrics, perfMetrics)
    t.Logf("Memory: %d MB", mem/(1024*1024))
}
```

---

## CI Integration

### GitHub Actions Workflows

**Regular CI** (`ci.yml`):
- Runs on: PRs, main pushes, tags
- Includes: Unit tests, integration tests
- Coverage: Upload to Codecov
- Duration: ~5-10 min

**Comprehensive Tests** (planned):
- Runs on: Weekly schedule, manual dispatch
- Includes: Recall validation, stress tests
- Datasets: SIFT10K, SIFT1M
- Duration: ~30-60 min

**Performance Regression** (planned):
- Tracks: QPS, latency, recall over time
- Alerts: >10% regression
- Stores: Historical benchmark data

### Running Tests Locally

```bash
# Quick validation (< 1 min)
go test -v -short ./...

# Full test suite (5-10 min)
go test -v ./...

# With datasets (30+ min first time, then cached)
./scripts/download_test_datasets.sh sift10k
go test -v ./... -timeout=30m

# Specific category
go test -v ./test/recall/...
go test -v ./test/scenarios/...
go test -v ./test/stress/...

# With coverage
go test -v -coverprofile=coverage.out ./...
go tool cover -func=coverage.out
go tool cover -html=coverage.out -o coverage.html
```

---

## Coverage Targets

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Line Coverage | ~85% | 95%+ | 🟡 In Progress |
| Branch Coverage | ~70% | 90%+ | 🟡 In Progress |
| Real-World Scenarios | 1 | 10+ | 🟢 Infrastructure Ready |
| Recall Validation | Basic | All Indexes | 🟢 Framework Ready |
| Scale Tests | 100K | 10M+ | 🟡 In Progress |
| Stress Tests | Basic | Comprehensive | 🟢 Planned |

---

## Contributing Tests

### Adding a New Test

1. **Choose the right category:**
   - Recall validation → `test/recall/`
   - Real-world scenario → `test/scenarios/`
   - Scale/stress → `test/stress/`
   - Property → `test/properties/`

2. **Follow the naming convention:**
   - File: `<category>_<feature>_test.go`
   - Function: `Test<Feature>_<Aspect>`

3. **Use existing helpers:**
   - Import `github.com/NerdMeNot/faiss-go/test/helpers`
   - Use recall calculator, data generators, etc.

4. **Document what you're testing:**
   ```go
   // TestHNSW_Recall_HighDimensions validates that HNSW maintains
   // high recall (>95%) even with high-dimensional vectors (1536-dim).
   // This tests the robustness of the HNSW algorithm with modern
   // embedding sizes like OpenAI ada-002.
   func TestHNSW_Recall_HighDimensions(t *testing.T) {
       // ...
   }
   ```

5. **Add to CI if appropriate:**
   - Fast tests (<10s): Run in regular CI
   - Medium tests (<1min): Run with dataset flag
   - Slow tests (>1min): Weekly comprehensive suite

---

## Troubleshooting

### "Dataset not found"
```bash
# Download the dataset first
./scripts/download_test_datasets.sh sift10k

# Or skip tests requiring datasets
go test -v -short ./...
```

### "Test timeout"
```bash
# Increase timeout for slow tests
go test -v ./... -timeout=30m

# Or run specific fast tests
go test -v -short ./...
```

### "Out of memory"
```bash
# Reduce test scale
go test -v -short ./...

# Or increase available memory
# Stress tests may need 8GB+ RAM for large datasets
```

### "Coverage not updating"
```bash
# Ensure you're testing all packages
go test -v -coverprofile=coverage.out ./...

# View coverage
go tool cover -func=coverage.out

# Generate HTML report
go tool cover -html=coverage.out -o coverage.html
open coverage.html
```

---

## Resources

- [Testing in Go](https://go.dev/doc/tutorial/add-a-test)
- [Table-Driven Tests](https://github.com/golang/go/wiki/TableDrivenTests)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [ANN Benchmarks](http://ann-benchmarks.com/)
- [Project VERSIONING.md](VERSIONING.md)
- [GitHub Actions Workflows](.github/workflows/README.md)

---

## Questions?

- Review this guide
- Check existing tests for examples
- Open an issue with `[Testing]` prefix
- See the comprehensive testing strategy plan
