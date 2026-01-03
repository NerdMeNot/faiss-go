# Testing Guide

Comprehensive guide to running and writing tests for faiss-go.

## Running Tests

### Quick Start

```bash
# Run all tests
go test ./...

# Run with verbose output
go test -v ./...

# Run specific test
go test -v -run TestIndexFactory

# Run tests in a specific package
go test -v ./test/recall/
```

### Test Modes

```bash
# Fast smoke test (skip slow tests)
go test -short ./...

# Full test suite
go test -v ./...

# With coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# With race detection
go test -race ./...

# With timeout (for long tests)
go test -v -timeout 30m ./test/recall/
```

## Test Organization

```
faiss-go/
├── *_test.go              # Unit tests (alongside source)
├── test/
│   ├── datasets/          # Dataset generators and loaders
│   │   ├── generators.go  # Synthetic data generation
│   │   └── loader.go      # SIFT/GIST dataset loading
│   ├── helpers/           # Test utilities
│   │   ├── recall_calculator.go  # Quality metrics
│   │   └── test_utils.go         # Common utilities
│   ├── recall/            # Recall validation tests
│   │   ├── framework.go   # Recall testing framework
│   │   ├── hnsw_recall_test.go
│   │   ├── ivf_recall_test.go
│   │   └── pq_recall_test.go
│   ├── scenarios/         # Real-world scenario tests
│   │   ├── semantic_search_test.go
│   │   ├── recommendations_test.go
│   │   └── streaming_test.go
│   └── integration/       # End-to-end tests
```

## Test Categories

### Unit Tests

Located alongside source files. Test individual functions and types:

```bash
go test -v -run TestIndexFactory
go test -v -run TestIndexFlat
```

### Recall Validation Tests

Located in `test/recall/`. Validate search quality:

```bash
go test -v ./test/recall/ -run TestHNSW
go test -v ./test/recall/ -run TestIVF
go test -v ./test/recall/ -run TestPQ
```

### Scenario Tests

Located in `test/scenarios/`. Test real-world use cases:

```bash
go test -v ./test/scenarios/ -run TestSemanticSearch
go test -v ./test/scenarios/ -run TestRecommendations
go test -v ./test/scenarios/ -run TestStreaming
```

---

## Recall Baselines

Expected recall values for properly configured indexes. Use these to validate your index is working correctly.

### HNSW

| M | efSearch | Recall@10 | Notes |
|---|----------|-----------|-------|
| 16 | 32 | ~90-93% | Fast, lower memory |
| 32 | 64 | ~95-97% | **Recommended default** |
| 48 | 128 | ~97-98% | High accuracy |
| 64 | 256 | ~98-99% | Maximum accuracy |

### IVF

| nlist | nprobe | Recall@10 | Notes |
|-------|--------|-----------|-------|
| 100 | 1 | ~60-70% | Very fast |
| 100 | 10 | ~85-90% | **Balanced** |
| 100 | 20 | ~90-93% | Higher accuracy |
| 100 | 50 | ~95-97% | High accuracy |

Rule of thumb: `nlist ≈ sqrt(n_vectors)`, `nprobe ≈ nlist/10`

### PQ

| M (subquantizers) | Recall@10 | Compression |
|-------------------|-----------|-------------|
| 64 | ~80-85% | 4x |
| 32 | ~75-80% | 8x |
| 16 | ~70-75% | 16x |
| 8 | ~65-70% | 32x |

### IVF+PQ

| nlist | M | nprobe | Recall@10 |
|-------|---|--------|-----------|
| 100 | 16 | 10 | ~70-75% |
| 1000 | 16 | 20 | ~75-80% |
| 4096 | 16 | 32 | ~78-83% |

---

## Using the Recall Framework

The `test/recall/framework.go` provides a reusable framework for recall tests.

### Basic Usage

```go
import "github.com/NerdMeNot/faiss-go/test/recall"

func TestMyIndex_Recall(t *testing.T) {
    config := recall.RecallTestConfig{
        Name:          "HNSW_M32",
        Description:   "HNSW with M=32",
        N:             10000,    // Number of vectors
        D:             128,      // Dimensions
        NQ:            100,      // Number of queries
        K:             10,       // Top-k results
        MinRecall10:   0.90,     // Minimum acceptable recall@10
        NeedsTraining: false,
        BuildIndex: func(d int) (faiss.Index, error) {
            return faiss.IndexFactory(d, "HNSW32", faiss.MetricL2)
        },
    }

    result := recall.RunRecallTest(t, config)
    t.Logf("Recall@10: %.2f%%, QPS: %.0f", result.Recall10*100, result.QPS)
}
```

### Parameter Sweep

```go
func TestHNSW_ParameterSweep(t *testing.T) {
    mValues := []int{16, 32, 48, 64}

    for _, m := range mValues {
        t.Run(fmt.Sprintf("M=%d", m), func(t *testing.T) {
            config := recall.RecallTestConfig{
                Name: fmt.Sprintf("HNSW_M%d", m),
                // ... other config
                BuildIndex: func(d int) (faiss.Index, error) {
                    return faiss.IndexFactory(d, fmt.Sprintf("HNSW%d", m), faiss.MetricL2)
                },
            }
            recall.RunRecallTest(t, config)
        })
    }
}
```

### With Training

```go
func TestIVF_Recall(t *testing.T) {
    config := recall.RecallTestConfig{
        Name:          "IVF100",
        NeedsTraining: true,  // Framework will call Train()
        BuildIndex: func(d int) (faiss.Index, error) {
            return faiss.IndexFactory(d, "IVF100,Flat", faiss.MetricL2)
        },
    }
    recall.RunRecallTest(t, config)
}
```

---

## Scenario Tests

Real-world use case validation.

### Semantic Search

**File:** `test/scenarios/semantic_search_test.go`

Tests document retrieval with text embeddings:
- 768-dimensional BERT-style embeddings
- 100K document corpus
- Query-document relevance
- Expected recall: >90%

```bash
go test -v ./test/scenarios/ -run TestSemanticSearch
```

### Recommendations

**File:** `test/scenarios/recommendations_test.go`

Tests recommendation systems:
- Item-to-item similarity
- User preference matching
- 128-dimensional embeddings
- Expected recall: >80%

```bash
go test -v ./test/scenarios/ -run TestRecommendations
```

### Streaming Updates

**File:** `test/scenarios/streaming_test.go`

Tests concurrent operations:
- Simultaneous add and search
- Batch updates
- ID mapping with IndexIDMap
- Latency stability under load

```bash
go test -v ./test/scenarios/ -run TestStreaming
```

---

## Test Datasets

### Synthetic Data

Use `test/datasets/generators.go` for controlled test data:

```go
import "github.com/NerdMeNot/faiss-go/test/datasets"

// Simple random vectors
vectors := datasets.GenerateRandomVectors(10000, 128)

// Clustered data (more realistic)
config := datasets.GeneratorConfig{
    N:            10000,
    D:            128,
    Distribution: datasets.GaussianClustered,
    NumClusters:  100,
    Seed:         42,  // Reproducible
}
data := datasets.GenerateSyntheticData(config)

// Normalized vectors (for inner product)
config.Normalize = true
data := datasets.GenerateSyntheticData(config)
```

### Standard Benchmarks

Download real benchmark datasets:

```bash
# SIFT10K - Quick tests (5MB)
./scripts/download_test_datasets.sh sift10k

# SIFT1M - Comprehensive tests (160MB)
./scripts/download_test_datasets.sh sift1m

# GIST1M - High-dimensional tests (4GB)
./scripts/download_test_datasets.sh gist1m
```

**Dataset Details:**

| Dataset | Vectors | Dimensions | Queries | Size |
|---------|---------|------------|---------|------|
| SIFT10K | 10,000 | 128 | 100 | 5 MB |
| SIFT1M | 1,000,000 | 128 | 10,000 | 160 MB |
| GIST1M | 1,000,000 | 960 | 1,000 | 4 GB |

**Loading in tests:**

```go
dataset, err := datasets.LoadDataset("SIFT10K", "testdata")
if err != nil {
    t.Skip("dataset not available: run ./scripts/download_test_datasets.sh sift10k")
}

// Use dataset
index.Add(dataset.Vectors)
distances, labels, _ := index.Search(dataset.Queries, 10)

// Compare with ground truth
recall := helpers.CalculateRecall(dataset.GroundTruth, labels, 10)
```

---

## Writing Tests

### Basic Pattern

```go
func TestMyFeature(t *testing.T) {
    // Setup
    index, err := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
    if err != nil {
        t.Fatalf("failed to create index: %v", err)
    }
    defer index.Close()

    // Prepare data
    vectors := make([]float32, 100*128)
    for i := range vectors {
        vectors[i] = float32(i % 100)
    }

    // Add
    if err := index.Add(vectors); err != nil {
        t.Fatalf("failed to add vectors: %v", err)
    }

    // Search
    query := vectors[:128]
    distances, labels, err := index.Search(query, 5)
    if err != nil {
        t.Fatalf("search failed: %v", err)
    }

    // Assert
    if len(labels) != 5 {
        t.Errorf("expected 5 results, got %d", len(labels))
    }
    if labels[0] != 0 {
        t.Errorf("expected first result to be vector 0, got %d", labels[0])
    }
}
```

### Table-Driven Tests

```go
func TestIndexTypes(t *testing.T) {
    tests := []struct {
        name          string
        factoryString string
        metric        faiss.MetricType
        needsTraining bool
    }{
        {"Flat_L2", "Flat", faiss.MetricL2, false},
        {"Flat_IP", "Flat", faiss.MetricInnerProduct, false},
        {"HNSW32", "HNSW32", faiss.MetricL2, false},
        {"IVF100", "IVF100,Flat", faiss.MetricL2, true},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            index, err := faiss.IndexFactory(128, tt.factoryString, tt.metric)
            if err != nil {
                t.Fatalf("failed to create: %v", err)
            }
            defer index.Close()

            vectors := generateTestVectors(1000, 128)

            if tt.needsTraining {
                if err := index.Train(vectors); err != nil {
                    t.Fatalf("training failed: %v", err)
                }
            }

            if err := index.Add(vectors); err != nil {
                t.Fatalf("add failed: %v", err)
            }

            _, labels, err := index.Search(vectors[:128], 5)
            if err != nil {
                t.Fatalf("search failed: %v", err)
            }

            if len(labels) != 5 {
                t.Errorf("expected 5 results, got %d", len(labels))
            }
        })
    }
}
```

### Skipping Slow Tests

```go
func TestLargeDataset(t *testing.T) {
    if testing.Short() {
        t.Skip("skipping slow test in short mode")
    }

    // Test with 1M+ vectors
    vectors := generateTestVectors(1000000, 128)
    // ...
}
```

### Skipping When Dataset Missing

```go
func TestWithRealData(t *testing.T) {
    dataset, err := datasets.LoadDataset("SIFT1M", "testdata")
    if err != nil {
        t.Skip("SIFT1M not available: " + err.Error())
    }

    // Test with real data
}
```

---

## Metrics

### Quality Metrics

```go
import "github.com/NerdMeNot/faiss-go/test/helpers"

// Calculate recall@k
recall := helpers.CalculateRecall(groundTruth, results, k)

// Calculate all metrics
metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
t.Logf("Recall@1: %.2f", metrics.Recall1)
t.Logf("Recall@10: %.2f", metrics.Recall10)
t.Logf("Recall@100: %.2f", metrics.Recall100)
t.Logf("MRR: %.4f", metrics.MRR)
```

### Performance Metrics

```go
// Search with timing
results, latencies, err := helpers.SearchWithTiming(index, queries, k)

// Calculate stats
perf := helpers.MeasureLatencies(latencies)
t.Logf("P50 Latency: %v", perf.P50)
t.Logf("P95 Latency: %v", perf.P95)
t.Logf("P99 Latency: %v", perf.P99)
t.Logf("QPS: %.0f", perf.QPS)
```

---

## Troubleshooting

### Test Timeout

**Problem:** Tests hang or exceed timeout.

**Solution:**
```bash
# Increase timeout
go test -v -timeout 30m ./test/recall/

# Or skip slow tests
go test -short ./...
```

### Dataset Not Found

**Problem:** `dataset not available` skip messages.

**Solution:**
```bash
# Download the dataset
./scripts/download_test_datasets.sh sift10k

# Check testdata directory
ls testdata/sift/
```

### Out of Memory

**Problem:** Tests crash with OOM.

**Solution:**
```bash
# Run tests individually
go test -v ./test/recall/ -run TestHNSW

# Or use smaller datasets
go test -short ./...
```

### Flaky Recall Tests

**Problem:** Recall tests sometimes fail.

**Causes:**
1. Random data generation without fixed seed
2. Threshold too close to actual performance
3. Insufficient training data

**Solution:**
```go
// Use fixed seed for reproducibility
config := datasets.GeneratorConfig{
    Seed: 42,
    // ...
}

// Set threshold below expected (e.g., 90% when expecting 95%)
config := recall.RecallTestConfig{
    MinRecall10: 0.90,  // Expect 95%, require 90%
}

// Ensure enough training data
trainingVectors := generateTestVectors(10000, 128)  // 10x nlist
```

### CGO Errors

**Problem:** Crashes during tests with CGO errors.

**Solution:**
```bash
# Check for memory issues
go test -race ./...

# Run with verbose CGO debugging
CGO_DEBUG=1 go test -v ./...
```

---

## Best Practices

1. **Always `defer index.Close()`** - Prevent memory leaks
2. **Use table-driven tests** - Cover multiple configurations
3. **Skip slow tests properly** - Use `testing.Short()`
4. **Test error cases** - Not just happy paths
5. **Use fixed random seeds** - Reproducible tests
6. **Set realistic thresholds** - Account for variance
7. **Log useful metrics** - Help debug failures
8. **Clean test names** - `TestIndexType_Operation_Condition`

---

## CI Integration

Tests run automatically on every push and PR:

- **Go versions:** 1.21, 1.22, 1.23, 1.24, 1.25
- **Platforms:** Ubuntu (AMD64), macOS (ARM64)
- **Test command:** `go test -v ./...`

See `.github/workflows/ci.yml` for configuration.

### Running CI Locally

```bash
# Simulate CI environment
go test -v ./...
golangci-lint run
```
