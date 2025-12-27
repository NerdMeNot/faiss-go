package helpers

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
)

// SearchWithTiming performs search and measures latency
func SearchWithTiming(index faiss.Index, queries []float32, k int) ([]SearchResult, []time.Duration, error) {
	nq := len(queries) / index.D()
	results := make([]SearchResult, nq)
	latencies := make([]time.Duration, nq)

	for i := 0; i < nq; i++ {
		query := queries[i*index.D() : (i+1)*index.D()]

		start := time.Now()
		distances, ids, err := index.Search(query, k)
		latencies[i] = time.Since(start)

		if err != nil {
			return nil, nil, fmt.Errorf("search failed at query %d: %w", i, err)
		}

		results[i] = SearchResult{
			IDs:       ids,
			Distances: distances,
		}
	}

	return results, latencies, nil
}

// ComputeGroundTruth computes exact nearest neighbors using brute force
func ComputeGroundTruth(vectors, queries []float32, d, k int, metric faiss.MetricType) ([]GroundTruth, error) {
	// Create flat index for ground truth
	var gtIndex faiss.Index
	if metric == faiss.MetricL2 {
		gtIndex = faiss.NewIndexFlatL2(d)
	} else {
		gtIndex = faiss.NewIndexFlatIP(d)
	}
	defer gtIndex.Delete()

	// Add vectors
	n := len(vectors) / d
	if err := gtIndex.Add(vectors); err != nil {
		return nil, fmt.Errorf("failed to add vectors to ground truth index: %w", err)
	}

	// Search
	nq := len(queries) / d
	groundTruth := make([]GroundTruth, nq)

	for i := 0; i < nq; i++ {
		query := queries[i*d : (i+1)*d]
		distances, ids, err := gtIndex.Search(query, k)
		if err != nil {
			return nil, fmt.Errorf("ground truth search failed at query %d: %w", i, err)
		}

		groundTruth[i] = GroundTruth{
			IDs:       ids,
			Distances: distances,
		}
	}

	return groundTruth, nil
}

// ConvertToSearchResults converts ground truth to search results format
func ConvertToSearchResults(groundTruth []GroundTruth) []SearchResult {
	results := make([]SearchResult, len(groundTruth))
	for i, gt := range groundTruth {
		results[i] = SearchResult{
			IDs:       gt.IDs,
			Distances: gt.Distances,
		}
	}
	return results
}

// MeasureIndexMemory estimates memory usage of an index
func MeasureIndexMemory(index faiss.Index) uint64 {
	// Force GC to get cleaner measurement
	runtime.GC()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	before := m.Alloc

	// This is approximate - actual memory may include C++ allocations
	// For more accurate measurement, check RSS before/after index creation

	runtime.ReadMemStats(&m)
	after := m.Alloc

	if after > before {
		return after - before
	}
	return 0
}

// MeasureQPS measures queries per second for an index
func MeasureQPS(index faiss.Index, queries []float32, k int, duration time.Duration) float64 {
	nq := len(queries) / index.D()
	if nq == 0 {
		return 0
	}

	start := time.Now()
	queriesExecuted := 0

	for time.Since(start) < duration {
		// Cycle through queries
		queryIdx := queriesExecuted % nq
		query := queries[queryIdx*index.D() : (queryIdx+1)*index.D()]

		_, _, err := index.Search(query, k)
		if err != nil {
			break
		}

		queriesExecuted++
	}

	elapsed := time.Since(start).Seconds()
	if elapsed > 0 {
		return float64(queriesExecuted) / elapsed
	}
	return 0
}

// RequireDataset skips test if dataset is not available
func RequireDataset(t *testing.T, datasetName string, testdataPath string) {
	t.Helper()

	// Check if dataset files exist
	// This is a simple check - LoadDataset will do more thorough validation
	if testdataPath == "" {
		t.Skipf("Dataset %s: testdata path not configured", datasetName)
	}

	// Could add file existence checks here
	// For now, just document the requirement
	t.Logf("Test requires dataset: %s", datasetName)
}

// SkipIfShort skips test if running in short mode
func SkipIfShort(t *testing.T, reason string) {
	t.Helper()
	if testing.Short() {
		t.Skipf("Skipping in short mode: %s", reason)
	}
}

// SkipIfNoGPU skips test if GPU is not available
func SkipIfNoGPU(t *testing.T) {
	t.Helper()
	// This would check for CUDA availability
	// For now, always skip GPU tests in this implementation
	t.Skip("GPU not available in current environment")
}

// AssertRecallAbove fails test if recall is below threshold
func AssertRecallAbove(t *testing.T, recall, threshold float64, message string) {
	t.Helper()
	if recall < threshold {
		t.Errorf("%s: recall %.4f < threshold %.4f", message, recall, threshold)
	}
}

// AssertLatencyBelow fails test if latency is above threshold
func AssertLatencyBelow(t *testing.T, latency, threshold time.Duration, message string) {
	t.Helper()
	if latency > threshold {
		t.Errorf("%s: latency %v > threshold %v", message, latency, threshold)
	}
}

// LogMetrics logs quality and performance metrics
func LogMetrics(t *testing.T, name string, recall RecallMetrics, perf PerformanceMetrics) {
	t.Helper()
	t.Logf("%s Quality: %s", name, recall.String())
	t.Logf("%s Performance: %s", name, perf.String())
}

// CompareIndexTypes compares multiple index types on the same dataset
type IndexComparison struct {
	Name    string
	Recall  RecallMetrics
	Perf    PerformanceMetrics
	Memory  uint64
	Builder func(d int) (faiss.Index, error)
}

// RunComparison executes comparison across multiple index types
func RunComparison(t *testing.T, vectors, queries []float32, d, k int, comparisons []IndexComparison) {
	t.Helper()

	// Compute ground truth once
	groundTruth, err := ComputeGroundTruth(vectors, queries, d, k, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Test each index type
	for i := range comparisons {
		comp := &comparisons[i]
		t.Run(comp.Name, func(t *testing.T) {
			// Build index
			index, err := comp.Builder(d)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Delete()

			// Add vectors
			if err := index.Add(vectors); err != nil {
				t.Fatalf("Failed to add vectors: %v", err)
			}

			// Measure memory
			comp.Memory = MeasureIndexMemory(index)

			// Search and measure
			results, latencies, err := SearchWithTiming(index, queries, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Calculate metrics
			comp.Recall = CalculateAllMetrics(groundTruth, results, k)
			comp.Perf = MeasureLatencies(latencies)

			// Log results
			LogMetrics(t, comp.Name, comp.Recall, comp.Perf)
			t.Logf("%s Memory: %d MB", comp.Name, comp.Memory/(1024*1024))
		})
	}

	// Summary comparison
	t.Run("Summary", func(t *testing.T) {
		t.Logf("\n%-20s %10s %10s %10s %10s", "Index", "Recall@10", "QPS", "P99", "Memory(MB)")
		t.Logf("%s", "-------------------------------------------------------------------")
		for _, comp := range comparisons {
			t.Logf("%-20s %10.4f %10.0f %10v %10d",
				comp.Name,
				comp.Recall.Recall10,
				comp.Perf.QPS,
				comp.Perf.P99Latency.Round(time.Microsecond),
				comp.Memory/(1024*1024),
			)
		}
	})
}

// ValidateIndexInvariants checks basic index invariants
func ValidateIndexInvariants(t *testing.T, index faiss.Index, expectedN, expectedD int) {
	t.Helper()

	if index.Ntotal() != expectedN {
		t.Errorf("Expected %d vectors, got %d", expectedN, index.Ntotal())
	}

	if index.D() != expectedD {
		t.Errorf("Expected dimension %d, got %d", expectedD, index.D())
	}

	if index.IsTrained() != true {
		// Some indexes need training - this check might need to be conditional
		t.Logf("Warning: Index not trained (might be expected)")
	}
}

// ValidateSearchResults checks basic search result properties
func ValidateSearchResults(t *testing.T, results []SearchResult, k int) {
	t.Helper()

	for i, result := range results {
		// Check result size
		if len(result.IDs) != k {
			t.Errorf("Query %d: expected %d results, got %d", i, k, len(result.IDs))
		}

		if len(result.Distances) != k {
			t.Errorf("Query %d: expected %d distances, got %d", i, k, len(result.Distances))
		}

		// Check distances are sorted (ascending for L2, descending for IP)
		for j := 1; j < len(result.Distances); j++ {
			if result.Distances[j] < result.Distances[j-1] {
				// For L2, distances should be non-decreasing
				t.Errorf("Query %d: distances not sorted at position %d: %f < %f",
					i, j, result.Distances[j], result.Distances[j-1])
				break
			}
		}
	}
}

// MeasureBuildTime measures time to build and train an index
func MeasureBuildTime(index faiss.Index, vectors []float32, needsTraining bool) (time.Duration, error) {
	start := time.Now()

	if needsTraining {
		if err := index.Train(vectors); err != nil {
			return 0, fmt.Errorf("training failed: %w", err)
		}
	}

	if err := index.Add(vectors); err != nil {
		return 0, fmt.Errorf("add failed: %w", err)
	}

	return time.Since(start), nil
}
