package recall

import (
	"fmt"
	"testing"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// RecallTestConfig configures a recall validation test
type RecallTestConfig struct {
	// Test identification
	Name        string // Test name (e.g., "HNSW_M32_efSearch64")
	IndexType   string // Index type description

	// Index builder
	BuildIndex func(d int, metric faiss.MetricType) (faiss.Index, error)

	// Training configuration
	NeedsTraining bool   // Whether index requires training
	TrainSize     int    // Number of vectors for training (0 = use all)

	// Dataset
	UseDataset string // Dataset name (e.g., "SIFT10K") or empty for synthetic
	N          int    // Number of vectors (for synthetic data)
	D          int    // Dimension (for synthetic data)
	NQ         int    // Number of queries (for synthetic data)

	// Quality targets
	MinRecall1   float64 // Minimum recall@1 (0.0 = skip check)
	MinRecall10  float64 // Minimum recall@10
	MinRecall100 float64 // Minimum recall@100 (0.0 = skip check)

	// Performance targets (0 = skip check)
	MaxP99Latency time.Duration // Maximum P99 latency
	MinQPS        float64        // Minimum queries per second

	// Test configuration
	K              int                      // Number of neighbors to retrieve
	Metric         faiss.MetricType         // Distance metric
	Distribution   datasets.DataDistribution // For synthetic data
	TestdataPath   string                   // Path to testdata directory
	SkipIfNoData   bool                     // Skip instead of fail if dataset missing
}

// RecallTestResult contains test results
type RecallTestResult struct {
	Config  RecallTestConfig
	Metrics helpers.RecallMetrics
	Perf    helpers.PerformanceMetrics
	Memory  uint64
	Passed  bool
	Error   error
}

// RunRecallTest executes a single recall validation test
func RunRecallTest(t *testing.T, config RecallTestConfig) RecallTestResult {
	t.Helper()

	result := RecallTestResult{
		Config: config,
		Passed: false,
	}

	// Load or generate dataset
	var vectors, queries []float32
	var groundTruthData []helpers.GroundTruth
	var n, nq, d int

	if config.UseDataset != "" {
		// Load real dataset
		dataset, err := datasets.LoadDataset(config.UseDataset, config.TestdataPath)
		if err != nil {
			if config.SkipIfNoData {
				t.Skipf("Dataset %s not available: %v. Run: ./scripts/download_test_datasets.sh",
					config.UseDataset, err)
				return result
			}
			result.Error = fmt.Errorf("failed to load dataset: %w", err)
			t.Error(result.Error)
			return result
		}

		vectors = dataset.Vectors
		queries = dataset.Queries
		n = dataset.N
		nq = dataset.NQ
		d = dataset.D

		// Convert ground truth format
		groundTruthData = make([]helpers.GroundTruth, nq)
		for i := 0; i < nq && i < len(dataset.GroundTruth); i++ {
			groundTruthData[i] = helpers.GroundTruth{
				IDs: dataset.GroundTruth[i],
			}
		}
	} else {
		// Generate synthetic dataset
		genConfig := datasets.GeneratorConfig{
			N:            config.N,
			D:            config.D,
			Distribution: config.Distribution,
			NumClusters:  int(float64(config.N) * 0.1), // ~10% clusters
			Seed:         42, // Reproducible
		}

		synData := datasets.GenerateSyntheticData(genConfig)
		synData.GenerateQueries(config.NQ, config.Distribution)

		vectors = synData.Vectors
		queries = synData.Queries
		n = config.N
		nq = config.NQ
		d = config.D

		// Compute ground truth
		var err error
		groundTruthData, err = helpers.ComputeGroundTruth(vectors, queries, d, config.K, config.Metric)
		if err != nil {
			result.Error = fmt.Errorf("failed to compute ground truth: %w", err)
			t.Error(result.Error)
			return result
		}
	}

	t.Logf("Testing %s with %d vectors (%d-dim), %d queries", config.Name, n, d, nq)

	// Build approximate index
	index, err := config.BuildIndex(d, config.Metric)
	if err != nil {
		result.Error = fmt.Errorf("failed to create index: %w", err)
		t.Error(result.Error)
		return result
	}
	defer index.Delete()

	// Train if needed
	if config.NeedsTraining {
		trainVectors := vectors
		if config.TrainSize > 0 && config.TrainSize < n {
			trainVectors = vectors[:config.TrainSize*d]
		}

		t.Logf("Training index with %d vectors...", len(trainVectors)/d)
		if err := index.Train(trainVectors); err != nil {
			result.Error = fmt.Errorf("training failed: %w", err)
			t.Error(result.Error)
			return result
		}

		if !index.IsTrained() {
			result.Error = fmt.Errorf("index not trained after Train() call")
			t.Error(result.Error)
			return result
		}
	}

	// Add vectors
	t.Logf("Adding %d vectors to index...", n)
	if err := index.Add(vectors); err != nil {
		result.Error = fmt.Errorf("add failed: %w", err)
		t.Error(result.Error)
		return result
	}

	if index.Ntotal() != n {
		result.Error = fmt.Errorf("expected %d vectors in index, got %d", n, index.Ntotal())
		t.Error(result.Error)
		return result
	}

	// Measure memory
	result.Memory = helpers.MeasureIndexMemory(index)

	// Search and measure performance
	t.Logf("Searching %d queries...", nq)
	results, latencies, err := helpers.SearchWithTiming(index, queries, config.K)
	if err != nil {
		result.Error = fmt.Errorf("search failed: %w", err)
		t.Error(result.Error)
		return result
	}

	// Calculate metrics
	result.Metrics = helpers.CalculateAllMetrics(groundTruthData, results, config.K)
	result.Perf = helpers.MeasureLatencies(latencies)

	// Log results
	t.Logf("Results for %s:", config.Name)
	t.Logf("  Quality:     %s", result.Metrics.String())
	t.Logf("  Performance: %s", result.Perf.String())
	t.Logf("  Memory:      %d MB", result.Memory/(1024*1024))

	// Validate quality targets
	passed := true

	if config.MinRecall1 > 0 && result.Metrics.Recall1 < config.MinRecall1 {
		t.Errorf("Recall@1 too low: %.4f < %.4f", result.Metrics.Recall1, config.MinRecall1)
		passed = false
	}

	if config.MinRecall10 > 0 && result.Metrics.Recall10 < config.MinRecall10 {
		t.Errorf("Recall@10 too low: %.4f < %.4f", result.Metrics.Recall10, config.MinRecall10)
		passed = false
	}

	if config.MinRecall100 > 0 && result.Metrics.Recall100 < config.MinRecall100 {
		t.Errorf("Recall@100 too low: %.4f < %.4f", result.Metrics.Recall100, config.MinRecall100)
		passed = false
	}

	// Validate performance targets
	if config.MaxP99Latency > 0 && result.Perf.P99Latency > config.MaxP99Latency {
		t.Errorf("P99 latency too high: %v > %v", result.Perf.P99Latency, config.MaxP99Latency)
		passed = false
	}

	if config.MinQPS > 0 && result.Perf.QPS < config.MinQPS {
		t.Errorf("QPS too low: %.0f < %.0f", result.Perf.QPS, config.MinQPS)
		passed = false
	}

	result.Passed = passed

	if passed {
		t.Logf("✓ %s passed all targets", config.Name)
	} else {
		t.Logf("✗ %s failed some targets", config.Name)
	}

	return result
}

// RunParameterSweep runs tests across multiple parameter combinations
func RunParameterSweep(t *testing.T, sweepName string, configs []RecallTestConfig) []RecallTestResult {
	t.Helper()

	results := make([]RecallTestResult, 0, len(configs))

	t.Run(sweepName, func(t *testing.T) {
		for _, config := range configs {
			t.Run(config.Name, func(t *testing.T) {
				result := RunRecallTest(t, config)
				results = append(results, result)
			})
		}

		// Summary
		t.Run("Summary", func(t *testing.T) {
			t.Logf("\n%s Parameter Sweep Summary:", sweepName)
			t.Logf("%-30s %10s %10s %10s %10s %10s",
				"Configuration", "Recall@1", "Recall@10", "Recall@100", "QPS", "P99")
			t.Logf("%s", "-------------------------------------------------------------------------------------")

			for _, result := range results {
				if result.Error != nil {
					t.Logf("%-30s %10s", result.Config.Name, "ERROR")
					continue
				}

				status := "✓"
				if !result.Passed {
					status = "✗"
				}

				t.Logf("%s %-28s %10.4f %10.4f %10.4f %10.0f %10v",
					status,
					result.Config.Name,
					result.Metrics.Recall1,
					result.Metrics.Recall10,
					result.Metrics.Recall100,
					result.Perf.QPS,
					result.Perf.P99Latency.Round(time.Microsecond),
				)
			}
		})
	})

	return results
}

// DefaultTestdataPath returns the default path to testdata directory
func DefaultTestdataPath() string {
	return "../../testdata" // From test/recall/ to testdata/
}

// StandardRecallTargets returns recommended recall targets for different use cases
func StandardRecallTargets() map[string]struct{ R1, R10, R100 float64 } {
	return map[string]struct{ R1, R10, R100 float64 }{
		"high-precision":  {0.99, 0.99, 0.98}, // Requires very accurate results
		"balanced":        {0.95, 0.95, 0.90}, // Good balance of speed and accuracy
		"high-throughput": {0.80, 0.85, 0.80}, // Prioritize speed over accuracy
		"approximate":     {0.70, 0.75, 0.70}, // Maximum speed, acceptable accuracy
	}
}
