package recall

import (
	"fmt"
	"strings"
	"testing"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// RecallTestConfig configures a recall validation test
type RecallTestConfig struct {
	// Test identification
	Name      string // Test name (e.g., "HNSW_M32_efSearch64")
	IndexType string // Index type description

	// Index builder
	BuildIndex func(d int, metric faiss.MetricType) (faiss.Index, error)

	// Training configuration
	NeedsTraining bool // Whether index requires training
	TrainSize     int  // Number of vectors for training (0 = use all)

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
	MinQPS        float64       // Minimum queries per second

	// Test configuration
	K            int                       // Number of neighbors to retrieve
	Metric       faiss.MetricType          // Distance metric
	Distribution datasets.DataDistribution // For synthetic data
	TestdataPath string                    // Path to testdata directory
	SkipIfNoData bool                      // Skip instead of fail if dataset missing
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

// datasetResult contains loaded/generated dataset
type datasetResult struct {
	vectors        []float32
	queries        []float32
	groundTruth    []helpers.GroundTruth
	n, nq, d       int
}

// loadOrGenerateDataset loads a real dataset or generates synthetic data
func loadOrGenerateDataset(t *testing.T, config RecallTestConfig) (datasetResult, error) {
	var res datasetResult

	if config.UseDataset != "" {
		// Load real dataset
		dataset, err := datasets.LoadDataset(config.UseDataset, config.TestdataPath)
		if err != nil {
			if config.SkipIfNoData {
				t.Skipf("Dataset %s not available: %v. Run: ./scripts/download_test_datasets.sh",
					config.UseDataset, err)
			}
			return res, fmt.Errorf("failed to load dataset: %w", err)
		}

		res.vectors = dataset.Vectors
		res.queries = dataset.Queries
		res.n = dataset.N
		res.nq = dataset.NQ
		res.d = dataset.D

		// Convert ground truth format
		res.groundTruth = make([]helpers.GroundTruth, res.nq)
		for i := 0; i < res.nq && i < len(dataset.GroundTruth); i++ {
			res.groundTruth[i] = helpers.GroundTruth{
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
			Seed:         42,                           // Reproducible
		}

		synData := datasets.GenerateSyntheticData(genConfig)
		synData.GenerateQueries(config.NQ, config.Distribution)

		res.vectors = synData.Vectors
		res.queries = synData.Queries
		res.n = config.N
		res.nq = config.NQ
		res.d = config.D

		// Compute ground truth
		groundTruth, err := helpers.ComputeGroundTruth(res.vectors, res.queries, res.d, config.K, config.Metric)
		if err != nil {
			return res, fmt.Errorf("failed to compute ground truth: %w", err)
		}
		res.groundTruth = groundTruth
	}

	return res, nil
}

// trainIndexIfNeeded trains the index if required by configuration
func trainIndexIfNeeded(t *testing.T, index faiss.Index, config RecallTestConfig, vectors []float32, n int) error {
	if !config.NeedsTraining {
		return nil
	}

	trainVectors := vectors
	if config.TrainSize > 0 && config.TrainSize < n {
		trainVectors = vectors[:config.TrainSize*config.D]
	}

	t.Logf("Training index with %d vectors...", len(trainVectors)/config.D)
	if err := index.Train(trainVectors); err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	if !index.IsTrained() {
		return fmt.Errorf("index not trained after Train() call")
	}

	return nil
}

// validateTargets checks if results meet quality and performance targets
func validateTargets(t *testing.T, config RecallTestConfig, metrics helpers.RecallMetrics, perf helpers.PerformanceMetrics) bool {
	passed := true

	if config.MinRecall1 > 0 && metrics.Recall1 < config.MinRecall1 {
		t.Errorf("Recall@1 too low: %.4f < %.4f", metrics.Recall1, config.MinRecall1)
		passed = false
	}

	if config.MinRecall10 > 0 && metrics.Recall10 < config.MinRecall10 {
		t.Errorf("Recall@10 too low: %.4f < %.4f", metrics.Recall10, config.MinRecall10)
		passed = false
	}

	if config.MinRecall100 > 0 && metrics.Recall100 < config.MinRecall100 {
		t.Errorf("Recall@100 too low: %.4f < %.4f", metrics.Recall100, config.MinRecall100)
		passed = false
	}

	if config.MaxP99Latency > 0 && perf.P99Latency > config.MaxP99Latency {
		t.Errorf("P99 latency too high: %v > %v", perf.P99Latency, config.MaxP99Latency)
		passed = false
	}

	if config.MinQPS > 0 && perf.QPS < config.MinQPS {
		t.Errorf("QPS too low: %.0f < %.0f", perf.QPS, config.MinQPS)
		passed = false
	}

	return passed
}

// RunRecallTest executes a single recall validation test
func RunRecallTest(t *testing.T, config RecallTestConfig) RecallTestResult {
	t.Helper()

	result := RecallTestResult{
		Config: config,
		Passed: false,
	}

	// Load or generate dataset
	data, err := loadOrGenerateDataset(t, config)
	if err != nil {
		result.Error = err
		t.Error(err)
		return result
	}

	t.Logf("Testing %s with %d vectors (%d-dim), %d queries", config.Name, data.n, data.d, data.nq)

	// Build approximate index
	index, err := config.BuildIndex(data.d, config.Metric)
	if err != nil {
		result.Error = fmt.Errorf("failed to create index: %w", err)
		t.Error(result.Error)
		return result
	}
	defer index.Close()

	// Train if needed
	if err := trainIndexIfNeeded(t, index, config, data.vectors, data.n); err != nil {
		// Skip test if insufficient training data (common with synthetic data)
		if strings.Contains(err.Error(), "insufficient training data") {
			t.Skipf("Skipping test due to insufficient training data: %v", err)
			return result
		}
		result.Error = err
		t.Error(err)
		return result
	}

	// Add vectors
	t.Logf("Adding %d vectors to index...", data.n)
	if err := index.Add(data.vectors); err != nil {
		result.Error = fmt.Errorf("add failed: %w", err)
		t.Error(result.Error)
		return result
	}

	if index.Ntotal() != int64(data.n) {
		result.Error = fmt.Errorf("expected %d vectors in index, got %d", data.n, index.Ntotal())
		t.Error(result.Error)
		return result
	}

	// Measure memory
	result.Memory = helpers.MeasureIndexMemory(index)

	// Search and measure performance
	t.Logf("Searching %d queries...", data.nq)
	results, latencies, err := helpers.SearchWithTiming(index, data.queries, config.K)
	if err != nil {
		result.Error = fmt.Errorf("search failed: %w", err)
		t.Error(result.Error)
		return result
	}

	// Calculate metrics
	result.Metrics = helpers.CalculateAllMetrics(data.groundTruth, results, config.K)
	result.Perf = helpers.MeasureLatencies(latencies)

	// Log results
	t.Logf("Results for %s:", config.Name)
	t.Logf("  Quality:     %s", result.Metrics.String())
	t.Logf("  Performance: %s", result.Perf.String())
	t.Logf("  Memory:      %d MB", result.Memory/(1024*1024))

	// Validate targets
	result.Passed = validateTargets(t, config, result.Metrics, result.Perf)

	if result.Passed {
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
