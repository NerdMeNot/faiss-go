package recall

import (
	"fmt"
	"testing"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
)

// TestHNSW_Recall_Synthetic tests HNSW with synthetic data
func TestHNSW_Recall_Synthetic(t *testing.T) {
	config := RecallTestConfig{
		Name:         "HNSW_M32_efSearch64",
		IndexType:    "IndexHNSWFlat",
		BuildIndex:   buildHNSW(32, 64),
		N:            10000,
		D:            128,
		NQ:           100,
		MinRecall10:  0.80,
		K:            10,
		Metric:       faiss.MetricL2,
		Distribution: datasets.UniformRandom,
	}

	RunRecallTest(t, config)
}

// TestHNSW_Recall_SIFT10K tests HNSW with real SIFT dataset
func TestHNSW_Recall_SIFT10K(t *testing.T) {
	config := RecallTestConfig{
		Name:         "HNSW_M32_efSearch64_SIFT10K",
		IndexType:    "IndexHNSWFlat",
		BuildIndex:   buildHNSW(32, 64),
		UseDataset:   "SIFT10K",
		MinRecall1:   0.90,
		MinRecall10:  0.80,
		MinRecall100: 0.90,
		K:            100,
		Metric:       faiss.MetricL2,
		TestdataPath: DefaultTestdataPath(),
		SkipIfNoData: true,
	}

	RunRecallTest(t, config)
}

// TestHNSW_Recall_SIFT1M tests HNSW with full SIFT1M dataset
func TestHNSW_Recall_SIFT1M(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping SIFT1M test in short mode")
	}

	config := RecallTestConfig{
		Name:         "HNSW_M32_efSearch64_SIFT1M",
		IndexType:    "IndexHNSWFlat",
		BuildIndex:   buildHNSW(32, 64),
		UseDataset:   "SIFT1M",
		MinRecall10:  0.80,
		K:            10,
		Metric:       faiss.MetricL2,
		TestdataPath: DefaultTestdataPath(),
		SkipIfNoData: true,
	}

	RunRecallTest(t, config)
}

// TestHNSW_ParameterSweep_M tests different M values
func TestHNSW_ParameterSweep_M(t *testing.T) {
	mValues := []int{16, 32, 48, 64}
	configs := make([]RecallTestConfig, 0, len(mValues))

	for _, m := range mValues {
		config := RecallTestConfig{
			Name:         fmt.Sprintf("HNSW_M%d_efSearch64", m),
			IndexType:    "IndexHNSWFlat",
			BuildIndex:   buildHNSW(m, 64),
			N:            10000,
			D:            128,
			NQ:           100,
			MinRecall10:  0.75, // Lower target for sweep
			K:            10,
			Metric:       faiss.MetricL2,
			Distribution: datasets.UniformRandom,
		}
		configs = append(configs, config)
	}

	results := RunParameterSweep(t, "HNSW_M_Sweep", configs)

	// Verify that higher M generally improves recall
	t.Run("Validation", func(t *testing.T) {
		if len(results) < 2 {
			t.Skip("Not enough results for validation")
		}

		// Check that M=64 has better or equal recall than M=16
		if results[len(results)-1].Metrics.Recall10 < results[0].Metrics.Recall10-0.05 {
			t.Logf("Note: Higher M did not improve recall as expected")
			t.Logf("M=16: Recall@10=%.4f, M=64: Recall@10=%.4f",
				results[0].Metrics.Recall10,
				results[len(results)-1].Metrics.Recall10)
		}
	})
}

// TestHNSW_ParameterSweep_efSearch tests different efSearch values
func TestHNSW_ParameterSweep_efSearch(t *testing.T) {
	efSearchValues := []int{16, 32, 64, 128, 256}
	configs := make([]RecallTestConfig, 0, len(efSearchValues))

	for _, ef := range efSearchValues {
		config := RecallTestConfig{
			Name:         fmt.Sprintf("HNSW_M32_efSearch%d", ef),
			IndexType:    "IndexHNSWFlat",
			BuildIndex:   buildHNSW(32, ef),
			N:            10000,
			D:            128,
			NQ:           100,
			MinRecall10:  0.85, // Lower target for sweep
			K:            10,
			Metric:       faiss.MetricL2,
			Distribution: datasets.UniformRandom,
		}
		configs = append(configs, config)
	}

	results := RunParameterSweep(t, "HNSW_efSearch_Sweep", configs)

	// Verify recall-speed tradeoff
	t.Run("Validation", func(t *testing.T) {
		if len(results) < 2 {
			t.Skip("Not enough results for validation")
		}

		// Higher efSearch should improve recall but reduce QPS
		t.Logf("Recall-Speed Tradeoff Analysis:")
		for i, result := range results {
			t.Logf("efSearch=%d: Recall@10=%.4f, QPS=%.0f",
				efSearchValues[i],
				result.Metrics.Recall10,
				result.Perf.QPS)
		}

		// Check that higher efSearch improves recall
		if results[len(results)-1].Metrics.Recall10 < results[0].Metrics.Recall10 {
			t.Errorf("Higher efSearch should improve recall")
		}

		// Check that lower efSearch has higher QPS
		if results[0].Perf.QPS < results[len(results)-1].Perf.QPS {
			t.Logf("Note: Lower efSearch did not achieve higher QPS as expected")
		}
	})
}

// TestHNSW_HighDimensional tests HNSW with high-dimensional vectors
func TestHNSW_HighDimensional(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping high-dimensional test in short mode")
	}

	// Test with dimensions similar to modern embeddings
	dimensions := []int{768, 1536}

	for _, d := range dimensions {
		t.Run(fmt.Sprintf("Dim%d", d), func(t *testing.T) {
			config := RecallTestConfig{
				Name:         fmt.Sprintf("HNSW_M32_efSearch64_Dim%d", d),
				IndexType:    "IndexHNSWFlat",
				BuildIndex:   buildHNSW(32, 64),
				N:            5000, // Smaller N for high dimensions
				D:            d,
				NQ:           50,
				MinRecall10:  0.75,
				K:            10,
				Metric:       faiss.MetricL2,
				Distribution: datasets.Normalized, // Normalized like real embeddings
			}

			RunRecallTest(t, config)
		})
	}
}

// TestHNSW_InnerProduct tests HNSW with IP metric (cosine similarity)
func TestHNSW_InnerProduct(t *testing.T) {
	config := RecallTestConfig{
		Name:         "HNSW_M32_efSearch64_IP",
		IndexType:    "IndexHNSWFlat_IP",
		BuildIndex:   buildHNSW_IP(32, 64),
		N:            10000,
		D:            128,
		NQ:           100,
		MinRecall10:  0.80,
		K:            10,
		Metric:       faiss.MetricInnerProduct,
		Distribution: datasets.Normalized, // Must normalize for IP
	}

	RunRecallTest(t, config)
}

// TestHNSW_Clustered tests HNSW with clustered data
func TestHNSW_Clustered(t *testing.T) {
	config := RecallTestConfig{
		Name:         "HNSW_M32_efSearch64_Clustered",
		IndexType:    "IndexHNSWFlat",
		BuildIndex:   buildHNSW(32, 64),
		N:            10000,
		D:            128,
		NQ:           100,
		MinRecall10:  0.80,
		K:            10,
		Metric:       faiss.MetricL2,
		Distribution: datasets.GaussianClustered,
	}

	RunRecallTest(t, config)
}

// TestHNSW_LargeK tests HNSW with large K values
func TestHNSW_LargeK(t *testing.T) {
	kValues := []int{1, 10, 50, 100}
	configs := make([]RecallTestConfig, 0, len(kValues))

	for _, k := range kValues {
		var minRecall float64
		switch k {
		case 1:
			minRecall = 0.95
		case 10:
			minRecall = 0.95
		case 50:
			minRecall = 0.90
		case 100:
			minRecall = 0.85
		}

		config := RecallTestConfig{
			Name:         fmt.Sprintf("HNSW_M32_efSearch128_K%d", k),
			IndexType:    "IndexHNSWFlat",
			BuildIndex:   buildHNSW(32, 128), // Higher efSearch for large K
			N:            10000,
			D:            128,
			NQ:           100,
			MinRecall10:  minRecall,
			K:            k,
			Metric:       faiss.MetricL2,
			Distribution: datasets.UniformRandom,
		}
		configs = append(configs, config)
	}

	RunParameterSweep(t, "HNSW_K_Sweep", configs)
}

// Helper functions to build HNSW indexes with different parameters

func buildHNSW(m int, efSearch int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		index, err := faiss.NewIndexHNSWFlat(d, m, metric)
		if err != nil {
			return nil, err
		}
		index.SetEfSearch(efSearch)
		return index, nil
	}
}

func buildHNSW_IP(m int, efSearch int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		index, err := faiss.NewIndexHNSWFlat(d, m, faiss.MetricInnerProduct)
		if err != nil {
			return nil, err
		}
		index.SetEfSearch(efSearch)
		return index, nil
	}
}
