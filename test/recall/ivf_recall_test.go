package recall

import (
	"fmt"
	"testing"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
)

// TestIVF_Recall_Synthetic tests IVF with synthetic data
func TestIVF_Recall_Synthetic(t *testing.T) {
	config := RecallTestConfig{
		Name:          "IVF100_nprobe10",
		IndexType:     "IndexIVFFlat",
		BuildIndex:    buildIVF(100, 10),
		NeedsTraining: true,
		N:             10000,
		D:             128,
		NQ:            100,
		MinRecall10:   0.85,
		K:             10,
		Metric:        faiss.MetricL2,
		Distribution:  datasets.UniformRandom,
	}

	RunRecallTest(t, config)
}

// TestIVF_Recall_SIFT10K tests IVF with real SIFT dataset
func TestIVF_Recall_SIFT10K(t *testing.T) {
	config := RecallTestConfig{
		Name:          "IVF100_nprobe10_SIFT10K",
		IndexType:     "IndexIVFFlat",
		BuildIndex:    buildIVF(100, 10),
		NeedsTraining: true,
		TrainSize:     5000, // Use subset for training
		UseDataset:    "SIFT10K",
		MinRecall10:   0.85,
		K:             10,
		Metric:        faiss.MetricL2,
		TestdataPath:  DefaultTestdataPath(),
		SkipIfNoData:  true,
	}

	RunRecallTest(t, config)
}

// TestIVF_Recall_SIFT1M tests IVF with full SIFT1M dataset
func TestIVF_Recall_SIFT1M(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping SIFT1M test in short mode")
	}

	config := RecallTestConfig{
		Name:          "IVF4096_nprobe16_SIFT1M",
		IndexType:     "IndexIVFFlat",
		BuildIndex:    buildIVF(4096, 16), // More clusters for 1M vectors
		NeedsTraining: true,
		TrainSize:     100000, // 100K training vectors
		UseDataset:    "SIFT1M",
		MinRecall10:   0.85,
		K:             10,
		Metric:        faiss.MetricL2,
		TestdataPath:  DefaultTestdataPath(),
		SkipIfNoData:  true,
	}

	RunRecallTest(t, config)
}

// TestIVF_ParameterSweep_nlist tests different nlist values
func TestIVF_ParameterSweep_nlist(t *testing.T) {
	nlistValues := []int{50, 100, 200, 400}
	configs := make([]RecallTestConfig, 0, len(nlistValues))

	for _, nlist := range nlistValues {
		// nprobe should be proportional to nlist
		nprobe := nlist / 10
		if nprobe < 1 {
			nprobe = 1
		}

		config := RecallTestConfig{
			Name:          fmt.Sprintf("IVF%d_nprobe%d", nlist, nprobe),
			IndexType:     "IndexIVFFlat",
			BuildIndex:    buildIVF(nlist, nprobe),
			NeedsTraining: true,
			N:             10000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.75, // Lower target for sweep
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.GaussianClustered,
		}
		configs = append(configs, config)
	}

	results := RunParameterSweep(t, "IVF_nlist_Sweep", configs)

	// Analysis
	t.Run("Validation", func(t *testing.T) {
		if len(results) < 2 {
			t.Skip("Not enough results for validation")
		}

		t.Logf("nlist Impact on Recall and Performance:")
		for i, result := range results {
			t.Logf("nlist=%d: Recall@10=%.4f, QPS=%.0f, Memory=%dMB",
				nlistValues[i],
				result.Metrics.Recall10,
				result.Perf.QPS,
				result.Memory/(1024*1024))
		}
	})
}

// TestIVF_ParameterSweep_nprobe tests different nprobe values
func TestIVF_ParameterSweep_nprobe(t *testing.T) {
	nlist := 100
	nprobeValues := []int{1, 5, 10, 20, 40, 80}
	configs := make([]RecallTestConfig, 0, len(nprobeValues))

	for _, nprobe := range nprobeValues {
		config := RecallTestConfig{
			Name:          fmt.Sprintf("IVF100_nprobe%d", nprobe),
			IndexType:     "IndexIVFFlat",
			BuildIndex:    buildIVF(nlist, nprobe),
			NeedsTraining: true,
			N:             10000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.70, // Lower target for sweep
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		}
		configs = append(configs, config)
	}

	results := RunParameterSweep(t, "IVF_nprobe_Sweep", configs)

	// Validate recall-speed tradeoff
	t.Run("Validation", func(t *testing.T) {
		if len(results) < 2 {
			t.Skip("Not enough results for validation")
		}

		t.Logf("\nRecall-Speed Tradeoff Analysis:")
		t.Logf("nprobe: Recall@10, QPS")
		for i, result := range results {
			t.Logf("%6d: %.4f, %.0f",
				nprobeValues[i],
				result.Metrics.Recall10,
				result.Perf.QPS)
		}

		// Higher nprobe should improve recall
		firstRecall := results[0].Metrics.Recall10
		lastRecall := results[len(results)-1].Metrics.Recall10

		if lastRecall < firstRecall {
			t.Errorf("Higher nprobe should improve recall: nprobe=%d (%.4f) < nprobe=%d (%.4f)",
				nprobeValues[len(nprobeValues)-1], lastRecall,
				nprobeValues[0], firstRecall)
		}

		// Lower nprobe should have higher QPS
		firstQPS := results[0].Perf.QPS
		lastQPS := results[len(results)-1].Perf.QPS

		if firstQPS < lastQPS {
			t.Logf("Note: Lower nprobe did not achieve higher QPS as expected")
			t.Logf("nprobe=%d: QPS=%.0f, nprobe=%d: QPS=%.0f",
				nprobeValues[0], firstQPS,
				nprobeValues[len(nprobeValues)-1], lastQPS)
		}
	})
}

// TestIVF_OptimalConfiguration tests recommended nlist/nprobe combinations
func TestIVF_OptimalConfiguration(t *testing.T) {
	// Test recommended configurations for different dataset sizes
	configs := []RecallTestConfig{
		{
			Name:          "IVF_1K_optimal",
			IndexType:     "IndexIVFFlat",
			BuildIndex:    buildIVF(10, 2),
			NeedsTraining: true,
			N:             1000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.85,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
		{
			Name:          "IVF_10K_optimal",
			IndexType:     "IndexIVFFlat",
			BuildIndex:    buildIVF(100, 10),
			NeedsTraining: true,
			N:             10000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.85,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
		{
			Name:          "IVF_100K_optimal",
			IndexType:     "IndexIVFFlat",
			BuildIndex:    buildIVF(1000, 20),
			NeedsTraining: true,
			N:             100000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.85,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
	}

	if !testing.Short() {
		configs = append(configs, RecallTestConfig{
			Name:          "IVF_1M_optimal",
			IndexType:     "IndexIVFFlat",
			BuildIndex:    buildIVF(4096, 32),
			NeedsTraining: true,
			N:             1000000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.85,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		})
	}

	RunParameterSweep(t, "IVF_Optimal_Configurations", configs)
}

// TestIVF_Clustered tests IVF with clustered data (ideal case)
func TestIVF_Clustered(t *testing.T) {
	config := RecallTestConfig{
		Name:          "IVF100_nprobe10_Clustered",
		IndexType:     "IndexIVFFlat",
		BuildIndex:    buildIVF(100, 10),
		NeedsTraining: true,
		N:             10000,
		D:             128,
		NQ:            100,
		MinRecall10:   0.90, // Should be higher for clustered data
		K:             10,
		Metric:        faiss.MetricL2,
		Distribution:  datasets.GaussianClustered,
	}

	result := RunRecallTest(t, config)

	if result.Metrics.Recall10 > 0.95 {
		t.Logf("✓ IVF performs well on clustered data: Recall@10=%.4f", result.Metrics.Recall10)
	}
}

// TestIVF_InnerProduct tests IVF with IP metric
func TestIVF_InnerProduct(t *testing.T) {
	config := RecallTestConfig{
		Name:          "IVF100_nprobe10_IP",
		IndexType:     "IndexIVFFlat_IP",
		BuildIndex:    buildIVF_IP(100, 10),
		NeedsTraining: true,
		N:             10000,
		D:             128,
		NQ:            100,
		MinRecall10:   0.85,
		K:             10,
		Metric:        faiss.MetricInnerProduct,
		Distribution:  datasets.Normalized, // Must normalize for IP
	}

	RunRecallTest(t, config)
}

// TestIVF_HighDimensional tests IVF with high-dimensional vectors
func TestIVF_HighDimensional(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping high-dimensional test in short mode")
	}

	dimensions := []int{768, 1536}

	for _, d := range dimensions {
		t.Run(fmt.Sprintf("Dim%d", d), func(t *testing.T) {
			// Adjust nlist based on dimension (curse of dimensionality)
			nlist := 100
			if d > 500 {
				nlist = 200
			}

			config := RecallTestConfig{
				Name:          fmt.Sprintf("IVF%d_nprobe10_Dim%d", nlist, d),
				IndexType:     "IndexIVFFlat",
				BuildIndex:    buildIVF(nlist, 10),
				NeedsTraining: true,
				N:             5000,
				D:             d,
				NQ:            50,
				MinRecall10:   0.80, // Lower target for high dimensions
				K:             10,
				Metric:        faiss.MetricL2,
				Distribution:  datasets.Normalized,
			}

			RunRecallTest(t, config)
		})
	}
}

// TestIVF_TrainingSize tests impact of training set size
func TestIVF_TrainingSize(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping training size test in short mode")
	}

	n := 20000
	trainSizes := []int{1000, 2000, 5000, 10000, 20000}
	configs := make([]RecallTestConfig, 0, len(trainSizes))

	for _, trainSize := range trainSizes {
		config := RecallTestConfig{
			Name:          fmt.Sprintf("IVF100_train%d", trainSize),
			IndexType:     "IndexIVFFlat",
			BuildIndex:    buildIVF(100, 10),
			NeedsTraining: true,
			TrainSize:     trainSize,
			N:             n,
			D:             128,
			NQ:            100,
			MinRecall10:   0.75,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		}
		configs = append(configs, config)
	}

	results := RunParameterSweep(t, "IVF_Training_Size", configs)

	// Analysis
	t.Run("Validation", func(t *testing.T) {
		if len(results) < 2 {
			t.Skip("Not enough results for validation")
		}

		t.Logf("\nTraining Set Size Impact:")
		for i, result := range results {
			t.Logf("Train=%d: Recall@10=%.4f",
				trainSizes[i],
				result.Metrics.Recall10)
		}

		// More training data should generally improve recall
		if results[len(results)-1].Metrics.Recall10 < results[0].Metrics.Recall10-0.05 {
			t.Logf("Note: Larger training set did not significantly improve recall")
		}
	})
}

// Helper functions to build IVF indexes

func buildIVF(nlist int, nprobe int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		quantizer, err := faiss.NewIndexFlatL2(d)
		if err != nil {
			return nil, err
		}
		index, err := faiss.NewIndexIVFFlat(quantizer, d, nlist, metric)
		if err != nil {
			return nil, err
		}
		index.SetNprobe(nprobe)
		return index, nil
	}
}

func buildIVF_IP(nlist int, nprobe int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		quantizer, err := faiss.NewIndexFlatIP(d)
		if err != nil {
			return nil, err
		}
		index, err := faiss.NewIndexIVFFlat(quantizer, d, nlist, faiss.MetricInnerProduct)
		if err != nil {
			return nil, err
		}
		index.SetNprobe(nprobe)
		return index, nil
	}
}
