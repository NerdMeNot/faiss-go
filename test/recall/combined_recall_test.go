package recall

import (
	"fmt"
	"testing"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
)

// TestIVFPQ_Recall_Synthetic tests IVFPQ (combination of IVF and PQ)
func TestIVFPQ_Recall_Synthetic(t *testing.T) {
	config := RecallTestConfig{
		Name:          "IVFPQ_nlist100_M8_nprobe10",
		IndexType:     "IndexIVFPQ",
		BuildIndex:    buildIVFPQ(100, 8, 8, 10),
		NeedsTraining: true,
		N:             10000,
		D:             128,
		NQ:            100,
		MinRecall10:   0.75,
		K:             10,
		Metric:        faiss.MetricL2,
		Distribution:  datasets.UniformRandom,
	}

	RunRecallTest(t, config)
}

// TestIVFPQ_Recall_SIFT10K tests IVFPQ with real dataset
func TestIVFPQ_Recall_SIFT10K(t *testing.T) {
	config := RecallTestConfig{
		Name:          "IVFPQ_nlist100_M8_nprobe10_SIFT10K",
		IndexType:     "IndexIVFPQ",
		BuildIndex:    buildIVFPQ(100, 8, 8, 10),
		NeedsTraining: true,
		TrainSize:     5000,
		UseDataset:    "SIFT10K",
		MinRecall10:   0.70,
		K:             10,
		Metric:        faiss.MetricL2,
		TestdataPath:  DefaultTestdataPath(),
		SkipIfNoData:  true,
	}

	RunRecallTest(t, config)
}

// TestIVFPQ_BestPractices tests recommended IVFPQ configurations
func TestIVFPQ_BestPractices(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping best practices test in short mode")
	}

	configs := []RecallTestConfig{
		{
			Name:          "IVFPQ_10K_recommended",
			IndexType:     "IndexIVFPQ",
			BuildIndex:    buildIVFPQ(100, 8, 8, 10),
			NeedsTraining: true,
			N:             10000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.75,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
		{
			Name:          "IVFPQ_100K_recommended",
			IndexType:     "IndexIVFPQ",
			BuildIndex:    buildIVFPQ(1000, 16, 8, 20),
			NeedsTraining: true,
			N:             100000,
			D:             256,
			NQ:            100,
			MinRecall10:   0.75,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
	}

	RunParameterSweep(t, "IVFPQ_Best_Practices", configs)
}

// TestScalarQuantizer_Recall tests SQ (8-bit quantization)
func TestScalarQuantizer_Recall(t *testing.T) {
	config := RecallTestConfig{
		Name:          "SQ8",
		IndexType:     "IndexScalarQuantizer",
		BuildIndex:    buildSQ(faiss.QuantizerType_QT_8bit),
		NeedsTraining: true,
		N:             10000,
		D:             128,
		NQ:            100,
		MinRecall10:   0.95, // SQ should have high recall (minimal quantization loss)
		K:             10,
		Metric:        faiss.MetricL2,
		Distribution:  datasets.UniformRandom,
	}

	RunRecallTest(t, config)
}

// TestIVFSQ_Recall tests IVF + Scalar Quantizer
func TestIVFSQ_Recall(t *testing.T) {
	config := RecallTestConfig{
		Name:          "IVFSQ_nlist100_nprobe10",
		IndexType:     "IndexIVFScalarQuantizer",
		BuildIndex:    buildIVFSQ(100, faiss.QuantizerType_QT_8bit, 10),
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

// TestPQFastScan_Recall tests PQ with SIMD optimizations
func TestPQFastScan_Recall(t *testing.T) {
	config := RecallTestConfig{
		Name:          "PQFastScan_M8",
		IndexType:     "IndexPQFastScan",
		BuildIndex:    buildPQFastScan(8),
		NeedsTraining: true,
		N:             10000,
		D:             128,
		NQ:            100,
		MinRecall10:   0.70,
		K:             10,
		Metric:        faiss.MetricL2,
		Distribution:  datasets.UniformRandom,
	}

	RunRecallTest(t, config)
}

// TestIndexComparison compares major index types on the same dataset
func TestIndexComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping comparison test in short mode")
	}

	configs := []RecallTestConfig{
		{
			Name:      "Flat_L2_baseline",
			IndexType: "IndexFlatL2",
			BuildIndex: func(d int, metric faiss.MetricType) (faiss.Index, error) {
				return faiss.NewIndexFlatL2(d)
			},
			N:            10000,
			D:            128,
			NQ:           100,
			MinRecall10:  1.0,
			K:            10,
			Metric:       faiss.MetricL2,
			Distribution: datasets.UniformRandom,
		},
		{
			Name:         "HNSW_M32_efSearch64",
			IndexType:    "IndexHNSWFlat",
			BuildIndex:   buildHNSW(32, 64),
			N:            10000,
			D:            128,
			NQ:           100,
			MinRecall10:  0.95,
			K:            10,
			Metric:       faiss.MetricL2,
			Distribution: datasets.UniformRandom,
		},
		{
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
		},
		{
			Name:          "PQ_M8_nbits8",
			IndexType:     "IndexPQ",
			BuildIndex:    buildPQ(8, 8),
			NeedsTraining: true,
			N:             10000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.70,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
		{
			Name:          "IVFPQ_nlist100_M8_nprobe10",
			IndexType:     "IndexIVFPQ",
			BuildIndex:    buildIVFPQ(100, 8, 8, 10),
			NeedsTraining: true,
			N:             10000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.75,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
	}

	results := RunParameterSweep(t, "Index_Type_Comparison", configs)

	// Detailed comparison summary
	t.Run("Analysis", func(t *testing.T) {
		t.Logf("\n=== Index Type Comparison ===\n")
		t.Logf("Dataset: 10K vectors, 128-dim, 100 queries, K=10\n")
		t.Logf("%-25s %10s %10s %8s %10s %10s",
			"Index", "Recall@10", "Recall@1", "QPS", "P99", "Memory(MB)")
		t.Logf("%s", fmt.Sprintf("%s", "----------------------------------------------------------------------------"))

		for _, result := range results {
			t.Logf("%-25s %10.4f %10.4f %8.0f %10v %10d",
				result.Config.IndexType,
				result.Metrics.Recall10,
				result.Metrics.Recall1,
				result.Perf.QPS,
				result.Perf.P99Latency.Round(1000),
				result.Memory/(1024*1024))
		}

		t.Logf("\nKey Insights:")
		t.Logf("- Flat: 100%% recall (baseline), lowest QPS")
		t.Logf("- HNSW: Best recall/speed tradeoff for most use cases")
		t.Logf("- IVF: Good for large datasets, tunable with nprobe")
		t.Logf("- PQ: Best memory efficiency, lower recall")
		t.Logf("- IVFPQ: Best balance for very large datasets (1M+)")
	})
}

// Helper functions

func buildIVFPQ(nlist int, m int, nbits int, nprobe int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		quantizer, err := faiss.NewIndexFlatL2(d)
		if err != nil {
			return nil, err
		}
		index, err := faiss.NewIndexIVFPQ(quantizer, d, nlist, m, nbits, metric)
		if err != nil {
			return nil, err
		}
		index.SetNprobe(nprobe)
		return index, nil
	}
}

func buildSQ(qtype faiss.QuantizerType) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		return faiss.NewIndexScalarQuantizer(d, qtype, metric)
	}
}

func buildIVFSQ(nlist int, qtype faiss.QuantizerType, nprobe int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		quantizer, err := faiss.NewIndexFlatL2(d)
		if err != nil {
			return nil, err
		}
		index, err := faiss.NewIndexIVFScalarQuantizer(quantizer, d, nlist, qtype, metric)
		if err != nil {
			return nil, err
		}
		index.SetNprobe(nprobe)
		return index, nil
	}
}

func buildPQFastScan(m int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		return faiss.NewIndexPQFastScan(d, m, 4, metric) // 4-bit codes for FastScan
	}
}
