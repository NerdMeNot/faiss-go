package recall

import (
	"fmt"
	"strings"
	"testing"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
)

// TestPQ_Recall_Synthetic tests PQ with synthetic data
func TestPQ_Recall_Synthetic(t *testing.T) {
	config := RecallTestConfig{
		Name:          "PQ_M8_nbits8",
		IndexType:     "IndexPQ",
		BuildIndex:    buildPQ(8, 8),
		NeedsTraining: true,
		N:             10000,
		D:             128, // Must be divisible by M
		NQ:            100,
		MinRecall10:   0.70, // PQ typically has lower recall than HNSW/IVF
		K:             10,
		Metric:        faiss.MetricL2,
		Distribution:  datasets.UniformRandom,
	}

	RunRecallTest(t, config)
}

// TestPQ_Recall_SIFT10K tests PQ with real SIFT dataset
func TestPQ_Recall_SIFT10K(t *testing.T) {
	config := RecallTestConfig{
		Name:          "PQ_M8_nbits8_SIFT10K",
		IndexType:     "IndexPQ",
		BuildIndex:    buildPQ(8, 8),
		NeedsTraining: true,
		TrainSize:     5000,
		UseDataset:    "SIFT10K",
		MinRecall10:   0.65,
		K:             10,
		Metric:        faiss.MetricL2,
		TestdataPath:  DefaultTestdataPath(),
		SkipIfNoData:  true,
	}

	RunRecallTest(t, config)
}

// TestPQ_ParameterSweep_M tests different M values (subquantizers)
func TestPQ_ParameterSweep_M(t *testing.T) {
	// D must be divisible by M
	d := 128
	mValues := []int{4, 8, 16, 32}
	configs := make([]RecallTestConfig, 0, len(mValues))

	for _, m := range mValues {
		// Compression ratio: d * 4 / (m * nbits / 8)
		// For d=128, nbits=8: compression = 128*4 / (m*1) = 512/m
		compressionRatio := float64(d*4) / float64(m*1)

		config := RecallTestConfig{
			Name:          fmt.Sprintf("PQ_M%d_nbits8_compression%.0fx", m, compressionRatio),
			IndexType:     "IndexPQ",
			BuildIndex:    buildPQ(m, 8),
			NeedsTraining: true,
			N:             10000,
			D:             d,
			NQ:            100,
			MinRecall10:   0.60, // Lower target for sweep
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		}
		configs = append(configs, config)
	}

	results := RunParameterSweep(t, "PQ_M_Sweep", configs)

	// Analyze compression vs recall tradeoff
	t.Run("Validation", func(t *testing.T) {
		if len(results) < 2 {
			t.Skip("Not enough results for validation")
		}

		t.Logf("\nCompression vs Recall Tradeoff:")
		t.Logf("%6s %15s %10s %10s", "M", "Compression", "Recall@10", "Memory(MB)")
		for i, result := range results {
			m := mValues[i]
			compression := float64(d*4) / float64(m*1)
			t.Logf("%6d %15.0fx %10.4f %10d",
				m,
				compression,
				result.Metrics.Recall10,
				result.Memory/(1024*1024))
		}

		// Higher M should improve recall but use more memory
		firstRecall := results[0].Metrics.Recall10
		lastRecall := results[len(results)-1].Metrics.Recall10

		if lastRecall < firstRecall-0.05 {
			t.Logf("Note: Higher M did not significantly improve recall")
		}
	})
}

// TestPQ_ParameterSweep_nbits tests different nbits values
func TestPQ_ParameterSweep_nbits(t *testing.T) {
	m := 8
	nbitsValues := []int{4, 6, 8}
	configs := make([]RecallTestConfig, 0, len(nbitsValues))

	for _, nbits := range nbitsValues {
		config := RecallTestConfig{
			Name:          fmt.Sprintf("PQ_M8_nbits%d", nbits),
			IndexType:     "IndexPQ",
			BuildIndex:    buildPQ(m, nbits),
			NeedsTraining: true,
			N:             10000,
			D:             128,
			NQ:            100,
			MinRecall10:   0.55, // Lower target for sweep
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		}
		configs = append(configs, config)
	}

	results := RunParameterSweep(t, "PQ_nbits_Sweep", configs)

	// Analyze precision vs recall
	t.Run("Validation", func(t *testing.T) {
		if len(results) < 2 {
			t.Skip("Not enough results for validation")
		}

		t.Logf("\nBits per Code vs Recall:")
		for i, result := range results {
			t.Logf("nbits=%d: Recall@10=%.4f, Memory=%dMB",
				nbitsValues[i],
				result.Metrics.Recall10,
				result.Memory/(1024*1024))
		}
	})
}

// TestPQ_CompressionRatios tests various compression levels
func TestPQ_CompressionRatios(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping compression ratio test in short mode")
	}

	d := 256
	// Different M values for different compression ratios
	configs := []RecallTestConfig{
		{
			Name:          "PQ_M64_nbits8_4x_compression",
			IndexType:     "IndexPQ",
			BuildIndex:    buildPQ(64, 8),
			NeedsTraining: true,
			N:             10000,
			D:             d,
			NQ:            100,
			MinRecall10:   0.80, // High M, less compression, better recall
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
		{
			Name:          "PQ_M32_nbits8_8x_compression",
			IndexType:     "IndexPQ",
			BuildIndex:    buildPQ(32, 8),
			NeedsTraining: true,
			N:             10000,
			D:             d,
			NQ:            100,
			MinRecall10:   0.75,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
		{
			Name:          "PQ_M16_nbits8_16x_compression",
			IndexType:     "IndexPQ",
			BuildIndex:    buildPQ(16, 8),
			NeedsTraining: true,
			N:             10000,
			D:             d,
			NQ:            100,
			MinRecall10:   0.70,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
		{
			Name:          "PQ_M8_nbits8_32x_compression",
			IndexType:     "IndexPQ",
			BuildIndex:    buildPQ(8, 8),
			NeedsTraining: true,
			N:             10000,
			D:             d,
			NQ:            100,
			MinRecall10:   0.65,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
	}

	results := RunParameterSweep(t, "PQ_Compression_Levels", configs)

	// Summary with compression ratios
	t.Run("Summary", func(t *testing.T) {
		t.Logf("\nCompression Ratio Summary:")
		t.Logf("%-35s %12s %10s %10s", "Configuration", "Compression", "Recall@10", "Memory(MB)")
		t.Logf("%s", strings.Repeat("-", 70))

		compressionRatios := []float64{4, 8, 16, 32}
		for i, result := range results {
			t.Logf("%-35s %12.0fx %10.4f %10d",
				result.Config.Name,
				compressionRatios[i],
				result.Metrics.Recall10,
				result.Memory/(1024*1024))
		}
	})
}

// TestPQ_HighDimensional tests PQ with high-dimensional vectors
func TestPQ_HighDimensional(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping high-dimensional test in short mode")
	}

	// Test with BERT and OpenAI embedding dimensions
	dimensions := []struct {
		d int
		m int
	}{
		{768, 48},  // BERT: 768-dim, M=48 (16x compression)
		{1536, 64}, // OpenAI: 1536-dim, M=64 (24x compression)
	}

	for _, dim := range dimensions {
		t.Run(fmt.Sprintf("Dim%d_M%d", dim.d, dim.m), func(t *testing.T) {
			config := RecallTestConfig{
				Name:          fmt.Sprintf("PQ_M%d_nbits8_Dim%d", dim.m, dim.d),
				IndexType:     "IndexPQ",
				BuildIndex:    buildPQ(dim.m, 8),
				NeedsTraining: true,
				N:             5000,
				D:             dim.d,
				NQ:            50,
				MinRecall10:   0.65,
				K:             10,
				Metric:        faiss.MetricL2,
				Distribution:  datasets.Normalized,
			}

			RunRecallTest(t, config)
		})
	}
}

// TestPQ_TrainingSize tests impact of training set size on codebook quality
func TestPQ_TrainingSize(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping training size test in short mode")
	}

	n := 20000
	trainSizes := []int{1000, 5000, 10000, 20000}
	configs := make([]RecallTestConfig, 0, len(trainSizes))

	for _, trainSize := range trainSizes {
		config := RecallTestConfig{
			Name:          fmt.Sprintf("PQ_M8_train%d", trainSize),
			IndexType:     "IndexPQ",
			BuildIndex:    buildPQ(8, 8),
			NeedsTraining: true,
			TrainSize:     trainSize,
			N:             n,
			D:             128,
			NQ:            100,
			MinRecall10:   0.60,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		}
		configs = append(configs, config)
	}

	results := RunParameterSweep(t, "PQ_Training_Size", configs)

	// Analysis
	t.Run("Validation", func(t *testing.T) {
		if len(results) < 2 {
			t.Skip("Not enough results for validation")
		}

		t.Logf("\nTraining Set Size Impact on PQ Codebook Quality:")
		for i, result := range results {
			t.Logf("Train=%d: Recall@10=%.4f",
				trainSizes[i],
				result.Metrics.Recall10)
		}

		// More training data should improve codebook and recall
		if results[len(results)-1].Metrics.Recall10 < results[0].Metrics.Recall10 {
			t.Logf("Note: Larger training set did not improve recall")
		}
	})
}

// TestPQ_Clustered tests PQ with clustered data
func TestPQ_Clustered(t *testing.T) {
	config := RecallTestConfig{
		Name:          "PQ_M8_nbits8_Clustered",
		IndexType:     "IndexPQ",
		BuildIndex:    buildPQ(8, 8),
		NeedsTraining: true,
		N:             10000,
		D:             128,
		NQ:            100,
		MinRecall10:   0.70,
		K:             10,
		Metric:        faiss.MetricL2,
		Distribution:  datasets.GaussianClustered,
	}

	result := RunRecallTest(t, config)

	if result.Metrics.Recall10 > 0.75 {
		t.Logf("✓ PQ performs reasonably well on clustered data: Recall@10=%.4f", result.Metrics.Recall10)
	}
}

// TestPQ_MemoryEfficiency tests PQ memory usage
func TestPQ_MemoryEfficiency(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory test in short mode")
	}

	n := 100000
	d := 128

	// Compare PQ memory vs Flat index
	configs := []RecallTestConfig{
		{
			Name:      "Flat_L2_baseline",
			IndexType: "IndexFlatL2",
			BuildIndex: func(d int, metric faiss.MetricType) (faiss.Index, error) {
				return faiss.NewIndexFlatL2(d)
			},
			N:            n,
			D:            d,
			NQ:           100,
			MinRecall10:  1.0,
			K:            10,
			Metric:       faiss.MetricL2,
			Distribution: datasets.UniformRandom,
		},
		{
			Name:          "PQ_M16_nbits8",
			IndexType:     "IndexPQ",
			BuildIndex:    buildPQ(16, 8),
			NeedsTraining: true,
			N:             n,
			D:             d,
			NQ:            100,
			MinRecall10:   0.70,
			K:             10,
			Metric:        faiss.MetricL2,
			Distribution:  datasets.UniformRandom,
		},
	}

	results := RunParameterSweep(t, "PQ_Memory_Efficiency", configs)

	// Compare memory usage
	t.Run("Validation", func(t *testing.T) {
		if len(results) < 2 {
			t.Skip("Not enough results for validation")
		}

		flatMemory := results[0].Memory
		pqMemory := results[1].Memory

		reduction := float64(flatMemory) / float64(pqMemory)

		t.Logf("\nMemory Efficiency Comparison:")
		t.Logf("Flat index:  %d MB (Recall@10=%.4f)", flatMemory/(1024*1024), results[0].Metrics.Recall10)
		t.Logf("PQ index:    %d MB (Recall@10=%.4f)", pqMemory/(1024*1024), results[1].Metrics.Recall10)
		t.Logf("Memory reduction: %.1fx", reduction)

		// PQ should use significantly less memory
		if reduction < 2 {
			t.Logf("Warning: PQ did not achieve expected memory reduction (got %.1fx, expected >2x)", reduction)
		} else {
			t.Logf("✓ PQ achieved %.1fx memory reduction with %.4f recall", reduction, results[1].Metrics.Recall10)
		}
	})
}

// Helper functions to build PQ indexes

func buildPQ(m int, nbits int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		return faiss.NewIndexPQ(d, m, nbits, metric)
	}
}
