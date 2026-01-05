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
		MinRecall10:   0.30,
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
		MinRecall10:   0.30,
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
		MinRecall10:   0.30,
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

		// FAISS needs ~39x nlist training vectors (e.g., nlist=400 needs 15.6K vectors)
		n := nlist * 40
		if n < 10000 {
			n = 10000
		}

		config := RecallTestConfig{
			Name:          fmt.Sprintf("IVF%d_nprobe%d", nlist, nprobe),
			IndexType:     "IndexIVFFlat",
			BuildIndex:    buildIVF(nlist, nprobe),
			NeedsTraining: true,
			N:             n, // Dynamic based on nlist
			D:             128,
			NQ:            100,
			MinRecall10:   0.40, // Lower target for sweep
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
		// For parameter sweeps, we're demonstrating tradeoffs, not validating quality
		// nprobe=1 means searching only 1/100 clusters, so recall ~1% is expected
		// Set thresholds to 0 to skip validation for sweep tests
		config := RecallTestConfig{
			Name:          fmt.Sprintf("IVF100_nprobe%d", nprobe),
			IndexType:     "IndexIVFFlat",
			BuildIndex:    buildIVF(nlist, nprobe),
			NeedsTraining: true,
			N:             10000,
			D:             128,
			NQ:            100,
			// No thresholds - this is a parameter sweep to demonstrate tradeoffs
			Thresholds: RecallThresholds{
				CI:    RecallTargets{MinRecall10: 0.0}, // 0 = skip check
				Local: RecallTargets{MinRecall10: 0.0}, // 0 = skip check
			},
			K:            10,
			Metric:       faiss.MetricL2,
			Distribution: datasets.UniformRandom,
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

// TestIVF_OptimalConfiguration tests recommended nlist/nprobe combinations using STRUCTURED data
// Uses clustered data where we KNOW neighbors should be from the same cluster
func TestIVF_OptimalConfiguration(t *testing.T) {
	// Test different dataset sizes with structured clustered data
	testCases := []struct {
		name        string
		nVectors    int
		numClusters int
		nlist       int
		nprobe      int
	}{
		{"IVF_1K", 1000, 10, 10, 2},
		{"IVF_10K", 10000, 50, 100, 10},
	}

	// Add larger configs only when not in short mode
	if !testing.Short() {
		testCases = append(testCases, struct {
			name        string
			nVectors    int
			numClusters int
			nlist       int
			nprobe      int
		}{"IVF_100K", 100000, 100, 1000, 20})
	}

	dim := 128
	k := 10
	nQueries := 50

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Generate STRUCTURED clustered data
			data := datasets.GenerateClusteredDataWithGroundTruth(tc.nVectors, dim, tc.numClusters, 42)
			data.GenerateQueriesFromClusters(nQueries, 2.0)

			// Build IVF index using factory
			desc := fmt.Sprintf("IVF%d,Flat", tc.nlist)
			index, err := faiss.IndexFactory(dim, desc, faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Train
			if err := index.Train(data.Vectors); err != nil {
				t.Fatalf("Failed to train: %v", err)
			}

			// Set nprobe
			if ivfIndex, ok := index.(interface{ SetNprobe(int) error }); ok {
				if err := ivfIndex.SetNprobe(tc.nprobe); err != nil {
					t.Logf("Warning: Could not set nprobe: %v", err)
				}
			}

			// Add vectors
			if err := index.Add(data.Vectors); err != nil {
				t.Fatalf("Failed to add vectors: %v", err)
			}

			// Search
			_, resultIDs, err := index.Search(data.Queries, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Calculate cluster consistency (deterministic validation)
			clusterMatches := 0
			totalNeighbors := 0
			nq := len(data.Queries) / dim
			for queryIdx := 0; queryIdx < nq; queryIdx++ {
				targetCluster := queryIdx % tc.numClusters
				for i := 0; i < k; i++ {
					neighborID := resultIDs[queryIdx*k+i]
					if neighborID >= 0 && int(neighborID) < len(data.Labels) {
						if data.Labels[neighborID] == targetCluster {
							clusterMatches++
						}
						totalNeighbors++
					}
				}
			}

			clusterConsistency := float64(clusterMatches) / float64(totalNeighbors)
			t.Logf("%s: nlist=%d, nprobe=%d, cluster consistency = %.2f%% (%d/%d)",
				tc.name, tc.nlist, tc.nprobe, clusterConsistency*100, clusterMatches, totalNeighbors)

			// DETERMINISTIC VALIDATION: At least 30% of neighbors should be from same cluster
			if clusterConsistency < 0.30 {
				t.Errorf("Cluster consistency too low: %.2f%% (expected: >30%%)", clusterConsistency*100)
			}

			t.Logf("✓ %s passed with %.2f%% cluster consistency", tc.name, clusterConsistency*100)
		})
	}
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
		MinRecall10:   0.40, // Should be higher for clustered data
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
		MinRecall10:   0.30,
		K:             10,
		Metric:        faiss.MetricInnerProduct,
		Distribution:  datasets.Normalized, // Must normalize for IP
	}

	RunRecallTest(t, config)
}

// TestIVF_HighDimensional tests IVF with high-dimensional vectors using STRUCTURED data
// Uses clustered data where we KNOW neighbors should be from the same cluster
func TestIVF_HighDimensional(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping high-dimensional IVF test in short mode")
	}

	// Test with dimensions similar to modern embeddings
	testCases := []struct {
		dim         int
		nVectors    int
		numClusters int
		nlist       int
		nprobe      int
	}{
		{768, 2000, 20, 100, 10},   // GPT-like embedding dimension
		{1536, 1000, 10, 50, 5},    // Larger embedding dimension
	}

	k := 10
	nQueries := 50

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("Dim%d", tc.dim), func(t *testing.T) {
			// Generate STRUCTURED clustered data
			data := datasets.GenerateClusteredDataWithGroundTruth(tc.nVectors, tc.dim, tc.numClusters, 42)
			data.GenerateQueriesFromClusters(nQueries, 2.0)

			// Build IVF index using factory
			desc := fmt.Sprintf("IVF%d,Flat", tc.nlist)
			index, err := faiss.IndexFactory(tc.dim, desc, faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Train
			if err := index.Train(data.Vectors); err != nil {
				t.Fatalf("Failed to train: %v", err)
			}

			// Set nprobe
			if ivfIndex, ok := index.(interface{ SetNprobe(int) error }); ok {
				if err := ivfIndex.SetNprobe(tc.nprobe); err != nil {
					t.Logf("Warning: Could not set nprobe: %v", err)
				}
			}

			// Add vectors
			if err := index.Add(data.Vectors); err != nil {
				t.Fatalf("Failed to add vectors: %v", err)
			}

			// Search
			_, resultIDs, err := index.Search(data.Queries, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Calculate cluster consistency (deterministic validation)
			clusterMatches := 0
			totalNeighbors := 0
			nq := len(data.Queries) / tc.dim
			for queryIdx := 0; queryIdx < nq; queryIdx++ {
				targetCluster := queryIdx % tc.numClusters
				for i := 0; i < k; i++ {
					neighborID := resultIDs[queryIdx*k+i]
					if neighborID >= 0 && int(neighborID) < len(data.Labels) {
						if data.Labels[neighborID] == targetCluster {
							clusterMatches++
						}
						totalNeighbors++
					}
				}
			}

			clusterConsistency := float64(clusterMatches) / float64(totalNeighbors)
			t.Logf("Dim=%d: nlist=%d, nprobe=%d, cluster consistency = %.2f%% (%d/%d)",
				tc.dim, tc.nlist, tc.nprobe, clusterConsistency*100, clusterMatches, totalNeighbors)

			// DETERMINISTIC VALIDATION: At least 30% of neighbors should be from same cluster
			if clusterConsistency < 0.30 {
				t.Errorf("Cluster consistency too low: %.2f%% (expected: >30%%)", clusterConsistency*100)
			}

			t.Logf("✓ IVF high-dimensional test passed with %.2f%% cluster consistency", clusterConsistency*100)
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
			MinRecall10:   0.40,
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

// Helper functions to build IVF indexes using factory pattern
// This avoids the known bug with direct IVF constructors

func buildIVF(nlist int, nprobe int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		// Use factory to create IVF index
		desc := fmt.Sprintf("IVF%d,Flat", nlist)
		index, err := faiss.IndexFactory(d, desc, metric)
		if err != nil {
			return nil, err
		}

		// Set nprobe parameter
		if genericIdx, ok := index.(*faiss.GenericIndex); ok {
			if err := genericIdx.SetNprobe(nprobe); err != nil {
				return nil, fmt.Errorf("failed to set nprobe: %w", err)
			}
		}

		return index, nil
	}
}

func buildIVF_IP(nlist int, nprobe int) func(d int, metric faiss.MetricType) (faiss.Index, error) {
	return func(d int, metric faiss.MetricType) (faiss.Index, error) {
		// Use factory to create IVF index with InnerProduct metric
		desc := fmt.Sprintf("IVF%d,Flat", nlist)
		index, err := faiss.IndexFactory(d, desc, faiss.MetricInnerProduct)
		if err != nil {
			return nil, err
		}

		// Set nprobe parameter
		if genericIdx, ok := index.(*faiss.GenericIndex); ok {
			if err := genericIdx.SetNprobe(nprobe); err != nil {
				return nil, fmt.Errorf("failed to set nprobe: %w", err)
			}
		}

		return index, nil
	}
}
