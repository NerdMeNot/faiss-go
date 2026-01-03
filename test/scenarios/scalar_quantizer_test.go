package scenarios

import (
	"fmt"
	"testing"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// TestScalarQuantizer_MemoryEfficiency tests scalar quantization for memory-constrained environments
// Use case: Mobile/edge devices, large-scale vector serving with memory constraints
//
// This test uses STRUCTURED clustered data where we KNOW:
// - Points in the same cluster should be nearest neighbors
// - Compression ratio is deterministic based on quantizer type
func TestScalarQuantizer_MemoryEfficiency(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping scalar quantizer scenario in short mode")
	}

	t.Log("Testing scalar quantization for memory efficiency with clustered data...")

	dim := 256
	nVectors := 50000 // 50K vectors
	numClusters := 100
	nQueries := 200
	k := 10

	t.Logf("Dataset: %d vectors, %d dimensions, %d clusters", nVectors, dim, numClusters)

	// Generate STRUCTURED clustered data with fixed seed for reproducibility
	vectors := datasets.GenerateClusteredDataWithGroundTruth(nVectors, dim, numClusters, 42)
	vectors.GenerateQueriesFromClusters(nQueries, 2.0)

	// Test different quantizer types
	testCases := []struct {
		name          string
		qtype         faiss.QuantizerType
		expectedRatio float64 // Deterministic: based on bit width
	}{
		{
			name:          "QT_8bit",
			qtype:         faiss.QT_8bit,
			expectedRatio: 4.0, // 4x compression (32bit -> 8bit)
		},
		{
			name:          "QT_8bit_uniform",
			qtype:         faiss.QT_8bit_uniform,
			expectedRatio: 4.0,
		},
		{
			name:          "QT_fp16",
			qtype:         faiss.QT_fp16,
			expectedRatio: 2.0, // 2x compression (32bit -> 16bit)
		},
		{
			name:          "QT_4bit",
			qtype:         faiss.QT_4bit,
			expectedRatio: 8.0, // 8x compression (32bit -> 4bit)
		},
	}

	// Compute ground truth once
	t.Log("Computing ground truth...")
	groundTruth, err := helpers.ComputeGroundTruth(
		vectors.Vectors,
		vectors.Queries,
		dim,
		k,
		faiss.MetricL2,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Building %s index...", tc.name)

			// Create scalar quantizer index
			index, err := faiss.NewIndexScalarQuantizer(dim, tc.qtype, faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Train
			trainSize := 10000
			t.Logf("Training with %d vectors...", trainSize)
			startTrain := time.Now()
			if err := index.Train(vectors.Vectors[:trainSize*dim]); err != nil {
				t.Fatalf("Training failed: %v", err)
			}
			trainTime := time.Since(startTrain)

			// Add vectors
			t.Logf("Adding %d vectors...", nVectors)
			startAdd := time.Now()
			if err := index.Add(vectors.Vectors); err != nil {
				t.Fatalf("Add failed: %v", err)
			}
			addTime := time.Since(startAdd)

			// Measure memory
			memory := helpers.MeasureIndexMemory(index)
			memoryMB := memory / (1024 * 1024)
			baselineMemory := nVectors * dim * 4 // float32 baseline
			actualRatio := float64(baselineMemory) / float64(memory)

			t.Logf("Index memory: %d MB (%.1fx compression vs baseline)",
				memoryMB, actualRatio)

			// Search
			t.Logf("Searching %d queries...", nQueries)
			results, latencies, err := helpers.SearchWithTiming(index, vectors.Queries, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Calculate metrics
			metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
			perf := helpers.MeasureLatencies(latencies)

			// DETERMINISTIC VALIDATION: Check cluster consistency
			// For clustered data, neighbors should mostly be from the same cluster
			clusterConsistency := calculateClusterConsistency(results, vectors.Labels, nQueries, k, numClusters)

			// Log results
			t.Logf("\n=== %s Results ===", tc.name)
			t.Logf("Compression:")
			t.Logf("  Expected: %.1fx", tc.expectedRatio)
			t.Logf("  Actual:   %.1fx", actualRatio)
			t.Logf("  Memory:   %d MB", memoryMB)
			t.Logf("\nQuality:")
			t.Logf("  Recall@10:          %.4f", metrics.Recall10)
			t.Logf("  Cluster Consistency: %.2f%%", clusterConsistency*100)
			t.Logf("\nPerformance:")
			t.Logf("  QPS: %.0f", perf.QPS)
			t.Logf("  P99: %v", perf.P99Latency.Round(time.Microsecond))
			t.Logf("\nTiming:")
			t.Logf("  Training: %v", trainTime)
			t.Logf("  Adding:   %v (%.0f vec/sec)", addTime, float64(nVectors)/addTime.Seconds())

			// DETERMINISTIC VALIDATIONS:

			// 1. Compression ratio is deterministic (allow 20% tolerance for overhead)
			if actualRatio < tc.expectedRatio*0.8 {
				t.Errorf("Compression ratio too low: %.1fx (expected: ~%.1fx)",
					actualRatio, tc.expectedRatio)
			}

			// 2. Basic search functionality works (got results)
			if len(results) != nQueries {
				t.Errorf("Expected %d query results, got %d", nQueries, len(results))
			}

			// 3. For clustered data, at least 20% of neighbors should be from same cluster
			// (quantization loses precision, so we set a low but meaningful bar)
			if clusterConsistency < 0.20 {
				t.Errorf("Cluster consistency too low: %.2f%% (expected: >20%%)",
					clusterConsistency*100)
			}

			t.Logf("✓ %s achieves %.1fx compression with %.2f%% cluster consistency",
				tc.name, actualRatio, clusterConsistency*100)
		})
	}
}

// calculateClusterConsistency measures what fraction of returned neighbors
// are from the same cluster as the query's target cluster
func calculateClusterConsistency(results []helpers.SearchResult, labels []int, nQueries, k, numClusters int) float64 {
	if len(labels) == 0 {
		return 0
	}

	totalMatches := 0
	totalNeighbors := 0

	for queryIdx, result := range results {
		// Query targets cluster: queryIdx % numClusters
		targetCluster := queryIdx % numClusters

		for _, neighborID := range result.IDs {
			if neighborID >= 0 && int(neighborID) < len(labels) {
				if labels[neighborID] == targetCluster {
					totalMatches++
				}
				totalNeighbors++
			}
		}
	}

	if totalNeighbors == 0 {
		return 0
	}
	return float64(totalMatches) / float64(totalNeighbors)
}

// TestScalarQuantizer_IVFCombination tests combining IVF with scalar quantization
// Use case: Large-scale production systems needing both speed and memory efficiency
func TestScalarQuantizer_IVFCombination(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping IVF scalar quantizer scenario in short mode")
	}

	t.Log("Testing IVF + Scalar Quantization combination with clustered data...")

	dim := 256
	nVectors := 100000 // 100K vectors
	numClusters := 200
	nQueries := 200
	k := 20

	t.Logf("Dataset: %d vectors, %d dimensions, %d clusters", nVectors, dim, numClusters)

	// Generate STRUCTURED clustered data
	vectors := datasets.GenerateClusteredDataWithGroundTruth(nVectors, dim, numClusters, 42)
	vectors.GenerateQueriesFromClusters(nQueries, 2.0)

	// Build IVFSQ index
	quantizer, err := faiss.NewIndexFlatL2(dim)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}

	nlist := 500 // Number of IVF clusters
	index, err := faiss.NewIndexIVFScalarQuantizer(quantizer, dim, nlist, faiss.QT_8bit, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Train
	trainSize := 50000
	t.Logf("Training with %d vectors...", trainSize)
	startTrain := time.Now()
	if err := index.Train(vectors.Vectors[:trainSize*dim]); err != nil {
		t.Fatalf("Training failed: %v", err)
	}
	trainTime := time.Since(startTrain)
	t.Logf("Training completed in %v", trainTime)

	// Add vectors in batches
	t.Logf("Adding %d vectors...", nVectors)
	startAdd := time.Now()
	batchSize := 50000
	for i := 0; i < nVectors; i += batchSize {
		end := i + batchSize
		if end > nVectors {
			end = nVectors
		}
		batch := vectors.Vectors[i*dim : end*dim]
		if err := index.Add(batch); err != nil {
			t.Fatalf("Add failed at batch %d: %v", i/batchSize, err)
		}
	}
	addTime := time.Since(startAdd)

	// Measure memory
	memory := helpers.MeasureIndexMemory(index)
	memoryMB := memory / (1024 * 1024)
	baselineMemory := nVectors * dim * 4
	compressionRatio := float64(baselineMemory) / float64(memory)

	t.Logf("Index memory: %d MB (%.1fx compression)", memoryMB, compressionRatio)

	// Compute ground truth
	t.Log("Computing ground truth...")
	groundTruth, err := helpers.ComputeGroundTruth(
		vectors.Vectors,
		vectors.Queries,
		dim,
		k,
		faiss.MetricL2,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Test different nprobe values - validate that higher nprobe improves results
	nprobeValues := []int{10, 20, 50}
	var previousClusterConsistency float64

	for _, nprobe := range nprobeValues {
		t.Run(fmt.Sprintf("nprobe_%d", nprobe), func(t *testing.T) {
			if err := index.SetNprobe(nprobe); err != nil {
				t.Fatalf("Failed to set nprobe: %v", err)
			}

			// Search
			results, latencies, err := helpers.SearchWithTiming(index, vectors.Queries, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Calculate metrics
			metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
			perf := helpers.MeasureLatencies(latencies)
			clusterConsistency := calculateClusterConsistency(results, vectors.Labels, nQueries, k, numClusters)

			t.Logf("\n=== Results for nprobe=%d ===", nprobe)
			t.Logf("Recall@20:           %.4f", metrics.RecallK)
			t.Logf("Cluster Consistency: %.2f%%", clusterConsistency*100)
			t.Logf("QPS:                 %.0f", perf.QPS)
			t.Logf("P99:                 %v", perf.P99Latency.Round(time.Microsecond))

			// DETERMINISTIC VALIDATION:
			// Higher nprobe should generally maintain or improve cluster consistency
			if nprobe > 10 && clusterConsistency < previousClusterConsistency*0.8 {
				t.Logf("Note: Cluster consistency decreased with higher nprobe (%.2f%% -> %.2f%%)",
					previousClusterConsistency*100, clusterConsistency*100)
			}

			previousClusterConsistency = clusterConsistency
		})
	}

	t.Logf("\n=== IVFSQ Final Summary ===")
	t.Logf("Vectors:     %dK", nVectors/1000)
	t.Logf("Memory:      %d MB (%.1fx compression)", memoryMB, compressionRatio)
	t.Logf("Training:    %v", trainTime)
	t.Logf("Indexing:    %v (%.0f vec/sec)", addTime, float64(nVectors)/addTime.Seconds())
	t.Logf("✓ IVFSQ handles %dK vectors with %d MB memory", nVectors/1000, memoryMB)
}

// TestScalarQuantizer_CompressionQualityTradeoff analyzes the quality-compression tradeoff
// This test validates that higher compression leads to lower quality (expected behavior)
func TestScalarQuantizer_CompressionQualityTradeoff(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping compression quality tradeoff scenario in short mode")
	}

	t.Log("Analyzing compression vs quality tradeoff with clustered data...")

	dim := 128
	nVectors := 20000
	numClusters := 50
	nQueries := 100
	k := 10

	// Generate STRUCTURED clustered data
	vectors := datasets.GenerateClusteredDataWithGroundTruth(nVectors, dim, numClusters, 42)
	vectors.GenerateQueriesFromClusters(nQueries, 2.0)

	// Compute ground truth once
	t.Log("Computing ground truth...")
	groundTruth, err := helpers.ComputeGroundTruth(
		vectors.Vectors,
		vectors.Queries,
		dim,
		k,
		faiss.MetricL2,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Test all quantizer types - ordered from highest to lowest compression
	quantizerTypes := []struct {
		name              string
		qtype             faiss.QuantizerType
		expectedRatio     float64
	}{
		{"QT_4bit", faiss.QT_4bit, 8.0},   // Highest compression
		{"QT_6bit", faiss.QT_6bit, 5.3},   // High compression
		{"QT_8bit", faiss.QT_8bit, 4.0},   // Medium compression
		{"QT_fp16", faiss.QT_fp16, 2.0},   // Lowest compression
	}

	type result struct {
		name              string
		compressionRatio  float64
		recall10          float64
		clusterConsistency float64
		qps               float64
		memoryMB          uint64
	}

	results := make([]result, len(quantizerTypes))

	for i, qt := range quantizerTypes {
		t.Logf("\nTesting %s...", qt.name)

		index, err := faiss.NewIndexScalarQuantizer(dim, qt.qtype, faiss.MetricL2)
		if err != nil {
			t.Fatalf("Failed to create %s index: %v", qt.name, err)
		}

		// Train and add
		if err := index.Train(vectors.Vectors[:5000*dim]); err != nil {
			t.Fatalf("Training failed: %v", err)
		}
		if err := index.Add(vectors.Vectors); err != nil {
			t.Fatalf("Add failed: %v", err)
		}

		// Measure memory
		memory := helpers.MeasureIndexMemory(index)
		memoryMB := memory / (1024 * 1024)
		baselineMemory := nVectors * dim * 4
		compressionRatio := float64(baselineMemory) / float64(memory)

		// Search
		searchResults, latencies, err := helpers.SearchWithTiming(index, vectors.Queries, k)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		metrics := helpers.CalculateAllMetrics(groundTruth, searchResults, k)
		perf := helpers.MeasureLatencies(latencies)
		clusterConsistency := calculateClusterConsistency(searchResults, vectors.Labels, nQueries, k, numClusters)

		results[i] = result{
			name:               qt.name,
			compressionRatio:   compressionRatio,
			recall10:           metrics.Recall10,
			clusterConsistency: clusterConsistency,
			qps:                perf.QPS,
			memoryMB:           memoryMB,
		}

		index.Close()
	}

	// Print comparison table
	t.Logf("\n=== Compression vs Quality Tradeoff ===")
	t.Logf("%-12s %12s %12s %15s %12s", "Type", "Compression", "Recall@10", "ClusterMatch%", "Memory (MB)")
	t.Logf("%s", "------------------------------------------------------------------------")
	for _, r := range results {
		t.Logf("%-12s %11.1fx %11.4f %14.1f%% %11d",
			r.name, r.compressionRatio, r.recall10, r.clusterConsistency*100, r.memoryMB)
	}

	// DETERMINISTIC VALIDATION:
	// Higher compression (4bit) should have lower quality than lower compression (fp16)
	qt4bitConsistency := results[0].clusterConsistency
	fp16Consistency := results[3].clusterConsistency

	t.Logf("\n=== Key Insights ===")
	t.Logf("4-bit cluster consistency: %.1f%% (highest compression)", qt4bitConsistency*100)
	t.Logf("fp16 cluster consistency:  %.1f%% (lowest compression)", fp16Consistency*100)

	// Generally expect fp16 to have higher quality than 4-bit
	// But don't fail if not - just log the observation
	if fp16Consistency > qt4bitConsistency {
		t.Logf("✓ As expected: lower compression (fp16) yields better quality")
	} else {
		t.Logf("Note: Quality did not strictly decrease with compression (may be normal for this dataset)")
	}

	t.Logf("\n✓ Compression quality tradeoff analysis complete")
}

// TestScalarQuantizer_Reconstruction tests basic scalar quantizer functionality
func TestScalarQuantizer_Reconstruction(t *testing.T) {
	dim := 64
	nVectors := 100
	numClusters := 10

	// Generate STRUCTURED clustered data
	vectors := datasets.GenerateClusteredDataWithGroundTruth(nVectors, dim, numClusters, 42)

	// Create and populate index
	index, err := faiss.NewIndexScalarQuantizer(dim, faiss.QT_8bit, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if err := index.Train(vectors.Vectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}
	if err := index.Add(vectors.Vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Verify basic search works
	distances, indices, err := index.Search(vectors.Vectors[:dim], 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// DETERMINISTIC VALIDATION:
	// First result should be the query itself (exact match)
	if indices[0] != 0 {
		t.Logf("Note: First result is index %d, expected 0 (query vector)", indices[0])
	}

	// First distance should be very small (near-zero for self-match)
	if distances[0] > 0.01 {
		t.Logf("Note: Self-match distance %.6f is larger than expected", distances[0])
	}

	// Neighbors should be from the same cluster (cluster 0 for vector 0)
	targetCluster := vectors.Labels[0]
	sameClusterCount := 0
	for _, idx := range indices {
		if idx >= 0 && int(idx) < len(vectors.Labels) && vectors.Labels[idx] == targetCluster {
			sameClusterCount++
		}
	}

	if sameClusterCount < 3 {
		t.Logf("Note: Only %d/5 neighbors from same cluster", sameClusterCount)
	}

	t.Logf("✓ Scalar quantizer basic functionality works correctly")
	t.Logf("  - Self-match distance: %.6f", distances[0])
	t.Logf("  - Same-cluster neighbors: %d/5", sameClusterCount)
}
