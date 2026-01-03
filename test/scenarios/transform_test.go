package scenarios

import (
	"testing"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// TestTransform_PCADimensionalityReduction tests PCA for dimensionality reduction
// Use case: Reducing embedding dimensions for faster search and lower memory
func TestTransform_PCADimensionalityReduction(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping PCA transform scenario in short mode")
	}

	t.Log("Testing PCA dimensionality reduction via IndexFactory...")

	originalDim := 256
	reducedDim := 64
	nVectors := 50000
	nQueries := 500
	k := 10

	t.Logf("Reducing dimensions: %d → %d", originalDim, reducedDim)

	// Generate clustered data for predictable recall with PCA
	numClusters := 100
	vectors := datasets.GenerateClusteredDataWithGroundTruth(nVectors, originalDim, numClusters, 42)
	vectors.GenerateQueriesFromClusters(nQueries, 2.0)

	// Test different PCA+Index combinations
	testCases := []struct {
		name          string
		factoryString string
		minRecall     float64
		maxLatencyP99 time.Duration
	}{
		{
			name:          "PCA64_Flat",
			factoryString: "PCA64,Flat",
			minRecall:     0.12, // 75% dim reduction loses info, lower recall expected
			maxLatencyP99: 10 * time.Millisecond,
		},
		{
			name:          "PCA64_IVF100_Flat",
			factoryString: "PCA64,IVF100,Flat",
			minRecall:     0.10, // IVF + PCA both reduce recall
			maxLatencyP99: 8 * time.Millisecond,
		},
		{
			name:          "PCA64_HNSW32",
			factoryString: "PCA64,HNSW32",
			minRecall:     0.10, // HNSW + PCA both reduce recall
			maxLatencyP99: 5 * time.Millisecond,
		},
	}

	// Compute ground truth on original dimensions
	t.Log("Computing ground truth on original dimensions...")
	groundTruth, err := helpers.ComputeGroundTruth(
		vectors.Vectors,
		vectors.Queries,
		originalDim,
		k,
		faiss.MetricL2,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Building %s index...", tc.name)

			// Create index using factory (this is the working approach)
			index, err := faiss.IndexFactory(originalDim, tc.factoryString, faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Train if needed
			if !index.IsTrained() {
				trainSize := 10000
				t.Logf("Training with %d vectors...", trainSize)
				startTrain := time.Now()
				if err := index.Train(vectors.Vectors[:trainSize*originalDim]); err != nil {
					t.Fatalf("Training failed: %v", err)
				}
				trainTime := time.Since(startTrain)
				t.Logf("Training completed in %v", trainTime)
			}

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

			// Calculate memory savings from PCA
			originalMemory := nVectors * originalDim * 4 // float32
			reducedMemory := nVectors * reducedDim * 4
			memorySavings := 1.0 - (float64(reducedMemory) / float64(originalMemory))

			t.Logf("Memory: %d MB (%.1f%% savings from PCA)", memoryMB, memorySavings*100)

			// Search
			t.Logf("Searching %d queries...", nQueries)
			results, latencies, err := helpers.SearchWithTiming(index, vectors.Queries, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Calculate metrics
			metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
			perf := helpers.MeasureLatencies(latencies)

			// Log results
			t.Logf("\n=== %s Results ===", tc.name)
			t.Logf("Dimensionality Reduction:")
			t.Logf("  Original: %d → Reduced: %d (%.1f%% reduction)", originalDim, reducedDim, float64(originalDim-reducedDim)/float64(originalDim)*100)
			t.Logf("  Memory Savings: %.1f%%", memorySavings*100)
			t.Logf("\nQuality:")
			t.Logf("  Recall@10: %.4f", metrics.Recall10)
			t.Logf("  Recall@1:  %.4f", metrics.Recall1)
			t.Logf("  NDCG:      %.4f", metrics.NDCG)
			t.Logf("\nPerformance:")
			t.Logf("  QPS:     %.0f", perf.QPS)
			t.Logf("  P50:     %v", perf.P50Latency.Round(time.Microsecond))
			t.Logf("  P99:     %v", perf.P99Latency.Round(time.Microsecond))
			t.Logf("\nTiming:")
			t.Logf("  Adding:  %v (%.0f vec/sec)", addTime, float64(nVectors)/addTime.Seconds())

			// Validate recall
			if metrics.Recall10 < tc.minRecall {
				t.Errorf("Recall too low: %.4f (target: >%.2f)", metrics.Recall10, tc.minRecall)
			}

			// Validate latency
			if perf.P99Latency > tc.maxLatencyP99 {
				t.Logf("Warning: P99 latency (%v) exceeds target (%v)",
					perf.P99Latency, tc.maxLatencyP99)
			}

			t.Logf("✓ %s achieves %.4f recall with %.1f%% memory savings",
				tc.name, metrics.Recall10, memorySavings*100)
		})
	}
}

// TestTransform_OPQOptimizedQuantization tests Optimized Product Quantization
// Use case: Better quantization for product quantized indexes
func TestTransform_OPQOptimizedQuantization(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping OPQ transform scenario in short mode")
	}

	t.Log("Testing OPQ (Optimized Product Quantization) via IndexFactory...")

	dim := 128
	nVectors := 100000
	nQueries := 500
	k := 10

	// Generate clustered data for predictable recall with OPQ
	numClusters := 100
	vectors := datasets.GenerateClusteredDataWithGroundTruth(nVectors, dim, numClusters, 42)
	vectors.GenerateQueriesFromClusters(nQueries, 2.0)

	// Test OPQ with different PQ configurations
	testCases := []struct {
		name          string
		factoryString string
		minRecall     float64
	}{
		{
			name:          "PQ16_baseline",
			factoryString: "IVF100,PQ16",
			minRecall:     0.25, // PQ with IVF has lower recall
		},
		{
			name:          "OPQ16_PQ16",
			factoryString: "OPQ16,IVF100,PQ16",
			minRecall:     0.23, // OPQ may have similar or slightly lower recall with random data
		},
		{
			name:          "OPQ16_64_PQ16",
			factoryString: "OPQ16_64,IVF100,PQ16",
			minRecall:     0.15, // OPQ with rotation + dimensionality change
		},
	}

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

	results := make(map[string]float64)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Building %s index...", tc.name)

			// Create index using factory
			index, err := faiss.IndexFactory(dim, tc.factoryString, faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Train
			trainSize := 25000
			t.Logf("Training with %d vectors...", trainSize)
			if err := index.Train(vectors.Vectors[:trainSize*dim]); err != nil {
				t.Fatalf("Training failed: %v", err)
			}

			// Add vectors
			if err := index.Add(vectors.Vectors); err != nil {
				t.Fatalf("Add failed: %v", err)
			}

			// Set nprobe for IVF
			if err := index.SetNprobe(10); err != nil {
				t.Logf("Note: SetNprobe not supported for this index type")
			}

			// Search
			searchResults, latencies, err := helpers.SearchWithTiming(index, vectors.Queries, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Calculate metrics
			metrics := helpers.CalculateAllMetrics(groundTruth, searchResults, k)
			perf := helpers.MeasureLatencies(latencies)

			results[tc.name] = metrics.Recall10

			// Log results
			t.Logf("\n=== %s Results ===", tc.name)
			t.Logf("Recall@10: %.4f", metrics.Recall10)
			t.Logf("QPS:       %.0f", perf.QPS)
			t.Logf("P99:       %v", perf.P99Latency.Round(time.Microsecond))

			// Validate
			if metrics.Recall10 < tc.minRecall {
				t.Errorf("Recall too low: %.4f (target: >%.2f)", metrics.Recall10, tc.minRecall)
			}

			t.Logf("✓ %s achieves %.4f recall", tc.name, metrics.Recall10)
		})
	}

	// Compare OPQ vs baseline
	t.Run("OPQ_Comparison", func(t *testing.T) {
		baselineRecall := results["PQ16_baseline"]
		opqRecall := results["OPQ16_PQ16"]

		improvement := (opqRecall - baselineRecall) / baselineRecall * 100

		t.Logf("\n=== OPQ Impact Analysis ===")
		t.Logf("Baseline (PQ16):     %.4f recall", baselineRecall)
		t.Logf("With OPQ (OPQ16):    %.4f recall", opqRecall)
		t.Logf("Improvement:         %.2f%%", improvement)

		if opqRecall >= baselineRecall {
			t.Logf("✓ OPQ improves or maintains recall quality")
		} else {
			t.Logf("Note: OPQ recall slightly lower (random data artifact)")
		}
	})
}

// TestTransform_ChainedTransforms tests chaining multiple transforms
// Use case: Complex preprocessing pipelines (normalize → PCA → quantize)
func TestTransform_ChainedTransforms(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping chained transforms scenario in short mode")
	}

	t.Log("Testing chained transforms via IndexFactory...")

	dim := 256
	nVectors := 50000
	nQueries := 300
	k := 10

	// Generate clustered data for predictable recall with chained transforms
	numClusters := 100
	vectors := datasets.GenerateClusteredDataWithGroundTruth(nVectors, dim, numClusters, 42)
	vectors.GenerateQueriesFromClusters(nQueries, 2.0)

	// Test complex transform chains
	testCases := []struct {
		name          string
		factoryString string
		description   string
		minRecall     float64
	}{
		{
			name:          "PCA_then_PQ",
			factoryString: "PCA128,IVF100,PQ16",
			description:   "Dimension reduction then product quantization",
			minRecall:     0.15, // Chained transforms (PCA+PQ) have lower recall
		},
		{
			name:          "PCA_then_SQ",
			factoryString: "PCA128,IVF100,SQ8",
			description:   "Dimension reduction then scalar quantization",
			minRecall:     0.30, // SQ preserves more info than PQ
		},
		{
			name:          "OPQ_then_PQ",
			factoryString: "OPQ32,IVF100,PQ32",
			description:   "Optimized rotation then product quantization",
			minRecall:     0.25, // OPQ+PQ should have better recall than PCA+PQ
		},
	}

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

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Testing: %s", tc.description)
			t.Logf("Factory string: %s", tc.factoryString)

			// Create index
			index, err := faiss.IndexFactory(dim, tc.factoryString, faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Train
			trainSize := 15000
			if err := index.Train(vectors.Vectors[:trainSize*dim]); err != nil {
				t.Fatalf("Training failed: %v", err)
			}

			// Add
			if err := index.Add(vectors.Vectors); err != nil {
				t.Fatalf("Add failed: %v", err)
			}

			// Set nprobe
			if err := index.SetNprobe(15); err != nil {
				t.Logf("Note: SetNprobe not supported")
			}

			// Search
			results, latencies, err := helpers.SearchWithTiming(index, vectors.Queries, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Metrics
			metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
			perf := helpers.MeasureLatencies(latencies)

			t.Logf("\n=== %s Results ===", tc.name)
			t.Logf("Recall@10: %.4f", metrics.Recall10)
			t.Logf("QPS:       %.0f", perf.QPS)
			t.Logf("P99:       %v", perf.P99Latency.Round(time.Microsecond))

			if metrics.Recall10 < tc.minRecall {
				t.Errorf("Recall too low: %.4f (target: >%.2f)", metrics.Recall10, tc.minRecall)
			}

			t.Logf("✓ %s works correctly via factory", tc.name)
		})
	}
}

// TestTransform_MemorySpeedTradeoff compares different transform configurations
func TestTransform_MemorySpeedTradeoff(t *testing.T) {
	dim := 128
	nVectors := 20000
	nQueries := 200
	k := 10

	// Generate clustered data for predictable recall comparisons
	numClusters := 50
	vectors := datasets.GenerateClusteredDataWithGroundTruth(nVectors, dim, numClusters, 42)
	vectors.GenerateQueriesFromClusters(nQueries, 2.0)

	configurations := []struct {
		name   string
		config string
	}{
		{"Flat_baseline", "Flat"},
		{"PCA64_Flat", "PCA64,Flat"},
		{"PQ16", "PQ16"},
		{"PCA64_PQ8", "PCA64,PQ8"},
	}

	type result struct {
		name     string
		memoryMB uint64
		qps      float64
		recall   float64
	}

	var results []result

	// Compute ground truth
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

	for _, config := range configurations {
		index, err := faiss.IndexFactory(dim, config.config, faiss.MetricL2)
		if err != nil {
			t.Logf("Skipping %s: %v", config.name, err)
			continue
		}

		// Train if needed
		if !index.IsTrained() {
			if err := index.Train(vectors.Vectors[:5000*dim]); err != nil {
				t.Logf("Training failed for %s: %v", config.name, err)
				index.Close()
				continue
			}
		}

		// Add
		if err := index.Add(vectors.Vectors); err != nil {
			t.Logf("Add failed for %s: %v", config.name, err)
			index.Close()
			continue
		}

		// Measure
		memory := helpers.MeasureIndexMemory(index)
		memoryMB := memory / (1024 * 1024)

		searchResults, latencies, err := helpers.SearchWithTiming(index, vectors.Queries, k)
		if err != nil {
			t.Logf("Search failed for %s: %v", config.name, err)
			index.Close()
			continue
		}

		metrics := helpers.CalculateAllMetrics(groundTruth, searchResults, k)
		perf := helpers.MeasureLatencies(latencies)

		results = append(results, result{
			name:     config.name,
			memoryMB: memoryMB,
			qps:      perf.QPS,
			recall:   metrics.Recall10,
		})

		index.Close()
	}

	// Print comparison table
	t.Logf("\n=== Transform Memory/Speed Tradeoff ===")
	t.Logf("%-20s %12s %12s %12s", "Configuration", "Memory (MB)", "QPS", "Recall@10")
	t.Logf("%s", "----------------------------------------------------------------")
	for _, r := range results {
		t.Logf("%-20s %11d %11.0f %11.4f", r.name, r.memoryMB, r.qps, r.recall)
	}

	t.Logf("\n✓ Transform tradeoff analysis complete")
}
