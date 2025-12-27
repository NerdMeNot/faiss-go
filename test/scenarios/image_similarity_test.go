package scenarios

import (
	"fmt"
	"testing"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// TestImageSimilarity_VisualSearch simulates image search with deep learning features
// Use case: E-commerce visual search with 1M product images
func TestImageSimilarity_VisualSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping image similarity scenario in short mode")
	}

	// Simulate ResNet50 features (2048-dim, L2-normalized)
	t.Log("Simulating visual search with ResNet50-like features...")

	nImages := 1000000 // 1M product images
	dim := 2048        // ResNet50 feature dimension
	nQueries := 500    // 500 user searches
	k := 20            // Top-20 similar products

	t.Logf("Dataset: %d images, %d dimensions", nImages, dim)

	// Generate realistic image embeddings (normalized)
	imageEmbeddings := datasets.GenerateRealisticEmbeddings(nImages, dim)
	imageEmbeddings.GenerateQueries(nQueries, datasets.Normalized)

	// Test index types suitable for large-scale image search
	testCases := []struct {
		name          string
		buildIndex    func() faiss.Index
		minRecall     float64
		maxLatencyP99 time.Duration
	}{
		{
			name: "HNSW_M48_efSearch128",
			buildIndex: func() faiss.Index {
				index, err := faiss.NewIndexHNSWFlat(dim, 48, faiss.MetricInnerProduct)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
				index.HnswSetEfSearch(128)
				return index
			},
			minRecall:     0.95,
			maxLatencyP99: 15 * time.Millisecond,
		},
		{
			name: "IVFPQ_nlist4096_M64",
			buildIndex: func() faiss.Index {
				quantizer, err := faiss.NewIndexFlatIP(dim)
			if err != nil {
				t.Fatalf("Failed to create quantizer: %v", err)
			}
				index, err := faiss.NewIndexIVFPQ(quantizer, dim, 4096, 64, 8, faiss.MetricInnerProduct)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
				index.SetNprobe(32)
				return index
			},
			minRecall:     0.80,
			maxLatencyP99: 10 * time.Millisecond,
		},
	}

	// Compute ground truth (sample for speed)
	sampleQueries := 100
	if nQueries < sampleQueries {
		sampleQueries = nQueries
	}

	t.Logf("Computing ground truth for %d queries...", sampleQueries)
	groundTruth, err := helpers.ComputeGroundTruth(
		imageEmbeddings.Vectors,
		imageEmbeddings.Queries[:sampleQueries*dim],
		dim,
		k,
		faiss.MetricInnerProduct,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Test each index type
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Building %s index for 1M images...", tc.name)

			index := tc.buildIndex()
			defer index.Close()

			// Train if needed
			if !index.IsTrained() {
				trainSize := 100000 // 100K training images
				t.Logf("Training index with %d images...", trainSize)

				startTrain := time.Now()
				if err := index.Train(imageEmbeddings.Vectors[:trainSize*dim]); err != nil {
					t.Fatalf("Training failed: %v", err)
				}
				trainTime := time.Since(startTrain)
				t.Logf("Training completed in %v", trainTime)
			}

			// Add images in batches
			t.Logf("Adding %d images to index...", nImages)
			startAdd := time.Now()

			batchSize := 100000
			for i := 0; i < nImages; i += batchSize {
				end := i + batchSize
				if end > nImages {
					end = nImages
				}

				batch := imageEmbeddings.Vectors[i*dim : end*dim]
				if err := index.Add(batch); err != nil {
					t.Fatalf("Add failed at batch %d: %v", i/batchSize, err)
				}

				if (i+batchSize)%500000 == 0 {
					t.Logf("  Added %d images...", i+batchSize)
				}
			}

			addTime := time.Since(startAdd)
			t.Logf("Added %d images in %v (%.0f images/sec)",
				nImages, addTime, float64(nImages)/addTime.Seconds())

			// Measure memory
			memory := helpers.MeasureIndexMemory(index)
			memoryMB := memory / (1024 * 1024)
			compressionRatio := float64(nImages*dim*4) / float64(memory)
			t.Logf("Index memory: %d MB (%.1fx compression)", memoryMB, compressionRatio)

			// Search with sample queries
			t.Logf("Searching %d queries...", sampleQueries)
			results, latencies, err := helpers.SearchWithTiming(
				index,
				imageEmbeddings.Queries[:sampleQueries*dim],
				k,
			)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Calculate metrics
			metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
			perf := helpers.MeasureLatencies(latencies)

			// Log results
			t.Logf("\n=== Results for %s ===", tc.name)
			t.Logf("Quality Metrics:")
			t.Logf("  Recall@20: %.4f", metrics.RecallK)
			t.Logf("  Recall@10: %.4f", metrics.Recall10)
			t.Logf("  Recall@1:  %.4f", metrics.Recall1)
			t.Logf("\nPerformance Metrics:")
			t.Logf("  QPS:        %.0f queries/sec", perf.QPS)
			t.Logf("  P99 Latency: %v", perf.P99Latency.Round(time.Microsecond))
			t.Logf("\nResource Usage:")
			t.Logf("  Memory:     %d MB", memoryMB)
			t.Logf("  Add Time:   %v", addTime)
			t.Logf("  Throughput: %.0f images/sec", float64(nImages)/addTime.Seconds())

			// Validate targets
			if metrics.RecallK < tc.minRecall {
				t.Errorf("Recall@%d (%.4f) below target (%.4f)",
					k, metrics.RecallK, tc.minRecall)
			}

			if perf.P99Latency > tc.maxLatencyP99 {
				t.Errorf("P99 latency (%v) exceeds target (%v)",
					perf.P99Latency, tc.maxLatencyP99)
			}

			if metrics.RecallK >= tc.minRecall && perf.P99Latency <= tc.maxLatencyP99 {
				t.Logf("✓ %s meets all targets for 1M image search", tc.name)
			}
		})
	}

	// Summary
	t.Run("Summary", func(t *testing.T) {
		t.Logf("\n=== Image Similarity Scenario Summary ===")
		t.Logf("Use Case: Visual search for e-commerce")
		t.Logf("Dataset: 1M images with ResNet50 features (2048-dim)")
		t.Logf("\nRecommendation:")
		t.Logf("  - For high accuracy: HNSW with M=48, efSearch=128")
		t.Logf("  - For scale + memory: IVFPQ with nlist=4096, M=64")
		t.Logf("  - IVFPQ provides 10-20x memory reduction with 80%% recall")
	})
}

// TestImageSimilarity_Deduplication simulates finding duplicate/near-duplicate images
func TestImageSimilarity_Deduplication(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping deduplication scenario in short mode")
	}

	// Smaller dataset for deduplication
	nImages := 50000
	dim := 512 // Smaller features for dedup
	nQueries := 1000
	k := 100 // Check top-100 for near-duplicates

	t.Logf("Simulating image deduplication with %d images", nImages)

	// Generate realistic embeddings
	imageEmbeddings := datasets.GenerateRealisticEmbeddings(nImages, dim)
	imageEmbeddings.GenerateQueries(nQueries, datasets.Normalized)

	// For deduplication, use Flat for perfect recall
	// (need to find ALL near-duplicates)
	index, err := faiss.NewIndexFlatIP(dim)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Add images
	if err := index.Add(imageEmbeddings.Vectors); err != nil {
		t.Fatalf("Failed to add images: %v", err)
	}

	// Search
	results, latencies, err := helpers.SearchWithTiming(index, imageEmbeddings.Queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	perf := helpers.MeasureLatencies(latencies)

	// For deduplication, analyze distance distribution
	t.Logf("\n=== Image Deduplication Results ===")
	t.Logf("Images: %d", nImages)
	t.Logf("QPS: %.0f", perf.QPS)
	t.Logf("P99: %v", perf.P99Latency.Round(time.Microsecond))

	// Count potential duplicates (top results with high similarity)
	duplicates := 0
	for _, result := range results {
		// Skip first result (query image itself)
		// Check if any other results have very high similarity (>0.95)
		for j := 1; j < len(result.Distances) && j < 10; j++ {
			if result.Distances[j] > 0.95 { // High inner product = similar
				duplicates++
				break
			}
		}
	}

	t.Logf("Potential duplicates found: %d / %d queries (%.1f%%)",
		duplicates, nQueries, float64(duplicates)*100/float64(nQueries))

	t.Logf("✓ Deduplication completed with perfect recall")
}

// TestImageSimilarity_ThumbnailSearch simulates searching with thumbnail images
func TestImageSimilarity_ThumbnailSearch(t *testing.T) {
	// Use case: User uploads photo, find similar product images
	nProducts := 100000
	dim := 1024 // Smaller features for mobile
	nUploads := 200
	k := 10

	t.Logf("Simulating thumbnail search with %d products", nProducts)

	// Generate realistic embeddings
	productEmbeddings := datasets.GenerateRealisticEmbeddings(nProducts, dim)
	productEmbeddings.GenerateQueries(nUploads, datasets.Normalized)

	// Use HNSW for fast search
	index, err := faiss.NewIndexHNSWFlat(dim, 32, faiss.MetricInnerProduct)
	index.HnswSetEfSearch(64)
	defer index.Close()

	// Add products
	if err := index.Add(productEmbeddings.Vectors); err != nil {
		t.Fatalf("Failed to add products: %v", err)
	}

	// Compute ground truth
	groundTruth, err := helpers.ComputeGroundTruth(
		productEmbeddings.Vectors,
		productEmbeddings.Queries,
		dim,
		k,
		faiss.MetricInnerProduct,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Search
	results, latencies, err := helpers.SearchWithTiming(index, productEmbeddings.Queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Calculate metrics
	metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
	perf := helpers.MeasureLatencies(latencies)

	// Log results
	t.Logf("\n=== Thumbnail Search Results ===")
	t.Logf("Recall@10: %.4f (target: >0.95)", metrics.Recall10)
	t.Logf("QPS:       %.0f", perf.QPS)
	t.Logf("P99:       %v (target: <3ms for mobile)", perf.P99Latency.Round(time.Microsecond))

	// Mobile requires very low latency
	if metrics.Recall10 < 0.95 {
		t.Errorf("Recall too low for thumbnail search: %.4f", metrics.Recall10)
	}

	if perf.P99Latency > 3*time.Millisecond {
		t.Logf("Warning: P99 latency (%v) may be too high for mobile", perf.P99Latency)
	}

	t.Logf("✓ Thumbnail search achieves %.4f recall with %v P99", metrics.Recall10, perf.P99Latency.Round(time.Microsecond))
}
