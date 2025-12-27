package scenarios

import (
	"testing"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// TestRecommendations_ItemToItem simulates item-to-item recommendations
// Use case: E-commerce "customers who bought X also bought Y"
func TestRecommendations_ItemToItem(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping recommendations scenario in short mode")
	}

	// Simulate product embeddings learned from user interactions
	t.Log("Simulating item-to-item recommendations...")

	nItems := 10000000 // 10M products
	dim := 128         // Embedding dimension
	nRequests := 1000  // 1K recommendation requests
	k := 50            // Top-50 recommendations

	t.Logf("Dataset: %d items, %d dimensions", nItems, dim)

	// Generate realistic item embeddings (clustered by category)
	itemEmbeddings := datasets.GenerateRealisticEmbeddings(nItems, dim)
	itemEmbeddings.GenerateQueries(nRequests, datasets.Normalized)

	// For 10M items, IVFPQ is the only practical choice
	t.Run("IVFPQ_Production", func(t *testing.T) {
		t.Log("Building IVFPQ index for 10M items...")

		// Build IVFPQ: nlist=16384, M=16, nprobe=64
		quantizer := faiss.NewIndexFlatIP(dim)
		index := faiss.NewIndexIVFPQ(quantizer, dim, 16384, 16, 8, faiss.MetricInnerProduct)
		index.SetNprobe(64)
		defer index.Delete()

		// Train with 200K items
		trainSize := 200000
		t.Logf("Training with %d items...", trainSize)
		startTrain := time.Now()
		if err := index.Train(itemEmbeddings.Vectors[:trainSize*dim]); err != nil {
			t.Fatalf("Training failed: %v", err)
		}
		trainTime := time.Since(startTrain)
		t.Logf("Training completed in %v", trainTime)

		// Add items in batches
		t.Log("Adding 10M items...")
		startAdd := time.Now()

		batchSize := 500000
		for i := 0; i < nItems; i += batchSize {
			end := i + batchSize
			if end > nItems {
				end = nItems
			}

			batch := itemEmbeddings.Vectors[i*dim : end*dim]
			if err := index.Add(batch); err != nil {
				t.Fatalf("Add failed at batch %d: %v", i/batchSize, err)
			}

			if (i+batchSize)%2000000 == 0 {
				t.Logf("  Added %dM items...", (i+batchSize)/1000000)
			}
		}

		addTime := time.Since(startAdd)
		t.Logf("Added %dM items in %v (%.0f items/sec)",
			nItems/1000000, addTime, float64(nItems)/addTime.Seconds())

		// Measure memory
		memory := helpers.MeasureIndexMemory(index)
		memoryMB := memory / (1024 * 1024)
		memoryGB := float64(memoryMB) / 1024
		compressionRatio := float64(nItems*dim*4) / float64(memory)
		t.Logf("Index memory: %.2f GB (%.1fx compression)", memoryGB, compressionRatio)

		// Compute ground truth for sample
		sampleQueries := 100
		t.Logf("Computing ground truth for %d queries...", sampleQueries)
		groundTruth, err := helpers.ComputeGroundTruth(
			itemEmbeddings.Vectors,
			itemEmbeddings.Queries[:sampleQueries*dim],
			dim,
			k,
			faiss.MetricInnerProduct,
		)
		if err != nil {
			t.Fatalf("Failed to compute ground truth: %v", err)
		}

		// Search
		t.Logf("Generating recommendations for %d items...", sampleQueries)
		results, latencies, err := helpers.SearchWithTiming(
			index,
			itemEmbeddings.Queries[:sampleQueries*dim],
			k,
		)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Calculate metrics
		metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
		perf := helpers.MeasureLatencies(latencies)

		// Log results
		t.Logf("\n=== 10M Item Recommendation Results ===")
		t.Logf("Quality Metrics:")
		t.Logf("  Recall@50: %.4f", metrics.RecallK)
		t.Logf("  Recall@10: %.4f", metrics.Recall10)
		t.Logf("\nPerformance Metrics:")
		t.Logf("  QPS:        %.0f recommendations/sec", perf.QPS)
		t.Logf("  P99 Latency: %v", perf.P99Latency.Round(time.Microsecond))
		t.Logf("\nResource Usage:")
		t.Logf("  Memory:     %.2f GB", memoryGB)
		t.Logf("  Training:   %v", trainTime)
		t.Logf("  Indexing:   %v (%.0f items/sec)", addTime, float64(nItems)/addTime.Seconds())

		// Validate targets
		if metrics.RecallK < 0.70 {
			t.Errorf("Recall@50 too low: %.4f (target: >0.70)", metrics.RecallK)
		}

		if perf.P99Latency > 20*time.Millisecond {
			t.Errorf("P99 latency too high: %v (target: <20ms)", perf.P99Latency)
		}

		if memoryGB > 5.0 {
			t.Logf("Warning: Memory usage (%.2f GB) is high for 10M items", memoryGB)
		}

		t.Logf("✓ IVFPQ handles 10M items with %.2f GB memory and %.4f recall", memoryGB, metrics.RecallK)
	})
}

// TestRecommendations_ContentBased simulates content-based recommendations
// Use case: Video/article recommendations based on content similarity
func TestRecommendations_ContentBased(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping content-based recommendations in short mode")
	}

	// Simulate video/article embeddings
	nContent := 500000 // 500K videos/articles
	dim := 512         // Content embedding dimension
	nUsers := 1000     // 1K active users
	k := 20            // Top-20 recommendations

	t.Logf("Simulating content-based recommendations for %d items", nContent)

	// Generate realistic content embeddings
	contentEmbeddings := datasets.GenerateRealisticEmbeddings(nContent, dim)
	contentEmbeddings.GenerateQueries(nUsers, datasets.Normalized)

	// Use HNSW for high-quality recommendations
	index := faiss.NewIndexHNSWFlat(dim, 32, faiss.MetricInnerProduct)
	index.HnswSetEfSearch(64)
	defer index.Delete()

	// Add content
	t.Log("Building content index...")
	if err := index.Add(contentEmbeddings.Vectors); err != nil {
		t.Fatalf("Failed to add content: %v", err)
	}

	// Compute ground truth
	groundTruth, err := helpers.ComputeGroundTruth(
		contentEmbeddings.Vectors,
		contentEmbeddings.Queries,
		dim,
		k,
		faiss.MetricInnerProduct,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Generate recommendations
	results, latencies, err := helpers.SearchWithTiming(index, contentEmbeddings.Queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Calculate metrics
	metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
	perf := helpers.MeasureLatencies(latencies)

	// Log results
	t.Logf("\n=== Content-Based Recommendation Results ===")
	t.Logf("Recall@20: %.4f (target: >0.95)", metrics.RecallK)
	t.Logf("QPS:       %.0f recommendations/sec", perf.QPS)
	t.Logf("P99:       %v (target: <5ms)", perf.P99Latency.Round(time.Microsecond))

	// Content recommendations need high accuracy
	if metrics.RecallK < 0.95 {
		t.Errorf("Recall too low for content recommendations: %.4f", metrics.RecallK)
	}

	if perf.P99Latency > 5*time.Millisecond {
		t.Logf("Warning: P99 latency (%v) may impact user experience", perf.P99Latency)
	}

	t.Logf("✓ Content recommendations achieve %.4f recall with %v P99",
		metrics.RecallK, perf.P99Latency.Round(time.Microsecond))
}

// TestRecommendations_PersonalizedRanking simulates personalized ranking
func TestRecommendations_PersonalizedRanking(t *testing.T) {
	// Use case: Re-rank candidate items based on user preference
	nCandidates := 1000 // Pre-filtered candidates
	nUsers := 500       // Active users
	dim := 256          // User/item embedding dimension
	k := 10             // Top-10 personalized results

	t.Logf("Simulating personalized ranking for %d users", nUsers)

	// Generate user and item embeddings
	itemEmbeddings := datasets.GenerateRealisticEmbeddings(nCandidates, dim)
	itemEmbeddings.GenerateQueries(nUsers, datasets.Normalized)

	// For small candidate set, use Flat for perfect ranking
	index := faiss.NewIndexFlatIP(dim)
	defer index.Delete()

	// Add candidates
	if err := index.Add(itemEmbeddings.Vectors); err != nil {
		t.Fatalf("Failed to add candidates: %v", err)
	}

	// Rank for each user
	results, latencies, err := helpers.SearchWithTiming(index, itemEmbeddings.Queries, k)
	if err != nil {
		t.Fatalf("Ranking failed: %v", err)
	}

	perf := helpers.MeasureLatencies(latencies)

	// Log results
	t.Logf("\n=== Personalized Ranking Results ===")
	t.Logf("Users ranked: %d", nUsers)
	t.Logf("QPS:          %.0f users/sec", perf.QPS)
	t.Logf("P99:          %v", perf.P99Latency.Round(time.Microsecond))

	// Ranking must be fast for real-time serving
	if perf.P99Latency > 2*time.Millisecond {
		t.Logf("Warning: P99 latency (%v) may be too high for real-time", perf.P99Latency)
	}

	t.Logf("✓ Personalized ranking achieves %v P99 for 1K candidates", perf.P99Latency.Round(time.Microsecond))
}

// TestRecommendations_CollaborativeFiltering simulates CF-based recommendations
func TestRecommendations_CollaborativeFiltering(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping CF recommendations in short mode")
	}

	// Use case: User-user or item-item collaborative filtering
	nItems := 100000 // 100K items
	dim := 128       // Latent factor dimension
	nQueries := 2000 // 2K recommendation requests
	k := 30          // Top-30 recommendations

	t.Logf("Simulating collaborative filtering with %d items", nItems)

	// Generate item latent factors
	itemFactors := datasets.GenerateRealisticEmbeddings(nItems, dim)
	itemFactors.GenerateQueries(nQueries, datasets.Normalized)

	// Use IVF for good balance
	quantizer := faiss.NewIndexFlatIP(dim)
	index := faiss.NewIndexIVFFlat(quantizer, dim, 1000, faiss.MetricInnerProduct)
	index.SetNprobe(20)
	defer index.Delete()

	// Train and add
	t.Log("Training collaborative filtering index...")
	if err := index.Train(itemFactors.Vectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if err := index.Add(itemFactors.Vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Compute ground truth
	sampleQueries := 200
	groundTruth, err := helpers.ComputeGroundTruth(
		itemFactors.Vectors,
		itemFactors.Queries[:sampleQueries*dim],
		dim,
		k,
		faiss.MetricInnerProduct,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Generate recommendations
	results, latencies, err := helpers.SearchWithTiming(
		index,
		itemFactors.Queries[:sampleQueries*dim],
		k,
	)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Calculate metrics
	metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
	perf := helpers.MeasureLatencies(latencies)

	// Log results
	t.Logf("\n=== Collaborative Filtering Results ===")
	t.Logf("Recall@30: %.4f (target: >0.85)", metrics.RecallK)
	t.Logf("QPS:       %.0f recommendations/sec", perf.QPS)
	t.Logf("P99:       %v", perf.P99Latency.Round(time.Microsecond))

	// CF can tolerate slightly lower recall
	if metrics.RecallK < 0.85 {
		t.Errorf("Recall too low for CF: %.4f", metrics.RecallK)
	}

	t.Logf("✓ Collaborative filtering achieves %.4f recall", metrics.RecallK)
}
