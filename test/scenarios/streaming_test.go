package scenarios

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// TestStreaming_ConcurrentAddAndSearch simulates real-time vector additions with concurrent queries
// Use case: Social media feed, news articles, real-time monitoring
func TestStreaming_ConcurrentAddAndSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping streaming scenario in short mode")
	}

	t.Log("Simulating real-time streaming with concurrent add + search...")

	dim := 256
	initialSize := 50000
	streamDuration := 10 * time.Second
	addRate := 1000  // 1K vectors/sec
	queryRate := 100 // 100 queries/sec
	k := 10

	t.Logf("Configuration:")
	t.Logf("  Initial size: %d vectors", initialSize)
	t.Logf("  Stream duration: %v", streamDuration)
	t.Logf("  Add rate: %d vectors/sec", addRate)
	t.Logf("  Query rate: %d queries/sec", queryRate)

	// Generate initial dataset
	initial := datasets.GenerateRealisticEmbeddings(initialSize, dim)

	// Generate streaming data
	streamingVectors := datasets.GenerateRealisticEmbeddings(
		addRate*int(streamDuration.Seconds()),
		dim,
	)

	// Generate query stream
	queries := datasets.GenerateRealisticEmbeddings(
		queryRate*int(streamDuration.Seconds()),
		dim,
	)

	// Test with HNSW (best for dynamic updates)
	t.Run("HNSW_Dynamic", func(t *testing.T) {
		index, err := faiss.NewIndexHNSWFlat(dim, 32, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
		index.SetEfSearch(64)
		defer index.Close()

		// Add initial vectors
		t.Log("Adding initial vectors...")
		if err := index.Add(initial.Vectors); err != nil {
			t.Fatalf("Initial add failed: %v", err)
		}

		// Metrics tracking
		var totalAdded atomic.Int64
		var totalQueries atomic.Int64
		var totalErrors atomic.Int64

		latencies := make([]time.Duration, 0, queryRate*int(streamDuration.Seconds()))
		var latencyMutex sync.Mutex

		// Start time
		startTime := time.Now()
		var wg sync.WaitGroup

		// Add goroutine - streams vectors
		wg.Add(1)
		go func() {
			defer wg.Done()

			addInterval := time.Second / time.Duration(addRate)
			ticker := time.NewTicker(addInterval)
			defer ticker.Stop()

			vectorIdx := 0
			for time.Since(startTime) < streamDuration {
				<-ticker.C

				if vectorIdx >= len(streamingVectors.Vectors)/dim {
					break
				}

				// Add single vector
				vec := streamingVectors.Vectors[vectorIdx*dim : (vectorIdx+1)*dim]
				if err := index.Add(vec); err != nil {
					totalErrors.Add(1)
				} else {
					totalAdded.Add(1)
				}

				vectorIdx++
			}
		}()

		// Query goroutine - continuous search
		wg.Add(1)
		go func() {
			defer wg.Done()

			queryInterval := time.Second / time.Duration(queryRate)
			ticker := time.NewTicker(queryInterval)
			defer ticker.Stop()

			queryIdx := 0
			for time.Since(startTime) < streamDuration {
				<-ticker.C

				if queryIdx >= len(queries.Vectors)/dim {
					queryIdx = 0 // Wrap around
				}

				// Search
				query := queries.Vectors[queryIdx*dim : (queryIdx+1)*dim]

				searchStart := time.Now()
				_, _, err := index.Search(query, k)
				latency := time.Since(searchStart)

				if err != nil {
					totalErrors.Add(1)
				} else {
					totalQueries.Add(1)
					latencyMutex.Lock()
					latencies = append(latencies, latency)
					latencyMutex.Unlock()
				}

				queryIdx++
			}
		}()

		// Wait for completion
		wg.Wait()
		totalDuration := time.Since(startTime)

		// Calculate metrics
		added := totalAdded.Load()
		searched := totalQueries.Load()
		errors := totalErrors.Load()

		perf := helpers.MeasureLatencies(latencies)

		// Log results
		t.Logf("\n=== Streaming Results ===")
		t.Logf("Duration: %v", totalDuration)
		t.Logf("\nThroughput:")
		t.Logf("  Vectors added: %d (%.0f/sec)", added, float64(added)/totalDuration.Seconds())
		t.Logf("  Queries executed: %d (%.0f/sec)", searched, float64(searched)/totalDuration.Seconds())
		t.Logf("  Errors: %d", errors)
		t.Logf("\nQuery Latency:")
		t.Logf("  P50: %v", perf.P50Latency.Round(time.Microsecond))
		t.Logf("  P95: %v", perf.P95Latency.Round(time.Microsecond))
		t.Logf("  P99: %v", perf.P99Latency.Round(time.Microsecond))
		t.Logf("\nFinal index size: %d vectors", index.Ntotal())

		// Validate
		if errors > 0 {
			t.Errorf("Encountered %d errors during streaming", errors)
		}

		if added < int64(addRate*int(streamDuration.Seconds())*95/100) {
			t.Errorf("Add throughput too low: %d (target: ~%d)",
				added, addRate*int(streamDuration.Seconds()))
		}

		if searched < int64(queryRate*int(streamDuration.Seconds())*95/100) {
			t.Errorf("Query throughput too low: %d (target: ~%d)",
				searched, queryRate*int(streamDuration.Seconds()))
		}

		if perf.P99Latency > 10*time.Millisecond {
			t.Logf("Warning: P99 latency (%v) is high during streaming", perf.P99Latency)
		}

		t.Logf("✓ Streaming completed: %d adds/sec, %d queries/sec, P99=%v",
			int(float64(added)/totalDuration.Seconds()),
			int(float64(searched)/totalDuration.Seconds()),
			perf.P99Latency.Round(time.Microsecond))
	})
}

// TestStreaming_BatchUpdates simulates periodic batch updates
// Use case: Hourly/daily index updates with fresh content
func TestStreaming_BatchUpdates(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping batch updates scenario in short mode")
	}

	dim := 512
	initialSize := 100000
	batchSize := 10000
	nBatches := 5

	t.Logf("Simulating batch updates: %d batches of %d vectors", nBatches, batchSize)

	// Generate initial dataset
	initial := datasets.GenerateRealisticEmbeddings(initialSize, dim)

	// Use IVF for batch updates
	quantizer, err := faiss.NewIndexFlatL2(dim)
	index, err := faiss.NewIndexIVFFlat(quantizer, dim, 1000, faiss.MetricL2)
	index.SetNprobe(20)
	defer index.Close()

	// Train and add initial vectors
	t.Log("Building initial index...")
	if err := index.Train(initial.Vectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if err := index.Add(initial.Vectors); err != nil {
		t.Fatalf("Initial add failed: %v", err)
	}

	t.Logf("Initial index: %d vectors", index.Ntotal())

	// Simulate batch updates
	totalAddTime := time.Duration(0)

	for i := 0; i < nBatches; i++ {
		// Generate batch
		batch := datasets.GenerateRealisticEmbeddings(batchSize, dim)

		// Add batch
		t.Logf("Adding batch %d/%d (%d vectors)...", i+1, nBatches, batchSize)
		startAdd := time.Now()
		if err := index.Add(batch.Vectors); err != nil {
			t.Fatalf("Batch %d add failed: %v", i, err)
		}
		addTime := time.Since(startAdd)
		totalAddTime += addTime

		t.Logf("  Added in %v (%.0f vectors/sec)",
			addTime, float64(batchSize)/addTime.Seconds())
		t.Logf("  Total vectors: %d", index.Ntotal())
	}

	// Final stats
	finalSize := index.Ntotal()
	expectedSize := initialSize + (batchSize * nBatches)

	t.Logf("\n=== Batch Update Results ===")
	t.Logf("Initial size: %d", initialSize)
	t.Logf("Batches added: %d × %d = %d vectors", nBatches, batchSize, batchSize*nBatches)
	t.Logf("Final size: %d (expected: %d)", finalSize, expectedSize)
	t.Logf("Total add time: %v", totalAddTime)
	t.Logf("Average throughput: %.0f vectors/sec",
		float64(batchSize*nBatches)/totalAddTime.Seconds())

	if finalSize != expectedSize {
		t.Errorf("Final size mismatch: got %d, expected %d", finalSize, expectedSize)
	}

	t.Logf("✓ Batch updates completed successfully")
}

// TestStreaming_IDMapping simulates streaming with custom IDs
// Use case: Social media posts, documents with external IDs
func TestStreaming_IDMapping(t *testing.T) {
	dim := 128
	nDocuments := 10000
	nQueries := 100
	k := 10

	t.Log("Simulating streaming with custom ID mapping...")

	// Generate documents
	documents := datasets.GenerateRealisticEmbeddings(nDocuments, dim)
	documents.GenerateQueries(nQueries, datasets.Normalized)

	// Use IndexIDMap for custom IDs
	baseIndex, err := faiss.NewIndexFlatL2(dim)
	index, err := faiss.NewIndexIDMap(baseIndex)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Add documents with custom IDs
	customIDs := make([]int64, nDocuments)
	for i := 0; i < nDocuments; i++ {
		customIDs[i] = int64(1000000 + i) // Start from 1M
	}

	t.Log("Adding documents with custom IDs...")
	if err := index.AddWithIDs(documents.Vectors, customIDs); err != nil {
		t.Fatalf("AddWithIDs failed: %v", err)
	}

	// Search
	results, latencies, err := helpers.SearchWithTiming(index, documents.Queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	perf := helpers.MeasureLatencies(latencies)

	// Verify IDs are in custom range
	customIDCount := 0
	for _, result := range results {
		for _, id := range result.IDs {
			if id >= 1000000 {
				customIDCount++
			}
		}
	}

	t.Logf("\n=== ID Mapping Results ===")
	t.Logf("Documents: %d (IDs: 1000000-%d)", nDocuments, 1000000+nDocuments-1)
	t.Logf("Custom IDs in results: %d/%d (%.1f%%)",
		customIDCount, nQueries*k, float64(customIDCount)*100/float64(nQueries*k))
	t.Logf("QPS: %.0f", perf.QPS)
	t.Logf("P99: %v", perf.P99Latency.Round(time.Microsecond))

	// All results should have custom IDs
	if customIDCount < nQueries*k*95/100 {
		t.Errorf("Expected most IDs to be custom, got %d/%d", customIDCount, nQueries*k)
	}

	t.Logf("✓ ID mapping works correctly")
}

// TestStreaming_LatencyDegradation measures performance degradation over time
func TestStreaming_LatencyDegradation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping latency degradation test in short mode")
	}

	dim := 256
	nQueries := 100
	k := 10

	t.Log("Measuring latency degradation as index grows...")

	// Test at different index sizes
	sizes := []int{1000, 10000, 50000, 100000, 500000}
	results := make([]struct {
		size    int
		latency time.Duration
		qps     float64
	}, len(sizes))

	queries := datasets.GenerateRealisticEmbeddings(nQueries, dim)

	for i, size := range sizes {
		t.Logf("\nTesting with %d vectors...", size)

		// Generate data
		data := datasets.GenerateRealisticEmbeddings(size, dim)

		// Build index
		index, err := faiss.NewIndexHNSWFlat(dim, 32, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
		index.SetEfSearch(64)

		if err := index.Add(data.Vectors); err != nil {
			t.Fatalf("Add failed: %v", err)
		}

		// Search
		_, latencies, err := helpers.SearchWithTiming(index, queries.Vectors, k)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		perf := helpers.MeasureLatencies(latencies)

		results[i].size = size
		results[i].latency = perf.P99Latency
		results[i].qps = perf.QPS

		t.Logf("  P99: %v, QPS: %.0f", perf.P99Latency.Round(time.Microsecond), perf.QPS)

		index.Close()
	}

	// Analyze degradation
	t.Logf("\n=== Latency Degradation Analysis ===")
	t.Logf("%-10s %15s %15s", "Size", "P99 Latency", "QPS")
	t.Logf("%s", "--------------------------------------------")

	for _, r := range results {
		t.Logf("%-10d %15v %15.0f", r.size, r.latency.Round(time.Microsecond), r.qps)
	}

	// Check degradation
	firstLatency := results[0].latency
	lastLatency := results[len(results)-1].latency
	degradation := float64(lastLatency) / float64(firstLatency)

	t.Logf("\nLatency increase: %.2fx (from %v to %v)",
		degradation,
		firstLatency.Round(time.Microsecond),
		lastLatency.Round(time.Microsecond))

	// For HNSW, logarithmic degradation is expected
	if degradation > 5.0 {
		t.Logf("Warning: High latency degradation (%.2fx)", degradation)
	}

	t.Logf("✓ Measured latency degradation across index sizes")
}
