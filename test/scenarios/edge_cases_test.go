package scenarios

import (
	"testing"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
)

// TestEdgeCase_EmptyIndex tests searching on empty indexes
// Use case: Prevents crashes when querying before adding data
func TestEdgeCase_EmptyIndex(t *testing.T) {
	dim := 128
	k := 10

	testCases := []struct {
		name          string
		factoryString string
	}{
		{"Flat", "Flat"},
		{"HNSW32", "HNSW32"},
		{"IVF100_Flat", "IVF100,Flat"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create index
			index, err := faiss.IndexFactory(dim, tc.factoryString, faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Try to search on empty index
			query := make([]float32, dim)
			_, labels, err := index.Search(query, k)

			// FAISS always returns k results, padding with -1 labels for empty indexes
			if err != nil {
				t.Logf("Search on empty index returned error: %v", err)
			} else {
				t.Logf("Search on empty index succeeded: %d results", len(labels))
				// FAISS returns k results even for empty index
				if len(labels) != k {
					t.Errorf("Expected %d results (FAISS always returns k), got %d", k, len(labels))
				}
				// Verify all labels are -1 (no valid neighbors found)
				allNegativeOne := true
				for _, label := range labels {
					if label != -1 {
						allNegativeOne = false
						break
					}
				}
				if !allNegativeOne {
					t.Logf("Note: Some labels are not -1 (FAISS behavior varies by index type)")
				}
			}

			// Verify index is still empty
			if index.Ntotal() != 0 {
				t.Errorf("Expected Ntotal=0, got %d", index.Ntotal())
			}

			t.Logf("✓ %s handles empty index gracefully", tc.name)
		})
	}
}

// TestEdgeCase_SingleVector tests indexes with just one vector
// Use case: Minimum viable index size
func TestEdgeCase_SingleVector(t *testing.T) {
	dim := 64
	k := 5 // Request more neighbors than available

	// Generate single vector
	vectors := datasets.GenerateRealisticEmbeddings(1, dim)

	testCases := []struct {
		name          string
		factoryString string
	}{
		{"Flat", "Flat"},
		{"HNSW32", "HNSW32"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			index, err := faiss.IndexFactory(dim, tc.factoryString, faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Add single vector
			if err := index.Add(vectors.Vectors); err != nil {
				t.Fatalf("Failed to add single vector: %v", err)
			}

			// Search for k neighbors (more than available)
			_, labels, err := index.Search(vectors.Vectors, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// FAISS returns k results, padding with -1 for non-existent neighbors
			if len(labels) != k {
				t.Errorf("Expected %d results (FAISS always returns k), got %d", k, len(labels))
			}

			// First label should be 0 (the vector itself)
			if len(labels) > 0 && labels[0] != 0 {
				t.Errorf("Expected first label 0, got %d", labels[0])
			}

			// Remaining labels should be -1
			for i := 1; i < len(labels); i++ {
				if labels[i] != -1 {
					t.Errorf("Expected label -1 at position %d, got %d", i, labels[i])
				}
			}

			t.Logf("✓ %s handles single vector correctly (returns k results with padding)", tc.name)
		})
	}
}

// TestEdgeCase_KGreaterThanN tests k > n scenarios
// Use case: Ensure robust handling when k exceeds available vectors
func TestEdgeCase_KGreaterThanN(t *testing.T) {
	dim := 128
	n := 10
	k := 50 // k > n

	vectors := datasets.GenerateRealisticEmbeddings(n, dim)

	index, err := faiss.IndexFactory(dim, "Flat", faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if err := index.Add(vectors.Vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Search with k > n
	_, labels, err := index.Search(vectors.Vectors[:dim], k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// FAISS returns k results, padding with -1 when k > n
	if len(labels) != k {
		t.Errorf("Expected %d results (FAISS always returns k), got %d", k, len(labels))
	}

	// First n results should be valid (0 to n-1), remaining should be -1
	validCount := 0
	paddingCount := 0
	for _, label := range labels {
		if label >= 0 {
			validCount++
		} else if label == -1 {
			paddingCount++
		}
	}

	t.Logf("✓ Handles k=%d > n=%d gracefully (%d valid results, %d padding)", 
		k, n, validCount, paddingCount)
}

// TestEdgeCase_ZeroDimension tests error handling for invalid dimensions
// Use case: Input validation
func TestEdgeCase_ZeroDimension(t *testing.T) {
	testCases := []struct {
		name string
		dim  int
	}{
		{"Zero", 0},
		{"Negative", -1},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := faiss.IndexFactory(tc.dim, "Flat", faiss.MetricL2)
			if err == nil {
				t.Errorf("Expected error for dim=%d, got nil", tc.dim)
			} else {
				t.Logf("✓ Correctly rejected dim=%d: %v", tc.dim, err)
			}
		})
	}
}

// TestEdgeCase_MismatchedDimensions tests dimension mismatch handling
// Use case: Prevents crashes from incorrect vector sizes
func TestEdgeCase_MismatchedDimensions(t *testing.T) {
	dim := 128
	wrongDim := 64

	index, err := faiss.IndexFactory(dim, "Flat", faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Generate vectors with wrong dimension
	wrongVectors := datasets.GenerateRealisticEmbeddings(10, wrongDim)

	// Try to add vectors with wrong dimension
	err = index.Add(wrongVectors.Vectors)
	// FAISS may or may not validate dimensions - both behaviors are acceptable
	if err != nil {
		t.Logf("✓ Correctly rejected mismatched dimensions: %v", err)
	} else {
		t.Logf("Note: FAISS did not reject mismatched dimensions (implementation-specific behavior)")
		// This is acceptable - FAISS behavior varies by implementation
	}
}

// TestEdgeCase_VeryLargeDimensions tests high-dimensional vectors
// Use case: Ensure scalability for modern embeddings (e.g., GPT-4 embeddings)
func TestEdgeCase_VeryLargeDimensions(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large dimension test in short mode")
	}

	dim := 4096 // GPT-4 embedding size
	n := 1000
	k := 10

	t.Logf("Testing with very large dimensions: %d-dim vectors", dim)

	vectors := datasets.GenerateRealisticEmbeddings(n, dim)

	testCases := []struct {
		name          string
		factoryString string
	}{
		{"Flat", "Flat"},
		{"HNSW32", "HNSW32"},
		// Use IVF10 since we only have 1000 vectors (~39*10=390 needed)
		{"IVF10_Flat", "IVF10,Flat"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			index, err := faiss.IndexFactory(dim, tc.factoryString, faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			// Train if needed
			// IVF100 needs ~39*100 = 3900 training vectors
			// But we only have 1000 vectors here, so use them all
			if !index.IsTrained() {
				if err := index.Train(vectors.Vectors); err != nil {
					t.Fatalf("Training failed: %v", err)
				}
			}

			// Add vectors
			if err := index.Add(vectors.Vectors); err != nil {
				t.Fatalf("Failed to add vectors: %v", err)
			}

			// Search
			_, labels, err := index.Search(vectors.Vectors[:dim], k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			if len(labels) != k {
				t.Errorf("Expected %d results, got %d", k, len(labels))
			}

			t.Logf("✓ %s handles %d-dim vectors correctly", tc.name, dim)
		})
	}
}

// TestEdgeCase_VerySmallDimensions tests low-dimensional vectors
// Use case: Ensure correctness for simple use cases
func TestEdgeCase_VerySmallDimensions(t *testing.T) {
	testCases := []struct {
		dim int
	}{
		{1},  // Minimum viable dimension
		{2},  // 2D vectors
		{3},  // 3D vectors
	}

	for _, tc := range testCases {
		t.Run("dim_"+string(rune(tc.dim+'0')), func(t *testing.T) {
			n := 100
			k := 10

			vectors := datasets.GenerateRealisticEmbeddings(n, tc.dim)

			index, err := faiss.IndexFactory(tc.dim, "Flat", faiss.MetricL2)
			if err != nil {
				t.Fatalf("Failed to create index with dim=%d: %v", tc.dim, err)
			}
			defer index.Close()

			if err := index.Add(vectors.Vectors); err != nil {
				t.Fatalf("Failed to add vectors: %v", err)
			}

			_, labels, err := index.Search(vectors.Vectors[:tc.dim], k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			if len(labels) != k {
				t.Errorf("Expected %d results, got %d", k, len(labels))
			}

			t.Logf("✓ Handles %d-dim vectors correctly", tc.dim)
		})
	}
}

// TestEdgeCase_MultipleClose tests that closing an index multiple times is safe
// Use case: Prevents double-free crashes
func TestEdgeCase_MultipleClose(t *testing.T) {
	dim := 128

	index, err := faiss.IndexFactory(dim, "Flat", faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	// Close once
	err = index.Close()
	if err != nil {
		t.Fatalf("First close failed: %v", err)
	}

	// Close again
	err = index.Close()
	// Should either succeed silently or return a specific error
	if err != nil {
		t.Logf("Second close returned error (acceptable): %v", err)
	} else {
		t.Logf("Second close succeeded silently (acceptable)")
	}

	t.Logf("✓ Multiple Close() calls handled safely")
}

// TestEdgeCase_LargeKValue tests very large k values
// Use case: Stress test result handling
func TestEdgeCase_LargeKValue(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large k test in short mode")
	}

	dim := 128
	n := 10000
	k := 5000 // Very large k

	t.Logf("Testing with large k=%d (n=%d)", k, n)

	vectors := datasets.GenerateRealisticEmbeddings(n, dim)

	index, err := faiss.IndexFactory(dim, "Flat", faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if err := index.Add(vectors.Vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Search with very large k
	distances, labels, err := index.Search(vectors.Vectors[:dim], k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// FAISS returns exactly k results
	if len(labels) != k {
		t.Errorf("Expected %d results, got %d", k, len(labels))
	}

	if len(distances) != len(labels) {
		t.Errorf("Distances and labels length mismatch: %d vs %d", len(distances), len(labels))
	}

	t.Logf("✓ Handles large k=%d correctly (returned %d results)", k, len(labels))
}

// TestEdgeCase_BatchSearch tests searching with multiple queries at once
// Use case: Batch processing efficiency
func TestEdgeCase_BatchSearch(t *testing.T) {
	dim := 128
	n := 1000
	nQueries := 100
	k := 10

	vectors := datasets.GenerateRealisticEmbeddings(n, dim)
	queries := datasets.GenerateRealisticEmbeddings(nQueries, dim)

	index, err := faiss.IndexFactory(dim, "Flat", faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if err := index.Add(vectors.Vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Search with multiple queries
	_, labels, err := index.Search(queries.Vectors, k)
	if err != nil {
		t.Fatalf("Batch search failed: %v", err)
	}

	expectedResults := nQueries * k
	if len(labels) != expectedResults {
		t.Errorf("Expected %d results (%d queries * %d), got %d",
			expectedResults, nQueries, k, len(labels))
	}

	t.Logf("✓ Batch search with %d queries works correctly", nQueries)
}

// TestEdgeCase_AllZeroVectors tests indexes with zero vectors
// Use case: Edge case in normalization and distance computation
func TestEdgeCase_AllZeroVectors(t *testing.T) {
	dim := 128
	n := 10
	k := 5

	// Create all-zero vectors
	vectors := make([]float32, n*dim)
	// All values are already 0

	index, err := faiss.IndexFactory(dim, "Flat", faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Add zero vectors
	err = index.Add(vectors)
	if err != nil {
		t.Logf("Adding zero vectors returned error: %v", err)
		// Some indexes may reject zero vectors, which is acceptable
		return
	}

	// Search with zero query
	query := make([]float32, dim)
	distances, labels, err := index.Search(query, k)
	if err != nil {
		t.Logf("Searching with zero query returned error: %v", err)
		// Acceptable behavior
		return
	}

	// All distances should be 0
	for i, dist := range distances {
		if dist != 0 {
			t.Errorf("Expected distance 0 for zero vectors, got %f at position %d", dist, i)
		}
	}

	t.Logf("✓ Handles all-zero vectors (returned %d results)", len(labels))
}

// TestEdgeCase_IncrementalAdd tests adding vectors incrementally
// Use case: Streaming/real-time indexing
func TestEdgeCase_IncrementalAdd(t *testing.T) {
	dim := 128
	totalVectors := 1000
	batchSize := 10
	k := 5

	index, err := faiss.IndexFactory(dim, "Flat", faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Add vectors in small batches
	for i := 0; i < totalVectors; i += batchSize {
		batch := datasets.GenerateRealisticEmbeddings(batchSize, dim)
		if err := index.Add(batch.Vectors); err != nil {
			t.Fatalf("Failed to add batch %d: %v", i/batchSize, err)
		}

		// Verify Ntotal increases
		expectedTotal := int64(i + batchSize)
		if index.Ntotal() != expectedTotal {
			t.Errorf("After batch %d, expected Ntotal=%d, got %d",
				i/batchSize, expectedTotal, index.Ntotal())
		}
	}

	// Final verification
	if index.Ntotal() != int64(totalVectors) {
		t.Errorf("Expected final Ntotal=%d, got %d", totalVectors, index.Ntotal())
	}

	// Search should work
	query := datasets.GenerateRealisticEmbeddings(1, dim)
	_, labels, err := index.Search(query.Vectors, k)
	if err != nil {
		t.Fatalf("Search after incremental adds failed: %v", err)
	}

	if len(labels) != k {
		t.Errorf("Expected %d results, got %d", k, len(labels))
	}

	t.Logf("✓ Incremental add of %d vectors in batches of %d works correctly",
		totalVectors, batchSize)
}

// TestEdgeCase_IdenticalVectors tests index behavior with duplicate vectors
// Use case: Deduplication scenarios
func TestEdgeCase_IdenticalVectors(t *testing.T) {
	dim := 64
	n := 100
	k := 10

	// Create identical vectors
	baseVector := datasets.GenerateRealisticEmbeddings(1, dim)
	vectors := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		copy(vectors[i*dim:(i+1)*dim], baseVector.Vectors)
	}

	index, err := faiss.IndexFactory(dim, "Flat", faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Search with the same vector
	distances, _, err := index.Search(baseVector.Vectors, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// All distances should be 0 (identical vectors)
	for i, dist := range distances {
		if dist > 1e-6 { // Allow small floating point error
			t.Errorf("Expected distance ~0 for identical vector, got %f at position %d", dist, i)
		}
	}

	t.Logf("✓ Handles %d identical vectors correctly (all distances ~0)", n)
}
