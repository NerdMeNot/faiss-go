package faiss

import (
	"testing"
)

// ========================================
// Index Interface Tests
// ========================================

// TestIndex_InterfaceCompliance verifies that all index types implement the Index interface
func TestIndex_InterfaceCompliance(t *testing.T) {
	tests := []struct {
		name string
		idx  Index
	}{
		{
			name: "IndexFlat",
			idx:  mustCreateIndexFlatL2(t, 64),
		},
		{
			name: "IndexLSH",
			idx:  mustCreateIndexLSH(t, 64, 8),
		},
		{
			name: "IndexScalarQuantizer",
			idx:  mustCreateIndexSQ(t, 64),
		},
		{
			name: "GenericIndex_Flat",
			idx:  mustCreateGenericIndex(t, 64, "Flat"),
		},
		{
			name: "GenericIndex_HNSW",
			idx:  mustCreateGenericIndex(t, 64, "HNSW32"),
		},
		{
			name: "GenericIndex_IVF",
			idx:  mustCreateGenericIndex(t, 64, "IVF10,Flat"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer tt.idx.Close()

			// Test D() method
			if d := tt.idx.D(); d != 64 {
				t.Errorf("D() = %d, want 64", d)
			}

			// Test Ntotal() - should be 0 initially
			if ntotal := tt.idx.Ntotal(); ntotal != 0 {
				t.Errorf("Ntotal() = %d, want 0 before adding vectors", ntotal)
			}

			// Test MetricType()
			metric := tt.idx.MetricType()
			if metric != MetricL2 && metric != MetricInnerProduct {
				t.Errorf("MetricType() returned invalid metric: %v", metric)
			}

			// Test IsTrained()
			isTrained := tt.idx.IsTrained()
			if tt.name == "GenericIndex_IVF" || tt.name == "IndexScalarQuantizer" {
				// IVF and SQ indexes need training (SQ needs training to set quantization bounds)
				if isTrained {
					t.Errorf("IsTrained() = true before training %s index", tt.name)
				}
			} else {
				// Flat, LSH, HNSW don't need training
				if !isTrained {
					t.Error("IsTrained() = false for index that doesn't require training")
				}
			}
		})
	}
}

// TestIndex_BasicOperations tests Add, Search, and Reset on all index types
func TestIndex_BasicOperations(t *testing.T) {
	d := 64
	n := 100
	vectors := generateTestVectors(n, d)

	tests := []struct {
		name          string
		createIndex   func(t *testing.T) Index
		needsTraining bool
	}{
		{
			name:          "IndexFlat",
			createIndex:   func(t *testing.T) Index { return mustCreateIndexFlatL2(t, d) },
			needsTraining: false,
		},
		{
			name:          "IndexLSH",
			createIndex:   func(t *testing.T) Index { return mustCreateIndexLSH(t, d, 8) },
			needsTraining: false,
		},
		{
			name:          "GenericIndex_Flat",
			createIndex:   func(t *testing.T) Index { return mustCreateGenericIndex(t, d, "Flat") },
			needsTraining: false,
		},
		{
			name:          "GenericIndex_HNSW",
			createIndex:   func(t *testing.T) Index { return mustCreateGenericIndex(t, d, "HNSW16") },
			needsTraining: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := tt.createIndex(t)
			defer idx.Close()

			// Train if needed
			if tt.needsTraining {
				if err := idx.Train(vectors); err != nil {
					t.Fatalf("Train() failed: %v", err)
				}
			}

			// Test Add
			if err := idx.Add(vectors); err != nil {
				t.Fatalf("Add() failed: %v", err)
			}

			if idx.Ntotal() != int64(n) {
				t.Errorf("Ntotal() = %d after Add, want %d", idx.Ntotal(), n)
			}

			// Test Search
			query := vectors[:d]
			distances, labels, err := idx.Search(query, 5)
			if err != nil {
				t.Fatalf("Search() failed: %v", err)
			}

			if len(distances) != 5 || len(labels) != 5 {
				t.Errorf("Search() returned %d distances and %d labels, want 5 each",
					len(distances), len(labels))
			}

			// First result should be the query itself (distance ≈ 0)
			if distances[0] > 0.1 {
				t.Errorf("First search result distance = %.4f, want ≈ 0 (query to itself)",
					distances[0])
			}

			// Test Reset
			if err := idx.Reset(); err != nil {
				t.Fatalf("Reset() failed: %v", err)
			}

			if idx.Ntotal() != 0 {
				t.Errorf("Ntotal() = %d after Reset, want 0", idx.Ntotal())
			}
		})
	}
}

// TestIndex_EmptyOperations tests edge cases with empty inputs
func TestIndex_EmptyOperations(t *testing.T) {
	idx := mustCreateIndexFlatL2(t, 64)
	defer idx.Close()

	// Test Add with empty vectors
	err := idx.Add([]float32{})
	if err != nil {
		t.Errorf("Add(empty) returned error: %v, want nil", err)
	}

	// Test Search with empty query
	_, _, err = idx.Search([]float32{}, 5)
	// Implementation may return error or handle gracefully
	// Just verify it doesn't crash
	t.Logf("Search(empty) returned: %v", err)
}

// TestIndex_InvalidDimensions tests error handling for incorrect dimensions
func TestIndex_InvalidDimensions(t *testing.T) {
	idx := mustCreateIndexFlatL2(t, 64)
	defer idx.Close()

	// Test Add with wrong dimension
	wrongDimVectors := []float32{1.0, 2.0, 3.0} // Length 3, not multiple of 64
	err := idx.Add(wrongDimVectors)
	if err == nil {
		t.Error("Add() with wrong dimension should return error")
	}

	// Test Search with wrong dimension
	wrongDimQuery := []float32{1.0, 2.0, 3.0}
	_, _, err = idx.Search(wrongDimQuery, 5)
	if err == nil {
		t.Error("Search() with wrong dimension should return error")
	}
}

// TestIndex_MultipleSearches tests multiple search queries in one call
func TestIndex_MultipleSearches(t *testing.T) {
	d := 64
	n := 100
	idx := mustCreateIndexFlatL2(t, d)
	defer idx.Close()

	vectors := generateTestVectors(n, d)
	if err := idx.Add(vectors); err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	// Search with multiple queries
	nq := 5 // 5 queries
	queries := generateTestVectors(nq, d)
	k := 10

	distances, labels, err := idx.Search(queries, k)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	expectedLen := nq * k
	if len(distances) != expectedLen || len(labels) != expectedLen {
		t.Errorf("Search() returned %d distances and %d labels, want %d each",
			len(distances), len(labels), expectedLen)
	}

	// Verify results are structured correctly (nq * k)
	for i := 0; i < nq; i++ {
		startIdx := i * k
		endIdx := startIdx + k

		queryDistances := distances[startIdx:endIdx]
		queryLabels := labels[startIdx:endIdx]

		// Distances should be sorted in ascending order
		for j := 1; j < len(queryDistances); j++ {
			if queryDistances[j] < queryDistances[j-1] {
				t.Errorf("Distances not sorted for query %d: %.4f > %.4f",
					i, queryDistances[j-1], queryDistances[j])
			}
		}

		// Labels should be valid indices
		for j, label := range queryLabels {
			if label < 0 || label >= int64(n) {
				t.Errorf("Invalid label at query %d, position %d: %d (should be 0-%d)",
					i, j, label, n-1)
			}
		}
	}
}

// TestIndex_CloseMultipleTimes verifies Close() is idempotent
func TestIndex_CloseMultipleTimes(t *testing.T) {
	idx := mustCreateIndexFlatL2(t, 64)

	// Close multiple times should not crash or error
	if err := idx.Close(); err != nil {
		t.Errorf("First Close() failed: %v", err)
	}

	if err := idx.Close(); err != nil {
		t.Errorf("Second Close() failed: %v", err)
	}
}

// ========================================
// SearchResult Tests
// ========================================

func TestNewSearchResult(t *testing.T) {
	distances := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	labels := []int64{0, 1, 2, 3, 4, 5}
	nq := 2
	k := 3

	sr := NewSearchResult(distances, labels, nq, k)

	if sr == nil {
		t.Fatal("NewSearchResult returned nil")
	}
	if sr.Nq != nq {
		t.Errorf("Nq = %d, want %d", sr.Nq, nq)
	}
	if sr.K != k {
		t.Errorf("K = %d, want %d", sr.K, k)
	}
	if len(sr.Distances) != len(distances) {
		t.Errorf("len(Distances) = %d, want %d", len(sr.Distances), len(distances))
	}
	if len(sr.Labels) != len(labels) {
		t.Errorf("len(Labels) = %d, want %d", len(sr.Labels), len(labels))
	}
}

func TestSearchResult_Get(t *testing.T) {
	// 2 queries, 3 neighbors each
	distances := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	labels := []int64{10, 20, 30, 40, 50, 60}
	sr := NewSearchResult(distances, labels, 2, 3)

	tests := []struct {
		i, j         int
		wantDist     float32
		wantLabel    int64
	}{
		{0, 0, 0.1, 10},
		{0, 1, 0.2, 20},
		{0, 2, 0.3, 30},
		{1, 0, 0.4, 40},
		{1, 1, 0.5, 50},
		{1, 2, 0.6, 60},
	}

	for _, tt := range tests {
		dist, label := sr.Get(tt.i, tt.j)
		if dist != tt.wantDist {
			t.Errorf("Get(%d, %d) distance = %.2f, want %.2f", tt.i, tt.j, dist, tt.wantDist)
		}
		if label != tt.wantLabel {
			t.Errorf("Get(%d, %d) label = %d, want %d", tt.i, tt.j, label, tt.wantLabel)
		}
	}
}

func TestSearchResult_GetNeighbors(t *testing.T) {
	// 2 queries, 3 neighbors each
	distances := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	labels := []int64{10, 20, 30, 40, 50, 60}
	sr := NewSearchResult(distances, labels, 2, 3)

	// Get neighbors for query 0
	dists0, labs0 := sr.GetNeighbors(0)
	if len(dists0) != 3 || len(labs0) != 3 {
		t.Errorf("GetNeighbors(0) returned wrong length: %d distances, %d labels", len(dists0), len(labs0))
	}
	if dists0[0] != 0.1 || dists0[1] != 0.2 || dists0[2] != 0.3 {
		t.Errorf("GetNeighbors(0) distances = %v, want [0.1, 0.2, 0.3]", dists0)
	}
	if labs0[0] != 10 || labs0[1] != 20 || labs0[2] != 30 {
		t.Errorf("GetNeighbors(0) labels = %v, want [10, 20, 30]", labs0)
	}

	// Get neighbors for query 1
	dists1, labs1 := sr.GetNeighbors(1)
	if len(dists1) != 3 || len(labs1) != 3 {
		t.Errorf("GetNeighbors(1) returned wrong length: %d distances, %d labels", len(dists1), len(labs1))
	}
	if dists1[0] != 0.4 || dists1[1] != 0.5 || dists1[2] != 0.6 {
		t.Errorf("GetNeighbors(1) distances = %v, want [0.4, 0.5, 0.6]", dists1)
	}
	if labs1[0] != 40 || labs1[1] != 50 || labs1[2] != 60 {
		t.Errorf("GetNeighbors(1) labels = %v, want [40, 50, 60]", labs1)
	}
}

func TestSearchResult_Integration(t *testing.T) {
	// Test with actual search results from an index
	d := 64
	n := 100
	idx := mustCreateIndexFlatL2(t, d)
	defer idx.Close()

	vectors := generateTestVectors(n, d)
	if err := idx.Add(vectors); err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	nq := 3
	k := 5
	queries := generateTestVectors(nq, d)

	distances, labels, err := idx.Search(queries, k)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	sr := NewSearchResult(distances, labels, nq, k)

	// Verify we can iterate over all results
	for i := 0; i < nq; i++ {
		dists, labs := sr.GetNeighbors(i)
		if len(dists) != k || len(labs) != k {
			t.Errorf("Query %d: expected %d neighbors, got %d distances and %d labels",
				i, k, len(dists), len(labs))
		}

		for j := 0; j < k; j++ {
			dist, label := sr.Get(i, j)
			if dist != dists[j] || label != labs[j] {
				t.Errorf("Get(%d, %d) mismatch with GetNeighbors", i, j)
			}
		}
	}
}

// ========================================
// Helper Functions
// ========================================

func mustCreateIndexFlatL2(t *testing.T, d int) Index {
	t.Helper()
	idx, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("NewIndexFlatL2(%d) failed: %v", d, err)
	}
	return idx
}

func mustCreateIndexLSH(t *testing.T, d, nbits int) Index {
	t.Helper()
	idx, err := NewIndexLSH(d, nbits)
	if err != nil {
		t.Fatalf("NewIndexLSH(%d, %d) failed: %v", d, nbits, err)
	}
	return idx
}

func mustCreateIndexSQ(t *testing.T, d int) Index {
	t.Helper()
	idx, err := NewIndexScalarQuantizer(d, QT_8bit, MetricL2)
	if err != nil {
		t.Fatalf("NewIndexScalarQuantizer(%d) failed: %v", d, err)
	}
	return idx
}

func mustCreateGenericIndex(t *testing.T, d int, description string) Index {
	t.Helper()
	idx, err := IndexFactory(d, description, MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory(%d, %q) failed: %v", d, description, err)
	}
	return idx
}

func generateTestVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}
	return vectors
}
