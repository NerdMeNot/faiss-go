package faiss

import (
	"testing"
)

func TestNewIndexBinaryFlat(t *testing.T) {
	validDimensions := []int{8, 16, 32, 64, 128, 256}

	for _, d := range validDimensions {
		t.Run("dimension="+string(rune(d)), func(t *testing.T) {
			index, err := NewIndexBinaryFlat(d)
			if err != nil {
				t.Fatalf("Failed to create binary index: %v", err)
			}
			defer index.Close()

			if index.D() != d {
				t.Errorf("Expected dimension %d, got %d", d, index.D())
			}

			if !index.IsTrained() {
				t.Error("Binary flat index should always be trained")
			}

			if index.Ntotal() != 0 {
				t.Errorf("New index should have 0 vectors, got %d", index.Ntotal())
			}
		})
	}
}

func TestNewIndexBinaryFlat_InvalidDimension(t *testing.T) {
	invalidDimensions := []int{0, -1, 7, 15, 17, 33}

	for _, d := range invalidDimensions {
		t.Run("dimension="+string(rune(d)), func(t *testing.T) {
			_, err := NewIndexBinaryFlat(d)
			if err == nil {
				t.Errorf("Expected error for dimension %d (not multiple of 8)", d)
			}
		})
	}
}

func TestIndexBinaryFlat_AddSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping binary add/search test in short mode")
	}

	d := 256 // 256 bits = 32 bytes
	n := 100
	nq := 10
	k := 5

	index, err := NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create binary index: %v", err)
	}
	defer index.Close()

	// Add binary vectors
	bytesPerVector := d / 8
	vectors := make([]uint8, n*bytesPerVector)
	for i := range vectors {
		vectors[i] = uint8(i % 256)
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if index.Ntotal() != int64(n) {
		t.Errorf("Expected %d vectors, got %d", n, index.Ntotal())
	}

	// Search
	queries := make([]uint8, nq*bytesPerVector)
	for i := range queries {
		queries[i] = uint8((i * 2) % 256)
	}

	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != nq*k {
		t.Errorf("Expected %d distances, got %d", nq*k, len(distances))
	}

	if len(indices) != nq*k {
		t.Errorf("Expected %d indices, got %d", nq*k, len(indices))
	}

	// Verify indices are valid
	for i, idx := range indices {
		if idx < 0 || idx >= int64(n) {
			t.Errorf("Invalid index at position %d: %d", i, idx)
		}
	}

	// Verify Hamming distances are in valid range [0, d]
	for i, dist := range distances {
		if dist < 0 || dist > int32(d) {
			t.Errorf("Invalid Hamming distance at position %d: %d (should be 0-%d)", i, dist, d)
		}
	}
}

func TestIndexBinaryFlat_Train(t *testing.T) {
	// Binary flat index doesn't require training, but Train() should not error
	d := 128
	index, err := NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create binary index: %v", err)
	}
	defer index.Close()

	bytesPerVector := d / 8
	vectors := make([]uint8, 100*bytesPerVector)

	if err := index.Train(vectors); err != nil {
		t.Errorf("Train should not error for binary flat index: %v", err)
	}
}

func TestIndexBinaryFlat_InvalidVectorSize(t *testing.T) {
	d := 256
	bytesPerVector := d / 8

	index, err := NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create binary index: %v", err)
	}
	defer index.Close()

	// Try to add with invalid size (not multiple of bytesPerVector)
	invalidVectors := make([]uint8, bytesPerVector+1)
	err = index.Add(invalidVectors)
	if err == nil {
		t.Error("Expected error for invalid vector size")
	}

	// Try to search with invalid size
	invalidQuery := make([]uint8, bytesPerVector+1)
	_, _, err = index.Search(invalidQuery, 5)
	if err == nil {
		t.Error("Expected error for invalid query size")
	}
}

func TestIndexBinaryFlat_Reset(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping reset test in short mode")
	}

	d := 128
	n := 50
	bytesPerVector := d / 8

	index, err := NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create binary index: %v", err)
	}
	defer index.Close()

	// Add vectors
	vectors := make([]uint8, n*bytesPerVector)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if index.Ntotal() != int64(n) {
		t.Errorf("Expected %d vectors before reset, got %d", n, index.Ntotal())
	}

	// Reset
	if err := index.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}

	if index.Ntotal() != 0 {
		t.Errorf("Expected 0 vectors after reset, got %d", index.Ntotal())
	}
}

func TestIndexBinaryFlat_EmptyAdd(t *testing.T) {
	d := 64
	index, err := NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create binary index: %v", err)
	}
	defer index.Close()

	// Adding empty vectors should not error
	if err := index.Add([]uint8{}); err != nil {
		t.Errorf("Adding empty vectors should not error, got %v", err)
	}

	if index.Ntotal() != 0 {
		t.Errorf("Expected 0 vectors after empty add, got %d", index.Ntotal())
	}
}

func TestIndexBinaryFlat_HammingDistance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Hamming distance test in short mode")
	}

	d := 64 // 64 bits = 8 bytes
	bytesPerVector := d / 8

	index, err := NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create binary index: %v", err)
	}
	defer index.Close()

	// Add two vectors with known Hamming distance
	vectors := make([]uint8, 2*bytesPerVector)

	// Vector 0: all zeros
	for i := 0; i < bytesPerVector; i++ {
		vectors[i] = 0x00
	}

	// Vector 1: all ones
	for i := 0; i < bytesPerVector; i++ {
		vectors[bytesPerVector+i] = 0xFF
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search with all zeros (should match vector 0 perfectly)
	query := make([]uint8, bytesPerVector)
	for i := range query {
		query[i] = 0x00
	}

	distances, indices, err := index.Search(query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// First result should be vector 0 with distance 0
	if indices[0] != 0 {
		t.Errorf("Expected nearest neighbor to be vector 0, got %d", indices[0])
	}
	if distances[0] != 0 {
		t.Errorf("Expected distance 0 to identical vector, got %d", distances[0])
	}

	// Second result should be vector 1 with distance d (all bits different)
	if indices[1] != 1 {
		t.Errorf("Expected second nearest neighbor to be vector 1, got %d", indices[1])
	}
	if distances[1] != int32(d) {
		t.Errorf("Expected distance %d to fully different vector, got %d", d, distances[1])
	}
}

func TestIndexBinaryFlat_MultipleQueries(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping multiple queries test in short mode")
	}

	d := 128
	n := 200
	nq := 20
	k := 10
	bytesPerVector := d / 8

	index, err := NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create binary index: %v", err)
	}
	defer index.Close()

	// Add vectors
	vectors := make([]uint8, n*bytesPerVector)
	for i := range vectors {
		vectors[i] = uint8(i % 128)
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Multiple queries
	queries := make([]uint8, nq*bytesPerVector)
	for i := range queries {
		queries[i] = uint8((i * 3) % 128)
	}

	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Verify result dimensions
	if len(distances) != nq*k {
		t.Errorf("Expected %d distances, got %d", nq*k, len(distances))
	}

	if len(indices) != nq*k {
		t.Errorf("Expected %d indices, got %d", nq*k, len(indices))
	}

	// Verify each query has k results
	for q := 0; q < nq; q++ {
		queryResults := k
		actualResults := 0
		for i := 0; i < k; i++ {
			if indices[q*k+i] >= 0 {
				actualResults++
			}
		}
		if actualResults != queryResults {
			t.Errorf("Query %d: expected %d results, got %d", q, queryResults, actualResults)
		}
	}
}

func TestIndexBinaryFlat_DifferentDimensions(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping different dimensions test in short mode")
	}

	dimensions := []int{64, 128, 256, 512}

	for _, d := range dimensions {
		t.Run("dimension="+string(rune(d)), func(t *testing.T) {
			index, err := NewIndexBinaryFlat(d)
			if err != nil {
				t.Fatalf("Failed to create binary index with d=%d: %v", d, err)
			}
			defer index.Close()

			bytesPerVector := d / 8
			vectors := make([]uint8, 10*bytesPerVector)
			for i := range vectors {
				vectors[i] = uint8(i % 256)
			}

			if err := index.Add(vectors); err != nil {
				t.Fatalf("Add failed for d=%d: %v", d, err)
			}

			if index.Ntotal() != 10 {
				t.Errorf("Expected 10 vectors, got %d", index.Ntotal())
			}

			// Quick search test
			query := make([]uint8, bytesPerVector)
			distances, indices, err := index.Search(query, 5)
			if err != nil {
				t.Fatalf("Search failed for d=%d: %v", d, err)
			}

			if len(distances) != 5 || len(indices) != 5 {
				t.Errorf("Expected 5 results, got distances=%d, indices=%d", len(distances), len(indices))
			}
		})
	}
}
