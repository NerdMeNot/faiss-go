package faiss

import (
	"testing"
)

func TestNewIndexHNSWFlat(t *testing.T) {
	d := 64
	M := 32

	index, err := NewIndexHNSWFlat(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	if index.D() != d {
		t.Errorf("Expected dimension %d, got %d", d, index.D())
	}

	if index.MetricType() != MetricL2 {
		t.Errorf("Expected MetricL2, got %v", index.MetricType())
	}

	if index.Ntotal() != 0 {
		t.Errorf("New index should have 0 vectors, got %d", index.Ntotal())
	}

	if !index.IsTrained() {
		t.Error("HNSW index should always be trained")
	}

	if index.GetM() != M {
		t.Errorf("Expected M=%d, got %d", M, index.GetM())
	}

	// Check default values
	if index.GetEfConstruction() != 40 {
		t.Errorf("Expected default efConstruction=40, got %d", index.GetEfConstruction())
	}

	if index.GetEfSearch() != 16 {
		t.Errorf("Expected default efSearch=16, got %d", index.GetEfSearch())
	}
}

func TestNewIndexHNSWFlat_InnerProduct(t *testing.T) {
	d := 32
	M := 16

	index, err := NewIndexHNSWFlat(d, M, MetricInnerProduct)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	if index.MetricType() != MetricInnerProduct {
		t.Errorf("Expected MetricInnerProduct, got %v", index.MetricType())
	}
}

func TestNewIndexHNSWFlat_InvalidParameters(t *testing.T) {
	tests := []struct {
		name   string
		d      int
		M      int
		metric MetricType
	}{
		{"zero dimension", 0, 32, MetricL2},
		{"negative dimension", -1, 32, MetricL2},
		{"zero M", 64, 0, MetricL2},
		{"negative M", 64, -1, MetricL2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewIndexHNSWFlat(tt.d, tt.M, tt.metric)
			if err == nil {
				t.Error("Expected error for invalid parameters")
			}
		})
	}
}

func TestIndexHNSW_SetEfSearch(t *testing.T) {
	d := 64
	M := 32

	index, err := NewIndexHNSWFlat(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	// Test valid efSearch values
	validEfSearch := []int{16, 32, 64, 128, 256}
	for _, ef := range validEfSearch {
		if err := index.SetEfSearch(ef); err != nil {
			t.Errorf("SetEfSearch(%d) failed: %v", ef, err)
		}
		if index.GetEfSearch() != ef {
			t.Errorf("Expected efSearch=%d, got %d", ef, index.GetEfSearch())
		}
	}

	// Test invalid efSearch values
	invalidEfSearch := []int{0, -1}
	for _, ef := range invalidEfSearch {
		if err := index.SetEfSearch(ef); err == nil {
			t.Errorf("Expected error for efSearch=%d", ef)
		}
	}
}

func TestIndexHNSW_SetEfConstruction(t *testing.T) {
	d := 64
	M := 32

	index, err := NewIndexHNSWFlat(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	// Test valid efConstruction values
	validEfConstruction := []int{20, 40, 80, 160}
	for _, ef := range validEfConstruction {
		if err := index.SetEfConstruction(ef); err != nil {
			t.Errorf("SetEfConstruction(%d) failed: %v", ef, err)
		}
		if index.GetEfConstruction() != ef {
			t.Errorf("Expected efConstruction=%d, got %d", ef, index.GetEfConstruction())
		}
	}

	// Test invalid efConstruction values
	invalidEfConstruction := []int{0, -1}
	for _, ef := range invalidEfConstruction {
		if err := index.SetEfConstruction(ef); err == nil {
			t.Errorf("Expected error for efConstruction=%d", ef)
		}
	}
}

func TestIndexHNSW_AddSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping HNSW add/search test in short mode")
	}

	d := 32
	M := 16
	n := 1000
	nQuery := 10
	k := 5

	// Create index
	index, err := NewIndexHNSWFlat(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	// Set parameters
	if err := index.SetEfConstruction(40); err != nil {
		t.Fatalf("SetEfConstruction failed: %v", err)
	}
	if err := index.SetEfSearch(32); err != nil {
		t.Fatalf("SetEfSearch failed: %v", err)
	}

	// Add vectors
	vectors := make([]float32, d*n)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if index.Ntotal() != int64(n) {
		t.Errorf("Expected %d vectors, got %d", n, index.Ntotal())
	}

	// Search
	queries := make([]float32, d*nQuery)
	for i := range queries {
		queries[i] = float32(i % 50)
	}

	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != nQuery*k {
		t.Errorf("Expected %d distances, got %d", nQuery*k, len(distances))
	}

	if len(indices) != nQuery*k {
		t.Errorf("Expected %d indices, got %d", nQuery*k, len(indices))
	}

	// Verify indices are valid
	for i, idx := range indices {
		if idx < 0 || idx >= int64(n) {
			t.Errorf("Invalid index at position %d: %d", i, idx)
		}
	}

	// Verify distances are non-negative for L2
	for i, dist := range distances {
		if dist < 0 {
			t.Errorf("Negative distance at position %d: %f", i, dist)
		}
	}
}

func TestIndexHNSW_Train(t *testing.T) {
	// HNSW doesn't require training, but Train() should not error
	d := 32
	M := 16

	index, err := NewIndexHNSWFlat(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	vectors := make([]float32, d*100)
	if err := index.Train(vectors); err != nil {
		t.Errorf("Train should not error for HNSW: %v", err)
	}
}

func TestIndexHNSW_InvalidVectorSize(t *testing.T) {
	d := 32
	M := 16

	index, err := NewIndexHNSWFlat(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	// Try to add with invalid size
	invalidVectors := make([]float32, d+1)
	err = index.Add(invalidVectors)
	if err != ErrInvalidVectors {
		t.Errorf("Expected ErrInvalidVectors, got %v", err)
	}

	// Try to search with invalid size
	invalidQuery := make([]float32, d+1)
	_, _, err = index.Search(invalidQuery, 5)
	if err != ErrInvalidVectors {
		t.Errorf("Expected ErrInvalidVectors, got %v", err)
	}
}

func TestIndexHNSW_Reset(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping reset test in short mode")
	}

	d := 32
	M := 16
	n := 100

	index, err := NewIndexHNSWFlat(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	// Add vectors
	vectors := make([]float32, d*n)
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

func TestIndexHNSW_DifferentMValues(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping M values test in short mode")
	}

	d := 32
	mValues := []int{8, 16, 32, 48}

	for _, M := range mValues {
		t.Run("M="+string(rune(M)), func(t *testing.T) {
			index, err := NewIndexHNSWFlat(d, M, MetricL2)
			if err != nil {
				t.Fatalf("Failed to create HNSW index with M=%d: %v", M, err)
			}
			defer index.Close()

			if index.GetM() != M {
				t.Errorf("Expected M=%d, got %d", M, index.GetM())
			}

			// Add a few vectors
			vectors := make([]float32, d*100)
			for i := range vectors {
				vectors[i] = float32(i)
			}

			if err := index.Add(vectors); err != nil {
				t.Fatalf("Add failed for M=%d: %v", M, err)
			}

			if index.Ntotal() != 100 {
				t.Errorf("Expected 100 vectors, got %d", index.Ntotal())
			}
		})
	}
}

func TestIndexHNSW_EfSearchQualityTradeoff(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping efSearch quality test in short mode")
	}

	d := 32
	M := 32
	n := 500
	k := 10

	// Create index
	index, err := NewIndexHNSWFlat(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	// Add vectors
	vectors := make([]float32, d*n)
	for i := range vectors {
		vectors[i] = float32(i % 20)
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Test with different efSearch values
	query := make([]float32, d)
	for i := range query {
		query[i] = 5.0
	}

	efSearchValues := []int{16, 64, 256}
	for _, ef := range efSearchValues {
		if err := index.SetEfSearch(ef); err != nil {
			t.Fatalf("SetEfSearch(%d) failed: %v", ef, err)
		}

		distances, indices, err := index.Search(query, k)
		if err != nil {
			t.Fatalf("Search with efSearch=%d failed: %v", ef, err)
		}

		if len(distances) != k || len(indices) != k {
			t.Errorf("efSearch=%d: expected %d results, got distances=%d, indices=%d",
				ef, k, len(distances), len(indices))
		}
	}
}
