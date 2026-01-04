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

func TestIndexHNSW_AddSearch(t *testing.T) {
	d := 32
	M := 16
	n := 500
	nQuery := 10
	k := 5

	// Create index
	index, err := NewIndexHNSWFlat(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

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

func TestIndexHNSW_Reset(t *testing.T) {
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

func TestNewIndexHNSW_Alias(t *testing.T) {
	// NewIndexHNSW should be equivalent to NewIndexHNSWFlat
	d := 32
	M := 16

	index, err := NewIndexHNSW(d, M, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index via alias: %v", err)
	}
	defer index.Close()

	if index.D() != d {
		t.Errorf("Expected dimension %d, got %d", d, index.D())
	}

	if !index.IsTrained() {
		t.Error("HNSW index should always be trained")
	}
}

func TestIndexHNSW_DifferentMValues(t *testing.T) {
	d := 32
	mValues := []int{8, 16, 32, 48}

	for _, M := range mValues {
		t.Run("M="+string(rune('0'+M/10))+string(rune('0'+M%10)), func(t *testing.T) {
			index, err := NewIndexHNSWFlat(d, M, MetricL2)
			if err != nil {
				t.Fatalf("Failed to create HNSW index with M=%d: %v", M, err)
			}
			defer index.Close()

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
