package faiss

import (
	"testing"
)

func TestNewIndexIVFFlat(t *testing.T) {
	d := 64
	nlist := 10

	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
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

	if index.IsTrained() {
		t.Error("New IVF index should not be trained")
	}

	if index.Nlist() != nlist {
		t.Errorf("Expected nlist=%d, got %d", nlist, index.Nlist())
	}

	if index.Nprobe() != 1 {
		t.Errorf("Expected default nprobe=1, got %d", index.Nprobe())
	}
}

func TestNewIndexIVFFlat_InnerProduct(t *testing.T) {
	d := 32
	nlist := 5

	quantizer, err := NewIndexFlatIP(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricInnerProduct)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer index.Close()

	if index.MetricType() != MetricInnerProduct {
		t.Errorf("Expected MetricInnerProduct, got %v", index.MetricType())
	}
}

func TestNewIndexIVFFlat_InvalidParameters(t *testing.T) {
	d := 64

	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	tests := []struct {
		name   string
		d      int
		nlist  int
		metric MetricType
	}{
		{"zero dimension", 0, 10, MetricL2},
		{"negative dimension", -1, 10, MetricL2},
		{"zero nlist", 64, 0, MetricL2},
		{"negative nlist", 64, -1, MetricL2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewIndexIVFFlat(quantizer, tt.d, tt.nlist, tt.metric)
			if err == nil {
				t.Error("Expected error for invalid parameters")
			}
		})
	}
}

func TestIndexIVFFlat_SetNprobe(t *testing.T) {
	d := 64
	nlist := 20

	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer index.Close()

	// Test valid nprobe values
	validNprobes := []int{1, 5, 10, 15, 20}
	for _, nprobe := range validNprobes {
		if err := index.SetNprobe(nprobe); err != nil {
			t.Errorf("SetNprobe(%d) failed: %v", nprobe, err)
		}
		if index.Nprobe() != nprobe {
			t.Errorf("Expected nprobe=%d, got %d", nprobe, index.Nprobe())
		}
	}

	// Test invalid nprobe values
	invalidNprobes := []int{0, -1, nlist + 1}
	for _, nprobe := range invalidNprobes {
		if err := index.SetNprobe(nprobe); err == nil {
			t.Errorf("Expected error for nprobe=%d", nprobe)
		}
	}
}

func TestIndexIVFFlat_TrainAddSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping IVF train/add/search test in short mode")
	}

	d := 32
	nlist := 10
	nTrain := 500
	nAdd := 1000
	nQuery := 10
	k := 5

	// Create index
	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer index.Close()

	// Generate training data
	trainVectors := make([]float32, d*nTrain)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 100)
	}

	// Train
	if err := index.Train(trainVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if !index.IsTrained() {
		t.Error("Index should be trained after Train()")
	}

	// Add vectors
	addVectors := make([]float32, d*nAdd)
	for i := range addVectors {
		addVectors[i] = float32(i % 50)
	}

	if err := index.Add(addVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if index.Ntotal() != int64(nAdd) {
		t.Errorf("Expected %d vectors, got %d", nAdd, index.Ntotal())
	}

	// Set nprobe for search
	if err := index.SetNprobe(5); err != nil {
		t.Fatalf("SetNprobe failed: %v", err)
	}

	// Search
	queries := make([]float32, d*nQuery)
	for i := range queries {
		queries[i] = float32(i % 30)
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
		if idx < 0 || idx >= int64(nAdd) {
			t.Errorf("Invalid index at position %d: %d", i, idx)
		}
	}
}

func TestIndexIVFFlat_TrainBeforeAdd(t *testing.T) {
	d := 32
	nlist := 5

	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer index.Close()

	// Try to add without training
	vectors := make([]float32, d*10)
	err = index.Add(vectors)
	if err != ErrNotTrained {
		t.Errorf("Expected ErrNotTrained, got %v", err)
	}
}

func TestIndexIVFFlat_InvalidVectorSize(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping invalid vector test in short mode")
	}

	d := 32
	nlist := 5

	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer index.Close()

	// Train with valid data
	trainVectors := make([]float32, d*100)
	if err := index.Train(trainVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

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

func TestIndexIVFFlat_Reset(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping reset test in short mode")
	}

	d := 32
	nlist := 5
	n := 100

	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer index.Close()

	// Train and add
	vectors := make([]float32, d*n)
	if err := index.Train(vectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}
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

	// Should still be trained after reset
	if !index.IsTrained() {
		t.Error("Index should remain trained after reset")
	}
}

func TestIndexIVFFlat_Assign(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping assign test in short mode")
	}

	d := 32
	nlist := 5
	n := 100

	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer index.Close()

	// Train
	trainVectors := make([]float32, d*n)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 20)
	}

	if err := index.Train(trainVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Assign vectors to clusters
	assignments, err := index.Assign(trainVectors)
	if err != nil {
		t.Fatalf("Assign failed: %v", err)
	}

	if len(assignments) != n {
		t.Errorf("Expected %d assignments, got %d", n, len(assignments))
	}

	// Verify assignments are valid cluster IDs
	for i, clusterID := range assignments {
		if clusterID < 0 || clusterID >= int64(nlist) {
			t.Errorf("Invalid cluster ID at position %d: %d (should be 0-%d)", i, clusterID, nlist-1)
		}
	}
}
