package faiss

import (
	"os"
	"testing"
)

func TestWriteReadIndex_FlatL2(t *testing.T) {
	d := 64
	nb := 100

	// Create index and add vectors
	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := make([]float32, d*nb)
	for i := range vectors {
		vectors[i] = float32(i) * 0.1
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Write to file
	filename := "/tmp/test_index_flatl2.faiss"
	defer os.Remove(filename)

	if err := WriteIndex(index, filename); err != nil {
		t.Fatalf("WriteIndex failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		t.Fatalf("Index file was not created: %s", filename)
	}

	// Read index back
	loadedIndex, err := ReadIndex(filename)
	if err != nil {
		t.Fatalf("ReadIndex failed: %v", err)
	}
	defer loadedIndex.Close()

	// Verify properties
	if loadedIndex.D() != d {
		t.Errorf("Dimension mismatch: expected %d, got %d", d, loadedIndex.D())
	}

	if loadedIndex.Ntotal() != int64(nb) {
		t.Errorf("Vector count mismatch: expected %d, got %d", nb, loadedIndex.Ntotal())
	}

	if loadedIndex.MetricType() != MetricL2 {
		t.Errorf("Metric type mismatch: expected %v, got %v", MetricL2, loadedIndex.MetricType())
	}

	if !loadedIndex.IsTrained() {
		t.Error("Loaded index should be trained")
	}
}

func TestWriteReadIndex_FlatIP(t *testing.T) {
	d := 32
	nb := 50

	// Create IP index
	index, err := NewIndexFlatIP(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := make([]float32, d*nb)
	for i := range vectors {
		vectors[i] = float32(i%10) / 10.0
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Write and read
	filename := "/tmp/test_index_flatip.faiss"
	defer os.Remove(filename)

	if err := WriteIndex(index, filename); err != nil {
		t.Fatalf("WriteIndex failed: %v", err)
	}

	loadedIndex, err := ReadIndex(filename)
	if err != nil {
		t.Fatalf("ReadIndex failed: %v", err)
	}
	defer loadedIndex.Close()

	// Verify metric type
	if loadedIndex.MetricType() != MetricInnerProduct {
		t.Errorf("Expected MetricInnerProduct, got %v", loadedIndex.MetricType())
	}

	if loadedIndex.Ntotal() != int64(nb) {
		t.Errorf("Vector count mismatch: expected %d, got %d", nb, loadedIndex.Ntotal())
	}
}

func TestWriteReadIndex_IVFFlat(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping IVF test in short mode")
	}

	d := 64
	nlist := 10
	nb := 1000

	// Create quantizer
	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	// Create IVF index
	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer index.Close()

	// Train
	trainVectors := make([]float32, d*nb)
	for i := range trainVectors {
		trainVectors[i] = float32(i) * 0.01
	}

	if err := index.Train(trainVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if err := index.Add(trainVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Write and read
	filename := "/tmp/test_index_ivf.faiss"
	defer os.Remove(filename)

	if err := WriteIndex(index, filename); err != nil {
		t.Fatalf("WriteIndex failed: %v", err)
	}

	loadedIndex, err := ReadIndex(filename)
	if err != nil {
		t.Fatalf("ReadIndex failed: %v", err)
	}
	defer loadedIndex.Close()

	// Verify properties
	if loadedIndex.D() != d {
		t.Errorf("Dimension mismatch: expected %d, got %d", d, loadedIndex.D())
	}

	if loadedIndex.Ntotal() != int64(nb) {
		t.Errorf("Vector count mismatch: expected %d, got %d", nb, loadedIndex.Ntotal())
	}

	if !loadedIndex.IsTrained() {
		t.Error("Loaded IVF index should be trained")
	}
}

func TestWriteIndex_NilIndex(t *testing.T) {
	err := WriteIndex(nil, "/tmp/test.faiss")
	if err == nil {
		t.Error("Expected error for nil index")
	}
	if err != nil && err.Error() != "faiss: index cannot be nil" {
		t.Errorf("Unexpected error message: %v", err)
	}
}

func TestReadIndex_NonExistentFile(t *testing.T) {
	_, err := ReadIndex("/tmp/nonexistent_index_file.faiss")
	if err == nil {
		t.Error("Expected error for non-existent file")
	}
	if err != nil && !os.IsNotExist(err) {
		// Check if error message contains expected text
		if err.Error() == "" || len(err.Error()) == 0 {
			t.Errorf("Error message should not be empty")
		}
	}
}

func TestWriteReadIndex_SearchConsistency(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping search consistency test in short mode")
	}

	d := 32
	nb := 100
	nq := 5
	k := 10

	// Create and populate index
	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := make([]float32, d*nb)
	for i := range vectors {
		vectors[i] = float32(i%20) * 0.5
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Perform search before serialization
	queries := make([]float32, d*nq)
	for i := range queries {
		queries[i] = float32(i%10) * 0.3
	}

	dist1, idx1, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Save and load
	filename := "/tmp/test_search_consistency.faiss"
	defer os.Remove(filename)

	if err := WriteIndex(index, filename); err != nil {
		t.Fatalf("WriteIndex failed: %v", err)
	}

	loadedIndex, err := ReadIndex(filename)
	if err != nil {
		t.Fatalf("ReadIndex failed: %v", err)
	}
	defer loadedIndex.Close()

	// Perform search after deserialization
	dist2, idx2, err := loadedIndex.Search(queries, k)
	if err != nil {
		t.Fatalf("Search on loaded index failed: %v", err)
	}

	// Verify results match
	if len(dist1) != len(dist2) {
		t.Errorf("Distance array length mismatch: %d vs %d", len(dist1), len(dist2))
	}

	if len(idx1) != len(idx2) {
		t.Errorf("Index array length mismatch: %d vs %d", len(idx1), len(idx2))
	}

	// Check first few results for consistency
	for i := 0; i < min(10, len(dist1)); i++ {
		if dist1[i] != dist2[i] {
			t.Errorf("Distance mismatch at position %d: %f vs %f", i, dist1[i], dist2[i])
		}
		if idx1[i] != idx2[i] {
			t.Errorf("Index mismatch at position %d: %d vs %d", i, idx1[i], idx2[i])
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
