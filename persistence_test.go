package faiss

import (
	"os"
	"path/filepath"
	"testing"
)

// ========================================
// WriteIndexToFile Tests
// ========================================

func TestWriteIndexToFile_IndexFlat(t *testing.T) {
	idx, _ := NewIndexFlatL2(64)
	defer idx.Close()

	// Add some vectors
	vectors := make([]float32, 64*100)
	for i := range vectors {
		vectors[i] = float32(i % 50)
	}
	idx.Add(vectors)

	// Write to file
	tmpFile := filepath.Join(t.TempDir(), "test_flat.index")
	err := WriteIndexToFile(idx, tmpFile)
	if err != nil {
		t.Fatalf("WriteIndexToFile() failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Error("Index file was not created")
	}
}

func TestWriteIndexToFile_IndexFlatIP(t *testing.T) {
	idx, _ := NewIndexFlatIP(64)
	defer idx.Close()

	vectors := make([]float32, 64*50)
	for i := range vectors {
		vectors[i] = float32(i % 30)
	}
	idx.Add(vectors)

	tmpFile := filepath.Join(t.TempDir(), "test_flat_ip.index")
	err := WriteIndexToFile(idx, tmpFile)
	if err != nil {
		t.Fatalf("WriteIndexToFile() failed: %v", err)
	}
}

func TestWriteIndexToFile_IndexLSH(t *testing.T) {
	// Use factory since direct LSH constructor has known issues
	idx, err := IndexFactory(64, "LSH", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory(LSH) failed: %v", err)
	}
	defer idx.Close()

	vectors := make([]float32, 64*50)
	for i := range vectors {
		vectors[i] = float32(i % 30)
	}
	idx.Add(vectors)

	tmpFile := filepath.Join(t.TempDir(), "test_lsh.index")
	err = WriteIndexToFile(idx, tmpFile)
	if err != nil {
		t.Fatalf("WriteIndexToFile() failed: %v", err)
	}
}

func TestWriteIndexToFile_IndexScalarQuantizer(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(64, QT_8bit, MetricL2)
	defer idx.Close()

	// Train and add
	trainVectors := make([]float32, 64*200)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	idx.Train(trainVectors)
	idx.Add(trainVectors[:64*50])

	tmpFile := filepath.Join(t.TempDir(), "test_sq.index")
	err := WriteIndexToFile(idx, tmpFile)
	if err != nil {
		t.Fatalf("WriteIndexToFile() failed: %v", err)
	}
}

func TestWriteIndexToFile_GenericIndex(t *testing.T) {
	idx, _ := IndexFactory(64, "HNSW32", MetricL2)
	defer idx.Close()

	vectors := make([]float32, 64*100)
	for i := range vectors {
		vectors[i] = float32(i % 50)
	}
	idx.Add(vectors)

	tmpFile := filepath.Join(t.TempDir(), "test_hnsw.index")
	err := WriteIndexToFile(idx, tmpFile)
	if err != nil {
		t.Fatalf("WriteIndexToFile() failed: %v", err)
	}
}

func TestWriteIndexToFile_NilIndex(t *testing.T) {
	err := WriteIndexToFile(nil, "test.index")
	if err == nil {
		t.Error("WriteIndexToFile(nil) should return error")
	}
}

func TestWriteIndexToFile_EmptyIndex(t *testing.T) {
	idx, _ := NewIndexFlatL2(64)
	defer idx.Close()

	// Empty index (no vectors)
	tmpFile := filepath.Join(t.TempDir(), "test_empty.index")
	err := WriteIndexToFile(idx, tmpFile)
	if err != nil {
		t.Fatalf("WriteIndexToFile(empty) failed: %v", err)
	}
}

// ========================================
// ReadIndexFromFile Tests
// ========================================

func TestReadIndexFromFile_IndexFlat(t *testing.T) {
	// Create and save index
	idx, _ := NewIndexFlatL2(64)
	vectors := make([]float32, 64*100)
	for i := range vectors {
		vectors[i] = float32(i % 50)
	}
	idx.Add(vectors)

	tmpFile := filepath.Join(t.TempDir(), "test_flat_read.index")
	WriteIndexToFile(idx, tmpFile)
	idx.Close()

	// Read back
	loadedIdx, err := ReadIndexFromFile(tmpFile)
	if err != nil {
		t.Fatalf("ReadIndexFromFile() failed: %v", err)
	}
	defer loadedIdx.Close()

	// Verify properties
	if loadedIdx.D() != 64 {
		t.Errorf("D() = %d, want 64", loadedIdx.D())
	}
	if loadedIdx.Ntotal() != 100 {
		t.Errorf("Ntotal() = %d, want 100", loadedIdx.Ntotal())
	}
	if !loadedIdx.IsTrained() {
		t.Error("IsTrained() = false, want true")
	}
	if loadedIdx.MetricType() != MetricL2 {
		t.Errorf("MetricType() = %v, want MetricL2", loadedIdx.MetricType())
	}
}

func TestReadIndexFromFile_SearchAfterLoad(t *testing.T) {
	// Create, add vectors, and save
	idx, _ := NewIndexFlatL2(4)
	vectors := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	idx.Add(vectors)

	tmpFile := filepath.Join(t.TempDir(), "test_search.index")
	WriteIndexToFile(idx, tmpFile)
	idx.Close()

	// Load and search
	loadedIdx, _ := ReadIndexFromFile(tmpFile)
	defer loadedIdx.Close()

	query := []float32{1, 0, 0, 0}
	distances, indices, err := loadedIdx.Search(query, 2)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	if len(distances) != 2 || len(indices) != 2 {
		t.Errorf("Search() returned %d results, want 2", len(distances))
	}

	// First result should be index 0 (exact match)
	if indices[0] != 0 {
		t.Errorf("First result index = %d, want 0", indices[0])
	}
}

func TestReadIndexFromFile_FileNotFound(t *testing.T) {
	_, err := ReadIndexFromFile("nonexistent_file.index")
	if err == nil {
		t.Error("ReadIndexFromFile(nonexistent) should return error")
	}
}

func TestReadIndexFromFile_InnerProduct(t *testing.T) {
	// Create IP index
	idx, _ := NewIndexFlatIP(64)
	vectors := make([]float32, 64*50)
	for i := range vectors {
		vectors[i] = float32(i % 30)
	}
	idx.Add(vectors)

	tmpFile := filepath.Join(t.TempDir(), "test_ip.index")
	WriteIndexToFile(idx, tmpFile)
	idx.Close()

	// Load and verify
	loadedIdx, _ := ReadIndexFromFile(tmpFile)
	defer loadedIdx.Close()

	if loadedIdx.MetricType() != MetricInnerProduct {
		t.Errorf("MetricType() = %v, want MetricInnerProduct", loadedIdx.MetricType())
	}
}

// ========================================
// Roundtrip Tests
// ========================================

func TestPersistence_Roundtrip_HNSW(t *testing.T) {
	// Create HNSW index via factory
	idx, _ := IndexFactory(64, "HNSW16", MetricL2)

	vectors := make([]float32, 64*500)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}
	idx.Add(vectors)

	originalNtotal := idx.Ntotal()

	tmpFile := filepath.Join(t.TempDir(), "test_hnsw_roundtrip.index")
	WriteIndexToFile(idx, tmpFile)
	idx.Close()

	// Load and verify
	loadedIdx, err := ReadIndexFromFile(tmpFile)
	if err != nil {
		t.Fatalf("ReadIndexFromFile() failed: %v", err)
	}
	defer loadedIdx.Close()

	if loadedIdx.Ntotal() != originalNtotal {
		t.Errorf("Ntotal() = %d, want %d", loadedIdx.Ntotal(), originalNtotal)
	}
}

func TestPersistence_Roundtrip_IVF(t *testing.T) {
	// Create IVF index via factory
	idx, _ := IndexFactory(64, "IVF10,Flat", MetricL2)

	// Train
	trainVectors := make([]float32, 64*500)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 100)
	}
	idx.Train(trainVectors)

	// Add
	idx.Add(trainVectors[:64*100])
	originalNtotal := idx.Ntotal()

	tmpFile := filepath.Join(t.TempDir(), "test_ivf_roundtrip.index")
	WriteIndexToFile(idx, tmpFile)
	idx.Close()

	// Load and verify
	loadedIdx, err := ReadIndexFromFile(tmpFile)
	if err != nil {
		t.Fatalf("ReadIndexFromFile() failed: %v", err)
	}
	defer loadedIdx.Close()

	if loadedIdx.Ntotal() != originalNtotal {
		t.Errorf("Ntotal() = %d, want %d", loadedIdx.Ntotal(), originalNtotal)
	}
	if !loadedIdx.IsTrained() {
		t.Error("Loaded IVF index should be trained")
	}
}

func TestPersistence_Roundtrip_PQ(t *testing.T) {
	// Create PQ index via factory
	idx, _ := IndexFactory(128, "PQ8", MetricL2)

	// Train with enough vectors
	trainVectors := make([]float32, 128*10000)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 100)
	}
	idx.Train(trainVectors)

	// Add
	idx.Add(trainVectors[:128*100])

	tmpFile := filepath.Join(t.TempDir(), "test_pq_roundtrip.index")
	WriteIndexToFile(idx, tmpFile)
	idx.Close()

	// Load and verify
	loadedIdx, err := ReadIndexFromFile(tmpFile)
	if err != nil {
		t.Fatalf("ReadIndexFromFile() failed: %v", err)
	}
	defer loadedIdx.Close()

	if loadedIdx.Ntotal() != 100 {
		t.Errorf("Ntotal() = %d, want 100", loadedIdx.Ntotal())
	}
}

// ========================================
// Search Consistency Tests
// ========================================

func TestPersistence_SearchConsistency(t *testing.T) {
	// Create index and add vectors
	idx, _ := NewIndexFlatL2(16)

	n := 50
	vectors := make([]float32, 16*n)
	for i := range vectors {
		vectors[i] = float32(i % 20)
	}
	idx.Add(vectors)

	// Search before save
	query := vectors[:16]
	distBefore, idxBefore, _ := idx.Search(query, 5)

	// Save and load
	tmpFile := filepath.Join(t.TempDir(), "test_consistency.index")
	WriteIndexToFile(idx, tmpFile)
	idx.Close()

	loadedIdx, _ := ReadIndexFromFile(tmpFile)
	defer loadedIdx.Close()

	// Search after load
	distAfter, idxAfter, _ := loadedIdx.Search(query, 5)

	// Compare results
	if len(distBefore) != len(distAfter) {
		t.Errorf("Distance slice lengths differ: %d vs %d", len(distBefore), len(distAfter))
	}

	for i := range distBefore {
		if distBefore[i] != distAfter[i] {
			t.Errorf("Distance[%d] differs: %f vs %f", i, distBefore[i], distAfter[i])
		}
		if idxBefore[i] != idxAfter[i] {
			t.Errorf("Index[%d] differs: %d vs %d", i, idxBefore[i], idxAfter[i])
		}
	}
}

// ========================================
// Edge Cases
// ========================================

func TestPersistence_LargeIndex(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large index test in short mode")
	}

	idx, _ := NewIndexFlatL2(128)
	defer idx.Close()

	// Add 10000 vectors
	n := 10000
	vectors := make([]float32, 128*n)
	for i := range vectors {
		vectors[i] = float32(i % 1000)
	}
	idx.Add(vectors)

	tmpFile := filepath.Join(t.TempDir(), "test_large.index")
	err := WriteIndexToFile(idx, tmpFile)
	if err != nil {
		t.Fatalf("WriteIndexToFile() failed: %v", err)
	}

	// Check file size is reasonable (should be at least n * d * 4 bytes)
	info, _ := os.Stat(tmpFile)
	minSize := int64(n * 128 * 4)
	if info.Size() < minSize {
		t.Errorf("File size %d seems too small (min expected: %d)", info.Size(), minSize)
	}
}

func TestPersistence_OverwriteFile(t *testing.T) {
	idx1, _ := NewIndexFlatL2(32)
	idx1.Add(make([]float32, 32*10))

	idx2, _ := NewIndexFlatL2(64)
	idx2.Add(make([]float32, 64*20))
	defer idx2.Close()

	tmpFile := filepath.Join(t.TempDir(), "test_overwrite.index")

	// Write first index
	WriteIndexToFile(idx1, tmpFile)
	idx1.Close()

	// Overwrite with second index
	err := WriteIndexToFile(idx2, tmpFile)
	if err != nil {
		t.Fatalf("Overwrite failed: %v", err)
	}

	// Load should give second index
	loadedIdx, _ := ReadIndexFromFile(tmpFile)
	defer loadedIdx.Close()

	if loadedIdx.D() != 64 {
		t.Errorf("D() = %d, want 64 (second index)", loadedIdx.D())
	}
	if loadedIdx.Ntotal() != 20 {
		t.Errorf("Ntotal() = %d, want 20 (second index)", loadedIdx.Ntotal())
	}
}

// ========================================
// Benchmark Tests
// ========================================

func BenchmarkWriteIndexToFile(b *testing.B) {
	idx, _ := NewIndexFlatL2(128)
	vectors := make([]float32, 128*1000)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}
	idx.Add(vectors)
	defer idx.Close()

	tmpDir := b.TempDir()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tmpFile := filepath.Join(tmpDir, "bench.index")
		WriteIndexToFile(idx, tmpFile)
	}
}

func BenchmarkReadIndexFromFile(b *testing.B) {
	idx, _ := NewIndexFlatL2(128)
	vectors := make([]float32, 128*1000)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}
	idx.Add(vectors)

	tmpFile := filepath.Join(b.TempDir(), "bench.index")
	WriteIndexToFile(idx, tmpFile)
	idx.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadedIdx, _ := ReadIndexFromFile(tmpFile)
		loadedIdx.Close()
	}
}
