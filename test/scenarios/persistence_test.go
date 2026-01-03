package scenarios

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	faiss "github.com/NerdMeNot/faiss-go"
)

// TestIndexPersistence_BasicSaveLoad tests saving and loading an index from disk
func TestIndexPersistence_BasicSaveLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping persistence test in short mode")
	}

	t.Log("=== Test: Index Persistence - Basic Save/Load ===")

	// Setup: Create temp directory for test files
	tempDir := t.TempDir()
	indexPath := filepath.Join(tempDir, "test_index.faiss")

	// 1. Create and populate an index
	d := 128
	index, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Add 1000 vectors with known patterns
	nVectors := 1000
	vectors := make([]float32, nVectors*d)
	for i := 0; i < nVectors; i++ {
		// Each vector has a unique "ID" encoded in first dimension
		vectors[i*d] = float32(i)
		// Fill rest with pattern
		for j := 1; j < d; j++ {
			vectors[i*d+j] = float32(i%10) * 0.1
		}
	}

	err = index.Add(vectors)
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}
	if index.Ntotal() != int64(nVectors) {
		t.Fatalf("Expected %d vectors, got %d", nVectors, index.Ntotal())
	}

	// 2. Perform a search to get baseline results
	queryVec := make([]float32, d)
	queryVec[0] = 500.0 // Should match vector 500
	for j := 1; j < d; j++ {
		queryVec[j] = 0.0 // Match the pattern
	}

	k := 10
	distances1, indices1, err := index.Search(queryVec, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(indices1) != k {
		t.Fatalf("Expected %d results, got %d", k, len(indices1))
	}
	t.Logf("Before save - Top result: index=%d, distance=%.4f", indices1[0], distances1[0])

	// 3. Save index to disk
	t.Logf("Saving index to: %s", indexPath)
	err = faiss.WriteIndexToFile(index, indexPath)
	if err != nil {
		t.Fatalf("Failed to save index: %v", err)
	}

	// Verify file was created
	stat, err := os.Stat(indexPath)
	if err != nil {
		t.Fatalf("Index file should exist: %v", err)
	}
	if stat.Size() == 0 {
		t.Fatal("Index file should not be empty")
	}
	t.Logf("Index file size: %d bytes", stat.Size())

	// 4. Close original index
	err = index.Close()
	if err != nil {
		t.Fatalf("Failed to close index: %v", err)
	}

	// 5. Load index from disk
	t.Logf("Loading index from: %s", indexPath)
	loadedIndex, err := faiss.ReadIndexFromFile(indexPath)
	if err != nil {
		t.Fatalf("Failed to load index: %v", err)
	}
	defer loadedIndex.Close()

	// 6. Verify loaded index properties
	if loadedIndex.D() != d {
		t.Errorf("Loaded index dimension: expected %d, got %d", d, loadedIndex.D())
	}
	if loadedIndex.Ntotal() != int64(nVectors) {
		t.Errorf("Loaded index vector count: expected %d, got %d", nVectors, loadedIndex.Ntotal())
	}
	t.Logf("Loaded index: d=%d, ntotal=%d", loadedIndex.D(), loadedIndex.Ntotal())

	// 7. Search on loaded index - should get same results
	distances2, indices2, err := loadedIndex.Search(queryVec, k)
	if err != nil {
		t.Fatalf("Search on loaded index failed: %v", err)
	}
	if len(indices2) != k {
		t.Fatalf("Expected %d results, got %d", k, len(indices2))
	}
	t.Logf("After load - Top result: index=%d, distance=%.4f", indices2[0], distances2[0])

	// 8. Verify results match
	for i := 0; i < k; i++ {
		if indices1[i] != indices2[i] {
			t.Errorf("Result %d: indices don't match (before=%d, after=%d)", i, indices1[i], indices2[i])
		}
		if math.Abs(float64(distances1[i]-distances2[i])) > 0.001 {
			t.Errorf("Result %d: distances don't match (before=%.4f, after=%.4f)", i, distances1[i], distances2[i])
		}
	}

	t.Log("✅ Index persistence verified - save/load works correctly")
}

// TestIndexPersistence_HNSW tests persistence with approximate index (HNSW)
func TestIndexPersistence_HNSW(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping HNSW persistence test in short mode")
	}

	t.Log("=== Test: Index Persistence - HNSW Index ===")

	tempDir := t.TempDir()
	indexPath := filepath.Join(tempDir, "hnsw_index.faiss")

	// 1. Create HNSW index
	d := 256
	index, err := faiss.NewIndexHNSWFlat(d, 32, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	defer index.Close()

	// 2. Add vectors
	nVectors := 10000
	vectors := make([]float32, nVectors*d)
	for i := 0; i < nVectors*d; i++ {
		vectors[i] = float32(i%100) * 0.01
	}

	err = index.Add(vectors)
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}
	t.Logf("Added %d vectors to HNSW index", nVectors)

	// 3. Search before save
	query := vectors[0:d] // Use first vector as query
	k := 20
	distances1, indices1, err := index.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	pos1 := findInSlice(0, indices1)
	t.Logf("Before save - Recall: vector 0 found at position %d", pos1)

	// 4. Save
	err = faiss.WriteIndexToFile(index, indexPath)
	if err != nil {
		t.Fatalf("Failed to save HNSW index: %v", err)
	}

	stat, _ := os.Stat(indexPath)
	t.Logf("HNSW index file size: %d bytes (%.2f MB)", stat.Size(), float64(stat.Size())/1024/1024)

	index.Close()

	// 5. Load
	loadedIndex, err := faiss.ReadIndexFromFile(indexPath)
	if err != nil {
		t.Fatalf("Failed to load HNSW index: %v", err)
	}
	defer loadedIndex.Close()

	if loadedIndex.Ntotal() != int64(nVectors) {
		t.Errorf("Vector count mismatch: expected %d, got %d", nVectors, loadedIndex.Ntotal())
	}

	// 6. Search after load
	distances2, indices2, err := loadedIndex.Search(query, k)
	if err != nil {
		t.Fatalf("Search on loaded index failed: %v", err)
	}
	pos2 := findInSlice(0, indices2)
	t.Logf("After load - Recall: vector 0 found at position %d", pos2)

	// 7. Results should be identical for HNSW (deterministic after build)
	for i := 0; i < k; i++ {
		if indices1[i] != indices2[i] {
			t.Errorf("HNSW result %d mismatch: before=%d, after=%d", i, indices1[i], indices2[i])
		}
		if math.Abs(float64(distances1[i]-distances2[i])) > 0.001 {
			t.Errorf("HNSW distance %d mismatch: before=%.4f, after=%.4f", i, distances1[i], distances2[i])
		}
	}

	t.Log("✅ HNSW index persistence verified")
}

// TestIndexPersistence_IVFPQTrained tests persistence with trained IVF index
func TestIndexPersistence_IVFPQTrained(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping IVFPQ persistence test in short mode")
	}

	t.Log("=== Test: Index Persistence - Trained IVFPQ ===")

	tempDir := t.TempDir()
	indexPath := filepath.Join(tempDir, "ivfpq_index.faiss")

	// 1. Create IVFPQ index
	d := 128
	nlist := 100
	M := 8
	nbits := 8

	// Use factory to create quantizer
	quantizer, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := faiss.NewIndexIVFPQ(quantizer, d, nlist, M, nbits)
	if err != nil {
		t.Fatalf("Failed to create IVFPQ index: %v", err)
	}
	defer index.Close()

	// 2. Generate training data
	nTrain := 10000
	trainVectors := make([]float32, nTrain*d)
	for i := 0; i < nTrain*d; i++ {
		trainVectors[i] = float32(i%100) * 0.1
	}

	// 3. Train the index
	t.Log("Training IVFPQ index...")
	err = index.Train(trainVectors)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}
	if !index.IsTrained() {
		t.Fatal("Index should be trained")
	}

	// 4. Add vectors
	err = index.Add(trainVectors)
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}
	t.Logf("Added %d vectors to trained IVFPQ index", nTrain)

	// 5. Set nprobe and search
	err = index.SetNprobe(10)
	if err != nil {
		t.Fatalf("SetNprobe failed: %v", err)
	}

	query := trainVectors[0:d]
	k := 10
	distances1, indices1, err := index.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	t.Logf("Before save - Top result: idx=%d, dist=%.4f", indices1[0], distances1[0])

	// 6. Save trained index
	err = faiss.WriteIndexToFile(index, indexPath)
	if err != nil {
		t.Fatalf("Failed to save IVFPQ index: %v", err)
	}

	stat, _ := os.Stat(indexPath)
	compressionRatio := float64(nTrain*d*4) / float64(stat.Size())
	t.Logf("IVFPQ index: %.2f MB (compression ratio: %.1fx)",
		float64(stat.Size())/1024/1024, compressionRatio)

	index.Close()

	// 7. Load trained index
	loadedIndex, err := faiss.ReadIndexFromFile(indexPath)
	if err != nil {
		t.Fatalf("Failed to load IVFPQ index: %v", err)
	}
	defer loadedIndex.Close()

	// 8. Verify trained state is preserved
	if !loadedIndex.IsTrained() {
		t.Error("Loaded index should be trained")
	}
	if loadedIndex.Ntotal() != int64(nTrain) {
		t.Errorf("Vector count mismatch: expected %d, got %d", nTrain, loadedIndex.Ntotal())
	}

	// 9. Set nprobe on loaded index and search
	err = loadedIndex.SetNprobe(10)
	if err != nil {
		t.Fatalf("SetNprobe on loaded index failed: %v", err)
	}

	distances2, indices2, err := loadedIndex.Search(query, k)
	if err != nil {
		t.Fatalf("Search on loaded index failed: %v", err)
	}
	t.Logf("After load - Top result: idx=%d, dist=%.4f", indices2[0], distances2[0])

	// 10. Results should match (IVFPQ with same nprobe is deterministic)
	for i := 0; i < k; i++ {
		if indices1[i] != indices2[i] {
			t.Errorf("Result %d mismatch: before=%d, after=%d", i, indices1[i], indices2[i])
		}
		if math.Abs(float64(distances1[i]-distances2[i])) > 0.1 {
			t.Logf("Warning: Distance %d differs slightly: before=%.4f, after=%.4f", i, distances1[i], distances2[i])
		}
	}

	t.Log("✅ Trained IVFPQ index persistence verified")
}

// Helper function to find position of value in slice
func findInSlice(val int64, slice []int64) int {
	for i, v := range slice {
		if v == val {
			return i
		}
	}
	return -1
}
