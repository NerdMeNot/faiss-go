package faiss

import (
	"testing"
)

// ========================================
// IndexIDMap Creation Tests
// ========================================

func TestNewIndexIDMap_WithFlatIndex(t *testing.T) {
	base, err := NewIndexFlatL2(64)
	if err != nil {
		t.Fatalf("Failed to create base index: %v", err)
	}
	defer base.Close()

	idmap, err := NewIndexIDMap(base)
	if err != nil {
		t.Fatalf("Failed to create IndexIDMap: %v", err)
	}
	defer idmap.Close()

	if idmap.D() != 64 {
		t.Errorf("D() = %d, want 64", idmap.D())
	}
	if idmap.Ntotal() != 0 {
		t.Errorf("Ntotal() = %d, want 0", idmap.Ntotal())
	}
	if idmap.MetricType() != MetricL2 {
		t.Errorf("MetricType() = %v, want MetricL2", idmap.MetricType())
	}
	if !idmap.IsTrained() {
		t.Error("IsTrained() = false, want true (Flat is always trained)")
	}
}

func TestNewIndexIDMap_WithLSHIndex(t *testing.T) {
	// Note: LSH direct constructor has known issues, skip if it fails
	base, err := NewIndexLSH(64, 128)
	if err != nil {
		t.Skipf("LSH direct constructor not working (known issue): %v", err)
		return
	}
	defer base.Close()

	idmap, err := NewIndexIDMap(base)
	if err != nil {
		t.Fatalf("Failed to create IndexIDMap with LSH: %v", err)
	}
	defer idmap.Close()

	if idmap.D() != 64 {
		t.Errorf("D() = %d, want 64", idmap.D())
	}
}

func TestNewIndexIDMap_NilIndex(t *testing.T) {
	_, err := NewIndexIDMap(nil)
	if err == nil {
		t.Error("NewIndexIDMap(nil) should return error")
	}
}

// ========================================
// IndexIDMap AddWithIDs Tests
// ========================================

func TestIndexIDMap_AddWithIDs(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	ids := []int64{100, 200, 300}

	err := idmap.AddWithIDs(vectors, ids)
	if err != nil {
		t.Fatalf("AddWithIDs() failed: %v", err)
	}

	if idmap.Ntotal() != 3 {
		t.Errorf("Ntotal() = %d, want 3", idmap.Ntotal())
	}
}

func TestIndexIDMap_AddWithIDs_Empty(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	err := idmap.AddWithIDs([]float32{}, []int64{})
	if err != nil {
		t.Errorf("AddWithIDs(empty) should not error: %v", err)
	}

	if idmap.Ntotal() != 0 {
		t.Errorf("Ntotal() = %d, want 0", idmap.Ntotal())
	}
}

func TestIndexIDMap_AddWithIDs_InvalidDimension(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	vectors := []float32{1, 2, 3} // Not a multiple of 4
	ids := []int64{100}

	err := idmap.AddWithIDs(vectors, ids)
	if err != ErrInvalidVectors {
		t.Errorf("AddWithIDs() with invalid dimension: got %v, want ErrInvalidVectors", err)
	}
}

func TestIndexIDMap_AddWithIDs_MismatchedCounts(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	vectors := []float32{1, 2, 3, 4, 5, 6, 7, 8} // 2 vectors
	ids := []int64{100, 200, 300}                 // 3 IDs

	err := idmap.AddWithIDs(vectors, ids)
	if err == nil {
		t.Error("AddWithIDs() with mismatched counts should error")
	}
}

// ========================================
// IndexIDMap Add Tests (auto-generated IDs)
// ========================================

func TestIndexIDMap_Add(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}

	err := idmap.Add(vectors)
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	if idmap.Ntotal() != 2 {
		t.Errorf("Ntotal() = %d, want 2", idmap.Ntotal())
	}
}

func TestIndexIDMap_Add_InvalidDimension(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	err := idmap.Add([]float32{1, 2, 3})
	if err != ErrInvalidVectors {
		t.Errorf("Add() with invalid dimension: got %v, want ErrInvalidVectors", err)
	}
}

// ========================================
// IndexIDMap Search Tests
// ========================================

func TestIndexIDMap_Search(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	vectors := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	ids := []int64{100, 200, 300}
	idmap.AddWithIDs(vectors, ids)

	// Search for first vector
	query := []float32{1, 0, 0, 0}
	distances, indices, err := idmap.Search(query, 2)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	if len(distances) != 2 || len(indices) != 2 {
		t.Errorf("Search() returned %d distances, %d indices, want 2 each",
			len(distances), len(indices))
	}

	// First result should be ID 100 (exact match)
	if indices[0] != 100 {
		t.Errorf("First result ID = %d, want 100", indices[0])
	}
	if distances[0] > 0.001 {
		t.Errorf("First result distance = %f, want ~0", distances[0])
	}
}

func TestIndexIDMap_Search_Empty(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	distances, indices, err := idmap.Search([]float32{}, 5)
	if err != nil {
		t.Errorf("Search(empty) should not error: %v", err)
	}
	if len(distances) != 0 || len(indices) != 0 {
		t.Error("Search(empty) should return empty results")
	}
}

func TestIndexIDMap_Search_InvalidDimension(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	_, _, err := idmap.Search([]float32{1, 2, 3}, 5)
	if err != ErrInvalidVectors {
		t.Errorf("Search() with invalid dimension: got %v, want ErrInvalidVectors", err)
	}
}

func TestIndexIDMap_Search_InvalidK(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	idmap.Add([]float32{1, 2, 3, 4})

	_, _, err := idmap.Search([]float32{1, 2, 3, 4}, 0)
	if err != ErrInvalidK {
		t.Errorf("Search() with k=0: got %v, want ErrInvalidK", err)
	}

	_, _, err = idmap.Search([]float32{1, 2, 3, 4}, -1)
	if err != ErrInvalidK {
		t.Errorf("Search() with k=-1: got %v, want ErrInvalidK", err)
	}
}

func TestIndexIDMap_Search_MultipleQueries(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	vectors := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	ids := []int64{100, 200, 300}
	idmap.AddWithIDs(vectors, ids)

	// 2 queries
	queries := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}
	k := 2

	distances, indices, err := idmap.Search(queries, k)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	expectedLen := 2 * k
	if len(distances) != expectedLen || len(indices) != expectedLen {
		t.Errorf("Search() returned %d distances, %d indices, want %d each",
			len(distances), len(indices), expectedLen)
	}
}

// ========================================
// IndexIDMap Reset Tests
// ========================================

func TestIndexIDMap_Reset(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	idmap.AddWithIDs([]float32{1, 2, 3, 4}, []int64{100})

	if idmap.Ntotal() != 1 {
		t.Errorf("Ntotal() before reset = %d, want 1", idmap.Ntotal())
	}

	err := idmap.Reset()
	if err != nil {
		t.Fatalf("Reset() failed: %v", err)
	}

	if idmap.Ntotal() != 0 {
		t.Errorf("Ntotal() after reset = %d, want 0", idmap.Ntotal())
	}
}

// ========================================
// IndexIDMap Close Tests
// ========================================

func TestIndexIDMap_Close(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)

	err := idmap.Close()
	if err != nil {
		t.Errorf("First Close() failed: %v", err)
	}

	// Second close should be safe
	err = idmap.Close()
	if err != nil {
		t.Errorf("Second Close() failed: %v", err)
	}
}

// ========================================
// IndexIDMap RemoveIDs Tests
// ========================================

func TestIndexIDMap_RemoveIDs_NotSupported(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	err := idmap.RemoveIDs([]int64{100})
	if err == nil {
		t.Error("RemoveIDs() should return error (not supported)")
	}
}

// ========================================
// IndexIDMap Delegation Tests
// ========================================

func TestIndexIDMap_SetNprobe(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	// Flat index doesn't support nprobe, should error
	err := idmap.SetNprobe(10)
	if err == nil {
		t.Error("SetNprobe() on Flat-based IDMap should error")
	}
}

func TestIndexIDMap_SetEfSearch(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	// Flat index doesn't support efSearch, should error
	err := idmap.SetEfSearch(10)
	if err == nil {
		t.Error("SetEfSearch() on Flat-based IDMap should error")
	}
}

func TestIndexIDMap_Train(t *testing.T) {
	base, _ := NewIndexFlatL2(4)
	defer base.Close()

	idmap, _ := NewIndexIDMap(base)
	defer idmap.Close()

	// Flat index doesn't need training, should be no-op
	err := idmap.Train([]float32{1, 2, 3, 4})
	if err != nil {
		t.Errorf("Train() on Flat-based IDMap failed: %v", err)
	}
}
