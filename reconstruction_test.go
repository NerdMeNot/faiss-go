package faiss

import (
	"math"
	"testing"
)

// ========================================
// IndexFlat Reconstruct Tests
// ========================================

func TestIndexFlat_Reconstruct(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	idx.Add(vectors)

	// Reconstruct first vector
	recons, err := idx.Reconstruct(0)
	if err != nil {
		t.Fatalf("Reconstruct(0) failed: %v", err)
	}

	if len(recons) != 4 {
		t.Errorf("Reconstructed vector length = %d, want 4", len(recons))
	}

	// Check values match
	expected := vectors[:4]
	for i := range expected {
		if math.Abs(float64(recons[i]-expected[i])) > 1e-6 {
			t.Errorf("Reconstructed[%d] = %f, want %f", i, recons[i], expected[i])
		}
	}
}

func TestIndexFlat_Reconstruct_AllVectors(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	idx.Add(vectors)

	for key := int64(0); key < 3; key++ {
		recons, err := idx.Reconstruct(key)
		if err != nil {
			t.Fatalf("Reconstruct(%d) failed: %v", key, err)
		}

		expected := vectors[key*4 : (key+1)*4]
		for i := range expected {
			if math.Abs(float64(recons[i]-expected[i])) > 1e-6 {
				t.Errorf("Vector %d: Reconstructed[%d] = %f, want %f",
					key, i, recons[i], expected[i])
			}
		}
	}
}

func TestIndexFlat_Reconstruct_InvalidKey(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	idx.Add([]float32{1, 2, 3, 4, 5, 6, 7, 8})

	// Negative key
	_, err := idx.Reconstruct(-1)
	if err == nil {
		t.Error("Reconstruct(-1) should return error")
	}

	// Key out of range
	_, err = idx.Reconstruct(10)
	if err == nil {
		t.Error("Reconstruct(10) should return error (only 2 vectors)")
	}
}

func TestIndexFlat_Reconstruct_EmptyIndex(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	_, err := idx.Reconstruct(0)
	if err == nil {
		t.Error("Reconstruct(0) on empty index should return error")
	}
}

// ========================================
// IndexFlat ReconstructN Tests
// ========================================

func TestIndexFlat_ReconstructN(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	idx.Add(vectors)

	// Reconstruct vectors 0 and 1
	recons, err := idx.ReconstructN(0, 2)
	if err != nil {
		t.Fatalf("ReconstructN(0, 2) failed: %v", err)
	}

	if len(recons) != 8 {
		t.Errorf("Reconstructed length = %d, want 8", len(recons))
	}

	// Check values match
	expected := vectors[:8]
	for i := range expected {
		if math.Abs(float64(recons[i]-expected[i])) > 1e-6 {
			t.Errorf("Reconstructed[%d] = %f, want %f", i, recons[i], expected[i])
		}
	}
}

func TestIndexFlat_ReconstructN_AllVectors(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	idx.Add(vectors)

	recons, err := idx.ReconstructN(0, 3)
	if err != nil {
		t.Fatalf("ReconstructN(0, 3) failed: %v", err)
	}

	if len(recons) != 12 {
		t.Errorf("Reconstructed length = %d, want 12", len(recons))
	}

	for i := range vectors {
		if math.Abs(float64(recons[i]-vectors[i])) > 1e-6 {
			t.Errorf("Reconstructed[%d] = %f, want %f", i, recons[i], vectors[i])
		}
	}
}

func TestIndexFlat_ReconstructN_Middle(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	idx.Add(vectors)

	// Reconstruct vector 1 only
	recons, err := idx.ReconstructN(1, 1)
	if err != nil {
		t.Fatalf("ReconstructN(1, 1) failed: %v", err)
	}

	expected := vectors[4:8]
	for i := range expected {
		if math.Abs(float64(recons[i]-expected[i])) > 1e-6 {
			t.Errorf("Reconstructed[%d] = %f, want %f", i, recons[i], expected[i])
		}
	}
}

func TestIndexFlat_ReconstructN_Zero(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	idx.Add([]float32{1, 2, 3, 4})

	recons, err := idx.ReconstructN(0, 0)
	if err != nil {
		t.Errorf("ReconstructN(0, 0) failed: %v", err)
	}
	if len(recons) != 0 {
		t.Errorf("ReconstructN(0, 0) should return empty, got len=%d", len(recons))
	}
}

func TestIndexFlat_ReconstructN_InvalidRange(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	idx.Add([]float32{1, 2, 3, 4, 5, 6, 7, 8})

	// Negative start
	_, err := idx.ReconstructN(-1, 1)
	if err == nil {
		t.Error("ReconstructN(-1, 1) should return error")
	}

	// Range extends beyond ntotal
	_, err = idx.ReconstructN(1, 5)
	if err == nil {
		t.Error("ReconstructN(1, 5) should return error (only 2 vectors)")
	}
}

// ========================================
// IndexFlat ReconstructBatch Tests
// ========================================

func TestIndexFlat_ReconstructBatch(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	idx.Add(vectors)

	// Reconstruct vectors 0 and 2 (not contiguous)
	keys := []int64{0, 2}
	recons, err := idx.ReconstructBatch(keys)
	if err != nil {
		t.Fatalf("ReconstructBatch() failed: %v", err)
	}

	if len(recons) != 8 {
		t.Errorf("Reconstructed length = %d, want 8", len(recons))
	}

	// Check first vector (key 0)
	expected0 := vectors[:4]
	for i := range expected0 {
		if math.Abs(float64(recons[i]-expected0[i])) > 1e-6 {
			t.Errorf("Key 0: Reconstructed[%d] = %f, want %f", i, recons[i], expected0[i])
		}
	}

	// Check second vector (key 2)
	expected2 := vectors[8:12]
	for i := range expected2 {
		if math.Abs(float64(recons[4+i]-expected2[i])) > 1e-6 {
			t.Errorf("Key 2: Reconstructed[%d] = %f, want %f", i, recons[4+i], expected2[i])
		}
	}
}

func TestIndexFlat_ReconstructBatch_Empty(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	idx.Add([]float32{1, 2, 3, 4})

	recons, err := idx.ReconstructBatch([]int64{})
	if err != nil {
		t.Errorf("ReconstructBatch(empty) failed: %v", err)
	}
	if len(recons) != 0 {
		t.Errorf("ReconstructBatch(empty) should return empty, got len=%d", len(recons))
	}
}

func TestIndexFlat_ReconstructBatch_InvalidKey(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	idx.Add([]float32{1, 2, 3, 4, 5, 6, 7, 8})

	_, err := idx.ReconstructBatch([]int64{0, 10})
	if err == nil {
		t.Error("ReconstructBatch with invalid key should return error")
	}

	_, err = idx.ReconstructBatch([]int64{-1})
	if err == nil {
		t.Error("ReconstructBatch with negative key should return error")
	}
}

func TestIndexFlat_ReconstructBatch_Duplicates(t *testing.T) {
	idx, _ := NewIndexFlatL2(4)
	defer idx.Close()

	vectors := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	idx.Add(vectors)

	// Reconstruct same key twice
	keys := []int64{0, 0, 1}
	recons, err := idx.ReconstructBatch(keys)
	if err != nil {
		t.Fatalf("ReconstructBatch() failed: %v", err)
	}

	if len(recons) != 12 {
		t.Errorf("Reconstructed length = %d, want 12 (3 vectors)", len(recons))
	}

	// First two should be identical
	for i := 0; i < 4; i++ {
		if recons[i] != recons[4+i] {
			t.Errorf("Duplicate keys should produce identical vectors")
			break
		}
	}
}

// ========================================
// IndexIVFFlat Reconstruct Tests
// ========================================

func TestIndexIVFFlat_Reconstruct(t *testing.T) {
	// Create via factory (direct constructor has known issues)
	idx, err := IndexFactory(4, "IVF5,Flat", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer idx.Close()

	// Train
	trainVectors := make([]float32, 4*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	idx.Train(trainVectors)

	// Add vectors
	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	idx.Add(vectors)

	// Reconstruct is available on GenericIndex via factory
	// but we need to test the IVFFlat type specifically
	// For now, just verify no panic
	t.Log("IVFFlat reconstruction test - using factory-created index")
}

// ========================================
// Reconstruction Accuracy Tests
// ========================================

func TestReconstruct_Accuracy(t *testing.T) {
	idx, _ := NewIndexFlatL2(128)
	defer idx.Close()

	// Generate random vectors
	n := 100
	vectors := make([]float32, 128*n)
	for i := range vectors {
		vectors[i] = float32(i%1000) / 100.0
	}
	idx.Add(vectors)

	// Reconstruct all and check accuracy
	for i := int64(0); i < int64(n); i++ {
		recons, err := idx.Reconstruct(i)
		if err != nil {
			t.Fatalf("Reconstruct(%d) failed: %v", i, err)
		}

		expected := vectors[i*128 : (i+1)*128]
		var mse float32
		for j := range expected {
			diff := recons[j] - expected[j]
			mse += diff * diff
		}
		mse /= 128

		if mse > 1e-10 {
			t.Errorf("Reconstruction MSE for vector %d = %e, want ~0", i, mse)
		}
	}
}

// ========================================
// Benchmark Tests
// ========================================

func BenchmarkIndexFlat_Reconstruct(b *testing.B) {
	idx, _ := NewIndexFlatL2(128)
	defer idx.Close()

	vectors := make([]float32, 128*1000)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}
	idx.Add(vectors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Reconstruct(int64(i % 1000))
	}
}

func BenchmarkIndexFlat_ReconstructN(b *testing.B) {
	idx, _ := NewIndexFlatL2(128)
	defer idx.Close()

	vectors := make([]float32, 128*1000)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}
	idx.Add(vectors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.ReconstructN(0, 100)
	}
}

func BenchmarkIndexFlat_ReconstructBatch(b *testing.B) {
	idx, _ := NewIndexFlatL2(128)
	defer idx.Close()

	vectors := make([]float32, 128*1000)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}
	idx.Add(vectors)

	keys := make([]int64, 100)
	for i := range keys {
		keys[i] = int64(i * 10)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.ReconstructBatch(keys)
	}
}
