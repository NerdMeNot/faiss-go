package faiss

import (
	"testing"
)

// ========================================
// IVF Index Tests (via Factory)
// ========================================

func TestIVFFlat_Factory(t *testing.T) {
	d := 64
	nb := 1000
	nq := 5

	// Create IVF index via factory
	index, err := IndexFactory(d, "IVF10,Flat", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer index.Close()

	// Verify properties
	if index.D() != d {
		t.Errorf("D() = %d, want %d", index.D(), d)
	}
	if index.MetricType() != MetricL2 {
		t.Errorf("MetricType() = %v, want MetricL2", index.MetricType())
	}

	// IVF requires training
	if index.IsTrained() {
		t.Error("IVF index should not be trained initially")
	}

	// Train the index
	trainingVectors := generateTestVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if !index.IsTrained() {
		t.Error("IVF index should be trained after Train()")
	}

	// Add vectors
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	if index.Ntotal() != int64(nb) {
		t.Errorf("Ntotal() = %d, want %d", index.Ntotal(), nb)
	}

	// Search
	queries := generateTestVectors(nq, d)
	k := 10
	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	expectedLen := nq * k
	if len(distances) != expectedLen || len(indices) != expectedLen {
		t.Errorf("Search() returned %d distances and %d indices, want %d each",
			len(distances), len(indices), expectedLen)
	}

	// Verify distances are sorted
	for i := 0; i < nq; i++ {
		for j := 1; j < k; j++ {
			idx := i*k + j
			if distances[idx] < distances[idx-1] {
				t.Errorf("Distances not sorted at query %d", i)
				break
			}
		}
	}

	// Reset
	if err := index.Reset(); err != nil {
		t.Fatalf("Reset() failed: %v", err)
	}

	if index.Ntotal() != 0 {
		t.Errorf("Ntotal() = %d after Reset, want 0", index.Ntotal())
	}
}

func TestIVFPQ_Factory(t *testing.T) {
	d := 128
	// PQ8 uses 256 centroids per sub-quantizer, needs ~39*256 = 9984 training vectors
	// IVF16 needs ~39*16 = 624 training vectors
	// Use 10000 to satisfy both requirements
	nb := 10000

	// Create IVFPQ index via factory
	// IVF16,PQ8 = 16 clusters, 8-bit product quantization (smaller nlist for faster test)
	index, err := IndexFactory(d, "IVF16,PQ8", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer index.Close()

	// Verify properties
	if index.D() != d {
		t.Errorf("D() = %d, want %d", index.D(), d)
	}

	// IVFPQ requires training
	if index.IsTrained() {
		t.Error("IVFPQ index should not be trained initially")
	}

	// Train with sufficient data for PQ (needs ~10k vectors)
	trainingVectors := generateTestVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if !index.IsTrained() {
		t.Error("IVFPQ index should be trained after Train()")
	}

	// Add vectors
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	// Search
	queries := generateTestVectors(5, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Errorf("Search() returned %d distances and %d indices, want 50 each",
			len(distances), len(indices))
	}
}

func TestIVF_DifferentNlist(t *testing.T) {
	d := 64
	// Need ~39*nlist training vectors. For nlist=50, need 1950 vectors
	nb := 2000

	tests := []struct {
		name  string
		nlist int
		desc  string
	}{
		{"IVF_small", 2, "IVF2,Flat"},
		{"IVF_medium", 10, "IVF10,Flat"},
		{"IVF_large", 50, "IVF50,Flat"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			index, err := IndexFactory(d, tt.desc, MetricL2)
			if err != nil {
				t.Fatalf("IndexFactory(%q) failed: %v", tt.desc, err)
			}
			defer index.Close()

			// Train and add
			vectors := generateTestVectors(nb, d)
			if err := index.Train(vectors); err != nil {
				t.Fatalf("Train() failed: %v", err)
			}

			if err := index.Add(vectors); err != nil {
				t.Fatalf("Add() failed: %v", err)
			}

			// Search should work
			queries := generateTestVectors(3, d)
			distances, indices, err := index.Search(queries, 5)
			if err != nil {
				t.Fatalf("Search() failed: %v", err)
			}

			if len(distances) != 15 || len(indices) != 15 {
				t.Errorf("Search() returned wrong number of results")
			}
		})
	}
}

func TestIVF_InsufficientTrainingData(t *testing.T) {
	d := 64

	index, err := IndexFactory(d, "IVF100,Flat", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer index.Close()

	// Try to train with too few vectors (need at least nlist vectors)
	insufficientVectors := generateTestVectors(50, d) // Only 50 vectors for 100 clusters

	// This should fail or succeed with a warning
	err = index.Train(insufficientVectors)
	// FAISS may handle this gracefully or error - just log the result
	if err != nil {
		t.Logf("Train() with insufficient data returned error (expected): %v", err)
	} else {
		t.Logf("Train() with insufficient data succeeded (FAISS may have reduced nlist)")
	}
}

func TestIVF_TrainBeforeAdd(t *testing.T) {
	d := 64
	nb := 500

	index, err := IndexFactory(d, "IVF10,Flat", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer index.Close()

	vectors := generateTestVectors(nb, d)

	// Try to add without training
	err = index.Add(vectors)
	if err == nil {
		t.Error("Add() should fail when index is not trained")
	}

	// Now train and add
	if err := index.Train(vectors); err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add() after training failed: %v", err)
	}
}

func TestIVF_SearchBeforeAdd(t *testing.T) {
	d := 64
	nb := 500

	index, err := IndexFactory(d, "IVF10,Flat", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer index.Close()

	vectors := generateTestVectors(nb, d)
	if err := index.Train(vectors); err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	// Search on empty index
	queries := generateTestVectors(1, d)
	distances, _, err := index.Search(queries, 5)

	// Should succeed but return no results or special values
	if err != nil {
		t.Logf("Search() on empty index returned error: %v", err)
	} else {
		t.Logf("Search() on empty index returned %d results", len(distances))
		// Distances might be inf or very large
	}
}

func TestIVF_MetricTypes(t *testing.T) {
	d := 64
	nb := 500

	tests := []struct {
		name   string
		metric MetricType
	}{
		{"L2", MetricL2},
		{"InnerProduct", MetricInnerProduct},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			index, err := IndexFactory(d, "IVF10,Flat", tt.metric)
			if err != nil {
				t.Fatalf("IndexFactory failed: %v", err)
			}
			defer index.Close()

			if index.MetricType() != tt.metric {
				t.Errorf("MetricType() = %v, want %v", index.MetricType(), tt.metric)
			}

			// Train and add
			vectors := generateTestVectors(nb, d)
			if err := index.Train(vectors); err != nil {
				t.Fatalf("Train() failed: %v", err)
			}

			if err := index.Add(vectors); err != nil {
				t.Fatalf("Add() failed: %v", err)
			}

			// Search should work
			queries := generateTestVectors(1, d)
			_, _, err = index.Search(queries, 5)
			if err != nil {
				t.Fatalf("Search() failed: %v", err)
			}
		})
	}
}

func TestIVF_AddInBatches(t *testing.T) {
	d := 64
	batchSize := 100
	numBatches := 5
	totalVectors := batchSize * numBatches

	index, err := IndexFactory(d, "IVF10,Flat", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer index.Close()

	// Train with sufficient data (IVF10 needs ~390 vectors)
	trainingData := generateTestVectors(400, d)
	if err := index.Train(trainingData); err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	// Add vectors in batches
	for i := 0; i < numBatches; i++ {
		batch := generateTestVectors(batchSize, d)
		if err := index.Add(batch); err != nil {
			t.Fatalf("Add() batch %d failed: %v", i, err)
		}

		expectedCount := int64((i + 1) * batchSize)
		if index.Ntotal() != expectedCount {
			t.Errorf("After batch %d: Ntotal() = %d, want %d",
				i, index.Ntotal(), expectedCount)
		}
	}

	if index.Ntotal() != int64(totalVectors) {
		t.Errorf("Final Ntotal() = %d, want %d", index.Ntotal(), totalVectors)
	}
}

func TestIVF_LargeK(t *testing.T) {
	d := 64
	// Need ~39*5 = 195 training vectors for IVF5
	nb := 200
	k := 400 // k larger than nb

	index, err := IndexFactory(d, "IVF5,Flat", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer index.Close()

	vectors := generateTestVectors(nb, d)
	if err := index.Train(vectors); err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	// Search with k > nb
	queries := generateTestVectors(1, d)
	distances, _, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	// FAISS returns k results even when k > nb, padding with invalid entries
	// Just verify Search doesn't crash
	if len(distances) != k {
		t.Errorf("Search() returned %d results, expected %d (FAISS pads results)", len(distances), k)
	}
}

func TestIVFSQ_Factory(t *testing.T) {
	d := 64
	nb := 500

	// Create IVF with scalar quantization
	index, err := IndexFactory(d, "IVF10,SQ8", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer index.Close()

	// Train
	vectors := generateTestVectors(nb, d)
	if err := index.Train(vectors); err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	// Add
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	// Search
	queries := generateTestVectors(3, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	if len(distances) != 30 || len(indices) != 30 {
		t.Errorf("Search() returned %d distances and %d indices, want 30 each",
			len(distances), len(indices))
	}
}

func TestIVF_SetGetNprobe(t *testing.T) {
	d := 64

	index, err := IndexFactory(d, "IVF10,Flat", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory failed: %v", err)
	}
	defer index.Close()

	genericIdx, ok := index.(*GenericIndex)
	if !ok {
		t.Fatalf("Expected *GenericIndex, got %T", index)
	}

	// Get initial nprobe (should be 1)
	nprobe, err := genericIdx.GetNprobe()
	if err != nil {
		t.Fatalf("GetNprobe() failed: %v", err)
	}
	t.Logf("Initial nprobe: %d", nprobe)

	// Set nprobe to 5
	if err := genericIdx.SetNprobe(5); err != nil {
		t.Fatalf("SetNprobe(5) failed: %v", err)
	}

	// Verify it was set
	nprobe, err = genericIdx.GetNprobe()
	if err != nil {
		t.Fatalf("GetNprobe() after set failed: %v", err)
	}

	if nprobe != 5 {
		t.Errorf("GetNprobe() = %d, want 5", nprobe)
	}
}
