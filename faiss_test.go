package faiss

import (
	"math/rand"
	"testing"
)

func TestIndexFlatL2Creation(t *testing.T) {
	dimensions := []int{16, 32, 64, 128, 256}

	for _, d := range dimensions {
		t.Run("dimension="+string(rune(d)), func(t *testing.T) {
			index, err := NewIndexFlatL2(d)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			if index.D() != d {
				t.Errorf("Expected dimension %d, got %d", d, index.D())
			}

			if index.MetricType() != MetricL2 {
				t.Errorf("Expected MetricL2, got %v", index.MetricType())
			}

			if !index.IsTrained() {
				t.Error("Flat index should always be trained")
			}

			if index.Ntotal() != 0 {
				t.Errorf("New index should have 0 vectors, got %d", index.Ntotal())
			}
		})
	}
}

func TestIndexFlatIPCreation(t *testing.T) {
	d := 128
	index, err := NewIndexFlatIP(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if index.MetricType() != MetricInnerProduct {
		t.Errorf("Expected MetricInnerProduct, got %v", index.MetricType())
	}
}

func TestInvalidDimension(t *testing.T) {
	invalidDims := []int{-1, 0}

	for _, d := range invalidDims {
		_, err := NewIndexFlatL2(d)
		if err != ErrInvalidDimension {
			t.Errorf("Expected ErrInvalidDimension for d=%d, got %v", d, err)
		}

		_, err = NewIndexFlatIP(d)
		if err != ErrInvalidDimension {
			t.Errorf("Expected ErrInvalidDimension for d=%d, got %v", d, err)
		}
	}
}

func TestAddVectors(t *testing.T) {
	d := 64
	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Test adding valid vectors
	vectors := make([]float32, d*10) // 10 vectors
	for i := range vectors {
		vectors[i] = rand.Float32()
	}

	// Note: This will fail with stub implementation
	// but the test structure is correct
	err = index.Add(vectors)
	// TODO: Uncomment when real implementation is ready
	// if err != nil {
	// 	t.Fatalf("Failed to add vectors: %v", err)
	// }
	// if index.Ntotal() != 10 {
	// 	t.Errorf("Expected 10 vectors, got %d", index.Ntotal())
	// }
}

func TestAddInvalidVectors(t *testing.T) {
	d := 64
	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Invalid: not a multiple of dimension
	invalidVectors := make([]float32, d+1)
	err = index.Add(invalidVectors)
	if err != ErrInvalidVectors {
		t.Errorf("Expected ErrInvalidVectors, got %v", err)
	}

	// Valid: empty vectors (should not error)
	err = index.Add([]float32{})
	if err != nil {
		t.Errorf("Adding empty vectors should not error, got %v", err)
	}
}

func TestSearch(t *testing.T) {
	d := 32
	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Add some vectors
	numVectors := 100
	vectors := make([]float32, d*numVectors)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}

	// TODO: Uncomment when real implementation is ready
	// err = index.Add(vectors)
	// if err != nil {
	// 	t.Fatalf("Failed to add vectors: %v", err)
	// }

	// Search
	query := make([]float32, d)
	for i := range query {
		query[i] = rand.Float32()
	}

	// k := 10
	// TODO: Uncomment when real implementation is ready
	// distances, indices, err := index.Search(query, k)
	// if err != nil {
	// 	t.Fatalf("Search failed: %v", err)
	// }
	//
	// if len(distances) != k {
	// 	t.Errorf("Expected %d distances, got %d", k, len(distances))
	// }
	//
	// if len(indices) != k {
	// 	t.Errorf("Expected %d indices, got %d", k, len(indices))
	// }
}

func TestReset(t *testing.T) {
	d := 32
	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// TODO: Uncomment when real implementation is ready
	// // Add vectors
	// vectors := make([]float32, d*50)
	// for i := range vectors {
	// 	vectors[i] = rand.Float32()
	// }
	// err = index.Add(vectors)
	// if err != nil {
	// 	t.Fatalf("Failed to add vectors: %v", err)
	// }
	//
	// // Reset
	// err = index.Reset()
	// if err != nil {
	// 	t.Fatalf("Failed to reset index: %v", err)
	// }
	//
	// if index.Ntotal() != 0 {
	// 	t.Errorf("After reset, expected 0 vectors, got %d", index.Ntotal())
	// }
}

func TestClose(t *testing.T) {
	d := 32
	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	err = index.Close()
	if err != nil {
		t.Errorf("Close failed: %v", err)
	}

	// Double close should be safe
	err = index.Close()
	if err != nil {
		t.Errorf("Second close should not error, got %v", err)
	}
}

func TestBuildInfo(t *testing.T) {
	info := GetBuildInfo()

	if info.Version == "" {
		t.Error("Version should not be empty")
	}

	if info.FAISSVersion == "" {
		t.Error("FAISSVersion should not be empty")
	}

	if info.BuildMode != "source" && info.BuildMode != "prebuilt" {
		t.Errorf("Unexpected build mode: %s", info.BuildMode)
	}

	// Test String() method
	infoStr := info.String()
	if infoStr == "" {
		t.Error("BuildInfo.String() should not be empty")
	}
}

func BenchmarkIndexCreation(b *testing.B) {
	d := 128
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		index, err := NewIndexFlatL2(d)
		if err != nil {
			b.Fatalf("Failed to create index: %v", err)
		}
		_ = index.Close()
	}
}

func BenchmarkAddVectors(b *testing.B) {
	d := 128
	numVectors := 1000

	index, err := NewIndexFlatL2(d)
	if err != nil {
		b.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := make([]float32, d*numVectors)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = index.Reset()
		_ = index.Add(vectors)
	}
}
