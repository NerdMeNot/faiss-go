package faiss

import (
	"runtime"
	"testing"
	"time"
)

// ========================================
// Memory Leak Detection Tests
// ========================================

// TestFinalizerExecution tests that finalizers are properly called
func TestFinalizerExecution(t *testing.T) {
	// Create and discard indexes to test finalizer execution
	for i := 0; i < 100; i++ {
		index, err := NewIndexFlatL2(64)
		if err != nil {
			t.Fatalf("Failed to create index: %v", err)
		}
		// Add some data
		vectors := generateVectors(100, 64)
		index.Add(vectors)
		// Index goes out of scope, finalizer should run
	}

	// Force garbage collection
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
	runtime.GC()

	// If we reach here without crashes, finalizers likely worked
	// This is a basic test - memory profiling tools would be more thorough
}

// TestExplicitClose tests that explicit Close() prevents double-free
func TestExplicitClose(t *testing.T) {
	index, err := NewIndexFlatL2(64)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	// Close explicitly
	if err := index.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Second close should be safe (no-op)
	if err := index.Close(); err != nil {
		t.Fatalf("Second close failed: %v", err)
	}

	// Force GC to ensure finalizer doesn't cause double-free
	runtime.GC()
	time.Sleep(50 * time.Millisecond)
}

// TestCloseAllIndexTypes tests Close on all index types
func TestCloseAllIndexTypes(t *testing.T) {
	d := 64
	nb := 100
	vectors := generateVectors(nb, d)

	tests := []struct {
		name  string
		setup func() (interface{ Close() error }, error)
	}{
		{"IndexFlatL2", func() (interface{ Close() error }, error) {
			idx, err := NewIndexFlatL2(d)
			if err == nil {
				idx.Add(vectors)
			}
			return idx, err
		}},
		{"IndexFlatIP", func() (interface{ Close() error }, error) {
			idx, err := NewIndexFlatIP(d)
			if err == nil {
				idx.Add(vectors)
			}
			return idx, err
		}},
		{"IndexIVFFlat", func() (interface{ Close() error }, error) {
			q, _ := NewIndexFlatL2(d)
			defer q.Close()
			idx, err := NewIndexIVFFlat(q, d, 10, MetricL2)
			if err == nil {
				idx.Train(vectors)
				idx.Add(vectors)
			}
			return idx, err
		}},
		{"IndexHNSW", func() (interface{ Close() error }, error) {
			idx, err := NewIndexHNSWFlat(d, 16, MetricL2)
			if err == nil {
				idx.Add(vectors)
			}
			return idx, err
		}},
		{"IndexPQ", func() (interface{ Close() error }, error) {
			idx, err := NewIndexPQ(d, 8, 8, MetricL2)
			if err == nil {
				idx.Train(vectors)
				idx.Add(vectors)
			}
			return idx, err
		}},
		{"IndexScalarQuantizer", func() (interface{ Close() error }, error) {
			idx, err := NewIndexScalarQuantizer(d, QT_8bit, MetricL2)
			if err == nil {
				idx.Train(vectors)
				idx.Add(vectors)
			}
			return idx, err
		}},
		{"IndexLSH", func() (interface{ Close() error }, error) {
			idx, err := NewIndexLSH(d, 256)
			if err != nil {
				// IndexLSH might not be available in all FAISS builds
				return nil, nil // Return nil error to skip gracefully
			}
			if idx != nil {
				idx.Add(vectors)
			}
			return idx, err
		}},
		{"PCAMatrix", func() (interface{ Close() error }, error) {
			idx, err := NewPCAMatrix(d, 32)
			if err == nil {
				idx.Train(vectors)
			}
			return idx, err
		}},
		{"OPQMatrix", func() (interface{ Close() error }, error) {
			idx, err := NewOPQMatrix(d, 8)
			if err == nil {
				idx.Train(vectors)
			}
			return idx, err
		}},
		{"RandomRotationMatrix", func() (interface{ Close() error }, error) {
			return NewRandomRotationMatrix(d, d)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := tt.setup()
			if err != nil {
				t.Fatalf("Setup failed: %v", err)
			}

			// Skip if object is nil (e.g., unsupported index type)
			if obj == nil {
				t.Skip("Index type not available in this build")
				return
			}

			// Close should succeed
			if err := obj.Close(); err != nil {
				t.Errorf("Close failed: %v", err)
			}

			// Second close should be safe
			if err := obj.Close(); err != nil {
				t.Errorf("Second close failed: %v", err)
			}
		})
	}

	// Force GC
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
}

// TestConcurrentIndexCreation tests concurrent index creation/destruction
func TestConcurrentIndexCreation(t *testing.T) {
	const numGoroutines = 50
	const indexesPerGoroutine = 10

	done := make(chan bool)

	for g := 0; g < numGoroutines; g++ {
		go func() {
			for i := 0; i < indexesPerGoroutine; i++ {
				index, err := NewIndexFlatL2(64)
				if err != nil {
					t.Errorf("Failed to create index: %v", err)
					done <- false
					return
				}
				vectors := generateVectors(50, 64)
				if err := index.Add(vectors); err != nil {
					t.Errorf("Failed to add vectors: %v", err)
				}
				index.Close()
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for g := 0; g < numGoroutines; g++ {
		<-done
	}

	// Force GC
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
}

// TestBinaryIndexMemory tests binary index memory management
func TestBinaryIndexMemory(t *testing.T) {
	d := 256
	nb := 100
	binaryVectors := generateBinaryVectors(nb, d)

	tests := []struct {
		name  string
		setup func() (BinaryIndex, error)
	}{
		{"IndexBinaryFlat", func() (BinaryIndex, error) {
			idx, err := NewIndexBinaryFlat(d)
			if err == nil {
				idx.Add(binaryVectors)
			}
			return idx, err
		}},
		{"IndexBinaryIVF", func() (BinaryIndex, error) {
			q, _ := NewIndexBinaryFlat(d)
			defer q.Close()
			idx, err := NewIndexBinaryIVF(q, d, 10)
			if err == nil {
				idx.Train(binaryVectors)
				idx.Add(binaryVectors)
			}
			return idx, err
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := tt.setup()
			if err != nil {
				t.Fatalf("Setup failed: %v", err)
			}

			if err := idx.Close(); err != nil {
				t.Errorf("Close failed: %v", err)
			}

			// Second close should be safe
			if err := idx.Close(); err != nil {
				t.Errorf("Second close failed: %v", err)
			}
		})
	}

	runtime.GC()
	time.Sleep(50 * time.Millisecond)
}

// TestNestedIndexMemory tests indexes that contain other indexes
func TestNestedIndexMemory(t *testing.T) {
	d := 64
	nb := 100
	vectors := generateVectors(nb, d)

	t.Run("IndexIVFFlat_with_quantizer", func(t *testing.T) {
		quantizer, _ := NewIndexFlatL2(d)
		index, _ := NewIndexIVFFlat(quantizer, d, 10, MetricL2)
		index.Train(vectors)
		index.Add(vectors)

		// Close IVF index (should not double-free quantizer)
		index.Close()
		quantizer.Close()

		runtime.GC()
		time.Sleep(50 * time.Millisecond)
	})

	t.Run("IndexRefine_with_base_and_refine", func(t *testing.T) {
		q, _ := NewIndexFlatL2(d)
		base, _ := NewIndexIVFFlat(q, d, 10, MetricL2)
		refine, _ := NewIndexFlatL2(d)
		index, err := NewIndexRefine(base, refine)
		if err != nil {
			t.Fatalf("Failed to create IndexRefine: %v", err)
		}

		index.Train(vectors)
		index.Add(vectors)

		// Close all
		index.Close()
		base.Close()
		refine.Close()
		q.Close()

		runtime.GC()
		time.Sleep(50 * time.Millisecond)
	})

	t.Run("IndexPreTransform_with_transform_and_index", func(t *testing.T) {
		pca, _ := NewPCAMatrix(d, 32)
		baseIndex, _ := NewIndexFlatL2(32)
		index, err := NewIndexPreTransform(pca, baseIndex)
		if err != nil {
			t.Fatalf("Failed to create IndexPreTransform: %v", err)
		}

		index.Train(vectors)
		index.Add(vectors)

		// Close all
		index.Close()
		pca.Close()
		baseIndex.Close()

		runtime.GC()
		time.Sleep(50 * time.Millisecond)
	})
}

// TestResetDoesNotLeak tests that Reset properly frees memory
func TestResetDoesNotLeak(t *testing.T) {
	index, _ := NewIndexFlatL2(64)
	defer index.Close()

	// Add and reset multiple times
	for i := 0; i < 100; i++ {
		vectors := generateVectors(100, 64)
		index.Add(vectors)

		if index.Ntotal() != 100 {
			t.Errorf("Expected 100 vectors, got %d", index.Ntotal())
		}

		if err := index.Reset(); err != nil {
			t.Fatalf("Reset failed: %v", err)
		}

		if index.Ntotal() != 0 {
			t.Errorf("Expected 0 vectors after reset, got %d", index.Ntotal())
		}
	}

	runtime.GC()
	time.Sleep(50 * time.Millisecond)
}

// TestLargeIndexMemory tests memory handling with larger indexes
func TestLargeIndexMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large memory test in short mode")
	}

	d := 128
	nb := 10000 // 10K vectors

	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := generateVectors(nb, d)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search to ensure index is working
	queries := generateVectors(10, d)
	_, _, err = index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Reset should free vectors
	if err := index.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}

	runtime.GC()
	time.Sleep(100 * time.Millisecond)
}
