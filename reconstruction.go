package faiss

import "fmt"

// Reconstruct reconstructs a single vector by its index
//
// This is useful for debugging and verifying what's stored in the index.
// Not all index types support reconstruction.
//
// Python equivalent: vector = index.reconstruct(key)
//
// Example:
//   vector, err := index.Reconstruct(42)
//   fmt.Printf("Vector 42: %v\n", vector)
func (idx *IndexFlat) Reconstruct(key int64) ([]float32, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if key < 0 || key >= idx.ntotal {
		return nil, fmt.Errorf("faiss: key %d out of range [0, %d)", key, idx.ntotal)
	}

	recons := make([]float32, idx.d)
	if err := faissIndexReconstruct(idx.ptr, key, recons); err != nil {
		return nil, fmt.Errorf("faiss: reconstruction failed: %w", err)
	}

	return recons, nil
}

// ReconstructN reconstructs multiple consecutive vectors
//
// Parameters:
//   - i0: starting index
//   - n: number of vectors to reconstruct
//
// Returns a flattened array of n vectors (length = n * d)
//
// Python equivalent: vectors = index.reconstruct_n(i0, n)
func (idx *IndexFlat) ReconstructN(i0, n int64) ([]float32, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if i0 < 0 || i0+n > idx.ntotal {
		return nil, fmt.Errorf("faiss: range [%d, %d) out of bounds [0, %d)",
			i0, i0+n, idx.ntotal)
	}
	if n <= 0 {
		return []float32{}, nil
	}

	recons := make([]float32, n*int64(idx.d))
	if err := faissIndexReconstructN(idx.ptr, i0, n, recons); err != nil {
		return nil, fmt.Errorf("faiss: reconstruction failed: %w", err)
	}

	return recons, nil
}

// ReconstructBatch reconstructs multiple vectors by their indices
//
// Python equivalent: vectors = index.reconstruct_batch(keys)
func (idx *IndexFlat) ReconstructBatch(keys []int64) ([]float32, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if len(keys) == 0 {
		return []float32{}, nil
	}

	recons := make([]float32, len(keys)*idx.d)
	offset := 0

	for _, key := range keys {
		if key < 0 || key >= idx.ntotal {
			return nil, fmt.Errorf("faiss: key %d out of range [0, %d)", key, idx.ntotal)
		}

		vec, err := idx.Reconstruct(key)
		if err != nil {
			return nil, err
		}

		copy(recons[offset:], vec)
		offset += idx.d
	}

	return recons, nil
}

// Reconstruction for IVF indexes
func (idx *IndexIVFFlat) Reconstruct(key int64) ([]float32, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if key < 0 || key >= idx.ntotal {
		return nil, fmt.Errorf("faiss: key %d out of range [0, %d)", key, idx.ntotal)
	}

	recons := make([]float32, idx.d)
	if err := faissIndexReconstruct(idx.ptr, key, recons); err != nil {
		return nil, fmt.Errorf("faiss: reconstruction failed: %w", err)
	}

	return recons, nil
}

func (idx *IndexIVFFlat) ReconstructN(i0, n int64) ([]float32, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if i0 < 0 || i0+n > idx.ntotal {
		return nil, fmt.Errorf("faiss: range [%d, %d) out of bounds [0, %d)",
			i0, i0+n, idx.ntotal)
	}
	if n <= 0 {
		return []float32{}, nil
	}

	recons := make([]float32, n*int64(idx.d))
	if err := faissIndexReconstructN(idx.ptr, i0, n, recons); err != nil {
		return nil, fmt.Errorf("faiss: reconstruction failed: %w", err)
	}

	return recons, nil
}

func (idx *IndexIVFFlat) ReconstructBatch(keys []int64) ([]float32, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if len(keys) == 0 {
		return []float32{}, nil
	}

	recons := make([]float32, len(keys)*idx.d)
	offset := 0

	for _, key := range keys {
		vec, err := idx.Reconstruct(key)
		if err != nil {
			return nil, err
		}

		copy(recons[offset:], vec)
		offset += idx.d
	}

	return recons, nil
}

// HNSW doesn't support reconstruction
// Attempting to reconstruct will return an error

func (idx *IndexHNSW) Reconstruct(key int64) ([]float32, error) {
	return nil, fmt.Errorf("faiss: IndexHNSW does not support reconstruction")
}

func (idx *IndexHNSW) ReconstructN(i0, n int64) ([]float32, error) {
	return nil, fmt.Errorf("faiss: IndexHNSW does not support reconstruction")
}

func (idx *IndexHNSW) ReconstructBatch(keys []int64) ([]float32, error) {
	return nil, fmt.Errorf("faiss: IndexHNSW does not support reconstruction")
}
