package faiss

import (
	"fmt"
	"runtime"
)

// IndexLSH performs locality-sensitive hashing on float vectors
//
// Python equivalent: faiss.IndexLSH
//
// Example:
//   // 128-dim vectors, 256 hash bits
//   index, _ := faiss.NewIndexLSH(128, 256)
//   index.Add(vectors)
//   distances, indices, _ := index.Search(queries, 10)
//
// Note: LSH is an approximate method that's fast but less accurate than
// other methods. Best for very high-dimensional data or when speed is critical.
type IndexLSH struct {
	ptr       uintptr // C pointer
	d         int     // dimension
	nbits     int     // number of hash bits
	ntotal    int64   // number of vectors
	isTrained bool    // training status
	rotateData bool   // whether to rotate data before hashing
}

// Ensure IndexLSH implements Index
var _ Index = (*IndexLSH)(nil)

// NewIndexLSH creates a new LSH index
// d is the vector dimension
// nbits is the number of hash bits (higher = more accurate but slower)
func NewIndexLSH(d, nbits int) (*IndexLSH, error) {
	if d <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if nbits <= 0 {
		return nil, fmt.Errorf("nbits must be positive")
	}

	var ptr uintptr
	ret := faiss_IndexLSH_new(&ptr, int64(d), int64(nbits), false, false)
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexLSH")
	}

	idx := &IndexLSH{
		ptr:        ptr,
		d:          d,
		nbits:      nbits,
		ntotal:     0,
		isTrained:  true, // LSH doesn't require training by default
		rotateData: false,
	}

	runtime.SetFinalizer(idx, func(idx *IndexLSH) {
		idx.Close()
	})

	return idx, nil
}

// NewIndexLSHWithRotation creates an LSH index with random rotation
// Random rotation can improve hash quality for some datasets
func NewIndexLSHWithRotation(d, nbits int) (*IndexLSH, error) {
	if d <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if nbits <= 0 {
		return nil, fmt.Errorf("nbits must be positive")
	}

	var ptr uintptr
	ret := faiss_IndexLSH_new(&ptr, int64(d), int64(nbits), true, false)
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexLSH with rotation")
	}

	idx := &IndexLSH{
		ptr:        ptr,
		d:          d,
		nbits:      nbits,
		ntotal:     0,
		isTrained:  true,
		rotateData: true,
	}

	runtime.SetFinalizer(idx, func(idx *IndexLSH) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension of the index
func (idx *IndexLSH) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexLSH) Ntotal() int64 {
	ntotal := faiss_Index_ntotal(idx.ptr)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained (always true for LSH)
func (idx *IndexLSH) IsTrained() bool {
	return idx.isTrained
}

// MetricType returns the distance metric (LSH uses Hamming on hash codes)
func (idx *IndexLSH) MetricType() MetricType {
	// LSH internally uses Hamming distance on hash codes
	// but accepts float vectors as input
	return MetricL2
}

// Nbits returns the number of hash bits
func (idx *IndexLSH) Nbits() int {
	return idx.nbits
}

// Train is a no-op for LSH (unless using rotation with training)
func (idx *IndexLSH) Train(vectors []float32) error {
	if len(vectors) == 0 {
		return nil
	}
	if len(vectors)%idx.d != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension %d", idx.d)
	}

	// If using rotation, we might want to train it
	if idx.rotateData {
		n := int64(len(vectors) / idx.d)
		ret := faiss_Index_train(idx.ptr, n, &vectors[0])
		if ret != 0 {
			return fmt.Errorf("training failed")
		}
	}

	idx.isTrained = true
	return nil
}

// Add adds vectors to the index
func (idx *IndexLSH) Add(vectors []float32) error {
	if len(vectors) == 0 {
		return nil
	}
	if len(vectors)%idx.d != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension %d", idx.d)
	}

	n := int64(len(vectors) / idx.d)
	ret := faiss_Index_add(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("add failed")
	}

	idx.ntotal += n
	return nil
}

// Search performs k-NN search using LSH
func (idx *IndexLSH) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
	if len(queries) == 0 {
		return nil, nil, fmt.Errorf("empty query vectors")
	}
	if len(queries)%idx.d != 0 {
		return nil, nil, fmt.Errorf("queries length must be multiple of dimension %d", idx.d)
	}

	nq := int64(len(queries) / idx.d)
	distances = make([]float32, nq*int64(k))
	indices = make([]int64, nq*int64(k))

	ret := faiss_Index_search(idx.ptr, nq, &queries[0], int64(k), &distances[0], &indices[0])
	if ret != 0 {
		return nil, nil, fmt.Errorf("search failed")
	}

	return distances, indices, nil
}

// SetNprobe is not supported for LSH indexes (not an IVF index)
func (idx *IndexLSH) SetNprobe(nprobe int) error {
	return fmt.Errorf("faiss: SetNprobe not supported for IndexLSH (not an IVF index)")
}

// SetEfSearch is not supported for LSH indexes (not an HNSW index)
func (idx *IndexLSH) SetEfSearch(efSearch int) error {
	return fmt.Errorf("faiss: SetEfSearch not supported for IndexLSH (not an HNSW index)")
}

// Reset removes all vectors from the index
func (idx *IndexLSH) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexLSH) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}
