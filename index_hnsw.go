package faiss

import (
	"fmt"
	"runtime"
)

// IndexHNSW implements the Hierarchical Navigable Small World graph index
// This is one of the best indexes for high recall and fast search.
// It does not require training.
//
// Python equivalent: faiss.IndexHNSWFlat
type IndexHNSW struct {
	ptr       uintptr    // C pointer
	d         int        // dimension
	metric    MetricType // metric type
	ntotal    int64      // number of vectors
	isTrained bool       // always true for HNSW
	M         int        // number of connections per layer
	efConstruction int   // construction time search depth
	efSearch  int        // search time search depth
}

// Ensure IndexHNSW implements Index
var _ Index = (*IndexHNSW)(nil)

// NewIndexHNSWFlat creates a new HNSW index with flat (uncompressed) storage
//
// Parameters:
//   - d: dimension of vectors
//   - M: number of connections per layer (typical values: 12-48)
//       Higher M = better recall, more memory
//   - metric: distance metric (MetricL2 or MetricInnerProduct)
//
// The index does not require training.
func NewIndexHNSWFlat(d, M int, metric MetricType) (*IndexHNSW, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}
	if M <= 0 {
		return nil, fmt.Errorf("faiss: M must be positive")
	}

	ptr, err := faissIndexHNSWFlatNew(d, M, int(metric))
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to create IndexHNSWFlat: %w", err)
	}

	idx := &IndexHNSW{
		ptr:            ptr,
		d:              d,
		metric:         metric,
		ntotal:         0,
		isTrained:      true, // HNSW doesn't require training
		M:              M,
		efConstruction: 40,   // FAISS default
		efSearch:       16,   // FAISS default
	}

	runtime.SetFinalizer(idx, func(i *IndexHNSW) {
		if i.ptr != 0 {
			_ = i.Close()
		}
	})

	return idx, nil
}

// D returns the dimension of vectors
func (idx *IndexHNSW) D() int {
	return idx.d
}

// Ntotal returns the total number of vectors
func (idx *IndexHNSW) Ntotal() int64 {
	return idx.ntotal
}

// IsTrained returns true (HNSW doesn't require training)
func (idx *IndexHNSW) IsTrained() bool {
	return idx.isTrained
}

// MetricType returns the metric type
func (idx *IndexHNSW) MetricType() MetricType {
	return idx.metric
}

// GetM returns the number of connections per layer
func (idx *IndexHNSW) GetM() int {
	return idx.M
}

// GetEfConstruction returns the construction time search depth
func (idx *IndexHNSW) GetEfConstruction() int {
	return idx.efConstruction
}

// SetEfConstruction sets the construction time search depth
// This affects the quality of the index during building
// Higher values = better index quality but slower construction
// Typical values: 40-200, default is 40
func (idx *IndexHNSW) SetEfConstruction(ef int) error {
	if ef <= 0 {
		return fmt.Errorf("faiss: efConstruction must be positive")
	}

	if err := faissIndexHNSWSetEfConstruction(idx.ptr, ef); err != nil {
		return err
	}

	idx.efConstruction = ef
	return nil
}

// GetEfSearch returns the search time search depth
func (idx *IndexHNSW) GetEfSearch() int {
	return idx.efSearch
}

// SetEfSearch sets the search time search depth
// This affects search quality and speed
// Higher values = better recall but slower search
// Typical values: 16-256, default is 16
func (idx *IndexHNSW) SetEfSearch(ef int) error {
	if ef <= 0 {
		return fmt.Errorf("faiss: efSearch must be positive")
	}

	if err := faissIndexHNSWSetEfSearch(idx.ptr, ef); err != nil {
		return err
	}

	idx.efSearch = ef
	return nil
}

// Train is a no-op for HNSW (included for interface compatibility)
func (idx *IndexHNSW) Train(vectors []float32) error {
	// HNSW doesn't require training
	return nil
}

// Add adds vectors to the index
func (idx *IndexHNSW) Add(vectors []float32) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}
	if len(vectors) == 0 {
		return nil
	}
	if len(vectors)%idx.d != 0 {
		return ErrInvalidVectors
	}

	n := len(vectors) / idx.d

	if err := faissIndexAdd(idx.ptr, vectors, n); err != nil {
		return fmt.Errorf("faiss: failed to add vectors: %w", err)
	}

	idx.ntotal += int64(n)
	return nil
}

// Search searches for k nearest neighbors
func (idx *IndexHNSW) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
	if idx.ptr == 0 {
		return nil, nil, ErrNullPointer
	}
	if len(queries) == 0 {
		return []float32{}, []int64{}, nil
	}
	if len(queries)%idx.d != 0 {
		return nil, nil, ErrInvalidVectors
	}
	if k <= 0 {
		return nil, nil, ErrInvalidK
	}

	nq := len(queries) / idx.d
	distances = make([]float32, nq*k)
	indices = make([]int64, nq*k)

	if err := faissIndexSearch(idx.ptr, queries, nq, k, distances, indices); err != nil {
		return nil, nil, fmt.Errorf("faiss: search failed: %w", err)
	}

	return distances, indices, nil
}

// Reset removes all vectors from the index
func (idx *IndexHNSW) Reset() error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}

	if err := faissIndexReset(idx.ptr); err != nil {
		return fmt.Errorf("faiss: reset failed: %w", err)
	}

	idx.ntotal = 0
	return nil
}

// Close releases resources
func (idx *IndexHNSW) Close() error {
	if idx.ptr == 0 {
		return nil
	}

	err := faissIndexFree(idx.ptr)
	idx.ptr = 0
	idx.ntotal = 0

	if err != nil {
		return fmt.Errorf("faiss: failed to free index: %w", err)
	}

	return nil
}
