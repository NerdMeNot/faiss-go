package faiss

import (
	"fmt"
	"runtime"
)

// IndexIDMap wraps another index to support custom IDs
// This allows you to use your own IDs instead of sequential indices
//
// Python equivalent: faiss.IndexIDMap, faiss.IndexIDMap2
type IndexIDMap struct {
	ptr       uintptr    // C pointer
	baseIndex Index      // wrapped index
	d         int        // dimension
	metric    MetricType // metric type
	ntotal    int64      // number of vectors
}

// Ensure IndexIDMap implements IndexWithIDs
var _ Index = (*IndexIDMap)(nil)
var _ IndexWithIDs = (*IndexIDMap)(nil)

// NewIndexIDMap creates a new index with ID mapping
//
// Parameters:
//   - baseIndex: the underlying index to wrap
//
// Example:
//   flatIndex, _ := faiss.NewIndexFlatL2(128)
//   idmapIndex, _ := faiss.NewIndexIDMap(flatIndex)
//   idmapIndex.AddWithIDs(vectors, []int64{100, 200, 300})
func NewIndexIDMap(baseIndex Index) (*IndexIDMap, error) {
	if baseIndex == nil {
		return nil, fmt.Errorf("faiss: baseIndex cannot be nil")
	}

	var basePtr uintptr
	switch idx := baseIndex.(type) {
	case *IndexFlat:
		basePtr = idx.ptr
	case *IndexIVFFlat:
		basePtr = idx.ptr
	case *IndexHNSW:
		basePtr = idx.ptr
	default:
		return nil, fmt.Errorf("faiss: unsupported base index type")
	}

	ptr, err := faissIndexIDMapNew(basePtr)
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to create IndexIDMap: %w", err)
	}

	idx := &IndexIDMap{
		ptr:       ptr,
		baseIndex: baseIndex,
		d:         baseIndex.D(),
		metric:    baseIndex.MetricType(),
		ntotal:    0,
	}

	runtime.SetFinalizer(idx, func(i *IndexIDMap) {
		if i.ptr != 0 {
			_ = i.Close()
		}
	})

	return idx, nil
}

// D returns the dimension of vectors
func (idx *IndexIDMap) D() int {
	return idx.d
}

// Ntotal returns the total number of vectors
func (idx *IndexIDMap) Ntotal() int64 {
	return idx.ntotal
}

// IsTrained returns whether the base index is trained
func (idx *IndexIDMap) IsTrained() bool {
	return idx.baseIndex.IsTrained()
}

// MetricType returns the metric type
func (idx *IndexIDMap) MetricType() MetricType {
	return idx.metric
}

// Train trains the base index
func (idx *IndexIDMap) Train(vectors []float32) error {
	return idx.baseIndex.Train(vectors)
}

// Add adds vectors with auto-generated sequential IDs
// For custom IDs, use AddWithIDs instead
func (idx *IndexIDMap) Add(vectors []float32) error {
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

	// Generate sequential IDs starting from current ntotal
	ids := make([]int64, n)
	for i := range ids {
		ids[i] = idx.ntotal + int64(i)
	}

	return idx.AddWithIDs(vectors, ids)
}

// AddWithIDs adds vectors with custom IDs
//
// Parameters:
//   - vectors: flattened vector data (length must be d * len(ids))
//   - ids: custom IDs for each vector
//
// Example:
//   vectors := []float32{ /* 3 vectors of dimension d */ }
//   ids := []int64{1000, 2000, 3000}
//   index.AddWithIDs(vectors, ids)
func (idx *IndexIDMap) AddWithIDs(vectors []float32, ids []int64) error {
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
	if len(ids) != n {
		return fmt.Errorf("faiss: number of IDs (%d) must match number of vectors (%d)", len(ids), n)
	}

	if err := faissIndexAddWithIDs(idx.ptr, vectors, ids, n); err != nil {
		return fmt.Errorf("faiss: failed to add vectors with IDs: %w", err)
	}

	idx.ntotal += int64(n)
	return nil
}

// Search searches for k nearest neighbors
// Returns distances and the custom IDs
func (idx *IndexIDMap) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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

// RemoveIDs removes vectors by their custom IDs
//
// Parameters:
//   - ids: slice of IDs to remove
//
// Returns the number of vectors actually removed
func (idx *IndexIDMap) RemoveIDs(ids []int64) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}
	if len(ids) == 0 {
		return nil
	}

	nRemoved, err := faissIndexRemoveIDs(idx.ptr, ids, len(ids))
	if err != nil {
		return fmt.Errorf("faiss: failed to remove IDs: %w", err)
	}

	idx.ntotal -= int64(nRemoved)
	return nil
}

// Reset removes all vectors
func (idx *IndexIDMap) Reset() error {
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
func (idx *IndexIDMap) Close() error {
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
