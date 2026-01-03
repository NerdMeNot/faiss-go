package faiss

import (
	"fmt"
	"runtime"
)

// GenericIndex wraps any index created by the factory function.
// This provides a universal interface for all FAISS index types,
// including those that don't have specific constructors in the C API
// (like HNSW, PQ, IVFPQ, etc.)
//
// Python equivalent: Using faiss.index_factory() returns a generic Index object
//
// Example:
//
//	// Create HNSW index (no direct constructor available)
//	index, _ := faiss.IndexFactory(128, "HNSW32", faiss.MetricL2)
//
//	// Create PQ index
//	index, _ := faiss.IndexFactory(128, "PQ8", faiss.MetricL2)
//
//	// Create IVF+PQ index
//	index, _ := faiss.IndexFactory(128, "IVF100,PQ8", faiss.MetricL2)
type GenericIndex struct {
	ptr         uintptr    // C pointer to FaissIndex
	d           int        // dimension
	metric      MetricType // metric type
	ntotal      int64      // number of vectors
	isTrained   bool       // training status (cached)
	description string     // factory description string
}

// Ensure GenericIndex implements Index interface
var _ Index = (*GenericIndex)(nil)

// D returns the dimension of vectors
func (idx *GenericIndex) D() int {
	return idx.d
}

// Ntotal returns the total number of vectors in the index
func (idx *GenericIndex) Ntotal() int64 {
	if idx.ptr == 0 {
		return 0
	}
	// Query from C API for accuracy
	return faissIndexNtotal(idx.ptr)
}

// IsTrained returns whether the index has been trained
func (idx *GenericIndex) IsTrained() bool {
	if idx.ptr == 0 {
		return false
	}
	// Query from C API for accuracy
	return faissIndexIsTrained(idx.ptr)
}

// MetricType returns the metric type used by this index
func (idx *GenericIndex) MetricType() MetricType {
	return idx.metric
}

// Train trains the index on a set of vectors
//
// Not all indexes require training (e.g., Flat indexes don't need it),
// but IVF-based and quantization-based indexes do.
func (idx *GenericIndex) Train(vectors []float32) error {
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

	timer := StartTimer()
	if err := faissIndexTrain(idx.ptr, vectors, n); err != nil {
		return fmt.Errorf("training failed: %w", err)
	}
	timer.RecordTrain(n)

	idx.isTrained = true
	return nil
}

// Add adds vectors to the index
//
// For indexes that require training, Train() must be called first.
func (idx *GenericIndex) Add(vectors []float32) error {
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

	timer := StartTimer()
	if err := faissIndexAdd(idx.ptr, vectors, n); err != nil {
		return fmt.Errorf("add failed: %w", err)
	}
	timer.RecordAdd(n)

	idx.ntotal += int64(n)
	return nil
}

// Search performs k-nearest neighbor search
//
// Parameters:
//   - queries: flattened query vectors (length must be d * nq)
//   - k: number of nearest neighbors to find
//
// Returns:
//   - distances: distances to nearest neighbors (nq * k)
//   - labels: IDs of nearest neighbors (nq * k)
//   - error: any error that occurred
func (idx *GenericIndex) Search(queries []float32, k int) (distances []float32, labels []int64, err error) {
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
	labels = make([]int64, nq*k)

	timer := StartTimer()
	if err := faissIndexSearch(idx.ptr, queries, nq, k, distances, labels); err != nil {
		return nil, nil, fmt.Errorf("search failed: %w", err)
	}
	timer.RecordSearch(nq, nq*k)

	return distances, labels, nil
}

// Reset removes all vectors from the index
func (idx *GenericIndex) Reset() error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}

	timer := StartTimer()
	if err := faissIndexReset(idx.ptr); err != nil {
		return fmt.Errorf("reset failed: %w", err)
	}
	timer.RecordReset()

	idx.ntotal = 0
	return nil
}

// Close releases the resources used by the index
func (idx *GenericIndex) Close() error {
	if idx.ptr == 0 {
		return nil
	}

	err := faissIndexFree(idx.ptr)
	idx.ptr = 0
	idx.ntotal = 0

	if err != nil {
		return fmt.Errorf("failed to free index: %w", err)
	}

	return nil
}

// SetNprobe sets the number of lists to probe during search (IVF indexes only)
//
// This parameter controls the speed/accuracy tradeoff for IVF indexes:
//   - nprobe=1: Fastest search, lowest recall
//   - nprobe=nlist: Exhaustive search, highest recall (equivalent to brute force)
//   - Recommended: nprobe = 10-20 for balanced performance
//
// Only works for IVF-based indexes (IVFFlat, IVFPQ, IVFSQ, etc.)
// Returns an error if called on non-IVF indexes.
//
// Example:
//
//	index, _ := faiss.IndexFactory(128, "IVF100,Flat", faiss.MetricL2)
//	index.SetNprobe(10) // Search 10 of 100 clusters
func (idx *GenericIndex) SetNprobe(nprobe int) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}

	if err := faissIndexIVFSetNprobe(idx.ptr, nprobe); err != nil {
		return fmt.Errorf("failed to set nprobe (index may not be IVF-based): %w", err)
	}

	return nil
}

// GetNprobe gets the number of lists to probe during search (IVF indexes only)
func (idx *GenericIndex) GetNprobe() (int, error) {
	if idx.ptr == 0 {
		return 0, ErrNullPointer
	}

	nprobe, err := faissIndexIVFGetNprobe(idx.ptr)
	if err != nil {
		return 0, fmt.Errorf("failed to get nprobe (index may not be IVF-based): %w", err)
	}

	return nprobe, nil
}

// SetEfSearch sets the search-time effort parameter for HNSW indexes.
//
// The efSearch parameter controls how many nodes are visited during search.
// Higher values give better recall but slower search.
//
// Recommended values:
//   - efSearch=16: Fast search, lower recall (default)
//   - efSearch=32-64: Balanced (recommended)
//   - efSearch=128+: High recall, slower search
//
// Note: This only works for HNSW indexes. Returns error for other index types.
//
// Example:
//
//	index, _ := faiss.IndexFactory(128, "HNSW32", faiss.MetricL2)
//	index.SetEfSearch(64) // Increase search effort for better recall
func (idx *GenericIndex) SetEfSearch(efSearch int) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}
	// Try to set efSearch - will fail for non-HNSW indexes
	return faissIndexHNSWSetEfSearch(idx.ptr, efSearch)
}

// Description returns the factory description string used to create this index
func (idx *GenericIndex) Description() string {
	return idx.description
}

// WriteToFile writes the index to a file
func (idx *GenericIndex) WriteToFile(filename string) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}

	if err := faissWriteIndex(idx.ptr, filename); err != nil {
		return fmt.Errorf("failed to write index: %w", err)
	}

	return nil
}

// newGenericIndex creates a new GenericIndex wrapper
// This is an internal function used by IndexFactory
func newGenericIndex(ptr uintptr, d int, metric MetricType, description string) *GenericIndex {
	idx := &GenericIndex{
		ptr:         ptr,
		d:           d,
		metric:      metric,
		ntotal:      0,
		isTrained:   false,
		description: description,
	}

	// Set up finalizer to automatically free the index when garbage collected
	runtime.SetFinalizer(idx, func(i *GenericIndex) {
		if i.ptr != 0 {
			_ = i.Close()
		}
	})

	return idx
}
