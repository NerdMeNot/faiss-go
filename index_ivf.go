package faiss

import (
	"errors"
	"fmt"
	"runtime"
)

// IndexIVFFlat is an inverted file index with flat (uncompressed) vectors
// This is one of the most commonly used indexes in production.
// It requires training before adding vectors.
//
// Python equivalent: faiss.IndexIVFFlat
type IndexIVFFlat struct {
	ptr       uintptr     // C pointer
	quantizer Index       // quantizer index (must be kept alive)
	d         int         // dimension
	metric    MetricType  // metric type
	ntotal    int64       // number of vectors
	isTrained bool        // training status
	nlist     int         // number of inverted lists
	nprobe    int         // number of lists to probe during search
}

// Ensure IndexIVFFlat implements Index and related interfaces
var _ Index = (*IndexIVFFlat)(nil)
var _ IndexWithAssign = (*IndexIVFFlat)(nil)

// NewIndexIVFFlat creates a new IVF index with flat storage
//
// Parameters:
//   - quantizer: a trained index used for coarse quantization (typically IndexFlat)
//   - d: dimension of vectors
//   - nlist: number of inverted lists (clusters)
//   - metric: distance metric (MetricL2 or MetricInnerProduct)
//
// The index must be trained before adding vectors.
func NewIndexIVFFlat(quantizer Index, d, nlist int, metric MetricType) (*IndexIVFFlat, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}
	if nlist <= 0 {
		return nil, fmt.Errorf("faiss: nlist must be positive")
	}

	// Get the quantizer pointer
	var quantizerPtr uintptr
	switch q := quantizer.(type) {
	case *IndexFlat:
		quantizerPtr = q.ptr
	default:
		return nil, fmt.Errorf("faiss: unsupported quantizer type")
	}

	ptr, err := faissIndexIVFFlatNew(quantizerPtr, d, nlist, int(metric))
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to create IndexIVFFlat: %w", err)
	}

	idx := &IndexIVFFlat{
		ptr:       ptr,
		quantizer: quantizer, // Keep reference to prevent GC
		d:         d,
		metric:    metric,
		ntotal:    0,
		isTrained: false,
		nlist:     nlist,
		nprobe:    1, // default nprobe
	}

	runtime.SetFinalizer(idx, func(i *IndexIVFFlat) {
		if i.ptr != 0 {
			_ = i.Close()
		}
	})

	return idx, nil
}

// D returns the dimension of vectors
func (idx *IndexIVFFlat) D() int {
	return idx.d
}

// Ntotal returns the total number of vectors
func (idx *IndexIVFFlat) Ntotal() int64 {
	return idx.ntotal
}

// IsTrained returns whether the index has been trained
func (idx *IndexIVFFlat) IsTrained() bool {
	return idx.isTrained
}

// MetricType returns the metric type
func (idx *IndexIVFFlat) MetricType() MetricType {
	return idx.metric
}

// Nlist returns the number of inverted lists
func (idx *IndexIVFFlat) Nlist() int {
	return idx.nlist
}

// Nprobe returns the number of lists to probe during search
func (idx *IndexIVFFlat) Nprobe() int {
	return idx.nprobe
}

// SetNprobe sets the number of lists to probe during search
// Higher values = better recall but slower search
// Default is 1, typical values are 1-nlist
func (idx *IndexIVFFlat) SetNprobe(nprobe int) error {
	if nprobe <= 0 || nprobe > idx.nlist {
		return fmt.Errorf("faiss: nprobe must be between 1 and %d", idx.nlist)
	}

	if err := faissIndexIVFSetNprobe(idx.ptr, nprobe); err != nil {
		return err
	}

	idx.nprobe = nprobe
	return nil
}

// Train trains the index on a representative set of vectors
// This is REQUIRED before adding vectors to IVF indexes
func (idx *IndexIVFFlat) Train(vectors []float32) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}
	if idx.isTrained {
		return nil // already trained
	}
	if len(vectors) == 0 {
		return errors.New("faiss: cannot train on empty vectors")
	}
	if len(vectors)%idx.d != 0 {
		return ErrInvalidVectors
	}

	n := len(vectors) / idx.d

	// Recommend at least 30*nlist training vectors
	minTraining := 30 * idx.nlist
	if n < minTraining {
		return fmt.Errorf("faiss: insufficient training data (have %d, recommend at least %d)", n, minTraining)
	}

	if err := faissIndexTrain(idx.ptr, vectors, n); err != nil {
		return fmt.Errorf("faiss: training failed: %w", err)
	}

	idx.isTrained = true
	return nil
}

// Add adds vectors to the index
// The index must be trained before calling this
func (idx *IndexIVFFlat) Add(vectors []float32) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}
	if !idx.isTrained {
		return ErrNotTrained
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
func (idx *IndexIVFFlat) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
	if idx.ptr == 0 {
		return nil, nil, ErrNullPointer
	}
	if !idx.isTrained {
		return nil, nil, ErrNotTrained
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

// Assign assigns vectors to their nearest cluster (inverted list)
func (idx *IndexIVFFlat) Assign(vectors []float32) ([]int64, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if !idx.isTrained {
		return nil, ErrNotTrained
	}
	if len(vectors) == 0 {
		return []int64{}, nil
	}
	if len(vectors)%idx.d != 0 {
		return nil, ErrInvalidVectors
	}

	n := len(vectors) / idx.d
	labels := make([]int64, n)

	if err := faissIndexAssign(idx.ptr, vectors, n, labels); err != nil {
		return nil, fmt.Errorf("faiss: assignment failed: %w", err)
	}

	return labels, nil
}

// Reset removes all vectors from the index
func (idx *IndexIVFFlat) Reset() error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}

	if err := faissIndexReset(idx.ptr); err != nil {
		return fmt.Errorf("faiss: reset failed: %w", err)
	}

	idx.ntotal = 0
	// Note: training is preserved after reset
	return nil
}

// Close releases resources
func (idx *IndexIVFFlat) Close() error {
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
