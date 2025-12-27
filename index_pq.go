package faiss

import (
	"fmt"
	"runtime"
)

// IndexPQ is a Product Quantization index
// This index compresses vectors using product quantization,
// trading accuracy for significant memory savings.
//
// Python equivalent: faiss.IndexPQ
//
// Example:
//   // 128-dim vectors, 8 subquantizers, 8 bits each
//   index, _ := faiss.NewIndexPQ(128, 8, 8, faiss.MetricL2)
//   index.Train(trainingVectors)
//   index.Add(vectors)
type IndexPQ struct {
	ptr       uintptr    // C pointer
	d         int        // dimension
	metric    MetricType // metric type
	ntotal    int64      // number of vectors
	isTrained bool       // training status
	M         int        // number of subquantizers
	nbits     int        // bits per subquantizer
}

// Ensure IndexPQ implements Index
var _ Index = (*IndexPQ)(nil)

// NewIndexPQ creates a new Product Quantization index
//
// Parameters:
//   - d: dimension of vectors
//   - M: number of subquantizers (d must be divisible by M)
//   - nbits: bits per subquantizer (typically 8)
//   - metric: distance metric
//
// The index compresses each vector from d*4 bytes to M*nbits/8 bytes.
// For example: 128-dim float32 (512 bytes) -> 8 subquant * 8 bits (8 bytes) = 64x compression
//
// Typical configurations:
//   - M=8, nbits=8: good compression, reasonable accuracy
//   - M=16, nbits=8: better accuracy, less compression
//   - M=32, nbits=8: high accuracy, moderate compression
func NewIndexPQ(d, M, nbits int, metric MetricType) (*IndexPQ, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}
	if M <= 0 {
		return nil, fmt.Errorf("faiss: M must be positive")
	}
	if d%M != 0 {
		return nil, fmt.Errorf("faiss: dimension %d must be divisible by M=%d", d, M)
	}
	if nbits <= 0 || nbits > 16 {
		return nil, fmt.Errorf("faiss: nbits must be in range [1, 16]")
	}

	ptr, err := faissIndexPQNew(d, M, nbits, int(metric))
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to create IndexPQ: %w", err)
	}

	idx := &IndexPQ{
		ptr:       ptr,
		d:         d,
		metric:    metric,
		ntotal:    0,
		isTrained: false,
		M:         M,
		nbits:     nbits,
	}

	runtime.SetFinalizer(idx, func(i *IndexPQ) {
		if i.ptr != 0 {
			_ = i.Close()
		}
	})

	return idx, nil
}

// D returns the dimension
func (idx *IndexPQ) D() int {
	return idx.d
}

// Ntotal returns the number of vectors
func (idx *IndexPQ) Ntotal() int64 {
	return idx.ntotal
}

// IsTrained returns training status
func (idx *IndexPQ) IsTrained() bool {
	return idx.isTrained
}

// MetricType returns the metric
func (idx *IndexPQ) MetricType() MetricType {
	return idx.metric
}

// GetM returns the number of subquantizers
func (idx *IndexPQ) GetM() int {
	return idx.M
}

// GetNbits returns bits per subquantizer
func (idx *IndexPQ) GetNbits() int {
	return idx.nbits
}

// CompressionRatio returns the compression ratio (original size / compressed size)
func (idx *IndexPQ) CompressionRatio() float64 {
	originalSize := float64(idx.d * 4) // float32 = 4 bytes
	compressedSize := float64(idx.M * idx.nbits / 8)
	return originalSize / compressedSize
}

// Train trains the product quantizer
//
// PQ requires training to learn the codebooks for each subquantizer.
// Training vectors should be representative of the data distribution.
//
// Recommended: At least 256 * M training vectors
func (idx *IndexPQ) Train(vectors []float32) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}
	if idx.isTrained {
		return nil // already trained
	}
	if len(vectors) == 0 {
		return fmt.Errorf("faiss: cannot train on empty vectors")
	}
	if len(vectors)%idx.d != 0 {
		return ErrInvalidVectors
	}

	n := len(vectors) / idx.d

	// Recommend enough training data
	minTraining := 256 * idx.M
	if n < minTraining {
		return fmt.Errorf("faiss: insufficient training data (have %d, recommend at least %d)",
			n, minTraining)
	}

	if err := faissIndexTrain(idx.ptr, vectors, n); err != nil {
		return fmt.Errorf("faiss: training failed: %w", err)
	}

	idx.isTrained = true
	return nil
}

// Add adds vectors to the index (must be trained first)
func (idx *IndexPQ) Add(vectors []float32) error {
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
func (idx *IndexPQ) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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

// Reset removes all vectors
func (idx *IndexPQ) Reset() error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}

	if err := faissIndexReset(idx.ptr); err != nil {
		return fmt.Errorf("faiss: reset failed: %w", err)
	}

	idx.ntotal = 0
	// Training is preserved
	return nil
}

// Close releases resources
func (idx *IndexPQ) Close() error {
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

// IndexIVFPQ combines IVF with PQ for large-scale search with compression
//
// Python equivalent: faiss.IndexIVFPQ
//
// This is one of the most commonly used indexes in production for
// billion-scale search with limited memory.
type IndexIVFPQ struct {
	ptr       uintptr    // C pointer
	quantizer Index      // quantizer index (must be kept alive)
	d         int        // dimension
	metric    MetricType // metric type
	ntotal    int64      // number of vectors
	isTrained bool       // training status
	nlist     int        // number of inverted lists
	nprobe    int        // number of lists to probe
	M         int        // number of subquantizers
	nbits     int        // bits per subquantizer
}

// Ensure IndexIVFPQ implements Index
var _ Index = (*IndexIVFPQ)(nil)

// NewIndexIVFPQ creates a new IVF+PQ index
//
// Parameters:
//   - quantizer: coarse quantizer (typically IndexFlat)
//   - d: dimension
//   - nlist: number of inverted lists
//   - M: number of PQ subquantizers
//   - nbits: bits per subquantizer
//
// This combines the benefits of IVF (fast search) and PQ (memory compression).
//
// Example for 1B vectors, 128-dim:
//   quantizer, _ := faiss.NewIndexFlatL2(128)
//   index, _ := faiss.NewIndexIVFPQ(quantizer, 128, 16384, 8, 8)
//   // 16384 clusters, 8 subquant, 8 bits = 64x compression
func NewIndexIVFPQ(quantizer Index, d, nlist, M, nbits int) (*IndexIVFPQ, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}
	if nlist <= 0 {
		return nil, fmt.Errorf("faiss: nlist must be positive")
	}
	if M <= 0 {
		return nil, fmt.Errorf("faiss: M must be positive")
	}
	if d%M != 0 {
		return nil, fmt.Errorf("faiss: dimension %d must be divisible by M=%d", d, M)
	}
	if nbits <= 0 || nbits > 16 {
		return nil, fmt.Errorf("faiss: nbits must be in range [1, 16]")
	}

	var quantizerPtr uintptr
	switch q := quantizer.(type) {
	case *IndexFlat:
		quantizerPtr = q.ptr
	default:
		return nil, fmt.Errorf("faiss: unsupported quantizer type")
	}

	ptr, err := faissIndexIVFPQNew(quantizerPtr, d, nlist, M, nbits)
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to create IndexIVFPQ: %w", err)
	}

	idx := &IndexIVFPQ{
		ptr:       ptr,
		quantizer: quantizer, // Keep reference to prevent GC
		d:         d,
		metric:    quantizer.MetricType(),
		ntotal:    0,
		isTrained: false,
		nlist:     nlist,
		nprobe:    1,
		M:         M,
		nbits:     nbits,
	}

	runtime.SetFinalizer(idx, func(i *IndexIVFPQ) {
		if i.ptr != 0 {
			_ = i.Close()
		}
	})

	return idx, nil
}

// D returns the dimension
func (idx *IndexIVFPQ) D() int {
	return idx.d
}

// Ntotal returns the number of vectors
func (idx *IndexIVFPQ) Ntotal() int64 {
	return idx.ntotal
}

// IsTrained returns training status
func (idx *IndexIVFPQ) IsTrained() bool {
	return idx.isTrained
}

// MetricType returns the metric
func (idx *IndexIVFPQ) MetricType() MetricType {
	return idx.metric
}

// SetNprobe sets the number of lists to probe during search
func (idx *IndexIVFPQ) SetNprobe(nprobe int) error {
	if nprobe <= 0 || nprobe > idx.nlist {
		return fmt.Errorf("faiss: nprobe must be between 1 and %d", idx.nlist)
	}

	if err := faissIndexIVFSetNprobe(idx.ptr, nprobe); err != nil {
		return err
	}

	idx.nprobe = nprobe
	return nil
}

// Train trains the index
func (idx *IndexIVFPQ) Train(vectors []float32) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}
	if idx.isTrained {
		return nil
	}
	if len(vectors) == 0 {
		return fmt.Errorf("faiss: cannot train on empty vectors")
	}
	if len(vectors)%idx.d != 0 {
		return ErrInvalidVectors
	}

	n := len(vectors) / idx.d

	// Need training for both IVF clustering and PQ codebooks
	minTraining := 30 * idx.nlist
	if n < minTraining {
		return fmt.Errorf("faiss: insufficient training data (have %d, recommend at least %d)",
			n, minTraining)
	}

	if err := faissIndexTrain(idx.ptr, vectors, n); err != nil {
		return fmt.Errorf("faiss: training failed: %w", err)
	}

	idx.isTrained = true
	return nil
}

// Add adds vectors to the index
func (idx *IndexIVFPQ) Add(vectors []float32) error {
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
func (idx *IndexIVFPQ) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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

// Reset removes all vectors
func (idx *IndexIVFPQ) Reset() error {
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
func (idx *IndexIVFPQ) Close() error {
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
