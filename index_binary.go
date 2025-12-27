package faiss

import (
	"fmt"
	"runtime"
)

// BinaryIndex is the interface for binary vector indexes
// Binary indexes use Hamming distance on binary vectors (uint8 arrays)
type BinaryIndex interface {
	D() int         // dimension in bits
	Ntotal() int64  // number of vectors
	IsTrained() bool
	Train(vectors []uint8) error
	Add(vectors []uint8) error
	Search(queries []uint8, k int) (distances []int32, indices []int64, err error)
	Reset() error
	Close() error
}

// IndexBinaryFlat is a flat (brute-force) index for binary vectors
// Uses Hamming distance for similarity
//
// Python equivalent: faiss.IndexBinaryFlat
//
// Example:
//   // 256-bit binary vectors
//   index, _ := faiss.NewIndexBinaryFlat(256)
//
//   // Binary vectors are stored as uint8 arrays (256 bits = 32 bytes)
//   vectors := make([]uint8, 1000 * 32)  // 1000 vectors
//   // ... fill with binary data ...
//   index.Add(vectors)
//
//   distances, indices, _ := index.Search(queries, 10)
type IndexBinaryFlat struct {
	ptr       uintptr // C pointer
	d         int     // dimension in bits
	ntotal    int64   // number of vectors
	isTrained bool    // always true for flat index
}

// Ensure IndexBinaryFlat implements BinaryIndex
var _ BinaryIndex = (*IndexBinaryFlat)(nil)

// NewIndexBinaryFlat creates a new flat binary index
// d is the dimension in bits (must be multiple of 8)
func NewIndexBinaryFlat(d int) (*IndexBinaryFlat, error) {
	if d <= 0 || d%8 != 0 {
		return nil, fmt.Errorf("dimension must be positive and multiple of 8")
	}

	var ptr uintptr
	ret := faiss_IndexBinaryFlat_new(&ptr, int64(d))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexBinaryFlat")
	}

	idx := &IndexBinaryFlat{
		ptr:       ptr,
		d:         d,
		ntotal:    0,
		isTrained: true, // flat index doesn't need training
	}

	runtime.SetFinalizer(idx, func(idx *IndexBinaryFlat) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension in bits
func (idx *IndexBinaryFlat) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexBinaryFlat) Ntotal() int64 {
	var ntotal int64
	faiss_IndexBinary_ntotal(idx.ptr, &ntotal)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained (always true for flat)
func (idx *IndexBinaryFlat) IsTrained() bool {
	return idx.isTrained
}

// Train is a no-op for flat indexes (they don't need training)
func (idx *IndexBinaryFlat) Train(vectors []uint8) error {
	// Flat index doesn't need training
	return nil
}

// Add adds binary vectors to the index
// Each vector should be d/8 bytes (d bits)
func (idx *IndexBinaryFlat) Add(vectors []uint8) error {
	if len(vectors) == 0 {
		return nil
	}

	bytesPerVec := idx.d / 8
	if len(vectors)%bytesPerVec != 0 {
		return fmt.Errorf("vectors length must be multiple of %d bytes (dimension %d bits)", bytesPerVec, idx.d)
	}

	n := int64(len(vectors) / bytesPerVec)
	ret := faiss_IndexBinary_add(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("add failed")
	}

	idx.ntotal += n
	return nil
}

// Search performs k-NN search on binary vectors
// Returns Hamming distances (int32) and indices
func (idx *IndexBinaryFlat) Search(queries []uint8, k int) (distances []int32, indices []int64, err error) {
	if len(queries) == 0 {
		return nil, nil, fmt.Errorf("empty query vectors")
	}

	bytesPerVec := idx.d / 8
	if len(queries)%bytesPerVec != 0 {
		return nil, nil, fmt.Errorf("queries length must be multiple of %d bytes (dimension %d bits)", bytesPerVec, idx.d)
	}

	nq := int64(len(queries) / bytesPerVec)
	distances = make([]int32, nq*int64(k))
	indices = make([]int64, nq*int64(k))

	ret := faiss_IndexBinary_search(idx.ptr, nq, &queries[0], int64(k), &distances[0], &indices[0])
	if ret != 0 {
		return nil, nil, fmt.Errorf("search failed")
	}

	return distances, indices, nil
}

// Reset removes all vectors from the index
func (idx *IndexBinaryFlat) Reset() error {
	ret := faiss_IndexBinary_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexBinaryFlat) Close() error {
	if idx.ptr != 0 {
		faiss_IndexBinary_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// ========================================
// IndexBinaryIVF
// ========================================

// IndexBinaryIVF is an inverted file index for binary vectors
//
// Python equivalent: faiss.IndexBinaryIVF
//
// Example:
//   quantizer, _ := faiss.NewIndexBinaryFlat(256)
//   index, _ := faiss.NewIndexBinaryIVF(quantizer, 256, 100)
//   index.Train(trainingVectors)
//   index.SetNprobe(10)
//   index.Add(vectors)
type IndexBinaryIVF struct {
	ptr       uintptr      // C pointer
	quantizer BinaryIndex  // coarse quantizer
	d         int          // dimension in bits
	ntotal    int64        // number of vectors
	isTrained bool         // training status
	nlist     int          // number of clusters
	nprobe    int          // number of clusters to probe
}

// Ensure IndexBinaryIVF implements BinaryIndex
var _ BinaryIndex = (*IndexBinaryIVF)(nil)

// NewIndexBinaryIVF creates a new IVF binary index
func NewIndexBinaryIVF(quantizer BinaryIndex, d, nlist int) (*IndexBinaryIVF, error) {
	if quantizer == nil {
		return nil, fmt.Errorf("quantizer cannot be nil")
	}
	if d <= 0 || d%8 != 0 {
		return nil, fmt.Errorf("dimension must be positive and multiple of 8")
	}
	if nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive")
	}

	// Get the quantizer pointer based on type
	var quantizerPtr uintptr
	switch q := quantizer.(type) {
	case *IndexBinaryFlat:
		quantizerPtr = q.ptr
	default:
		return nil, fmt.Errorf("unsupported quantizer type")
	}

	var ptr uintptr
	ret := faiss_IndexBinaryIVF_new(&ptr, quantizerPtr, int64(d), int64(nlist))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexBinaryIVF")
	}

	idx := &IndexBinaryIVF{
		ptr:       ptr,
		quantizer: quantizer,
		d:         d,
		ntotal:    0,
		isTrained: false,
		nlist:     nlist,
		nprobe:    1,
	}

	runtime.SetFinalizer(idx, func(idx *IndexBinaryIVF) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension in bits
func (idx *IndexBinaryIVF) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexBinaryIVF) Ntotal() int64 {
	var ntotal int64
	faiss_IndexBinary_ntotal(idx.ptr, &ntotal)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained
func (idx *IndexBinaryIVF) IsTrained() bool {
	var isTrained int
	faiss_IndexBinary_is_trained(idx.ptr, &isTrained)
	idx.isTrained = (isTrained != 0)
	return idx.isTrained
}

// Nlist returns the number of clusters
func (idx *IndexBinaryIVF) Nlist() int {
	return idx.nlist
}

// Nprobe returns the number of clusters to probe during search
func (idx *IndexBinaryIVF) Nprobe() int {
	return idx.nprobe
}

// SetNprobe sets the number of clusters to probe during search
func (idx *IndexBinaryIVF) SetNprobe(nprobe int) error {
	if nprobe < 1 || nprobe > idx.nlist {
		return fmt.Errorf("nprobe must be between 1 and %d", idx.nlist)
	}

	ret := faiss_IndexBinaryIVF_set_nprobe(idx.ptr, int64(nprobe))
	if ret != 0 {
		return fmt.Errorf("failed to set nprobe")
	}

	idx.nprobe = nprobe
	return nil
}

// Train trains the index on the given binary vectors
func (idx *IndexBinaryIVF) Train(vectors []uint8) error {
	if len(vectors) == 0 {
		return fmt.Errorf("empty training vectors")
	}

	bytesPerVec := idx.d / 8
	if len(vectors)%bytesPerVec != 0 {
		return fmt.Errorf("vectors length must be multiple of %d bytes (dimension %d bits)", bytesPerVec, idx.d)
	}

	n := int64(len(vectors) / bytesPerVec)
	if n < int64(idx.nlist) {
		return fmt.Errorf("need at least %d training vectors for %d clusters", idx.nlist, idx.nlist)
	}

	ret := faiss_IndexBinary_train(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("training failed")
	}

	idx.isTrained = true
	return nil
}

// Add adds binary vectors to the index
func (idx *IndexBinaryIVF) Add(vectors []uint8) error {
	if !idx.IsTrained() {
		return fmt.Errorf("index must be trained before adding vectors")
	}
	if len(vectors) == 0 {
		return nil
	}

	bytesPerVec := idx.d / 8
	if len(vectors)%bytesPerVec != 0 {
		return fmt.Errorf("vectors length must be multiple of %d bytes (dimension %d bits)", bytesPerVec, idx.d)
	}

	n := int64(len(vectors) / bytesPerVec)
	ret := faiss_IndexBinary_add(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("add failed")
	}

	idx.ntotal += n
	return nil
}

// Search performs k-NN search on binary vectors
func (idx *IndexBinaryIVF) Search(queries []uint8, k int) (distances []int32, indices []int64, err error) {
	if len(queries) == 0 {
		return nil, nil, fmt.Errorf("empty query vectors")
	}

	bytesPerVec := idx.d / 8
	if len(queries)%bytesPerVec != 0 {
		return nil, nil, fmt.Errorf("queries length must be multiple of %d bytes (dimension %d bits)", bytesPerVec, idx.d)
	}

	nq := int64(len(queries) / bytesPerVec)
	distances = make([]int32, nq*int64(k))
	indices = make([]int64, nq*int64(k))

	ret := faiss_IndexBinary_search(idx.ptr, nq, &queries[0], int64(k), &distances[0], &indices[0])
	if ret != 0 {
		return nil, nil, fmt.Errorf("search failed")
	}

	return distances, indices, nil
}

// Reset removes all vectors from the index
func (idx *IndexBinaryIVF) Reset() error {
	ret := faiss_IndexBinary_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexBinaryIVF) Close() error {
	if idx.ptr != 0 {
		faiss_IndexBinary_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// ========================================
// IndexBinaryHash (LSH for binary vectors)
// ========================================

// IndexBinaryHash is a locality-sensitive hashing index for binary vectors
//
// Python equivalent: faiss.IndexBinaryHash
//
// Example:
//   index, _ := faiss.NewIndexBinaryHash(256, 64)  // 256-bit vectors, 64-bit hash
//   index.Add(vectors)
//   distances, indices, _ := index.Search(queries, 10)
type IndexBinaryHash struct {
	ptr       uintptr // C pointer
	d         int     // dimension in bits
	nbits     int     // number of hash bits
	ntotal    int64   // number of vectors
	isTrained bool    // always true (no training needed)
}

// Ensure IndexBinaryHash implements BinaryIndex
var _ BinaryIndex = (*IndexBinaryHash)(nil)

// NewIndexBinaryHash creates a new binary LSH index
// d is the dimension in bits
// nbits is the number of hash bits to use
func NewIndexBinaryHash(d, nbits int) (*IndexBinaryHash, error) {
	if d <= 0 || d%8 != 0 {
		return nil, fmt.Errorf("dimension must be positive and multiple of 8")
	}
	if nbits <= 0 || nbits > d {
		return nil, fmt.Errorf("nbits must be positive and <= d")
	}

	var ptr uintptr
	ret := faiss_IndexBinaryHash_new(&ptr, int64(d), int64(nbits))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexBinaryHash")
	}

	idx := &IndexBinaryHash{
		ptr:       ptr,
		d:         d,
		nbits:     nbits,
		ntotal:    0,
		isTrained: true,
	}

	runtime.SetFinalizer(idx, func(idx *IndexBinaryHash) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension in bits
func (idx *IndexBinaryHash) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexBinaryHash) Ntotal() int64 {
	var ntotal int64
	faiss_IndexBinary_ntotal(idx.ptr, &ntotal)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained (always true)
func (idx *IndexBinaryHash) IsTrained() bool {
	return idx.isTrained
}

// Nbits returns the number of hash bits
func (idx *IndexBinaryHash) Nbits() int {
	return idx.nbits
}

// Train is a no-op (binary hash doesn't need training)
func (idx *IndexBinaryHash) Train(vectors []uint8) error {
	return nil
}

// Add adds binary vectors to the index
func (idx *IndexBinaryHash) Add(vectors []uint8) error {
	if len(vectors) == 0 {
		return nil
	}

	bytesPerVec := idx.d / 8
	if len(vectors)%bytesPerVec != 0 {
		return fmt.Errorf("vectors length must be multiple of %d bytes (dimension %d bits)", bytesPerVec, idx.d)
	}

	n := int64(len(vectors) / bytesPerVec)
	ret := faiss_IndexBinary_add(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("add failed")
	}

	idx.ntotal += n
	return nil
}

// Search performs k-NN search on binary vectors
func (idx *IndexBinaryHash) Search(queries []uint8, k int) (distances []int32, indices []int64, err error) {
	if len(queries) == 0 {
		return nil, nil, fmt.Errorf("empty query vectors")
	}

	bytesPerVec := idx.d / 8
	if len(queries)%bytesPerVec != 0 {
		return nil, nil, fmt.Errorf("queries length must be multiple of %d bytes (dimension %d bits)", bytesPerVec, idx.d)
	}

	nq := int64(len(queries) / bytesPerVec)
	distances = make([]int32, nq*int64(k))
	indices = make([]int64, nq*int64(k))

	ret := faiss_IndexBinary_search(idx.ptr, nq, &queries[0], int64(k), &distances[0], &indices[0])
	if ret != 0 {
		return nil, nil, fmt.Errorf("search failed")
	}

	return distances, indices, nil
}

// Reset removes all vectors from the index
func (idx *IndexBinaryHash) Reset() error {
	ret := faiss_IndexBinary_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexBinaryHash) Close() error {
	if idx.ptr != 0 {
		faiss_IndexBinary_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}
