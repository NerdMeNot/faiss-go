package faiss

import (
	"fmt"
	"runtime"
)

// IndexPQFastScan is a SIMD-optimized Product Quantization index
// Uses AVX2/AVX-512 instructions for 2-4x faster search than regular IndexPQ
//
// Python equivalent: faiss.IndexPQFastScan
//
// Example:
//   // 128-dim vectors, 8 subquantizers, 4 bits each
//   index, _ := faiss.NewIndexPQFastScan(128, 8, 4, faiss.MetricL2)
//   index.Train(trainingVectors)
//   index.Add(vectors)
//
// Note: FastScan requires nbits=4 for optimal SIMD performance
type IndexPQFastScan struct {
	ptr       uintptr    // C pointer
	d         int        // dimension
	metric    MetricType // metric type
	ntotal    int64      // number of vectors
	isTrained bool       // training status
	M         int        // number of subquantizers
	nbits     int        // bits per subquantizer (4 recommended)
	bbs       int        // block size for SIMD (default 32)
}

// Ensure IndexPQFastScan implements Index
var _ Index = (*IndexPQFastScan)(nil)

// NewIndexPQFastScan creates a new fast scan PQ index
// nbits should be 4 for optimal SIMD performance
func NewIndexPQFastScan(d, M, nbits int, metric MetricType) (*IndexPQFastScan, error) {
	if d <= 0 || M <= 0 {
		return nil, fmt.Errorf("d and M must be positive")
	}
	if d%M != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M=%d", d, M)
	}
	if nbits != 4 && nbits != 5 && nbits != 6 {
		return nil, fmt.Errorf("nbits must be 4, 5, or 6 for FastScan (4 is optimal)")
	}

	var ptr uintptr
	ret := faiss_IndexPQFastScan_new(&ptr, int64(d), int64(M), int64(nbits), int(metric))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexPQFastScan")
	}

	idx := &IndexPQFastScan{
		ptr:       ptr,
		d:         d,
		metric:    metric,
		ntotal:    0,
		isTrained: false,
		M:         M,
		nbits:     nbits,
		bbs:       32, // default block size
	}

	runtime.SetFinalizer(idx, func(idx *IndexPQFastScan) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension of the index
func (idx *IndexPQFastScan) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexPQFastScan) Ntotal() int64 {
	ntotal := faiss_Index_ntotal(idx.ptr)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained
func (idx *IndexPQFastScan) IsTrained() bool {
	isTrained := faiss_Index_is_trained(idx.ptr)
	idx.isTrained = (isTrained != 0)
	return idx.isTrained
}

// MetricType returns the distance metric used
func (idx *IndexPQFastScan) MetricType() MetricType {
	return idx.metric
}

// M returns the number of subquantizers
func (idx *IndexPQFastScan) GetM() int {
	return idx.M
}

// Nbits returns the number of bits per subquantizer
func (idx *IndexPQFastScan) Nbits() int {
	return idx.nbits
}

// BlockSize returns the SIMD block size
func (idx *IndexPQFastScan) BlockSize() int {
	return idx.bbs
}

// SetBlockSize sets the SIMD block size (must be multiple of 32)
func (idx *IndexPQFastScan) SetBlockSize(bbs int) error {
	if bbs <= 0 || bbs%32 != 0 {
		return fmt.Errorf("block size must be positive multiple of 32")
	}
	ret := faiss_IndexPQFastScan_set_bbs(idx.ptr, int64(bbs))
	if ret != 0 {
		return fmt.Errorf("failed to set block size")
	}
	idx.bbs = bbs
	return nil
}

// Train trains the index on the given vectors
func (idx *IndexPQFastScan) Train(vectors []float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("empty training vectors")
	}
	if len(vectors)%idx.d != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension %d", idx.d)
	}

	n := int64(len(vectors) / idx.d)
	ret := faiss_Index_train(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("training failed")
	}

	idx.isTrained = true
	return nil
}

// Add adds vectors to the index
func (idx *IndexPQFastScan) Add(vectors []float32) error {
	if !idx.IsTrained() {
		return fmt.Errorf("index must be trained before adding vectors")
	}
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

// Search performs k-NN search with SIMD acceleration
func (idx *IndexPQFastScan) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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

// Reset removes all vectors from the index
func (idx *IndexPQFastScan) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexPQFastScan) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// CompressionRatio returns the compression ratio achieved
func (idx *IndexPQFastScan) CompressionRatio() float64 {
	// 32 bits per float divided by bits used
	return 32.0 / float64(idx.nbits)
}

// ========================================
// IndexIVFPQFastScan
// ========================================

// IndexIVFPQFastScan combines IVF with SIMD-optimized PQ
//
// Python equivalent: faiss.IndexIVFPQFastScan
//
// Example:
//   quantizer, _ := faiss.NewIndexFlatL2(128)
//   index, _ := faiss.NewIndexIVFPQFastScan(quantizer, 128, 100, 8, 4, faiss.MetricL2)
//   index.Train(trainingVectors)
//   index.SetNprobe(10)
//   index.Add(vectors)
type IndexIVFPQFastScan struct {
	ptr       uintptr    // C pointer
	quantizer Index      // coarse quantizer
	d         int        // dimension
	metric    MetricType // metric type
	ntotal    int64      // number of vectors
	isTrained bool       // training status
	nlist     int        // number of clusters
	nprobe    int        // number of clusters to probe
	M         int        // number of subquantizers
	nbits     int        // bits per subquantizer
	bbs       int        // block size for SIMD
}

// Ensure IndexIVFPQFastScan implements Index
var _ Index = (*IndexIVFPQFastScan)(nil)

// NewIndexIVFPQFastScan creates a new IVF + fast scan PQ index
func NewIndexIVFPQFastScan(quantizer Index, d, nlist, M, nbits int, metric MetricType) (*IndexIVFPQFastScan, error) {
	if quantizer == nil {
		return nil, fmt.Errorf("quantizer cannot be nil")
	}
	if d <= 0 || nlist <= 0 || M <= 0 {
		return nil, fmt.Errorf("d, nlist, and M must be positive")
	}
	if d%M != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M=%d", d, M)
	}
	if nbits != 4 && nbits != 5 && nbits != 6 {
		return nil, fmt.Errorf("nbits must be 4, 5, or 6 for FastScan (4 is optimal)")
	}

	// Get the quantizer pointer based on type
	var quantizerPtr uintptr
	switch q := quantizer.(type) {
	case *IndexFlat:
		quantizerPtr = q.ptr
	default:
		return nil, fmt.Errorf("unsupported quantizer type")
	}

	var ptr uintptr
	ret := faiss_IndexIVFPQFastScan_new(&ptr, quantizerPtr, int64(d), int64(nlist), int64(M), int64(nbits), int(metric))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexIVFPQFastScan")
	}

	idx := &IndexIVFPQFastScan{
		ptr:       ptr,
		quantizer: quantizer,
		d:         d,
		metric:    metric,
		ntotal:    0,
		isTrained: false,
		nlist:     nlist,
		nprobe:    1,
		M:         M,
		nbits:     nbits,
		bbs:       32,
	}

	runtime.SetFinalizer(idx, func(idx *IndexIVFPQFastScan) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension of the index
func (idx *IndexIVFPQFastScan) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexIVFPQFastScan) Ntotal() int64 {
	ntotal := faiss_Index_ntotal(idx.ptr)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained
func (idx *IndexIVFPQFastScan) IsTrained() bool {
	isTrained := faiss_Index_is_trained(idx.ptr)
	idx.isTrained = (isTrained != 0)
	return idx.isTrained
}

// MetricType returns the distance metric used
func (idx *IndexIVFPQFastScan) MetricType() MetricType {
	return idx.metric
}

// Nlist returns the number of clusters
func (idx *IndexIVFPQFastScan) Nlist() int {
	return idx.nlist
}

// Nprobe returns the number of clusters to probe during search
func (idx *IndexIVFPQFastScan) Nprobe() int {
	return idx.nprobe
}

// SetNprobe sets the number of clusters to probe during search
func (idx *IndexIVFPQFastScan) SetNprobe(nprobe int) error {
	if nprobe < 1 || nprobe > idx.nlist {
		return fmt.Errorf("nprobe must be between 1 and %d", idx.nlist)
	}

	ret := faiss_IndexIVF_set_nprobe(idx.ptr, int64(nprobe))
	if ret != 0 {
		return fmt.Errorf("failed to set nprobe")
	}

	idx.nprobe = nprobe
	return nil
}

// GetM returns the number of subquantizers
func (idx *IndexIVFPQFastScan) GetM() int {
	return idx.M
}

// Nbits returns the number of bits per subquantizer
func (idx *IndexIVFPQFastScan) Nbits() int {
	return idx.nbits
}

// BlockSize returns the SIMD block size
func (idx *IndexIVFPQFastScan) BlockSize() int {
	return idx.bbs
}

// SetBlockSize sets the SIMD block size
func (idx *IndexIVFPQFastScan) SetBlockSize(bbs int) error {
	if bbs <= 0 || bbs%32 != 0 {
		return fmt.Errorf("block size must be positive multiple of 32")
	}
	ret := faiss_IndexPQFastScan_set_bbs(idx.ptr, int64(bbs))
	if ret != 0 {
		return fmt.Errorf("failed to set block size")
	}
	idx.bbs = bbs
	return nil
}

// Train trains the index on the given vectors
func (idx *IndexIVFPQFastScan) Train(vectors []float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("empty training vectors")
	}
	if len(vectors)%idx.d != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension %d", idx.d)
	}

	n := int64(len(vectors) / idx.d)
	if n < int64(idx.nlist) {
		return fmt.Errorf("need at least %d training vectors for %d clusters", idx.nlist, idx.nlist)
	}

	ret := faiss_Index_train(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("training failed")
	}

	idx.isTrained = true
	return nil
}

// Add adds vectors to the index
func (idx *IndexIVFPQFastScan) Add(vectors []float32) error {
	if !idx.IsTrained() {
		return fmt.Errorf("index must be trained before adding vectors")
	}
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

// Search performs k-NN search with IVF + SIMD acceleration
func (idx *IndexIVFPQFastScan) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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

// Reset removes all vectors from the index
func (idx *IndexIVFPQFastScan) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexIVFPQFastScan) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// CompressionRatio returns the compression ratio achieved
func (idx *IndexIVFPQFastScan) CompressionRatio() float64 {
	return 32.0 / float64(idx.nbits)
}
