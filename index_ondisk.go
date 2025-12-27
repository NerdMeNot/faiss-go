package faiss

import (
	"fmt"
	"runtime"
)

// IndexIVFFlatOnDisk is an IVF index with vectors stored on disk
// Useful for datasets larger than RAM
//
// Python equivalent: faiss.IndexIVFFlat with ondisk_invlists
//
// Example:
//   quantizer, _ := faiss.NewIndexFlatL2(128)
//   index, _ := faiss.NewIndexIVFFlatOnDisk(quantizer, 128, 1000, "index.ivfdata", faiss.MetricL2)
//   index.Train(trainingVectors)
//   index.Add(vectors)  // Vectors stored on disk, not in RAM
type IndexIVFFlatOnDisk struct {
	ptr       uintptr    // C pointer
	quantizer Index      // coarse quantizer
	d         int        // dimension
	metric    MetricType // metric type
	ntotal    int64      // number of vectors
	isTrained bool       // training status
	nlist     int        // number of clusters
	nprobe    int        // number of clusters to probe
	filename  string     // on-disk storage filename
}

// Ensure IndexIVFFlatOnDisk implements Index
var _ Index = (*IndexIVFFlatOnDisk)(nil)

// NewIndexIVFFlatOnDisk creates a new on-disk IVF Flat index
// filename is where the inverted lists will be stored
func NewIndexIVFFlatOnDisk(quantizer Index, d, nlist int, filename string, metric MetricType) (*IndexIVFFlatOnDisk, error) {
	if quantizer == nil {
		return nil, fmt.Errorf("quantizer cannot be nil")
	}
	if d <= 0 || nlist <= 0 {
		return nil, fmt.Errorf("d and nlist must be positive")
	}
	if filename == "" {
		return nil, fmt.Errorf("filename cannot be empty")
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
	ret := faiss_IndexIVFFlatOnDisk_new(&ptr, quantizerPtr, int64(d), int64(nlist), filename, int(metric))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexIVFFlatOnDisk")
	}

	idx := &IndexIVFFlatOnDisk{
		ptr:       ptr,
		quantizer: quantizer,
		d:         d,
		metric:    metric,
		ntotal:    0,
		isTrained: false,
		nlist:     nlist,
		nprobe:    1,
		filename:  filename,
	}

	runtime.SetFinalizer(idx, func(idx *IndexIVFFlatOnDisk) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension of the index
func (idx *IndexIVFFlatOnDisk) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexIVFFlatOnDisk) Ntotal() int64 {
	var ntotal int64
	faiss_Index_ntotal(idx.ptr, &ntotal)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained
func (idx *IndexIVFFlatOnDisk) IsTrained() bool {
	var isTrained int
	faiss_Index_is_trained(idx.ptr, &isTrained)
	idx.isTrained = (isTrained != 0)
	return idx.isTrained
}

// MetricType returns the distance metric used
func (idx *IndexIVFFlatOnDisk) MetricType() MetricType {
	return idx.metric
}

// Nlist returns the number of clusters
func (idx *IndexIVFFlatOnDisk) Nlist() int {
	return idx.nlist
}

// Nprobe returns the number of clusters to probe during search
func (idx *IndexIVFFlatOnDisk) Nprobe() int {
	return idx.nprobe
}

// SetNprobe sets the number of clusters to probe during search
func (idx *IndexIVFFlatOnDisk) SetNprobe(nprobe int) error {
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

// Filename returns the on-disk storage filename
func (idx *IndexIVFFlatOnDisk) Filename() string {
	return idx.filename
}

// Train trains the index on the given vectors
func (idx *IndexIVFFlatOnDisk) Train(vectors []float32) error {
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

// Add adds vectors to the index (stored on disk)
func (idx *IndexIVFFlatOnDisk) Add(vectors []float32) error {
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

// Search performs k-NN search (reading from disk as needed)
func (idx *IndexIVFFlatOnDisk) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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
func (idx *IndexIVFFlatOnDisk) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index and flushes data to disk
func (idx *IndexIVFFlatOnDisk) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// ========================================
// IndexIVFPQOnDisk
// ========================================

// IndexIVFPQOnDisk is an IVF+PQ index with compressed vectors stored on disk
// Best for billion-scale datasets with limited RAM
//
// Python equivalent: faiss.IndexIVFPQ with ondisk_invlists
//
// Example:
//   quantizer, _ := faiss.NewIndexFlatL2(128)
//   index, _ := faiss.NewIndexIVFPQOnDisk(quantizer, 128, 1000, 8, 8, "index.ivfpq", faiss.MetricL2)
//   index.Train(trainingVectors)
//   index.Add(vectors)
type IndexIVFPQOnDisk struct {
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
	filename  string     // on-disk storage filename
}

// Ensure IndexIVFPQOnDisk implements Index
var _ Index = (*IndexIVFPQOnDisk)(nil)

// NewIndexIVFPQOnDisk creates a new on-disk IVF+PQ index
func NewIndexIVFPQOnDisk(quantizer Index, d, nlist, M, nbits int, filename string, metric MetricType) (*IndexIVFPQOnDisk, error) {
	if quantizer == nil {
		return nil, fmt.Errorf("quantizer cannot be nil")
	}
	if d <= 0 || nlist <= 0 || M <= 0 || nbits <= 0 {
		return nil, fmt.Errorf("d, nlist, M, and nbits must be positive")
	}
	if d%M != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M=%d", d, M)
	}
	if filename == "" {
		return nil, fmt.Errorf("filename cannot be empty")
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
	ret := faiss_IndexIVFPQOnDisk_new(&ptr, quantizerPtr, int64(d), int64(nlist), int64(M), int64(nbits), filename, int(metric))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexIVFPQOnDisk")
	}

	idx := &IndexIVFPQOnDisk{
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
		filename:  filename,
	}

	runtime.SetFinalizer(idx, func(idx *IndexIVFPQOnDisk) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension of the index
func (idx *IndexIVFPQOnDisk) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexIVFPQOnDisk) Ntotal() int64 {
	var ntotal int64
	faiss_Index_ntotal(idx.ptr, &ntotal)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained
func (idx *IndexIVFPQOnDisk) IsTrained() bool {
	var isTrained int
	faiss_Index_is_trained(idx.ptr, &isTrained)
	idx.isTrained = (isTrained != 0)
	return idx.isTrained
}

// MetricType returns the distance metric used
func (idx *IndexIVFPQOnDisk) MetricType() MetricType {
	return idx.metric
}

// Nlist returns the number of clusters
func (idx *IndexIVFPQOnDisk) Nlist() int {
	return idx.nlist
}

// Nprobe returns the number of clusters to probe during search
func (idx *IndexIVFPQOnDisk) Nprobe() int {
	return idx.nprobe
}

// SetNprobe sets the number of clusters to probe during search
func (idx *IndexIVFPQOnDisk) SetNprobe(nprobe int) error {
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
func (idx *IndexIVFPQOnDisk) GetM() int {
	return idx.M
}

// Nbits returns the number of bits per subquantizer
func (idx *IndexIVFPQOnDisk) Nbits() int {
	return idx.nbits
}

// Filename returns the on-disk storage filename
func (idx *IndexIVFPQOnDisk) Filename() string {
	return idx.filename
}

// Train trains the index on the given vectors
func (idx *IndexIVFPQOnDisk) Train(vectors []float32) error {
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

// Add adds vectors to the index (compressed and stored on disk)
func (idx *IndexIVFPQOnDisk) Add(vectors []float32) error {
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

// Search performs k-NN search
func (idx *IndexIVFPQOnDisk) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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
func (idx *IndexIVFPQOnDisk) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index and flushes data to disk
func (idx *IndexIVFPQOnDisk) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// CompressionRatio returns the compression ratio achieved
func (idx *IndexIVFPQOnDisk) CompressionRatio() float64 {
	// 32 bits per float divided by bits used per dimension
	bitsPerDim := float64(idx.nbits)
	return 32.0 / bitsPerDim
}
