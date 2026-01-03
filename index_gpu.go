//go:build gpu
// +build gpu

package faiss

import (
	"fmt"
	"runtime"
)

// ========================================
// GpuIndexFlat
// ========================================

// GpuIndexFlat is a flat (brute-force) GPU index
// 10-100x faster than CPU for large batches
//
// Python equivalent: faiss.GpuIndexFlatL2 / GpuIndexFlatIP
//
// Example:
//   res, _ := faiss.NewStandardGpuResources()
//   index, _ := faiss.NewGpuIndexFlatL2(res, 128, 0)
//   index.Add(vectors)
//   distances, indices, _ := index.Search(queries, 10)
type GpuIndexFlat struct {
	ptr       uintptr
	resources *StandardGpuResources
	deviceID  int
	d         int
	metric    MetricType
	ntotal    int64
}

// Ensure GpuIndexFlat implements Index
var _ Index = (*GpuIndexFlat)(nil)

// NewGpuIndexFlatL2 creates a GPU flat index with L2 metric
func NewGpuIndexFlatL2(res *StandardGpuResources, d, device int) (*GpuIndexFlat, error) {
	return newGpuIndexFlat(res, d, device, MetricL2)
}

// NewGpuIndexFlatIP creates a GPU flat index with inner product metric
func NewGpuIndexFlatIP(res *StandardGpuResources, d, device int) (*GpuIndexFlat, error) {
	return newGpuIndexFlat(res, d, device, MetricInnerProduct)
}

func newGpuIndexFlat(res *StandardGpuResources, d, device int, metric MetricType) (*GpuIndexFlat, error) {
	if res == nil {
		return nil, fmt.Errorf("GPU resources cannot be nil")
	}
	if d <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	var ptr uintptr
	err := faiss_GpuIndexFlat_new(&ptr, res.ptr, device, int64(d), int(metric))
	if err != nil {
		return nil, fmt.Errorf("failed to create GPU flat index: %w", err)
	}

	idx := &GpuIndexFlat{
		ptr:       ptr,
		resources: res,
		deviceID:  device,
		d:         d,
		metric:    metric,
		ntotal:    0,
	}

	runtime.SetFinalizer(idx, func(idx *GpuIndexFlat) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension
func (idx *GpuIndexFlat) D() int {
	return idx.d
}

// Ntotal returns the number of vectors
func (idx *GpuIndexFlat) Ntotal() int64 {
	ntotal := faiss_Index_ntotal(idx.ptr)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns true (flat indexes don't need training)
func (idx *GpuIndexFlat) IsTrained() bool {
	return true
}

// MetricType returns the distance metric
func (idx *GpuIndexFlat) MetricType() MetricType {
	return idx.metric
}

// Train is a no-op for flat indexes
func (idx *GpuIndexFlat) Train(vectors []float32) error {
	return nil
}

// Add adds vectors to the GPU index
func (idx *GpuIndexFlat) Add(vectors []float32) error {
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

// Search performs k-NN search on GPU
func (idx *GpuIndexFlat) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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

// SetNprobe is not supported for GPU flat indexes (not an IVF index)
func (idx *GpuIndexFlat) SetNprobe(nprobe int) error {
	return fmt.Errorf("faiss: SetNprobe not supported for GpuIndexFlat (not an IVF index)")
}

// SetEfSearch is not supported for GPU flat indexes (not an HNSW index)
func (idx *GpuIndexFlat) SetEfSearch(efSearch int) error {
	return fmt.Errorf("faiss: SetEfSearch not supported for GpuIndexFlat (not an HNSW index)")
}

// Reset removes all vectors
func (idx *GpuIndexFlat) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the GPU index
func (idx *GpuIndexFlat) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// ========================================
// GpuIndexIVFFlat
// ========================================

// GpuIndexIVFFlat is a GPU IVF flat index
//
// Python equivalent: faiss.GpuIndexIVFFlat
//
// Example:
//   res, _ := faiss.NewStandardGpuResources()
//   quantizer, _ := faiss.NewGpuIndexFlatL2(res, 128, 0)
//   index, _ := faiss.NewGpuIndexIVFFlat(res, quantizer, 128, 100, 0, faiss.MetricL2)
//   index.Train(trainingVectors)
//   index.SetNprobe(10)
//   index.Add(vectors)
type GpuIndexIVFFlat struct {
	ptr       uintptr
	resources *StandardGpuResources
	quantizer Index
	deviceID  int
	d         int
	metric    MetricType
	ntotal    int64
	isTrained bool
	nlist     int
	nprobe    int
}

// Ensure GpuIndexIVFFlat implements Index
var _ Index = (*GpuIndexIVFFlat)(nil)

// NewGpuIndexIVFFlat creates a GPU IVF flat index
func NewGpuIndexIVFFlat(res *StandardGpuResources, quantizer Index, d, nlist, device int, metric MetricType) (*GpuIndexIVFFlat, error) {
	if res == nil {
		return nil, fmt.Errorf("GPU resources cannot be nil")
	}
	if quantizer == nil {
		return nil, fmt.Errorf("quantizer cannot be nil")
	}
	if d <= 0 || nlist <= 0 {
		return nil, fmt.Errorf("d and nlist must be positive")
	}

	var quantizerPtr uintptr
	switch q := quantizer.(type) {
	case *GpuIndexFlat:
		quantizerPtr = q.ptr
	case *IndexFlat:
		quantizerPtr = q.ptr
	default:
		return nil, fmt.Errorf("unsupported quantizer type")
	}

	var ptr uintptr
	err := faiss_GpuIndexIVFFlat_new(&ptr, res.ptr, device, quantizerPtr, int64(d), int64(nlist), int(metric))
	if err != nil {
		return nil, fmt.Errorf("failed to create GPU IVF flat index: %w", err)
	}

	idx := &GpuIndexIVFFlat{
		ptr:       ptr,
		resources: res,
		quantizer: quantizer,
		deviceID:  device,
		d:         d,
		metric:    metric,
		ntotal:    0,
		isTrained: false,
		nlist:     nlist,
		nprobe:    1,
	}

	runtime.SetFinalizer(idx, func(idx *GpuIndexIVFFlat) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension
func (idx *GpuIndexIVFFlat) D() int {
	return idx.d
}

// Ntotal returns the number of vectors
func (idx *GpuIndexIVFFlat) Ntotal() int64 {
	ntotal := faiss_Index_ntotal(idx.ptr)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index is trained
func (idx *GpuIndexIVFFlat) IsTrained() bool {
	isTrained := faiss_Index_is_trained(idx.ptr)
	idx.isTrained = (isTrained != 0)
	return idx.isTrained
}

// MetricType returns the distance metric
func (idx *GpuIndexIVFFlat) MetricType() MetricType {
	return idx.metric
}

// Nlist returns the number of clusters
func (idx *GpuIndexIVFFlat) Nlist() int {
	return idx.nlist
}

// Nprobe returns the number of clusters to probe
func (idx *GpuIndexIVFFlat) Nprobe() int {
	return idx.nprobe
}

// SetNprobe sets the number of clusters to probe
func (idx *GpuIndexIVFFlat) SetNprobe(nprobe int) error {
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

// Train trains the index on GPU
func (idx *GpuIndexIVFFlat) Train(vectors []float32) error {
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

// Add adds vectors to the GPU index
func (idx *GpuIndexIVFFlat) Add(vectors []float32) error {
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

// Search performs k-NN search on GPU
func (idx *GpuIndexIVFFlat) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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

// SetEfSearch is not supported for GPU IVF indexes (not an HNSW index)
func (idx *GpuIndexIVFFlat) SetEfSearch(efSearch int) error {
	return fmt.Errorf("faiss: SetEfSearch not supported for GpuIndexIVFFlat (not an HNSW index)")
}

// Reset removes all vectors
func (idx *GpuIndexIVFFlat) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the GPU index
func (idx *GpuIndexIVFFlat) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}
