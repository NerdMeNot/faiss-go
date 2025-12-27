// +build !nogpu

package faiss

import (
	"fmt"
	"runtime"
)

// StandardGpuResources manages GPU memory and resources for FAISS
//
// Python equivalent: faiss.StandardGpuResources()
//
// Example:
//   res, _ := faiss.NewStandardGpuResources()
//   defer res.Close()
//
//   // Set temp memory to 512MB
//   res.SetTempMemory(512 * 1024 * 1024)
//
//   // Create GPU index
//   cpuIndex, _ := faiss.NewIndexFlatL2(128)
//   gpuIndex, _ := faiss.IndexCpuToGpu(res, 0, cpuIndex)
type StandardGpuResources struct {
	ptr        uintptr // C pointer
	tempMemory int64   // temp memory in bytes
	deviceID   int     // GPU device ID
}

// NewStandardGpuResources creates GPU resources with default settings
func NewStandardGpuResources() (*StandardGpuResources, error) {
	var ptr uintptr
	ret := faiss_StandardGpuResources_new(&ptr)
	if ret != 0 {
		return nil, fmt.Errorf("failed to create GPU resources (CUDA not available?)")
	}

	res := &StandardGpuResources{
		ptr:        ptr,
		tempMemory: 128 * 1024 * 1024, // default 128MB
		deviceID:   0,
	}

	runtime.SetFinalizer(res, func(r *StandardGpuResources) {
		r.Close()
	})

	return res, nil
}

// SetTempMemory sets the amount of temporary GPU memory (in bytes)
// Default is 128MB. Increase for large datasets.
func (res *StandardGpuResources) SetTempMemory(bytes int64) error {
	if bytes <= 0 {
		return fmt.Errorf("temp memory must be positive")
	}

	ret := faiss_StandardGpuResources_setTempMemory(res.ptr, bytes)
	if ret != 0 {
		return fmt.Errorf("failed to set temp memory")
	}

	res.tempMemory = bytes
	return nil
}

// GetTempMemory returns the amount of temporary GPU memory configured
func (res *StandardGpuResources) GetTempMemory() int64 {
	return res.tempMemory
}

// SetDefaultNullStreamAllDevices sets whether to use the null stream
func (res *StandardGpuResources) SetDefaultNullStreamAllDevices() error {
	ret := faiss_StandardGpuResources_setDefaultNullStreamAllDevices(res.ptr)
	if ret != 0 {
		return fmt.Errorf("failed to set null stream")
	}
	return nil
}

// Close frees GPU resources
func (res *StandardGpuResources) Close() error {
	if res.ptr != 0 {
		faiss_StandardGpuResources_free(res.ptr)
		res.ptr = 0
	}
	return nil
}

// ========================================
// GPU Configuration Options
// ========================================

// GpuClonerOptions controls how indexes are cloned to GPU
//
// Python equivalent: faiss.GpuClonerOptions
type GpuClonerOptions struct {
	useFloat16           bool // use float16 for vectors (saves memory)
	useFloat16CoarseQuantizer bool // use float16 for IVF quantizer
	usePrecomputed       bool // use precomputed tables
	indicesOptions       int  // how to handle indices
	verbose              bool // print debug info
}

// NewGpuClonerOptions creates default GPU cloner options
func NewGpuClonerOptions() *GpuClonerOptions {
	return &GpuClonerOptions{
		useFloat16:           false,
		useFloat16CoarseQuantizer: false,
		usePrecomputed:       false,
		indicesOptions:       0,
		verbose:              false,
	}
}

// SetUseFloat16 enables/disables float16 for vectors
// Saves 50% GPU memory but slightly reduces accuracy
func (opts *GpuClonerOptions) SetUseFloat16(enable bool) {
	opts.useFloat16 = enable
}

// SetUseFloat16CoarseQuantizer enables/disables float16 for IVF quantizer
func (opts *GpuClonerOptions) SetUseFloat16CoarseQuantizer(enable bool) {
	opts.useFloat16CoarseQuantizer = enable
}

// SetUsePrecomputed enables/disables precomputed tables
func (opts *GpuClonerOptions) SetUsePrecomputed(enable bool) {
	opts.usePrecomputed = enable
}

// SetVerbose enables/disables debug output
func (opts *GpuClonerOptions) SetVerbose(enable bool) {
	opts.verbose = enable
}

// ========================================
// GPU Index Transfer
// ========================================

// IndexCpuToGpu transfers a CPU index to GPU
//
// Python equivalent: faiss.index_cpu_to_gpu
//
// Example:
//   cpuIndex, _ := faiss.NewIndexFlatL2(128)
//   cpuIndex.Add(vectors)
//
//   res, _ := faiss.NewStandardGpuResources()
//   gpuIndex, _ := faiss.IndexCpuToGpu(res, 0, cpuIndex)
//   defer gpuIndex.Close()
//
//   // Search runs on GPU
//   distances, indices, _ := gpuIndex.Search(queries, 10)
func IndexCpuToGpu(res *StandardGpuResources, device int, index Index) (Index, error) {
	if res == nil {
		return nil, fmt.Errorf("GPU resources cannot be nil")
	}
	if index == nil {
		return nil, fmt.Errorf("index cannot be nil")
	}

	var indexPtr uintptr
	switch idx := index.(type) {
	case *IndexFlat:
		indexPtr = idx.ptr
	case *IndexIVFFlat:
		indexPtr = idx.ptr
	case *IndexPQ:
		indexPtr = idx.ptr
	case *IndexIVFPQ:
		indexPtr = idx.ptr
	default:
		return nil, fmt.Errorf("unsupported index type for GPU transfer")
	}

	var gpuPtr uintptr
	ret := faiss_index_cpu_to_gpu(res.ptr, int64(device), indexPtr, &gpuPtr)
	if ret != 0 {
		return nil, fmt.Errorf("failed to transfer index to GPU")
	}

	// Wrap in generic GPU index
	gpuIndex := &GpuIndex{
		ptr:       gpuPtr,
		resources: res,
		deviceID:  device,
		d:         index.D(),
		metric:    index.MetricType(),
		ntotal:    index.Ntotal(),
	}

	runtime.SetFinalizer(gpuIndex, func(idx *GpuIndex) {
		idx.Close()
	})

	return gpuIndex, nil
}

// IndexGpuToCpu transfers a GPU index back to CPU
//
// Python equivalent: faiss.index_gpu_to_cpu
func IndexGpuToCpu(gpuIndex Index) (Index, error) {
	if gpuIndex == nil {
		return nil, fmt.Errorf("GPU index cannot be nil")
	}

	var gpuPtr uintptr
	switch idx := gpuIndex.(type) {
	case *GpuIndex:
		gpuPtr = idx.ptr
	case *GpuIndexFlat:
		gpuPtr = idx.ptr
	case *GpuIndexIVFFlat:
		gpuPtr = idx.ptr
	default:
		return nil, fmt.Errorf("not a GPU index")
	}

	var cpuPtr uintptr
	var indexType [32]byte
	var d, metric int
	var ntotal int64

	ret := faiss_index_gpu_to_cpu(gpuPtr, &cpuPtr, &indexType[0], &d, &metric, &ntotal)
	if ret != 0 {
		return nil, fmt.Errorf("failed to transfer index to CPU")
	}

	// Determine index type and wrap appropriately
	// For now, return generic wrapper
	metricType := MetricType(metric)
	cpuIndex := &IndexFlat{
		ptr:       cpuPtr,
		d:         d,
		metric:    metricType,
		ntotal:    ntotal,
		isTrained: true,
	}

	runtime.SetFinalizer(cpuIndex, func(idx *IndexFlat) {
		idx.Close()
	})

	return cpuIndex, nil
}

// IndexCpuToAllGpus transfers an index to all available GPUs
//
// Python equivalent: faiss.index_cpu_to_all_gpus
func IndexCpuToAllGpus(index Index) (Index, error) {
	if index == nil {
		return nil, fmt.Errorf("index cannot be nil")
	}

	// Get number of GPUs
	var ngpus int
	ret := faiss_get_num_gpus(&ngpus)
	if ret != 0 || ngpus == 0 {
		return nil, fmt.Errorf("no GPUs available")
	}

	var indexPtr uintptr
	switch idx := index.(type) {
	case *IndexFlat:
		indexPtr = idx.ptr
	case *IndexIVFFlat:
		indexPtr = idx.ptr
	default:
		return nil, fmt.Errorf("unsupported index type for multi-GPU")
	}

	var gpuPtr uintptr
	ret = faiss_index_cpu_to_all_gpus(indexPtr, &gpuPtr)
	if ret != 0 {
		return nil, fmt.Errorf("failed to transfer index to all GPUs")
	}

	// Wrap in multi-GPU index
	multiGpuIndex := &GpuIndex{
		ptr:      gpuPtr,
		deviceID: -1, // indicates multi-GPU
		d:        index.D(),
		metric:   index.MetricType(),
		ntotal:   index.Ntotal(),
	}

	runtime.SetFinalizer(multiGpuIndex, func(idx *GpuIndex) {
		idx.Close()
	})

	return multiGpuIndex, nil
}

// GetNumGpus returns the number of available CUDA GPUs
//
// Python equivalent: faiss.get_num_gpus()
func GetNumGpus() int {
	var ngpus int
	faiss_get_num_gpus(&ngpus)
	return ngpus
}

// ========================================
// Generic GPU Index Wrapper
// ========================================

// GpuIndex is a generic wrapper for GPU indexes
type GpuIndex struct {
	ptr       uintptr
	resources *StandardGpuResources
	deviceID  int
	d         int
	metric    MetricType
	ntotal    int64
}

// Ensure GpuIndex implements Index
var _ Index = (*GpuIndex)(nil)

// D returns the dimension
func (idx *GpuIndex) D() int {
	return idx.d
}

// Ntotal returns the number of vectors
func (idx *GpuIndex) Ntotal() int64 {
	ntotal := faiss_Index_ntotal(idx.ptr)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether trained
func (idx *GpuIndex) IsTrained() bool {
	return true
}

// MetricType returns the metric
func (idx *GpuIndex) MetricType() MetricType {
	return idx.metric
}

// Train is typically not needed for GPU indexes
func (idx *GpuIndex) Train(vectors []float32) error {
	return nil
}

// Add adds vectors
func (idx *GpuIndex) Add(vectors []float32) error {
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
func (idx *GpuIndex) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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

// Reset removes all vectors
func (idx *GpuIndex) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the GPU index
func (idx *GpuIndex) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// DeviceID returns the GPU device ID (-1 for multi-GPU)
func (idx *GpuIndex) DeviceID() int {
	return idx.deviceID
}
