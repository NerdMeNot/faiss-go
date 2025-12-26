package faiss

import (
	"fmt"
	"runtime"
)

// QuantizerType specifies the type of scalar quantization
type QuantizerType int

const (
	// QT_8bit uses 8 bits per dimension
	QT_8bit QuantizerType = 0
	// QT_4bit uses 4 bits per dimension
	QT_4bit QuantizerType = 1
	// QT_8bit_uniform uses uniform 8-bit quantization
	QT_8bit_uniform QuantizerType = 2
	// QT_4bit_uniform uses uniform 4-bit quantization
	QT_4bit_uniform QuantizerType = 3
	// QT_fp16 uses 16-bit floating point
	QT_fp16 QuantizerType = 4
	// QT_8bit_direct uses direct 8-bit mapping
	QT_8bit_direct QuantizerType = 5
	// QT_6bit uses 6 bits per dimension
	QT_6bit QuantizerType = 6
)

// IndexScalarQuantizer is a scalar quantization index
// This index quantizes each dimension independently using scalar quantization,
// trading accuracy for memory savings.
//
// Python equivalent: faiss.IndexScalarQuantizer
//
// Example:
//   index, _ := faiss.NewIndexScalarQuantizer(128, faiss.QT_8bit, faiss.MetricL2)
//   index.Train(trainingVectors)
//   index.Add(vectors)
type IndexScalarQuantizer struct {
	ptr       uintptr       // C pointer
	d         int           // dimension
	metric    MetricType    // metric type
	ntotal    int64         // number of vectors
	isTrained bool          // training status
	qtype     QuantizerType // quantizer type
}

// Ensure IndexScalarQuantizer implements Index
var _ Index = (*IndexScalarQuantizer)(nil)

// NewIndexScalarQuantizer creates a new scalar quantizer index
func NewIndexScalarQuantizer(d int, qtype QuantizerType, metric MetricType) (*IndexScalarQuantizer, error) {
	var ptr uintptr
	ret := faiss_IndexScalarQuantizer_new(&ptr, int64(d), int(qtype), int(metric))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexScalarQuantizer")
	}

	idx := &IndexScalarQuantizer{
		ptr:       ptr,
		d:         d,
		metric:    metric,
		ntotal:    0,
		isTrained: false,
		qtype:     qtype,
	}

	runtime.SetFinalizer(idx, func(idx *IndexScalarQuantizer) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension of the index
func (idx *IndexScalarQuantizer) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexScalarQuantizer) Ntotal() int64 {
	var ntotal int64
	faiss_Index_ntotal(idx.ptr, &ntotal)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained
func (idx *IndexScalarQuantizer) IsTrained() bool {
	var isTrained int
	faiss_Index_is_trained(idx.ptr, &isTrained)
	idx.isTrained = (isTrained != 0)
	return idx.isTrained
}

// MetricType returns the distance metric used
func (idx *IndexScalarQuantizer) MetricType() MetricType {
	return idx.metric
}

// QuantizerType returns the quantizer type
func (idx *IndexScalarQuantizer) QuantizerType() QuantizerType {
	return idx.qtype
}

// Train trains the index on the given vectors
func (idx *IndexScalarQuantizer) Train(vectors []float32) error {
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
func (idx *IndexScalarQuantizer) Add(vectors []float32) error {
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
func (idx *IndexScalarQuantizer) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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
func (idx *IndexScalarQuantizer) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexScalarQuantizer) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// CompressionRatio returns the compression ratio achieved by scalar quantization
func (idx *IndexScalarQuantizer) CompressionRatio() float64 {
	// Calculate based on quantizer type
	bitsPerDim := 32.0 // float32 baseline
	switch idx.qtype {
	case QT_8bit, QT_8bit_uniform, QT_8bit_direct:
		return bitsPerDim / 8.0 // 4x compression
	case QT_4bit, QT_4bit_uniform:
		return bitsPerDim / 4.0 // 8x compression
	case QT_6bit:
		return bitsPerDim / 6.0 // ~5.3x compression
	case QT_fp16:
		return bitsPerDim / 16.0 // 2x compression
	default:
		return 1.0
	}
}

// ========================================
// IndexIVFScalarQuantizer
// ========================================

// IndexIVFScalarQuantizer combines IVF with scalar quantization
//
// Python equivalent: faiss.IndexIVFScalarQuantizer
//
// Example:
//   quantizer, _ := faiss.NewIndexFlatL2(128)
//   index, _ := faiss.NewIndexIVFScalarQuantizer(quantizer, 128, 100, faiss.QT_8bit, faiss.MetricL2)
//   index.Train(trainingVectors)
//   index.SetNprobe(10)
//   index.Add(vectors)
type IndexIVFScalarQuantizer struct {
	ptr       uintptr       // C pointer
	quantizer Index         // coarse quantizer
	d         int           // dimension
	metric    MetricType    // metric type
	ntotal    int64         // number of vectors
	isTrained bool          // training status
	nlist     int           // number of clusters
	nprobe    int           // number of clusters to probe
	qtype     QuantizerType // quantizer type
}

// Ensure IndexIVFScalarQuantizer implements Index
var _ Index = (*IndexIVFScalarQuantizer)(nil)

// NewIndexIVFScalarQuantizer creates a new IVF scalar quantizer index
func NewIndexIVFScalarQuantizer(quantizer Index, d, nlist int, qtype QuantizerType, metric MetricType) (*IndexIVFScalarQuantizer, error) {
	if quantizer == nil {
		return nil, fmt.Errorf("quantizer cannot be nil")
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
	ret := faiss_IndexIVFScalarQuantizer_new(&ptr, quantizerPtr, int64(d), int64(nlist), int(qtype), int(metric))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexIVFScalarQuantizer")
	}

	idx := &IndexIVFScalarQuantizer{
		ptr:       ptr,
		quantizer: quantizer,
		d:         d,
		metric:    metric,
		ntotal:    0,
		isTrained: false,
		nlist:     nlist,
		nprobe:    1,
		qtype:     qtype,
	}

	runtime.SetFinalizer(idx, func(idx *IndexIVFScalarQuantizer) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension of the index
func (idx *IndexIVFScalarQuantizer) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexIVFScalarQuantizer) Ntotal() int64 {
	var ntotal int64
	faiss_Index_ntotal(idx.ptr, &ntotal)
	idx.ntotal = ntotal
	return ntotal
}

// IsTrained returns whether the index has been trained
func (idx *IndexIVFScalarQuantizer) IsTrained() bool {
	var isTrained int
	faiss_Index_is_trained(idx.ptr, &isTrained)
	idx.isTrained = (isTrained != 0)
	return idx.isTrained
}

// MetricType returns the distance metric used
func (idx *IndexIVFScalarQuantizer) MetricType() MetricType {
	return idx.metric
}

// Nlist returns the number of clusters
func (idx *IndexIVFScalarQuantizer) Nlist() int {
	return idx.nlist
}

// Nprobe returns the number of clusters to probe during search
func (idx *IndexIVFScalarQuantizer) Nprobe() int {
	return idx.nprobe
}

// SetNprobe sets the number of clusters to probe during search
func (idx *IndexIVFScalarQuantizer) SetNprobe(nprobe int) error {
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

// QuantizerType returns the quantizer type
func (idx *IndexIVFScalarQuantizer) QuantizerType() QuantizerType {
	return idx.qtype
}

// Train trains the index on the given vectors
func (idx *IndexIVFScalarQuantizer) Train(vectors []float32) error {
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
func (idx *IndexIVFScalarQuantizer) Add(vectors []float32) error {
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
func (idx *IndexIVFScalarQuantizer) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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
func (idx *IndexIVFScalarQuantizer) Reset() error {
	ret := faiss_Index_reset(idx.ptr)
	if ret != 0 {
		return fmt.Errorf("reset failed")
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexIVFScalarQuantizer) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// CompressionRatio returns the compression ratio achieved by scalar quantization
func (idx *IndexIVFScalarQuantizer) CompressionRatio() float64 {
	bitsPerDim := 32.0 // float32 baseline
	switch idx.qtype {
	case QT_8bit, QT_8bit_uniform, QT_8bit_direct:
		return bitsPerDim / 8.0
	case QT_4bit, QT_4bit_uniform:
		return bitsPerDim / 4.0
	case QT_6bit:
		return bitsPerDim / 6.0
	case QT_fp16:
		return bitsPerDim / 16.0
	default:
		return 1.0
	}
}
