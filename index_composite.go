package faiss

import (
	"fmt"
	"runtime"
)

// ========================================
// IndexRefine - Two-Stage Search
// ========================================

// IndexRefine performs a two-stage search: coarse with base_index, refine with refine_index
//
// Python equivalent: faiss.IndexRefine
//
// Example:
//   // Fast coarse search with IVF
//   baseIndex, _ := faiss.NewIndexIVFFlat(quantizer, 128, 100, faiss.MetricL2)
//
//   // Exact refinement with flat index
//   refineIndex, _ := faiss.NewIndexFlatL2(128)
//
//   // Combine
//   index, _ := faiss.NewIndexRefine(baseIndex, refineIndex)
//   index.SetK_factor(4)  // search 4x candidates in base, refine to k
type IndexRefine struct {
	ptr         uintptr // C pointer
	baseIndex   Index   // coarse index
	refineIndex Index   // refinement index
	d           int     // dimension
	metric      MetricType
	ntotal      int64
	isTrained   bool
	kFactor     float32 // multiplier for base search
}

// Ensure IndexRefine implements Index
var _ Index = (*IndexRefine)(nil)

// NewIndexRefine creates a new refine index
func NewIndexRefine(baseIndex, refineIndex Index) (*IndexRefine, error) {
	if baseIndex == nil || refineIndex == nil {
		return nil, fmt.Errorf("both base and refine indexes must be non-nil")
	}
	if baseIndex.D() != refineIndex.D() {
		return nil, fmt.Errorf("base and refine indexes must have same dimension")
	}
	if baseIndex.MetricType() != refineIndex.MetricType() {
		return nil, fmt.Errorf("base and refine indexes must use same metric")
	}

	// Get pointers based on type
	var basePtr, refinePtr uintptr
	switch b := baseIndex.(type) {
	case *IndexIVFFlat:
		basePtr = b.ptr
	case *IndexHNSW:
		basePtr = b.ptr
	case *IndexPQ:
		basePtr = b.ptr
	default:
		return nil, fmt.Errorf("unsupported base index type")
	}

	switch r := refineIndex.(type) {
	case *IndexFlat:
		refinePtr = r.ptr
	default:
		return nil, fmt.Errorf("unsupported refine index type")
	}

	var ptr uintptr
	ret := faiss_IndexRefine_new(&ptr, basePtr, refinePtr)
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexRefine")
	}

	idx := &IndexRefine{
		ptr:         ptr,
		baseIndex:   baseIndex,
		refineIndex: refineIndex,
		d:           baseIndex.D(),
		metric:      baseIndex.MetricType(),
		ntotal:      0,
		isTrained:   false,
		kFactor:     1.0,
	}

	runtime.SetFinalizer(idx, func(idx *IndexRefine) {
		idx.Close()
	})

	return idx, nil
}

// D returns the dimension of the index
func (idx *IndexRefine) D() int {
	return idx.d
}

// Ntotal returns the number of vectors in the index
func (idx *IndexRefine) Ntotal() int64 {
	return idx.baseIndex.Ntotal()
}

// IsTrained returns whether the index has been trained
func (idx *IndexRefine) IsTrained() bool {
	if idx.ptr == 0 {
		return false
	}
	return faiss_Index_is_trained(idx.ptr) != 0
}

// MetricType returns the distance metric used
func (idx *IndexRefine) MetricType() MetricType {
	return idx.metric
}

// SetK_factor sets the factor for base search candidates
func (idx *IndexRefine) SetK_factor(kFactor float32) error {
	if kFactor < 1.0 {
		return fmt.Errorf("k_factor must be >= 1.0")
	}
	ret := faiss_IndexRefine_set_k_factor(idx.ptr, kFactor)
	if ret != 0 {
		return fmt.Errorf("failed to set k_factor")
	}
	idx.kFactor = kFactor
	return nil
}

// Train trains the IndexRefine (which internally trains both base and refine indexes)
func (idx *IndexRefine) Train(vectors []float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("empty training vectors")
	}
	if len(vectors)%idx.d != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension %d", idx.d)
	}

	// Call faiss_Index_train on the IndexRefine pointer
	// FAISS will internally train both base and refine indexes and set is_trained flag
	n := int64(len(vectors) / idx.d)
	ret := faiss_Index_train(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("training failed")
	}

	idx.isTrained = true
	return nil
}

// Add adds vectors through IndexRefine (which delegates to both child indexes)
func (idx *IndexRefine) Add(vectors []float32) error {
	if !idx.IsTrained() {
		return fmt.Errorf("index must be trained before adding vectors")
	}
	if len(vectors)%idx.d != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension %d", idx.d)
	}

	// Call faiss_Index_add on the IndexRefine pointer
	// FAISS will internally add to both base and refine indexes
	n := int64(len(vectors) / idx.d)
	ret := faiss_Index_add(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("add failed")
	}

	idx.ntotal = idx.baseIndex.Ntotal()
	return nil
}

// Search performs two-stage search
func (idx *IndexRefine) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
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

// Reset removes all vectors from both indexes
func (idx *IndexRefine) Reset() error {
	if err := idx.baseIndex.Reset(); err != nil {
		return err
	}
	if err := idx.refineIndex.Reset(); err != nil {
		return err
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexRefine) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// ========================================
// IndexPreTransform - Index with Preprocessing
// ========================================

// IndexPreTransform applies a transformation before indexing
//
// Python equivalent: faiss.IndexPreTransform
//
// Example:
//   // PCA reduction then indexing
//   pca, _ := faiss.NewPCAMatrix(256, 64)
//   baseIndex, _ := faiss.NewIndexFlatL2(64)
//   index, _ := faiss.NewIndexPreTransform(pca, baseIndex)
//
//   // Training trains both PCA and index
//   index.Train(trainingVectors)
//
//   // Vectors are automatically reduced before indexing
//   index.Add(vectors)
type IndexPreTransform struct {
	ptr       uintptr        // C pointer
	transform VectorTransform // preprocessing transform
	index     Index          // underlying index
	dIn       int            // input dimension
	dOut      int            // output dimension (index dimension)
	metric    MetricType
	ntotal    int64
	isTrained bool
}

// Ensure IndexPreTransform implements Index
var _ Index = (*IndexPreTransform)(nil)

// NewIndexPreTransform creates a new index with preprocessing
func NewIndexPreTransform(transform VectorTransform, index Index) (*IndexPreTransform, error) {
	if transform == nil || index == nil {
		return nil, fmt.Errorf("both transform and index must be non-nil")
	}
	if transform.DOut() != index.D() {
		return nil, fmt.Errorf("transform output dimension (%d) must match index dimension (%d)",
			transform.DOut(), index.D())
	}

	// Get pointers based on type
	var transformPtr, indexPtr uintptr

	switch t := transform.(type) {
	case *PCAMatrix:
		transformPtr = t.ptr
	case *OPQMatrix:
		transformPtr = t.ptr
	case *RandomRotationMatrix:
		transformPtr = t.ptr
	default:
		return nil, fmt.Errorf("unsupported transform type")
	}

	switch i := index.(type) {
	case *IndexFlat:
		indexPtr = i.ptr
	case *IndexIVFFlat:
		indexPtr = i.ptr
	case *IndexHNSW:
		indexPtr = i.ptr
	default:
		return nil, fmt.Errorf("unsupported index type")
	}

	var ptr uintptr
	ret := faiss_IndexPreTransform_new(&ptr, transformPtr, indexPtr)
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexPreTransform")
	}

	idx := &IndexPreTransform{
		ptr:       ptr,
		transform: transform,
		index:     index,
		dIn:       transform.DIn(),
		dOut:      transform.DOut(),
		metric:    index.MetricType(),
		ntotal:    0,
		isTrained: false,
	}

	runtime.SetFinalizer(idx, func(idx *IndexPreTransform) {
		idx.Close()
	})

	return idx, nil
}

// D returns the input dimension (before transformation)
func (idx *IndexPreTransform) D() int {
	return idx.dIn
}

// Ntotal returns the number of vectors in the index
func (idx *IndexPreTransform) Ntotal() int64 {
	return idx.index.Ntotal()
}

// IsTrained returns whether both transform and index are trained
func (idx *IndexPreTransform) IsTrained() bool {
	if idx.ptr == 0 {
		return false
	}
	return faiss_Index_is_trained(idx.ptr) != 0
}

// MetricType returns the distance metric used
func (idx *IndexPreTransform) MetricType() MetricType {
	return idx.metric
}

// Train trains the IndexPreTransform (which internally trains transform and index)
func (idx *IndexPreTransform) Train(vectors []float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("empty training vectors")
	}
	if len(vectors)%idx.dIn != 0 {
		return fmt.Errorf("vectors length must be multiple of input dimension %d", idx.dIn)
	}

	// Call faiss_Index_train on the IndexPreTransform pointer
	// FAISS will internally train the transform chain and the underlying index
	n := int64(len(vectors) / idx.dIn)
	ret := faiss_Index_train(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("training failed")
	}

	idx.isTrained = true
	return nil
}

// Add adds vectors after applying transformation (FAISS handles transformation internally)
func (idx *IndexPreTransform) Add(vectors []float32) error {
	if !idx.IsTrained() {
		return fmt.Errorf("index must be trained before adding vectors")
	}
	if len(vectors) == 0 {
		return nil
	}
	if len(vectors)%idx.dIn != 0 {
		return fmt.Errorf("vectors length must be multiple of input dimension %d", idx.dIn)
	}

	// Call faiss_Index_add on the IndexPreTransform pointer
	// FAISS will internally apply transformation via apply_chain() and add to the index
	n := int64(len(vectors) / idx.dIn)
	ret := faiss_Index_add(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("add failed")
	}

	idx.ntotal = idx.index.Ntotal()
	return nil
}

// Search searches after applying transformation to queries (FAISS handles transformation internally)
func (idx *IndexPreTransform) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
	if len(queries) == 0 {
		return nil, nil, fmt.Errorf("empty query vectors")
	}
	if len(queries)%idx.dIn != 0 {
		return nil, nil, fmt.Errorf("queries length must be multiple of input dimension %d", idx.dIn)
	}

	// Call faiss_Index_search on the IndexPreTransform pointer
	// FAISS will internally apply transformation via apply_chain() before searching
	nq := int64(len(queries) / idx.dIn)
	distances = make([]float32, nq*int64(k))
	indices = make([]int64, nq*int64(k))

	ret := faiss_Index_search(idx.ptr, nq, &queries[0], int64(k), &distances[0], &indices[0])
	if ret != 0 {
		return nil, nil, fmt.Errorf("search failed")
	}

	return distances, indices, nil
}

// Reset removes all vectors from the underlying index
func (idx *IndexPreTransform) Reset() error {
	if err := idx.index.Reset(); err != nil {
		return err
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexPreTransform) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}

// ========================================
// IndexShards - Sharded Index
// ========================================

// IndexShards distributes vectors across multiple sub-indexes
//
// Python equivalent: faiss.IndexShards
//
// Example:
//   shards, _ := faiss.NewIndexShards(128, faiss.MetricL2)
//
//   // Add multiple sub-indexes
//   shard1, _ := faiss.NewIndexFlatL2(128)
//   shard2, _ := faiss.NewIndexFlatL2(128)
//   shards.AddShard(shard1)
//   shards.AddShard(shard2)
//
//   // Vectors distributed round-robin across shards
//   shards.Add(vectors)
type IndexShards struct {
	ptr       uintptr  // C pointer
	shards    []Index  // sub-indexes
	d         int      // dimension
	metric    MetricType
	ntotal    int64
	isTrained bool
}

// Ensure IndexShards implements Index
var _ Index = (*IndexShards)(nil)

// NewIndexShards creates a new sharded index
func NewIndexShards(d int, metric MetricType) (*IndexShards, error) {
	if d <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	var ptr uintptr
	ret := faiss_IndexShards_new(&ptr, int64(d), int(metric))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create IndexShards")
	}

	idx := &IndexShards{
		ptr:       ptr,
		shards:    make([]Index, 0),
		d:         d,
		metric:    metric,
		ntotal:    0,
		isTrained: true,
	}

	runtime.SetFinalizer(idx, func(idx *IndexShards) {
		idx.Close()
	})

	return idx, nil
}

// AddShard adds a sub-index to the shards
func (idx *IndexShards) AddShard(shard Index) error {
	if shard == nil {
		return fmt.Errorf("shard cannot be nil")
	}
	if shard.D() != idx.d {
		return fmt.Errorf("shard dimension %d != index dimension %d", shard.D(), idx.d)
	}

	var shardPtr uintptr
	switch s := shard.(type) {
	case *IndexFlat:
		shardPtr = s.ptr
	case *IndexIVFFlat:
		shardPtr = s.ptr
	default:
		return fmt.Errorf("unsupported shard type")
	}

	ret := faiss_IndexShards_add_shard(idx.ptr, shardPtr)
	if ret != 0 {
		return fmt.Errorf("failed to add shard")
	}

	idx.shards = append(idx.shards, shard)
	return nil
}

// D returns the dimension
func (idx *IndexShards) D() int {
	return idx.d
}

// Ntotal returns the total number of vectors across all shards
func (idx *IndexShards) Ntotal() int64 {
	total := int64(0)
	for _, shard := range idx.shards {
		total += shard.Ntotal()
	}
	return total
}

// IsTrained returns whether all shards are trained
func (idx *IndexShards) IsTrained() bool {
	if idx.ptr == 0 {
		return false
	}
	// IndexShards is always considered trained (it's just a container)
	// But we should query FAISS to be consistent
	return faiss_Index_is_trained(idx.ptr) != 0
}

// MetricType returns the distance metric
func (idx *IndexShards) MetricType() MetricType {
	return idx.metric
}

// Train trains all shards via the IndexShards composite index
func (idx *IndexShards) Train(vectors []float32) error {
	if len(idx.shards) == 0 {
		return fmt.Errorf("no shards added")
	}
	if len(vectors) == 0 {
		return fmt.Errorf("empty training vectors")
	}
	if len(vectors)%idx.d != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension %d", idx.d)
	}

	// Call faiss_Index_train on the IndexShards pointer
	// FAISS will internally train all child shards and set is_trained flag
	n := int64(len(vectors) / idx.d)
	ret := faiss_Index_train(idx.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("training failed")
	}

	idx.isTrained = true
	return nil
}

// Add distributes vectors across shards
func (idx *IndexShards) Add(vectors []float32) error {
	if len(idx.shards) == 0 {
		return fmt.Errorf("no shards added")
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

	return nil
}

// Search searches across all shards
func (idx *IndexShards) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
	if len(idx.shards) == 0 {
		return nil, nil, fmt.Errorf("no shards added")
	}
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

// Reset removes all vectors from all shards
func (idx *IndexShards) Reset() error {
	for _, shard := range idx.shards {
		if err := shard.Reset(); err != nil {
			return err
		}
	}
	idx.ntotal = 0
	return nil
}

// Close frees the index
func (idx *IndexShards) Close() error {
	if idx.ptr != 0 {
		faiss_Index_free(idx.ptr)
		idx.ptr = 0
	}
	return nil
}
