//go:build !faiss_use_system
// +build !faiss_use_system

package faiss

/*
// Static library build mode (default)
// Uses pre-built static libraries from the faiss-go-bindings module
// This is the fastest build mode (~30 seconds)
//
// Supported platforms:
//   - linux/amd64, linux/arm64
//   - darwin/amd64, darwin/arm64
//
// For other platforms, use: go build -tags=faiss_use_system
//
// The faiss-go-bindings module provides:
//   - Pre-built FAISS static libraries
//   - Platform-specific CGO linking configuration
//   - FAISS C API headers
//
// See: https://github.com/NerdMeNot/faiss-go-bindings

// CGO linking is provided by the faiss-go-bindings module (blank import below)
// Only common flags needed here - platform-specific flags come from bindings module
#cgo LDFLAGS: -lstdc++ -lm

#include <stdlib.h>
#include <stdint.h>

// Forward declarations for FAISS C API
// These match the actual FAISS C API but are declared here
// until we generate the real amalgamation

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer types
typedef void* FaissIndex;
typedef void* FaissIndexBinary;
typedef void* FaissVectorTransform;
typedef void* FaissKmeans;
typedef void* FaissIndexIVF;
typedef void* FaissIndexIVFFlat;
typedef void* FaissIndexRefineFlat;
typedef void* FaissIndexPreTransform;
typedef void* FaissIndexShards;

// ==== Flat Index Functions ====
extern int faiss_IndexFlatL2_new_with(FaissIndex* p_index, int64_t d);
extern int faiss_IndexFlatIP_new_with(FaissIndex* p_index, int64_t d);

// ==== IVF Index Functions ====
extern FaissIndexIVF* faiss_IndexIVF_cast(FaissIndex index);
extern int faiss_IndexIVFFlat_new_with_metric(FaissIndexIVFFlat** p_index, FaissIndex quantizer, size_t d, size_t nlist, int metric_type);
extern void faiss_IndexIVF_set_nprobe(FaissIndexIVF* index, size_t nprobe);
extern size_t faiss_IndexIVF_nprobe(FaissIndexIVF* index);  // Note: getter has no "get_" prefix
extern void faiss_IndexIVF_set_own_fields(FaissIndexIVF* index, int own_fields);

// ==== Scalar Quantizer Index Functions ====
extern int faiss_IndexScalarQuantizer_new_with(FaissIndex* p_index, int64_t d, int qtype, int metric_type);
extern int faiss_IndexIVFScalarQuantizer_new_with_metric(FaissIndex* p_index, FaissIndex quantizer, int64_t d, int64_t nlist, int qtype, int metric_type, int encode_residual);
extern void faiss_IndexIVFScalarQuantizer_set_own_fields(FaissIndex index, int own_fields);

// ==== LSH Index Functions ====
// Simple constructor: faiss_IndexLSH_new(FaissIndexLSH** p_index, idx_t d, int nbits)
extern int faiss_IndexLSH_new(FaissIndex** p_index, int64_t d, int nbits);
// Full constructor with options
extern int faiss_IndexLSH_new_with_options(FaissIndex** p_index, int64_t d, int nbits, int rotate_data, int train_thresholds);

// ==== ID Map Functions ====
extern int faiss_IndexIDMap_new(FaissIndex* p_index, FaissIndex base_index);
extern void faiss_IndexIDMap_set_own_fields(FaissIndex index, int own_fields);
extern int faiss_IndexIDMap_add_with_ids(FaissIndex index, int64_t n, const float* x, const int64_t* ids);
// extern int faiss_IndexIDMap_remove_ids(FaissIndex index, const int64_t* ids, int64_t n_ids, int64_t* n_removed); // NOT AVAILABLE

// ==== Common Index Operations ====
extern int faiss_Index_add(FaissIndex index, int64_t n, const float* x);
extern int faiss_Index_add_with_ids(FaissIndex index, int64_t n, const float* x, const int64_t* ids);
extern int faiss_Index_search(FaissIndex index, int64_t n, const float* x, int64_t k, float* distances, int64_t* labels);

// ==== Range Search (using official FAISS C API) ====
// FaissRangeSearchResult is an opaque pointer type
typedef void* FaissRangeSearchResult;
// Create a new RangeSearchResult for nq queries
extern int faiss_RangeSearchResult_new(FaissRangeSearchResult* p_result, int64_t nq);
// Perform range search - result must be pre-allocated with faiss_RangeSearchResult_new
extern int faiss_Index_range_search(FaissIndex index, int64_t n, const float* x, float radius, FaissRangeSearchResult result);
// Get results from RangeSearchResult (individual accessors)
extern int faiss_RangeSearchResult_nq(FaissRangeSearchResult result);
extern size_t faiss_RangeSearchResult_buffer_size(FaissRangeSearchResult result);
extern int64_t* faiss_RangeSearchResult_lims(FaissRangeSearchResult result);
extern int64_t* faiss_RangeSearchResult_labels(FaissRangeSearchResult result);
extern float* faiss_RangeSearchResult_distances(FaissRangeSearchResult result);
// Get all arrays from RangeSearchResult at once (from our extension)
extern int faiss_RangeSearchResult_get(FaissRangeSearchResult result, int64_t** lims, int64_t** labels, float** distances);
extern void faiss_RangeSearchResult_free(FaissRangeSearchResult result);
extern int faiss_Index_train(FaissIndex index, int64_t n, const float* x);
// Note: faiss_Index_assign requires k parameter for number of nearest neighbors to assign
extern int faiss_Index_assign(FaissIndex* index, int64_t n, const float* x, int64_t* labels, int64_t k);
extern int faiss_Index_reconstruct(FaissIndex index, int64_t key, float* recons);
extern int faiss_Index_reconstruct_n(FaissIndex index, int64_t i0, int64_t ni, float* recons);
extern int faiss_Index_reset(FaissIndex index);
extern void faiss_Index_free(FaissIndex index);
extern int64_t faiss_Index_ntotal(FaissIndex index);
extern int faiss_Index_is_trained(FaissIndex index);
extern int faiss_Index_d(FaissIndex index);

// ==== Index Factory ====
extern int faiss_index_factory(FaissIndex* p_index, int d, const char* description, int metric_type);

// ==== Serialization Functions (file-based only) ====
// Note: Using faiss_write_index_fname for file path-based writes (faiss_write_index takes FILE*)
extern int faiss_write_index_fname(const FaissIndex* idx, const char* fname);
// Note: Using faiss_read_index_fname for file path-based reads (faiss_read_index takes FILE*)
extern int faiss_read_index_fname(const char* fname, int io_flags, FaissIndex** p_out);

// ==== Binary Index Functions ====
// Note: Only IndexBinaryFlat_new is available via our extension
// Other binary index constructors have ABI compatibility issues
extern int faiss_IndexBinaryFlat_new(FaissIndexBinary* p_index, int64_t d);
extern int faiss_IndexBinary_add(FaissIndexBinary index, int64_t n, const uint8_t* x);
extern int faiss_IndexBinary_search(FaissIndexBinary index, int64_t n, const uint8_t* x, int64_t k, int32_t* distances, int64_t* labels);
extern int faiss_IndexBinary_train(FaissIndexBinary index, int64_t n, const uint8_t* x);
extern int faiss_IndexBinary_reset(FaissIndexBinary index);
extern int faiss_IndexBinary_ntotal(FaissIndexBinary index, int64_t* ntotal);
extern int faiss_IndexBinary_is_trained(FaissIndexBinary index, int* is_trained);
extern int faiss_IndexBinaryIVF_set_nprobe(FaissIndexBinary index, int64_t nprobe);
extern void faiss_IndexBinary_free(FaissIndexBinary index);

// ==== HNSW Property Accessors (from our extension) ====
extern int faiss_IndexHNSW_set_efConstruction(FaissIndex index, int ef);
extern int faiss_IndexHNSW_set_efSearch(FaissIndex index, int ef);
extern int faiss_IndexHNSW_get_efConstruction(FaissIndex index, int* ef);
extern int faiss_IndexHNSW_get_efSearch(FaissIndex index, int* ef);

// ==== Index Assign (from our extension - works reliably) ====
extern int faiss_Index_assign_ext(FaissIndex index, int64_t n, const float* x, int64_t* labels, int64_t k);

// ==== Vector Transform Functions ====
extern int faiss_PCAMatrix_new_with(FaissVectorTransform* p_transform, int64_t d_in, int64_t d_out, float eigen_power, int random_rotation);
// Note: faiss_OPQMatrix_new_with takes (d, M, d2) - d2 is output dimension (usually same as d)
extern int faiss_OPQMatrix_new_with(FaissVectorTransform* p_transform, int d, int M, int d2);
extern int faiss_RandomRotationMatrix_new_with(FaissVectorTransform* p_transform, int64_t d_in, int64_t d_out);
// VectorTransform functions - using extension wrappers for ABI safety
extern int faiss_VectorTransform_train_ext(FaissVectorTransform vt, int64_t n, const float* x);
extern int faiss_VectorTransform_is_trained_ext(FaissVectorTransform vt, int* trained);
extern int faiss_VectorTransform_apply_noalloc_ext(FaissVectorTransform vt, int64_t n, const float* x, float* xt);
extern int faiss_VectorTransform_reverse_transform_ext(FaissVectorTransform vt, int64_t n, const float* xt, float* x);
extern void faiss_VectorTransform_free(FaissVectorTransform vt);

// ==== Clustering Functions ====
typedef void* FaissClustering;
extern int faiss_Clustering_new(FaissClustering* p_clustering, int d, int k);
extern int faiss_Clustering_train(FaissClustering clustering, int64_t n, const float* x, FaissIndex index);
// Note: faiss_Clustering_centroids returns pointer and size, not accepting pre-allocated buffer
extern void faiss_Clustering_centroids(FaissClustering* clustering, float** centroids, size_t* size);
extern void faiss_Clustering_free(FaissClustering clustering);
// Note: faiss_kmeans_clustering also returns quantization error
extern int faiss_kmeans_clustering(size_t d, size_t n, size_t k, const float* x, float* centroids, float* q_error);

// ==== Composite Index Functions ====
extern int faiss_IndexRefineFlat_new(FaissIndex* p_index, FaissIndex base_index);
extern void faiss_IndexRefineFlat_set_k_factor(FaissIndexRefineFlat index, float k_factor);
extern void faiss_IndexRefineFlat_set_own_fields(FaissIndex index, int own_fields);
extern int faiss_IndexPreTransform_new_with_transform(FaissIndexPreTransform** p_index, FaissVectorTransform* ltrans, FaissIndex* index);
extern void faiss_IndexPreTransform_set_own_fields(FaissIndex index, int own_fields);
extern int faiss_IndexShards_new(FaissIndexShards** p_index, int64_t d);
extern int faiss_IndexShards_add_shard(FaissIndexShards* index, FaissIndex* shard);
// extern void faiss_IndexShards_set_own_indices(FaissIndex index, int own_indices); // NOT AVAILABLE

// ==== GPU Support Functions ====
// NOTE: GPU function declarations are in faiss_gpu.go (requires -tags=gpu to build)

#ifdef __cplusplus
}
#endif

// Helper function to handle errors
static int check_error(int code) {
    return code;
}

*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	_ "github.com/NerdMeNot/faiss-go-bindings" // Links FAISS static libraries
)

// faissIndexFlatL2New creates a new IndexFlatL2
func faissIndexFlatL2New(d int) (uintptr, error) {
	var idx C.FaissIndex
	ret := C.faiss_IndexFlatL2_new_with(&idx, C.int64_t(d))
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, errors.New("null index pointer")
	}
	return uintptr(unsafe.Pointer(idx)), nil
}

// faissIndexFlatIPNew creates a new IndexFlatIP
func faissIndexFlatIPNew(d int) (uintptr, error) {
	var idx C.FaissIndex
	ret := C.faiss_IndexFlatIP_new_with(&idx, C.int64_t(d))
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, errors.New("null index pointer")
	}
	return uintptr(unsafe.Pointer(idx)), nil
}

// faissIndexAdd adds vectors to an index
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	ret := C.faiss_Index_add(idx, C.int64_t(n), vecPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// faissIndexSearch searches for nearest neighbors
func faissIndexSearch(ptr uintptr, queries []float32, nq, k int, distances []float32, indices []int64) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	queryPtr := (*C.float)(unsafe.Pointer(&queries[0]))
	distPtr := (*C.float)(unsafe.Pointer(&distances[0]))
	idxPtr := (*C.int64_t)(unsafe.Pointer(&indices[0]))

	ret := C.faiss_Index_search(idx, C.int64_t(nq), queryPtr, C.int64_t(k), distPtr, idxPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// faissIndexReset resets an index
func faissIndexReset(ptr uintptr) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	ret := C.faiss_Index_reset(idx)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// faissIndexFree frees an index
func faissIndexFree(ptr uintptr) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	C.faiss_Index_free(idx)
	return nil
}

// faissIndexNtotal returns the number of vectors in the index
func faissIndexNtotal(ptr uintptr) int64 {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	return int64(C.faiss_Index_ntotal(idx))
}

// faissIndexIsTrained returns whether the index is trained
func faissIndexIsTrained(ptr uintptr) bool {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	ret := C.faiss_Index_is_trained(idx)
	return ret != 0
}

// getBuildInfo returns build information for source build
func getBuildInfo() BuildInfo {
	return BuildInfo{
		Version:      Version,
		FAISSVersion: FAISSVersion,
		BuildMode:    "static",
		Compiler:     getCompilerVersion(),
		Platform:     fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		BLASBackend:  getBLASBackend(),
	}
}

func getCompilerVersion() string {
	// Pre-compiled static libraries
	return "Pre-built (static libraries)"
}

func getBLASBackend() string {
	switch runtime.GOOS {
	case "darwin":
		return "Accelerate Framework + OpenBLAS"
	case "linux":
		return "OpenBLAS"
	case "windows":
		return "OpenBLAS"
	default:
		return "unknown"
	}
}

// ==== IVF Index Functions ====
// NOTE: NewIndexIVFFlat uses the factory pattern internally (see index_ivf.go).
// Direct IVF constructor removed to avoid C pointer management bugs.

func faissIndexIVFSetNprobe(ptr uintptr, nprobe int) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))

	// Downcast to IndexIVF
	ivf := C.faiss_IndexIVF_cast(idx)
	if ivf == nil {
		return fmt.Errorf("index is not an IVF index (downcast failed)")
	}

	// Set nprobe (void function, no error return)
	C.faiss_IndexIVF_set_nprobe(ivf, C.size_t(nprobe))
	return nil
}

func faissIndexIVFGetNprobe(ptr uintptr) (int, error) {
	idx := C.FaissIndex(unsafe.Pointer(ptr))

	// Downcast to IndexIVF
	ivf := C.faiss_IndexIVF_cast(idx)
	if ivf == nil {
		return 0, fmt.Errorf("index is not an IVF index (downcast failed)")
	}

	// Get nprobe (note: getter function has no "get_" prefix)
	nprobe := C.faiss_IndexIVF_nprobe(ivf)
	return int(nprobe), nil
}

// ==== ID Map Functions ====

func faissIndexIDMapNew(basePtr uintptr) (uintptr, error) {
	var idx C.FaissIndex
	base := C.FaissIndex(unsafe.Pointer(basePtr))
	ret := C.faiss_IndexIDMap_new(&idx, base)
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, errors.New("null index pointer")
	}
	return uintptr(unsafe.Pointer(idx)), nil
}

// faiss_IndexIDMap_set_own_fields sets whether the IDMap index owns its base index
// Setting own_fields=0 prevents FAISS from freeing the base index (Go manages it)
func faiss_IndexIDMap_set_own_fields(ptr uintptr, own int) {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	C.faiss_IndexIDMap_set_own_fields(idx, C.int(own))
}

func faissIndexAddWithIDs(ptr uintptr, vectors []float32, ids []int64, n int) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	idPtr := (*C.int64_t)(unsafe.Pointer(&ids[0]))
	ret := C.faiss_Index_add_with_ids(idx, C.int64_t(n), vecPtr, idPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// NOT AVAILABLE in static library
// func faissIndexRemoveIDs(ptr uintptr, ids []int64, nids int) (int, error) {
// 	idx := C.FaissIndex(unsafe.Pointer(ptr))
// 	idPtr := (*C.int64_t)(unsafe.Pointer(&ids[0]))
// 	var nRemoved C.int64_t
// 	ret := C.faiss_IndexIDMap_remove_ids(idx, idPtr, C.int64_t(nids), &nRemoved)
// 	if ret != 0 {
// 		return 0, fmt.Errorf("FAISS error code: %d", ret)
// 	}
// 	return int(nRemoved), nil
// }

// ==== Training and Assignment ====

func faissIndexTrain(ptr uintptr, vectors []float32, n int) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	ret := C.faiss_Index_train(idx, C.int64_t(n), vecPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissIndexAssign(ptr uintptr, vectors []float32, n int, labels []int64, k int) error {
	// Use our custom extension which passes FaissIndex by value (works reliably)
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	labelPtr := (*C.int64_t)(unsafe.Pointer(&labels[0]))
	ret := C.faiss_Index_assign_ext(idx, C.int64_t(n), vecPtr, labelPtr, C.int64_t(k))
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// ==== Serialization Functions ====

func faissWriteIndex(ptr uintptr, filename string) error {
	if ptr == 0 {
		return errors.New("null index pointer")
	}
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	// Pass pointer to FaissIndex - same pattern as persistence.go
	idxPtr := (*C.FaissIndex)(unsafe.Pointer(ptr))
	ret := C.faiss_write_index_fname(idxPtr, cFilename)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// ==== Index Factory ====

// faissIndexFactory creates an index from a factory description string
// This is the KEY function that unlocks ALL index types including HNSW, PQ, IVFPQ, etc.
func faissIndexFactory(d int, description string, metric int) (uintptr, error) {
	var idx C.FaissIndex
	cDesc := C.CString(description)
	defer C.free(unsafe.Pointer(cDesc))

	ret := C.faiss_index_factory(&idx, C.int(d), cDesc, C.int(metric))
	if ret != 0 {
		return 0, fmt.Errorf("index factory failed for '%s': FAISS error code %d", description, ret)
	}
	if idx == nil {
		return 0, fmt.Errorf("index factory returned null pointer for '%s'", description)
	}

	return uintptr(unsafe.Pointer(idx)), nil
}

// ==== Range Search Functions ====

func faissIndexRangeSearch(ptr uintptr, queries []float32, nq int, radius float32) (uintptr, []int64, []int64, []float32, error) {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	queryPtr := (*C.float)(unsafe.Pointer(&queries[0]))

	// Step 1: Create a RangeSearchResult object for nq queries
	var resultPtr C.FaissRangeSearchResult
	ret := C.faiss_RangeSearchResult_new(&resultPtr, C.int64_t(nq))
	if ret != 0 {
		return 0, nil, nil, nil, fmt.Errorf("faiss_RangeSearchResult_new failed with code %d", ret)
	}

	// Step 2: Perform the range search with pre-allocated result
	ret = C.faiss_Index_range_search(idx, C.int64_t(nq), queryPtr, C.float(radius), resultPtr)
	if ret != 0 {
		C.faiss_RangeSearchResult_free(resultPtr)
		return 0, nil, nil, nil, fmt.Errorf("range_search failed with code %d", ret)
	}

	// Step 3: Get results from the RangeSearchResult
	var cLims, cLabels *C.int64_t
	var cDistances *C.float

	ret2 := C.faiss_RangeSearchResult_get(resultPtr, &cLims, &cLabels, &cDistances)
	if ret2 != 0 {
		C.faiss_RangeSearchResult_free(resultPtr)
		return 0, nil, nil, nil, fmt.Errorf("RangeSearchResult_get failed with code %d", ret2)
	}

	// Convert lims (nq+1 elements)
	lims := make([]int64, nq+1)
	cLimsSlice := (*[1 << 30]C.int64_t)(unsafe.Pointer(cLims))[:nq+1:nq+1]
	for i := 0; i <= nq; i++ {
		lims[i] = int64(cLimsSlice[i])
	}

	// Total number of results
	nTotal := int(lims[nq])
	if nTotal == 0 {
		return uintptr(unsafe.Pointer(resultPtr)), lims, []int64{}, []float32{}, nil
	}

	// Convert labels and distances
	labels := make([]int64, nTotal)
	distances := make([]float32, nTotal)

	cLabelsSlice := (*[1 << 30]C.int64_t)(unsafe.Pointer(cLabels))[:nTotal:nTotal]
	cDistancesSlice := (*[1 << 30]C.float)(unsafe.Pointer(cDistances))[:nTotal:nTotal]

	for i := 0; i < nTotal; i++ {
		labels[i] = int64(cLabelsSlice[i])
		distances[i] = float32(cDistancesSlice[i])
	}

	return uintptr(unsafe.Pointer(resultPtr)), lims, labels, distances, nil
}

func faissRangeSearchResultFree(ptr uintptr) {
	C.faiss_RangeSearchResult_free(C.FaissRangeSearchResult(unsafe.Pointer(ptr)))
}

// ==== Reconstruction Functions ====

func faissIndexReconstruct(ptr uintptr, key int64, recons []float32) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	reconsPtr := (*C.float)(unsafe.Pointer(&recons[0]))
	ret := C.faiss_Index_reconstruct(idx, C.int64_t(key), reconsPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissIndexReconstructN(ptr uintptr, i0, ni int64, recons []float32) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	reconsPtr := (*C.float)(unsafe.Pointer(&recons[0]))
	ret := C.faiss_Index_reconstruct_n(idx, C.int64_t(i0), C.int64_t(ni), reconsPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// ==== Binary Index Functions ====
// NOTE: Binary index support is experimental (see LIMITATIONS.md).
// The binary index type is not currently exposed in the public Go API.
// If you need binary indexes, use the IndexFactory with binary descriptions.

// ==== Generic Index Functions (for composite indexes) ====

func faiss_Index_search(index uintptr, n int64, x *float32, k int64, distances *float32, labels *int64) int {
	idx := C.FaissIndex(unsafe.Pointer(index))
	ret := C.faiss_Index_search(idx, C.int64_t(n), (*C.float)(unsafe.Pointer(x)),
		C.int64_t(k), (*C.float)(unsafe.Pointer(distances)), (*C.int64_t)(unsafe.Pointer(labels)))
	return int(ret)
}

func faiss_Index_add(index uintptr, n int64, x *float32) int {
	idx := C.FaissIndex(unsafe.Pointer(index))
	ret := C.faiss_Index_add(idx, C.int64_t(n), (*C.float)(unsafe.Pointer(x)))
	return int(ret)
}

func faiss_Index_train(index uintptr, n int64, x *float32) int {
	idx := C.FaissIndex(unsafe.Pointer(index))
	ret := C.faiss_Index_train(idx, C.int64_t(n), (*C.float)(unsafe.Pointer(x)))
	return int(ret)
}

func faiss_Index_reset(index uintptr) int {
	idx := C.FaissIndex(unsafe.Pointer(index))
	ret := C.faiss_Index_reset(idx)
	return int(ret)
}

func faiss_Index_ntotal(index uintptr) int64 {
	idx := C.FaissIndex(unsafe.Pointer(index))
	return int64(C.faiss_Index_ntotal(idx))
}

func faiss_Index_is_trained(index uintptr) int {
	idx := C.FaissIndex(unsafe.Pointer(index))
	return int(C.faiss_Index_is_trained(idx))
}

func faiss_Index_free(index uintptr) {
	idx := C.FaissIndex(unsafe.Pointer(index))
	C.faiss_Index_free(idx)
}

// ==== Composite Index Wrapper Functions ====

func faiss_IndexRefineFlat_new(p_index *uintptr, base_index uintptr) int {
	var idx C.FaissIndex
	base := C.FaissIndex(unsafe.Pointer(base_index))
	ret := C.faiss_IndexRefineFlat_new(&idx, base)
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexRefineFlat_set_k_factor(index uintptr, k_factor float32) error {
	idx := C.FaissIndexRefineFlat(unsafe.Pointer(index))
	C.faiss_IndexRefineFlat_set_k_factor(idx, C.float(k_factor))
	return nil
}

func faiss_IndexPreTransform_new(p_index *uintptr, transform, base_index uintptr) int {
	var idx *C.FaissIndexPreTransform
	transformHandle := (*C.FaissVectorTransform)(unsafe.Pointer(transform))
	indexHandle := (*C.FaissIndex)(unsafe.Pointer(base_index))
	ret := C.faiss_IndexPreTransform_new_with_transform(&idx, transformHandle, indexHandle)
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexShards_new(p_index *uintptr, d int64, metric_type int) int {
	var idx *C.FaissIndexShards
	// C API: int faiss_IndexShards_new(FaissIndexShards** p_index, idx_t d)
	// Note: metric_type is NOT used in the C API, it's set via the added indexes
	ret := C.faiss_IndexShards_new(&idx, C.int64_t(d))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexShards_add_shard(index, shard uintptr) int {
	idx := (*C.FaissIndexShards)(unsafe.Pointer(index))
	sh := (*C.FaissIndex)(unsafe.Pointer(shard))
	ret := C.faiss_IndexShards_add_shard(idx, sh)
	return int(ret)
}

// faiss_IndexShards_set_own_indices sets whether IndexShards owns its sub-indexes
// Setting own_indices=0 prevents FAISS from freeing the shards (Go manages them)
// NOT AVAILABLE in static library
// func faiss_IndexShards_set_own_indices(ptr uintptr, own int) {
// 	idx := C.FaissIndex(unsafe.Pointer(ptr))
// 	C.faiss_IndexShards_set_own_indices(idx, C.int(own))
// }

// faiss_IndexRefineFlat_set_own_fields sets whether IndexRefineFlat owns its sub-indexes
// Setting own_fields=0 prevents FAISS from freeing the indexes (Go manages them)
func faiss_IndexRefineFlat_set_own_fields(ptr uintptr, own int) {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	C.faiss_IndexRefineFlat_set_own_fields(idx, C.int(own))
}

// faiss_IndexPreTransform_set_own_fields sets whether IndexPreTransform owns its base index
// Setting own_fields=0 prevents FAISS from freeing the index (Go manages it)
func faiss_IndexPreTransform_set_own_fields(ptr uintptr, own int) {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	C.faiss_IndexPreTransform_set_own_fields(idx, C.int(own))
}

// ==== IVF Functions ====

func faiss_IndexIVF_set_nprobe(index uintptr, nprobe int64) int {
	idx := C.FaissIndex(unsafe.Pointer(index))

	// Downcast to IndexIVF
	ivf := C.faiss_IndexIVF_cast(idx)
	if ivf == nil {
		// Not an IVF index, return error code
		return -1
	}

	// Set nprobe (void function, no error return)
	C.faiss_IndexIVF_set_nprobe(ivf, C.size_t(nprobe))
	return 0 // Success
}

// ==== Vector Transform Wrapper Functions ====

func faiss_PCAMatrix_new_with(p_transform *uintptr, d_in, d_out int64, eigen_power float32, random_rotation int) int {
	var transform C.FaissVectorTransform
	ret := C.faiss_PCAMatrix_new_with(&transform, C.int64_t(d_in), C.int64_t(d_out), C.float(eigen_power), C.int(random_rotation))
	if ret == 0 && transform != nil {
		*p_transform = uintptr(unsafe.Pointer(transform))
	}
	return int(ret)
}

func faiss_OPQMatrix_new_with(p_transform *uintptr, d, M, d2 int) int {
	var transform C.FaissVectorTransform
	// Note: FAISS C API uses (d, M, d2) not (d, M, niter)
	ret := C.faiss_OPQMatrix_new_with(&transform, C.int(d), C.int(M), C.int(d2))
	if ret == 0 && transform != nil {
		*p_transform = uintptr(unsafe.Pointer(transform))
	}
	return int(ret)
}

func faiss_RandomRotationMatrix_new_with(p_transform *uintptr, d_in, d_out int64) int {
	var transform C.FaissVectorTransform
	ret := C.faiss_RandomRotationMatrix_new_with(&transform, C.int64_t(d_in), C.int64_t(d_out))
	if ret == 0 && transform != nil {
		*p_transform = uintptr(unsafe.Pointer(transform))
	}
	return int(ret)
}

func faiss_VectorTransform_train(transform uintptr, n int64, x *float32) int {
	t := C.FaissVectorTransform(unsafe.Pointer(transform))
	ret := C.faiss_VectorTransform_train_ext(t, C.int64_t(n), (*C.float)(unsafe.Pointer(x)))
	return int(ret)
}

func faiss_VectorTransform_apply(transform uintptr, n int64, x, xt *float32) {
	t := C.FaissVectorTransform(unsafe.Pointer(transform))
	C.faiss_VectorTransform_apply_noalloc_ext(t, C.int64_t(n), (*C.float)(unsafe.Pointer(x)), (*C.float)(unsafe.Pointer(xt)))
}

func faiss_VectorTransform_reverse_transform(transform uintptr, n int64, xt, x *float32) {
	t := C.FaissVectorTransform(unsafe.Pointer(transform))
	C.faiss_VectorTransform_reverse_transform_ext(t, C.int64_t(n), (*C.float)(unsafe.Pointer(xt)), (*C.float)(unsafe.Pointer(x)))
}

func faiss_VectorTransform_free(transform uintptr) {
	t := C.FaissVectorTransform(unsafe.Pointer(transform))
	C.faiss_VectorTransform_free(t)
}

// ==== LSH Index Wrapper Functions ====

func faiss_IndexLSH_new(p_index *uintptr, d, nbits int64, rotate_data, train_thresholds bool) int {
	var idx *C.FaissIndex
	var ret C.int

	if rotate_data || train_thresholds {
		// Use full constructor with options
		var rotInt, trainInt C.int
		if rotate_data {
			rotInt = 1
		}
		if train_thresholds {
			trainInt = 1
		}
		ret = C.faiss_IndexLSH_new_with_options(&idx, C.int64_t(d), C.int(nbits), rotInt, trainInt)
	} else {
		// Use simple constructor (more likely to work)
		ret = C.faiss_IndexLSH_new(&idx, C.int64_t(d), C.int(nbits))
	}

	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

// ==== Scalar Quantizer Index Wrapper Functions ====

func faiss_IndexScalarQuantizer_new(p_index *uintptr, d int64, qtype, metric_type int) int {
	var idx C.FaissIndex
	ret := C.faiss_IndexScalarQuantizer_new_with(&idx, C.int64_t(d), C.int(qtype), C.int(metric_type))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexIVFScalarQuantizer_new(p_index *uintptr, quantizer uintptr, d, nlist int64, qtype, metric_type int) int {
	var idx C.FaissIndex
	q := C.FaissIndex(unsafe.Pointer(quantizer))
	// encode_residual=1 is the default
	ret := C.faiss_IndexIVFScalarQuantizer_new_with_metric(&idx, q, C.int64_t(d), C.int64_t(nlist), C.int(qtype), C.int(metric_type), 1)
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

// ========================================
// Clustering Functions
// ========================================

func faiss_kmeans_clustering(d, n, k int, x []float32, centroids []float32) error {
	if len(x) == 0 || len(centroids) == 0 {
		return fmt.Errorf("empty input")
	}
	var qError C.float // quantization error (ignored in current API)
	ret := C.faiss_kmeans_clustering(
		C.size_t(d),
		C.size_t(n),
		C.size_t(k),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&centroids[0])),
		&qError,
	)
	if ret != 0 {
		return fmt.Errorf("kmeans_clustering failed with code %d", ret)
	}
	return nil
}

// ==== Clustering Functions ====
// NOTE: The Kmeans type uses faiss_kmeans_clustering directly.
// Low-level Clustering API removed as it's not used.

// ==== HNSW Property Accessors ====

func faissIndexHNSWSetEfSearch(ptr uintptr, ef int) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	ret := C.faiss_IndexHNSW_set_efSearch(idx, C.int(ef))
	if ret != 0 {
		return fmt.Errorf("failed to set efSearch: error code %d", ret)
	}
	return nil
}

// Note: Byte-level serialization functions removed due to ABI compatibility issues.
// Use persistence.go (WriteIndexToFile, ReadIndexFromFile) for serialization.
