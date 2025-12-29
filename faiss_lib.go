//go:build !faiss_use_system
// +build !faiss_use_system

package faiss

/*
// Static library build mode (default)
// Uses pre-built static libraries from libs/ directory
// This is the fastest build mode (~30 seconds)
//
// Supported platforms:
//   - linux/amd64, linux/arm64
//   - darwin/amd64, darwin/arm64
//   - windows/amd64
//
// For other platforms, use: go build -tags=faiss_use_system

#cgo LDFLAGS: -lstdc++ -lm

// Platform-specific library paths and flags
// Linux
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss_c -lfaiss -lopenblas -lgfortran -lgomp -lpthread
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/libs/linux_arm64 -lfaiss_c -lfaiss -lopenblas -lgfortran -lgomp -lpthread

// macOS
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/libs/darwin_amd64 -lfaiss_c -lfaiss -Wl,-framework,Accelerate
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/libs/darwin_arm64 -lfaiss_c -lfaiss -Wl,-framework,Accelerate

// Windows
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/libs/windows_amd64 -lfaiss_c -lfaiss -lopenblas -lgfortran -lquadmath -lpthread

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

// ==== Flat Index Functions ====
extern int faiss_IndexFlatL2_new(FaissIndex* p_index, int64_t d);
extern int faiss_IndexFlatIP_new(FaissIndex* p_index, int64_t d);

// ==== IVF Index Functions ====
extern int faiss_IndexIVFFlat_new(FaissIndex* p_index, FaissIndex quantizer, int64_t d, int64_t nlist, int metric_type);
extern int faiss_IndexIVF_set_nprobe(FaissIndex index, int64_t nprobe);
extern int faiss_IndexIVF_get_nprobe(FaissIndex index, int64_t* nprobe);

// ==== HNSW Index Functions ====
extern int faiss_IndexHNSWFlat_new(FaissIndex* p_index, int64_t d, int M, int metric_type);
extern int faiss_IndexHNSW_set_efConstruction(FaissIndex index, int ef);
extern int faiss_IndexHNSW_set_efSearch(FaissIndex index, int ef);
extern int faiss_IndexHNSW_get_efConstruction(FaissIndex index, int* ef);
extern int faiss_IndexHNSW_get_efSearch(FaissIndex index, int* ef);

// ==== PQ Index Functions ====
extern int faiss_IndexPQ_new(FaissIndex* p_index, int64_t d, int64_t M, int64_t nbits, int metric_type);
extern int faiss_IndexIVFPQ_new(FaissIndex* p_index, FaissIndex quantizer, int64_t d, int64_t nlist, int64_t M, int64_t nbits);

// ==== ID Map Functions ====
extern int faiss_IndexIDMap_new(FaissIndex* p_index, FaissIndex base_index);
extern int faiss_IndexIDMap_add_with_ids(FaissIndex index, int64_t n, const float* x, const int64_t* ids);
extern int faiss_IndexIDMap_remove_ids(FaissIndex index, const int64_t* ids, int64_t n_ids, int64_t* n_removed);

// ==== Common Index Operations ====
extern int faiss_Index_add(FaissIndex index, int64_t n, const float* x);
extern int faiss_Index_add_with_ids(FaissIndex index, int64_t n, const float* x, const int64_t* ids);
extern int faiss_Index_search(FaissIndex index, int64_t n, const float* x, int64_t k, float* distances, int64_t* labels);
extern int faiss_Index_range_search(FaissIndex index, int64_t n, const float* x, float radius, void** p_result);
extern int faiss_RangeSearchResult_get(void* result, int64_t** lims, int64_t** labels, float** distances);
extern void faiss_RangeSearchResult_free(void* result);
extern int faiss_Index_train(FaissIndex index, int64_t n, const float* x);
extern int faiss_Index_assign(FaissIndex index, int64_t n, const float* x, int64_t* labels);
extern int faiss_Index_reconstruct(FaissIndex index, int64_t key, float* recons);
extern int faiss_Index_reconstruct_n(FaissIndex index, int64_t i0, int64_t ni, float* recons);
extern int faiss_Index_reset(FaissIndex index);
extern void faiss_Index_free(FaissIndex index);
extern int64_t faiss_Index_ntotal(FaissIndex index);
extern int faiss_Index_is_trained(FaissIndex index);
extern int faiss_Index_d(FaissIndex index);

// ==== Serialization Functions ====
extern int faiss_write_index(FaissIndex index, const char* filename);
extern int faiss_read_index(const char* filename, FaissIndex* p_index, char* index_type, int* d, int* metric, int64_t* ntotal);
extern int faiss_serialize_index(FaissIndex index, uint8_t** data, size_t* size);
extern int faiss_deserialize_index(const uint8_t* data, size_t size, FaissIndex* p_index, char* index_type, int* d, int* metric, int64_t* ntotal);

// ==== Kmeans Functions ====
extern int faiss_Kmeans_new(FaissKmeans* p_kmeans, int64_t d, int64_t k);
extern int faiss_Kmeans_train(FaissKmeans kmeans, int64_t n, const float* x);
extern int faiss_Kmeans_assign(FaissKmeans kmeans, int64_t n, const float* x, int64_t* labels);
extern int faiss_Kmeans_get_centroids(FaissKmeans kmeans, float* centroids);
extern int faiss_Kmeans_set_niter(FaissKmeans kmeans, int niter);
extern int faiss_Kmeans_set_verbose(FaissKmeans kmeans, int verbose);
extern int faiss_Kmeans_set_seed(FaissKmeans kmeans, int64_t seed);
extern void faiss_Kmeans_free(FaissKmeans kmeans);

// ==== Scalar Quantizer Index Functions ====
extern int faiss_IndexScalarQuantizer_new(FaissIndex* p_index, int64_t d, int qtype, int metric_type);
extern int faiss_IndexIVFScalarQuantizer_new(FaissIndex* p_index, FaissIndex quantizer, int64_t d, int64_t nlist, int qtype, int metric_type);

// ==== Binary Index Functions ====
extern int faiss_IndexBinaryFlat_new(FaissIndexBinary* p_index, int64_t d);
extern int faiss_IndexBinaryIVF_new(FaissIndexBinary* p_index, FaissIndexBinary quantizer, int64_t d, int64_t nlist);
extern int faiss_IndexBinaryHash_new(FaissIndexBinary* p_index, int64_t d, int64_t nbits);
extern int faiss_IndexBinary_add(FaissIndexBinary index, int64_t n, const uint8_t* x);
extern int faiss_IndexBinary_search(FaissIndexBinary index, int64_t n, const uint8_t* x, int64_t k, int32_t* distances, int64_t* labels);
extern int faiss_IndexBinary_train(FaissIndexBinary index, int64_t n, const uint8_t* x);
extern int faiss_IndexBinary_reset(FaissIndexBinary index);
extern int faiss_IndexBinary_ntotal(FaissIndexBinary index, int64_t* ntotal);
extern int faiss_IndexBinary_is_trained(FaissIndexBinary index, int* is_trained);
extern int faiss_IndexBinaryIVF_set_nprobe(FaissIndexBinary index, int64_t nprobe);
extern void faiss_IndexBinary_free(FaissIndexBinary index);

// ==== LSH Index Functions ====
extern int faiss_IndexLSH_new(FaissIndex* p_index, int64_t d, int64_t nbits, int rotate_data, int train_thresholds);

// ==== Vector Transform Functions ====
extern int faiss_PCAMatrix_new(FaissVectorTransform* p_transform, int64_t d_in, int64_t d_out, float eigen_power, int random_rotation);
extern int faiss_OPQMatrix_new(FaissVectorTransform* p_transform, int64_t d, int64_t M);
extern int faiss_RandomRotationMatrix_new(FaissVectorTransform* p_transform, int64_t d_in, int64_t d_out);
extern int faiss_VectorTransform_train(FaissVectorTransform transform, int64_t n, const float* x);
extern int faiss_VectorTransform_apply(FaissVectorTransform transform, int64_t n, const float* x, float* xt);
extern int faiss_VectorTransform_reverse_transform(FaissVectorTransform transform, int64_t n, const float* xt, float* x);
extern void faiss_VectorTransform_free(FaissVectorTransform transform);

// ==== Composite Index Functions ====
extern int faiss_IndexRefine_new(FaissIndex* p_index, FaissIndex base_index, FaissIndex refine_index);
extern int faiss_IndexRefine_set_k_factor(FaissIndex index, float k_factor);
extern int faiss_IndexPreTransform_new(FaissIndex* p_index, FaissVectorTransform transform, FaissIndex base_index);
extern int faiss_IndexShards_new(FaissIndex* p_index, int64_t d, int metric_type);
extern int faiss_IndexShards_add_shard(FaissIndex index, FaissIndex shard);

// ==== PQFastScan Index Functions ====
extern int faiss_IndexPQFastScan_new(FaissIndex* p_index, int64_t d, int64_t M, int64_t nbits, int metric_type);
extern int faiss_IndexIVFPQFastScan_new(FaissIndex* p_index, FaissIndex quantizer, int64_t d, int64_t nlist, int64_t M, int64_t nbits, int metric_type);
extern int faiss_IndexPQFastScan_set_bbs(FaissIndex index, int64_t bbs);

// ==== OnDisk Index Functions ====
extern int faiss_IndexIVFFlatOnDisk_new(FaissIndex* p_index, FaissIndex quantizer, int64_t d, int64_t nlist, const char* filename, int metric_type);
extern int faiss_IndexIVFPQOnDisk_new(FaissIndex* p_index, FaissIndex quantizer, int64_t d, int64_t nlist, int64_t M, int64_t nbits, const char* filename, int metric_type);

// ==== GPU Support Functions ====
typedef void* FaissGpuResources;

extern int faiss_StandardGpuResources_new(FaissGpuResources* p_res);
extern int faiss_StandardGpuResources_setTempMemory(FaissGpuResources res, int64_t bytes);
extern int faiss_StandardGpuResources_setDefaultNullStreamAllDevices(FaissGpuResources res);
extern void faiss_StandardGpuResources_free(FaissGpuResources res);
extern int faiss_GpuIndexFlat_new(FaissIndex* p_index, FaissGpuResources res, int64_t d, int metric_type, int64_t device);
extern int faiss_GpuIndexIVFFlat_new(FaissIndex* p_index, FaissGpuResources res, FaissIndex quantizer, int64_t d, int64_t nlist, int metric_type, int64_t device);
extern int faiss_index_cpu_to_gpu(FaissGpuResources res, int64_t device, FaissIndex cpu_index, FaissIndex* p_gpu_index);
extern int faiss_index_gpu_to_cpu(FaissIndex gpu_index, FaissIndex* p_cpu_index, char* index_type, int* d, int* metric, int64_t* ntotal);
extern int faiss_index_cpu_to_all_gpus(FaissIndex cpu_index, FaissIndex* p_gpu_index);
extern int faiss_get_num_gpus(int* ngpus);

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
)

// faissIndexFlatL2New creates a new IndexFlatL2
func faissIndexFlatL2New(d int) (uintptr, error) {
	var idx C.FaissIndex
	ret := C.faiss_IndexFlatL2_new(&idx, C.int64_t(d))
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
	ret := C.faiss_IndexFlatIP_new(&idx, C.int64_t(d))
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

func faissIndexIVFFlatNew(quantizerPtr uintptr, d, nlist, metric int) (uintptr, error) {
	var idx C.FaissIndex
	quantizer := C.FaissIndex(unsafe.Pointer(quantizerPtr))
	ret := C.faiss_IndexIVFFlat_new(&idx, quantizer, C.int64_t(d), C.int64_t(nlist), C.int(metric))
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, errors.New("null index pointer")
	}
	return uintptr(unsafe.Pointer(idx)), nil
}

func faissIndexIVFSetNprobe(ptr uintptr, nprobe int) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	ret := C.faiss_IndexIVF_set_nprobe(idx, C.int64_t(nprobe))
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// ==== HNSW Index Functions ====

func faissIndexHNSWFlatNew(d, M, metric int) (uintptr, error) {
	var idx C.FaissIndex
	ret := C.faiss_IndexHNSWFlat_new(&idx, C.int64_t(d), C.int(M), C.int(metric))
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, errors.New("null index pointer")
	}
	return uintptr(unsafe.Pointer(idx)), nil
}

func faissIndexHNSWSetEfConstruction(ptr uintptr, ef int) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	ret := C.faiss_IndexHNSW_set_efConstruction(idx, C.int(ef))
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissIndexHNSWSetEfSearch(ptr uintptr, ef int) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	ret := C.faiss_IndexHNSW_set_efSearch(idx, C.int(ef))
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
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

func faissIndexRemoveIDs(ptr uintptr, ids []int64, nids int) (int, error) {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	idPtr := (*C.int64_t)(unsafe.Pointer(&ids[0]))
	var nRemoved C.int64_t
	ret := C.faiss_IndexIDMap_remove_ids(idx, idPtr, C.int64_t(nids), &nRemoved)
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	return int(nRemoved), nil
}

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

func faissIndexAssign(ptr uintptr, vectors []float32, n int, labels []int64) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	labelPtr := (*C.int64_t)(unsafe.Pointer(&labels[0]))
	ret := C.faiss_Index_assign(idx, C.int64_t(n), vecPtr, labelPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// ==== Serialization Functions ====

func faissWriteIndex(ptr uintptr, filename string) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	ret := C.faiss_write_index(idx, cFilename)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissReadIndex(filename string) (uintptr, string, int, int, int64, error) {
	var idx C.FaissIndex
	var indexType [256]C.char
	var d, metric C.int
	var ntotal C.int64_t

	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	ret := C.faiss_read_index(cFilename, &idx, &indexType[0], &d, &metric, &ntotal)
	if ret != 0 {
		return 0, "", 0, 0, 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, "", 0, 0, 0, errors.New("null index pointer")
	}

	return uintptr(unsafe.Pointer(idx)), C.GoString(&indexType[0]), int(d), int(metric), int64(ntotal), nil
}

func faissSerializeIndex(ptr uintptr) ([]byte, error) {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	var data *C.uint8_t
	var size C.size_t

	ret := C.faiss_serialize_index(idx, &data, &size)
	if ret != 0 {
		return nil, fmt.Errorf("FAISS error code: %d", ret)
	}

	// Copy C data to Go slice
	goData := C.GoBytes(unsafe.Pointer(data), C.int(size))
	C.free(unsafe.Pointer(data))

	return goData, nil
}

func faissDeserializeIndex(data []byte) (uintptr, string, int, int, int64, error) {
	var idx C.FaissIndex
	var indexType [256]C.char
	var d, metric C.int
	var ntotal C.int64_t

	dataPtr := (*C.uint8_t)(unsafe.Pointer(&data[0]))
	size := C.size_t(len(data))

	ret := C.faiss_deserialize_index(dataPtr, size, &idx, &indexType[0], &d, &metric, &ntotal)
	if ret != 0 {
		return 0, "", 0, 0, 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, "", 0, 0, 0, errors.New("null index pointer")
	}

	return uintptr(unsafe.Pointer(idx)), C.GoString(&indexType[0]), int(d), int(metric), int64(ntotal), nil
}

// ==== Kmeans Functions ====

func faissKmeansNew(d, k int) (uintptr, error) {
	var kmeans C.FaissKmeans
	ret := C.faiss_Kmeans_new(&kmeans, C.int64_t(d), C.int64_t(k))
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if kmeans == nil {
		return 0, errors.New("null kmeans pointer")
	}
	return uintptr(unsafe.Pointer(kmeans)), nil
}

func faissKmeansTrain(ptr uintptr, vectors []float32, n int) error {
	kmeans := C.FaissKmeans(unsafe.Pointer(ptr))
	vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	ret := C.faiss_Kmeans_train(kmeans, C.int64_t(n), vecPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissKmeansAssign(ptr uintptr, vectors []float32, n int, labels []int64) error {
	kmeans := C.FaissKmeans(unsafe.Pointer(ptr))
	vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	labelPtr := (*C.int64_t)(unsafe.Pointer(&labels[0]))
	ret := C.faiss_Kmeans_assign(kmeans, C.int64_t(n), vecPtr, labelPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissKmeansGetCentroids(ptr uintptr, centroids []float32) error {
	kmeans := C.FaissKmeans(unsafe.Pointer(ptr))
	centPtr := (*C.float)(unsafe.Pointer(&centroids[0]))
	ret := C.faiss_Kmeans_get_centroids(kmeans, centPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissKmeansSetNiter(ptr uintptr, niter int) error {
	kmeans := C.FaissKmeans(unsafe.Pointer(ptr))
	ret := C.faiss_Kmeans_set_niter(kmeans, C.int(niter))
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissKmeansSetVerbose(ptr uintptr, verbose int) error {
	kmeans := C.FaissKmeans(unsafe.Pointer(ptr))
	ret := C.faiss_Kmeans_set_verbose(kmeans, C.int(verbose))
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissKmeansSetSeed(ptr uintptr, seed int64) error {
	kmeans := C.FaissKmeans(unsafe.Pointer(ptr))
	ret := C.faiss_Kmeans_set_seed(kmeans, C.int64_t(seed))
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

func faissKmeansFree(ptr uintptr) error {
	kmeans := C.FaissKmeans(unsafe.Pointer(ptr))
	C.faiss_Kmeans_free(kmeans)
	return nil
}

// ==== PQ Index Functions ====

func faissIndexPQNew(d, M, nbits, metric int) (uintptr, error) {
	var idx C.FaissIndex
	ret := C.faiss_IndexPQ_new(&idx, C.int64_t(d), C.int64_t(M), C.int64_t(nbits), C.int(metric))
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, errors.New("null index pointer")
	}
	return uintptr(unsafe.Pointer(idx)), nil
}

func faissIndexIVFPQNew(quantizerPtr uintptr, d, nlist, M, nbits int) (uintptr, error) {
	var idx C.FaissIndex
	quantizer := C.FaissIndex(unsafe.Pointer(quantizerPtr))
	ret := C.faiss_IndexIVFPQ_new(&idx, quantizer, C.int64_t(d), C.int64_t(nlist),
		C.int64_t(M), C.int64_t(nbits))
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, errors.New("null index pointer")
	}
	return uintptr(unsafe.Pointer(idx)), nil
}

// ==== Range Search Functions ====

func faissIndexRangeSearch(ptr uintptr, queries []float32, nq int, radius float32) (uintptr, []int64, []int64, []float32, error) {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	queryPtr := (*C.float)(unsafe.Pointer(&queries[0]))

	var resultPtr unsafe.Pointer
	ret := C.faiss_Index_range_search(idx, C.int64_t(nq), queryPtr, C.float(radius), &resultPtr)
	if ret != 0 {
		return 0, nil, nil, nil, fmt.Errorf("FAISS error code: %d", ret)
	}

	// Get result arrays
	var cLims, cLabels *C.int64_t
	var cDistances *C.float

	ret2 := C.faiss_RangeSearchResult_get(resultPtr, &cLims, &cLabels, &cDistances)
	if ret2 != 0 {
		C.faiss_RangeSearchResult_free(resultPtr)
		return 0, nil, nil, nil, fmt.Errorf("FAISS error code: %d", ret2)
	}

	// Convert to Go slices (we need to copy the data)
	lims := (*[1 << 30]C.int64_t)(unsafe.Pointer(cLims))[:nq+1:nq+1]
	totalResults := int(lims[nq])

	goLims := make([]int64, nq+1)
	for i := 0; i <= nq; i++ {
		goLims[i] = int64(lims[i])
	}

	var goLabels []int64
	var goDistances []float32

	if totalResults > 0 {
		labels := (*[1 << 30]C.int64_t)(unsafe.Pointer(cLabels))[:totalResults:totalResults]
		distances := (*[1 << 30]C.float)(unsafe.Pointer(cDistances))[:totalResults:totalResults]

		goLabels = make([]int64, totalResults)
		goDistances = make([]float32, totalResults)

		for i := 0; i < totalResults; i++ {
			goLabels[i] = int64(labels[i])
			goDistances[i] = float32(distances[i])
		}
	} else {
		goLabels = []int64{}
		goDistances = []float32{}
	}

	return uintptr(resultPtr), goLims, goLabels, goDistances, nil
}

func faissRangeSearchResultFree(ptr uintptr) {
	C.faiss_RangeSearchResult_free(unsafe.Pointer(ptr))
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

// ==== Binary Index Wrapper Functions ====

func faiss_IndexBinaryFlat_new(p_index *uintptr, d int64) int {
	var idx C.FaissIndexBinary
	ret := C.faiss_IndexBinaryFlat_new(&idx, C.int64_t(d))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexBinaryIVF_new(p_index *uintptr, quantizer uintptr, d, nlist int64) int {
	var idx C.FaissIndexBinary
	q := C.FaissIndexBinary(unsafe.Pointer(quantizer))
	ret := C.faiss_IndexBinaryIVF_new(&idx, q, C.int64_t(d), C.int64_t(nlist))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexBinaryHash_new(p_index *uintptr, d, nbits int64) int {
	var idx C.FaissIndexBinary
	ret := C.faiss_IndexBinaryHash_new(&idx, C.int64_t(d), C.int64_t(nbits))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexBinary_add(index uintptr, n int64, x *uint8) int {
	idx := C.FaissIndexBinary(unsafe.Pointer(index))
	ret := C.faiss_IndexBinary_add(idx, C.int64_t(n), (*C.uint8_t)(unsafe.Pointer(x)))
	return int(ret)
}

func faiss_IndexBinary_search(index uintptr, n int64, x *uint8, k int64, distances *int32, labels *int64) int {
	idx := C.FaissIndexBinary(unsafe.Pointer(index))
	ret := C.faiss_IndexBinary_search(idx, C.int64_t(n), (*C.uint8_t)(unsafe.Pointer(x)),
		C.int64_t(k), (*C.int32_t)(unsafe.Pointer(distances)), (*C.int64_t)(unsafe.Pointer(labels)))
	return int(ret)
}

func faiss_IndexBinary_train(index uintptr, n int64, x *uint8) int {
	idx := C.FaissIndexBinary(unsafe.Pointer(index))
	ret := C.faiss_IndexBinary_train(idx, C.int64_t(n), (*C.uint8_t)(unsafe.Pointer(x)))
	return int(ret)
}

func faiss_IndexBinary_reset(index uintptr) int {
	idx := C.FaissIndexBinary(unsafe.Pointer(index))
	ret := C.faiss_IndexBinary_reset(idx)
	return int(ret)
}

func faiss_IndexBinary_ntotal(index uintptr, ntotal *int64) {
	idx := C.FaissIndexBinary(unsafe.Pointer(index))
	C.faiss_IndexBinary_ntotal(idx, (*C.int64_t)(unsafe.Pointer(ntotal)))
}

func faiss_IndexBinary_is_trained(index uintptr, is_trained *int) {
	idx := C.FaissIndexBinary(unsafe.Pointer(index))
	C.faiss_IndexBinary_is_trained(idx, (*C.int)(unsafe.Pointer(is_trained)))
}

func faiss_IndexBinaryIVF_set_nprobe(index uintptr, nprobe int64) int {
	idx := C.FaissIndexBinary(unsafe.Pointer(index))
	ret := C.faiss_IndexBinaryIVF_set_nprobe(idx, C.int64_t(nprobe))
	return int(ret)
}

func faiss_IndexBinary_free(index uintptr) {
	idx := C.FaissIndexBinary(unsafe.Pointer(index))
	C.faiss_IndexBinary_free(idx)
}

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

func faiss_IndexRefine_new(p_index *uintptr, base_index, refine_index uintptr) int {
	var idx C.FaissIndex
	base := C.FaissIndex(unsafe.Pointer(base_index))
	refine := C.FaissIndex(unsafe.Pointer(refine_index))
	ret := C.faiss_IndexRefine_new(&idx, base, refine)
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexRefine_set_k_factor(index uintptr, k_factor float32) int {
	idx := C.FaissIndex(unsafe.Pointer(index))
	ret := C.faiss_IndexRefine_set_k_factor(idx, C.float(k_factor))
	return int(ret)
}

func faiss_IndexPreTransform_new(p_index *uintptr, transform, base_index uintptr) int {
	var idx C.FaissIndex
	t := C.FaissVectorTransform(unsafe.Pointer(transform))
	base := C.FaissIndex(unsafe.Pointer(base_index))
	ret := C.faiss_IndexPreTransform_new(&idx, t, base)
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexShards_new(p_index *uintptr, d int64, metric_type int) int {
	var idx C.FaissIndex
	ret := C.faiss_IndexShards_new(&idx, C.int64_t(d), C.int(metric_type))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexShards_add_shard(index, shard uintptr) int {
	idx := C.FaissIndex(unsafe.Pointer(index))
	sh := C.FaissIndex(unsafe.Pointer(shard))
	ret := C.faiss_IndexShards_add_shard(idx, sh)
	return int(ret)
}

// ==== IVF Functions ====

func faiss_IndexIVF_set_nprobe(index uintptr, nprobe int64) int {
	idx := C.FaissIndex(unsafe.Pointer(index))
	ret := C.faiss_IndexIVF_set_nprobe(idx, C.int64_t(nprobe))
	return int(ret)
}

// ==== LSH Index Wrapper Functions ====

func faiss_IndexLSH_new(p_index *uintptr, d, nbits int64, rotate_data, train_thresholds bool) int {
	var idx C.FaissIndex
	var rotInt, trainInt C.int
	if rotate_data {
		rotInt = 1
	}
	if train_thresholds {
		trainInt = 1
	}
	ret := C.faiss_IndexLSH_new(&idx, C.int64_t(d), C.int64_t(nbits), rotInt, trainInt)
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

// ==== PQFastScan Index Wrapper Functions ====

func faiss_IndexPQFastScan_new(p_index *uintptr, d, M, nbits int64, metric_type int) int {
	var idx C.FaissIndex
	ret := C.faiss_IndexPQFastScan_new(&idx, C.int64_t(d), C.int64_t(M), C.int64_t(nbits), C.int(metric_type))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexIVFPQFastScan_new(p_index *uintptr, quantizer uintptr, d, nlist, M, nbits int64, metric_type int) int {
	var idx C.FaissIndex
	q := C.FaissIndex(unsafe.Pointer(quantizer))
	ret := C.faiss_IndexIVFPQFastScan_new(&idx, q, C.int64_t(d), C.int64_t(nlist), C.int64_t(M), C.int64_t(nbits), C.int(metric_type))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexPQFastScan_set_bbs(index uintptr, bbs int64) int {
	idx := C.FaissIndex(unsafe.Pointer(index))
	ret := C.faiss_IndexPQFastScan_set_bbs(idx, C.int64_t(bbs))
	return int(ret)
}

// ==== OnDisk Index Wrapper Functions ====

func faiss_IndexIVFFlatOnDisk_new(p_index *uintptr, quantizer uintptr, d, nlist int64, filename string, metric_type int) int {
	var idx C.FaissIndex
	q := C.FaissIndex(unsafe.Pointer(quantizer))
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	ret := C.faiss_IndexIVFFlatOnDisk_new(&idx, q, C.int64_t(d), C.int64_t(nlist), cFilename, C.int(metric_type))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexIVFPQOnDisk_new(p_index *uintptr, quantizer uintptr, d, nlist, M, nbits int64, filename string, metric_type int) int {
	var idx C.FaissIndex
	q := C.FaissIndex(unsafe.Pointer(quantizer))
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	ret := C.faiss_IndexIVFPQOnDisk_new(&idx, q, C.int64_t(d), C.int64_t(nlist), C.int64_t(M), C.int64_t(nbits), cFilename, C.int(metric_type))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

// ==== Scalar Quantizer Index Wrapper Functions ====

func faiss_IndexScalarQuantizer_new(p_index *uintptr, d int64, qtype, metric_type int) int {
	var idx C.FaissIndex
	ret := C.faiss_IndexScalarQuantizer_new(&idx, C.int64_t(d), C.int(qtype), C.int(metric_type))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

func faiss_IndexIVFScalarQuantizer_new(p_index *uintptr, quantizer uintptr, d, nlist int64, qtype, metric_type int) int {
	var idx C.FaissIndex
	q := C.FaissIndex(unsafe.Pointer(quantizer))
	ret := C.faiss_IndexIVFScalarQuantizer_new(&idx, q, C.int64_t(d), C.int64_t(nlist), C.int(qtype), C.int(metric_type))
	if ret == 0 && idx != nil {
		*p_index = uintptr(unsafe.Pointer(idx))
	}
	return int(ret)
}

// ==== Vector Transform Wrapper Functions ====

func faiss_PCAMatrix_new(p_transform *uintptr, d_in, d_out int64, eigen_power float32, random_rotation int) int {
	var transform C.FaissVectorTransform
	ret := C.faiss_PCAMatrix_new(&transform, C.int64_t(d_in), C.int64_t(d_out), C.float(eigen_power), C.int(random_rotation))
	if ret == 0 && transform != nil {
		*p_transform = uintptr(unsafe.Pointer(transform))
	}
	return int(ret)
}

func faiss_OPQMatrix_new(p_transform *uintptr, d, M int64) int {
	var transform C.FaissVectorTransform
	ret := C.faiss_OPQMatrix_new(&transform, C.int64_t(d), C.int64_t(M))
	if ret == 0 && transform != nil {
		*p_transform = uintptr(unsafe.Pointer(transform))
	}
	return int(ret)
}

func faiss_RandomRotationMatrix_new(p_transform *uintptr, d_in, d_out int64) int {
	var transform C.FaissVectorTransform
	ret := C.faiss_RandomRotationMatrix_new(&transform, C.int64_t(d_in), C.int64_t(d_out))
	if ret == 0 && transform != nil {
		*p_transform = uintptr(unsafe.Pointer(transform))
	}
	return int(ret)
}

func faiss_VectorTransform_train(transform uintptr, n int64, x *float32) int {
	t := C.FaissVectorTransform(unsafe.Pointer(transform))
	ret := C.faiss_VectorTransform_train(t, C.int64_t(n), (*C.float)(unsafe.Pointer(x)))
	return int(ret)
}

func faiss_VectorTransform_apply(transform uintptr, n int64, x, xt *float32) int {
	t := C.FaissVectorTransform(unsafe.Pointer(transform))
	ret := C.faiss_VectorTransform_apply(t, C.int64_t(n), (*C.float)(unsafe.Pointer(x)), (*C.float)(unsafe.Pointer(xt)))
	return int(ret)
}

func faiss_VectorTransform_reverse_transform(transform uintptr, n int64, xt, x *float32) int {
	t := C.FaissVectorTransform(unsafe.Pointer(transform))
	ret := C.faiss_VectorTransform_reverse_transform(t, C.int64_t(n), (*C.float)(unsafe.Pointer(xt)), (*C.float)(unsafe.Pointer(x)))
	return int(ret)
}

func faiss_VectorTransform_free(transform uintptr) {
	t := C.FaissVectorTransform(unsafe.Pointer(transform))
	C.faiss_VectorTransform_free(t)
}
