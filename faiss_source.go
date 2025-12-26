//go:build !faiss_use_lib
// +build !faiss_use_lib

package faiss

/*
// FAISS Installation Requirements:
// This build mode requires FAISS to be installed on your system.
//
// Installation:
//   Ubuntu/Debian: apt-get install libfaiss-dev
//   macOS: brew install faiss
//   From source: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
//
// The C++ bridge implementation (faiss/faiss_c_impl.cpp) will be compiled
// and linked against your system's FAISS installation.

#cgo CPPFLAGS: -I${SRCDIR}/faiss -std=c++17 -O3 -Wall -Wextra
#cgo CXXFLAGS: -std=c++17 -O3 -fopenmp
#cgo LDFLAGS: -lfaiss -lgomp -lstdc++ -lm

// Linux-specific flags
#cgo linux CPPFLAGS: -I/usr/include -I/usr/local/include
#cgo linux LDFLAGS: -L/usr/lib -L/usr/local/lib -lopenblas -lgfortran

// macOS-specific flags
#cgo darwin CPPFLAGS: -I/opt/homebrew/include -I/usr/local/include -I/opt/homebrew/opt/openblas/include
#cgo darwin LDFLAGS: -L/opt/homebrew/lib -L/usr/local/lib -L/opt/homebrew/opt/openblas/lib -lopenblas -Wl,-framework,Accelerate

// Windows-specific flags
#cgo windows CPPFLAGS: -IC:/faiss/include
#cgo windows LDFLAGS: -LC:/faiss/lib -lopenblas -lgfortran -lquadmath -lpthread

// Compile our C++ bridge implementation
// This file uses FAISS's C++ API and exposes a C interface for CGO

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
		BuildMode:    "source",
		Compiler:     getCompilerVersion(),
		Platform:     fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		BLASBackend:  getBLASBackend(),
	}
}

func getCompilerVersion() string {
	// This will be populated during build
	return "GCC/Clang (from amalgamation build)"
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
