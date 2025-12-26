//go:build !faiss_use_lib
// +build !faiss_use_lib

package faiss

/*
#cgo CPPFLAGS: -I${SRCDIR}/faiss -std=c++17 -O3 -Wall -Wextra
#cgo CXXFLAGS: -std=c++17 -O3 -fopenmp
#cgo LDFLAGS: -lgomp -lstdc++ -lm

// Linux-specific flags
#cgo linux LDFLAGS: -lopenblas -lgfortran

// macOS-specific flags
#cgo darwin CPPFLAGS: -I/opt/homebrew/opt/openblas/include -I/usr/local/opt/openblas/include
#cgo darwin LDFLAGS: -L/opt/homebrew/opt/openblas/lib -L/usr/local/opt/openblas/lib -lopenblas -Wl,-framework,Accelerate

// Windows-specific flags
#cgo windows LDFLAGS: -lopenblas -lgfortran -lquadmath -lpthread

// Source files - these will be compiled by CGO
// Note: faiss.cpp is the amalgamated source file
// We'll include it via a separate C++ wrapper to control compilation

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

// Index creation functions (placeholders - will be replaced with real FAISS C API)
// These are just declarations for now until we have the real amalgamation
extern int faiss_IndexFlatL2_new(FaissIndex* p_index, int64_t d);
extern int faiss_IndexFlatIP_new(FaissIndex* p_index, int64_t d);
extern int faiss_Index_add(FaissIndex index, int64_t n, const float* x);
extern int faiss_Index_search(FaissIndex index, int64_t n, const float* x, int64_t k, float* distances, int64_t* labels);
extern int faiss_Index_reset(FaissIndex index);
extern void faiss_Index_free(FaissIndex index);
extern int64_t faiss_Index_ntotal(FaissIndex index);
extern int faiss_Index_is_trained(FaissIndex index);
extern int faiss_Index_d(FaissIndex index);

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
